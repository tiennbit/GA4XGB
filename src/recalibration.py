# -*- coding: utf-8 -*-
"""Tầng 1 — mục 1.1: baseline hiệu chỉnh hậu kỳ (post-hoc recalibration).

VÌ SAO PHẢI CÓ: phản biện chỉ ra rằng nếu mô hình co dự đoán về trung bình, thì
chỉ cần GIÃN dự đoán ra là đủ — không cần GA, không cần fitness mới, không cần
gene thứ tám. Nếu một phép biến đổi một tham số đạt được cùng profile đuôi như
GA4XGB(alpha=1,+LW), thì câu "joint search reaches a strictly more tail-favorable
point" (Section IV-F) KHÔNG đứng vững và phải viết lại.

Ba biến thể, tất cả hiệu chuẩn CHỈ trên out-of-fold prediction của tập train:

  linear   ŷ' = μ + s(ŷ − μ), s chọn trên lưới để tối thiểu macro-MAE (trung bình
           MAE của 3 vùng) — đối xứng với chính tiêu chí mà alpha=1 tối ưu.
  isotonic ŷ' = g(ŷ) với g là hồi quy đơn điệu của y theo ŷ. Đây là hiệu chuẩn
           theo nghĩa E[Y|Ŷ] — nó CO THÊM chứ không giãn ra. Có mặt ở đây để đo
           hướng ngược lại của đánh đổi, không phải như một đối thủ.
  quantile ánh xạ theo hạng: ŷ' = F_y^{-1}(F_ŷ(ŷ)) — khớp phân phối dự đoán với
           phân phối điểm thật. Đây là cách khử co rút "đúng" về mặt phân phối.

Chạy: PYTHONPATH=src python3 src/recalibration.py [--n-jobs 2]
Kết quả: results_mseed/recalibration.json
"""
import argparse
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

import bins
import review_common as rc

SCALE_GRID = np.round(np.arange(1.00, 2.001, 0.01), 2)


def region_mae(y_true, pred):
    y_true, pred = np.asarray(y_true, float), np.asarray(pred, float)
    err = np.abs(pred - y_true)
    return {k: float(err[m].mean()) for k, m in bins.regions(y_true).items()}


def macro_mae(y_true, pred):
    """Trung bình KHÔNG trọng số của MAE 3 vùng — tiêu chí chọn s."""
    r = region_mae(y_true, pred)
    return float(np.mean([r["Low tail"], r["Middle"], r["High tail"]]))


def oof_predictions(xgb_params, beta, X_tr, y_tr, n_jobs):
    """Dự đoán out-of-fold trên tập train, dùng fold seed RIÊNG (xem review_common)."""
    oof = np.zeros(len(y_tr), dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=rc.OOF_FOLD_SEED)
    for tr, va in kf.split(X_tr):
        oof[va] = rc.fit_predict(xgb_params, beta,
                                 X_tr.iloc[tr], y_tr.iloc[tr], X_tr.iloc[va], n_jobs)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=2)
    args = ap.parse_args()

    X_tr, X_te, y_tr, y_te = rc.get_data()
    ytr_v, yte_v = y_tr.to_numpy(float), y_te.to_numpy(float)
    mu = float(ytr_v.mean())

    per_seed, sweep_per_seed = [], []
    for seed in rc.SEEDS:
        xgb_params, beta = rc.load_params("ga_rmse", seed)
        assert beta is None, "ga_rmse không được có gene beta"

        oof = oof_predictions(xgb_params, beta, X_tr, y_tr, args.n_jobs)
        pred = rc.fit_predict(xgb_params, beta, X_tr, y_tr, X_te, args.n_jobs)

        # --- linear: chọn s trên OOF của TRAIN, không đụng test ---
        costs = [macro_mae(ytr_v, mu + s * (oof - mu)) for s in SCALE_GRID]
        s_star = float(SCALE_GRID[int(np.argmin(costs))])

        # --- isotonic: g khớp trên OOF ---
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof, ytr_v)

        # --- quantile mapping: hạng của ŷ -> phân vị tương ứng của y (đều từ train) ---
        oof_sorted = np.sort(oof)
        y_sorted = np.sort(ytr_v)
        def qmap(p):
            r = np.searchsorted(oof_sorted, p, side="left") / max(len(oof_sorted) - 1, 1)
            return np.interp(np.clip(r, 0, 1), np.linspace(0, 1, len(y_sorted)), y_sorted)

        variants = {
            "raw":             pred,
            "linear":          mu + s_star * (pred - mu),
            "isotonic":        iso.predict(pred),
            "quantile_map":    qmap(pred),
        }
        row = {"seed": seed, "s_star": s_star,
               "regions": {k: region_mae(yte_v, p) for k, p in variants.items()}}
        per_seed.append(row)

        # quét s để phủ lên frontier của Fig. 6(c)
        sweep_per_seed.append({str(s): region_mae(yte_v, mu + s * (pred - mu))
                               for s in np.round(np.arange(1.0, 2.01, 0.05), 2)})
        print(f"[recal] seed {seed}: s*={s_star:.2f} "
              f"low {row['regions']['linear']['Low tail']:.3f} "
              f"high {row['regions']['linear']['High tail']:.3f} "
              f"all {row['regions']['linear']['All']:.3f}", flush=True)

    summary = {}
    for var in ("raw", "linear", "isotonic", "quantile_map"):
        summary[var] = {reg: rc.ms([r["regions"][var][reg] for r in per_seed])
                        for reg in bins.regions(yte_v)}
    sweep = {s: {reg: rc.ms([d[s][reg] for d in sweep_per_seed])
                 for reg in bins.regions(yte_v)}
             for s in sweep_per_seed[0]}

    out = {"protocol": ("s và g hiệu chuẩn CHỈ trên out-of-fold prediction của tập "
                        f"train, fold seed {rc.OOF_FOLD_SEED} (khác phân hoạch GA đã "
                        "tối ưu). Test chỉ dùng một lần để báo cáo."),
           "s_star": rc.ms([r["s_star"] for r in per_seed]),
           "per_seed": per_seed, "summary": summary, "scale_sweep": sweep}
    with open(f"{rc.MSEED_DIR}/recalibration.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\ns* = {out['s_star']['mean']:.3f} ± {out['s_star']['sd']:.3f}")
    print(f"{'variant':<14}{'Low tail':>18}{'Middle':>18}{'High tail':>18}{'All':>18}")
    for var, d in summary.items():
        cells = "".join(f"{d[r]['mean']:>11.3f} ±{d[r]['sd']:.3f}"
                        for r in ("Low tail", "Middle", "High tail", "All"))
        print(f"{var:<14}{cells}")


if __name__ == "__main__":
    main()
