# -*- coding: utf-8 -*-
"""Tầng 1 — mục 1.5: bảng hiệu chuẩn (calibration) của dự đoán.

VÌ SAO PHẢI CÓ: phản biện lĩnh vực đo lường chỉ ra hai điều bài chưa xử lý.

(1) Co rút về trung bình KHÔNG bắt nguồn từ loss bình phương như Section IV-F
    khẳng định, mà là hệ quả tất yếu của Var(E[Y|X]) = R²·Var(Y) khi tương quan
    < 1. Cột sd_ratio dưới đây đo trực tiếp mức co rút đó.

(2) Mọi phân tích trong bài chia bin theo ĐIỂM THẬT y. Với một mô hình hoàn toàn
    không thiên lệch, E[Ŷ − Y | Y ∈ bin] LUÔN dương ở bin thấp — nên độ chệch
    theo bin-của-y không phải bằng chứng mô hình sai. Khung đúng cho quyết định
    sàng lọc là điều kiện theo Ŷ (tư vấn viên thấy dự đoán, chưa thấy điểm thật).
    Vì vậy ở đây báo cáo MAE theo bin CỦA Ŷ song song với bin của y.

Dự đoán trước khi chạy: cấu hình loss-weighted sẽ THẮNG khi bin theo y và THUA
khi bin theo ŷ. Nếu đúng, đó là cách phát biểu chính xác hơn nhiều về việc alpha
và beta thật sự điều khiển cái gì.

Chạy: PYTHONPATH=src python3 src/calibration_table.py [--n-jobs 2]
Kết quả: results_mseed/calibration_table.json
"""
import argparse
import json

import numpy as np

import bins
import review_common as rc
from recalibration import oof_predictions, macro_mae, SCALE_GRID

METHODS = ["default", "ga_rmse", "ga_tail_a1", "ga_tail_a1_lw", "ga_rmse_recal"]


def calib_stats(y_true, pred):
    """sd_ratio, calibration slope/intercept của hồi quy y ~ a + b·ŷ.

    b = 1 nghĩa là hiệu chuẩn hoàn hảo theo nghĩa điều kiện-theo-ŷ.
    b > 1 nghĩa là dự đoán bị CO (under-dispersed) — cần giãn ra.
    b < 1 nghĩa là dự đoán bị GIÃN quá (over-dispersed)."""
    b, a = np.polyfit(pred, y_true, 1)
    return {"sd_ratio": float(pred.std(ddof=1) / y_true.std(ddof=1)),
            "calib_slope": float(b), "calib_intercept": float(a),
            "corr": float(np.corrcoef(pred, y_true)[0, 1])}


def mae_by_bin(y_true, pred, by):
    """MAE theo bin. by='y' -> chia theo điểm thật; by='pred' -> chia theo dự đoán."""
    idx = bins.bin_index(y_true if by == "y" else pred)
    err = np.abs(pred - y_true)
    out = {}
    for i, lab in enumerate(bins.LABELS):
        m = idx == i
        out[lab] = {"n": int(m.sum()),
                    "mae": float(err[m].mean()) if m.sum() else None}
    return out


def region_by(y_true, pred, by):
    ref = y_true if by == "y" else pred
    err = np.abs(pred - y_true)
    return {k: float(err[m].mean()) if m.sum() else None
            for k, m in bins.regions(ref).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=2)
    args = ap.parse_args()

    X_tr, X_te, y_tr, y_te = rc.get_data()
    ytr_v, yte_v = y_tr.to_numpy(float), y_te.to_numpy(float)
    mu = float(ytr_v.mean())

    per_seed = {m: [] for m in METHODS}
    for seed in rc.SEEDS:
        preds = {"default": rc.fit_predict({}, None, X_tr, y_tr, X_te, args.n_jobs)}
        for name in ("ga_rmse", "ga_tail_a1", "ga_tail_a1_lw"):
            xp, beta = rc.load_params(name, seed)
            preds[name] = rc.fit_predict(xp, beta, X_tr, y_tr, X_te, args.n_jobs)
        xp, beta = rc.load_params("ga_rmse", seed)
        oof = oof_predictions(xp, beta, X_tr, y_tr, args.n_jobs)
        s_star = float(SCALE_GRID[int(np.argmin(
            [macro_mae(ytr_v, mu + s * (oof - mu)) for s in SCALE_GRID]))])
        preds["ga_rmse_recal"] = mu + s_star * (preds["ga_rmse"] - mu)

        for m, p in preds.items():
            per_seed[m].append({
                "calib": calib_stats(yte_v, p),
                "region_by_y": region_by(yte_v, p, "y"),
                "region_by_pred": region_by(yte_v, p, "pred"),
                "bin_by_y": mae_by_bin(yte_v, p, "y"),
                "bin_by_pred": mae_by_bin(yte_v, p, "pred")})
        print(f"[calib] seed {seed} xong", flush=True)

    summary = {}
    for m in METHODS:
        rows = per_seed[m]
        summary[m] = {
            "calib": {k: rc.ms([r["calib"][k] for r in rows])
                      for k in rows[0]["calib"]},
            "region_by_y": {k: rc.ms([r["region_by_y"][k] for r in rows])
                            for k in rows[0]["region_by_y"]},
            "region_by_pred": {k: rc.ms([r["region_by_pred"][k] for r in rows])
                               for k in rows[0]["region_by_pred"]},
            "bin_by_y": {lab: rc.ms([r["bin_by_y"][lab]["mae"] for r in rows])
                         for lab in bins.LABELS},
            "bin_by_pred": {lab: rc.ms([r["bin_by_pred"][lab]["mae"] for r in rows])
                            for lab in bins.LABELS}}

    with open(f"{rc.MSEED_DIR}/calibration_table.json", "w") as f:
        json.dump({"protocol": "5 seed, mean ± sd; sd_y_test = %.4f" % yte_v.std(ddof=1),
                   "sd_y_test": float(yte_v.std(ddof=1)),
                   "summary": summary}, f, indent=2, ensure_ascii=False)

    print(f"\n{'method':<16}{'sd_ratio':>12}{'calib_slope':>14}"
          f"{'low|by y':>12}{'low|by ŷ':>12}{'high|by y':>12}{'high|by ŷ':>12}")
    for m in METHODS:
        d = summary[m]
        print(f"{m:<16}{d['calib']['sd_ratio']['mean']:>12.3f}"
              f"{d['calib']['calib_slope']['mean']:>14.3f}"
              f"{d['region_by_y']['Low tail']['mean']:>12.3f}"
              f"{d['region_by_pred']['Low tail']['mean']:>12.3f}"
              f"{d['region_by_y']['High tail']['mean']:>12.3f}"
              f"{d['region_by_pred']['High tail']['mean']:>12.3f}")


if __name__ == "__main__":
    main()
