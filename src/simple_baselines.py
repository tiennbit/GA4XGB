# -*- coding: utf-8 -*-
"""Tầng 1 — mục 1.7 + 1.8: baseline đơn giản và baseline đổi LOSS.

VÌ SAO PHẢI CÓ:

1.7 — Baseline yếu nhất trong bài hiện là default XGBoost (R² = 0,4756). Câu hỏi
      đầu tiên của mọi reviewer: OLS/ridge trên 186 feature cho R² bao nhiêu? Nếu
      0,49 thì toàn bộ bộ máy GA + XGBoost mua được 0,02 R². Không trả lời được
      câu này thì claim "hyperparameter optimization matters" không có mốc so.
      Thêm cả mean-predictor và một OLS 3 biến (GPA lớp 12, Toán cuối lớp 11,
      cờ trường chuyên) để biết học bạ thô một mình đi được tới đâu.

1.8 — Bài lập luận (Section IV-F) rằng co rút về trung bình bắt nguồn từ loss
      bình phương. Kết luận tự nhiên của lập luận đó là ĐỔI LOSS, không phải đổi
      fitness của search. Quantile regression và MAE loss là hai cách chuẩn, có
      sẵn trong XGBoost, không cần tune, và bài chưa bao giờ nhắc tới (từ
      "quantile" xuất hiện 0 lần). Đây là baseline cạnh tranh trực tiếp nhất với
      chính cơ chế mà bài đề xuất.

Chạy: PYTHONPATH=src python3 src/simple_baselines.py [--n-jobs 8]
Kết quả: results_mseed/simple_baselines.json
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import bins
import review_common as rc

# Ba biến mà một giáo viên chủ nhiệm có thể tra trong 30 giây: điểm tổng kết cả
# năm lớp 12, điểm Toán cả năm lớp 11 (feature mạnh nhất theo Fig. 7), cờ trường
# chuyên. Nếu OLS trên ba biến này đã gần R² của mô hình đầy đủ thì phải nói ra.
SIMPLE_COLS = ["12.Điểm tổng kết CN", "11.Toán CN", "truong_chuyen"]
QUANTILES = [0.1, 0.5, 0.9]


def evaluate(y_true, pred):
    y_true, pred = np.asarray(y_true, float), np.asarray(pred, float)
    err = np.abs(pred - y_true)
    out = {"rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
           "mae": float(mean_absolute_error(y_true, pred)),
           "r2": float(r2_score(y_true, pred))}
    out["regions"] = {k: float(err[m].mean()) for k, m in bins.regions(y_true).items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    X_tr, X_te, y_tr, y_te = rc.get_data()
    ytr_v, yte_v = y_tr.to_numpy(float), y_te.to_numpy(float)
    res = {}

    # ---------- 1.7 baseline đơn giản (tất định, không cần seed) ----------
    res["mean_predictor"] = evaluate(yte_v, np.full(len(yte_v), ytr_v.mean()))

    Xtr_n = X_tr.astype(float).to_numpy()
    Xte_n = X_te.astype(float).to_numpy()
    res["ols_full"] = evaluate(yte_v, LinearRegression().fit(Xtr_n, ytr_v).predict(Xte_n))
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xtr_n, ytr_v)
    res["ridge_full"] = evaluate(yte_v, ridge.predict(Xte_n))
    res["ridge_full"]["alpha"] = float(ridge.alpha_)

    have = [c for c in SIMPLE_COLS if c in X_tr.columns]
    res["_simple_cols_used"] = have
    if len(have) == len(SIMPLE_COLS):
        a = X_tr[have].astype(float).to_numpy()
        b = X_te[have].astype(float).to_numpy()
        res["ols_3feat"] = evaluate(yte_v, LinearRegression().fit(a, ytr_v).predict(b))
    else:
        missing = [c for c in SIMPLE_COLS if c not in X_tr.columns]
        res["ols_3feat"] = {"skipped": f"thiếu cột: {missing}"}
        print(f"[warn] bỏ ols_3feat, thiếu cột: {missing}", flush=True)

    # ---------- 1.8 baseline đổi LOSS, dùng cấu hình GA-RMSE của từng seed ----------
    per_seed = {f"quantile_q{q:g}": [] for q in QUANTILES}
    per_seed["mae_loss"] = []
    per_seed["quantile_spread_q10_q90"] = []
    for seed in rc.SEEDS:
        xgb_params, beta = rc.load_params("ga_rmse", seed)
        assert beta is None

        m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=args.n_jobs,
                         verbosity=0, objective="reg:absoluteerror", **xgb_params)
        m.fit(X_tr, y_tr)
        per_seed["mae_loss"].append(evaluate(yte_v, m.predict(X_te)))

        qpred = {}
        for q in QUANTILES:
            m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=args.n_jobs,
                             verbosity=0, objective="reg:quantileerror",
                             quantile_alpha=q, **xgb_params)
            m.fit(X_tr, y_tr)
            qpred[q] = np.asarray(m.predict(X_te), float).ravel()
            per_seed[f"quantile_q{q:g}"].append(evaluate(yte_v, qpred[q]))
        # độ rộng khoảng dự báo 10–90% — dùng cho future work về prediction intervals
        cov = float(((yte_v >= qpred[0.1]) & (yte_v <= qpred[0.9])).mean())
        per_seed["quantile_spread_q10_q90"].append(
            {"mean_width": float((qpred[0.9] - qpred[0.1]).mean()), "coverage": cov})
        print(f"[baselines] seed {seed} xong", flush=True)

    agg = {}
    for name, rows in per_seed.items():
        if name == "quantile_spread_q10_q90":
            agg[name] = {k: rc.ms([r[k] for r in rows]) for k in rows[0]}
            continue
        agg[name] = {k: rc.ms([r[k] for r in rows]) for k in ("rmse", "mae", "r2")}
        agg[name]["regions"] = {k: rc.ms([r["regions"][k] for r in rows])
                                for k in rows[0]["regions"]}
    res["loss_variants"] = agg

    with open(f"{rc.MSEED_DIR}/simple_baselines.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n{'baseline':<24}{'RMSE':>9}{'MAE':>9}{'R2':>9}"
          f"{'low':>9}{'mid':>9}{'high':>9}")
    for k in ("mean_predictor", "ols_3feat", "ols_full", "ridge_full"):
        d = res.get(k, {})
        if "rmse" not in d:
            print(f"{k:<24}  {d.get('skipped','—')}")
            continue
        print(f"{k:<24}{d['rmse']:>9.4f}{d['mae']:>9.4f}{d['r2']:>9.4f}"
              f"{d['regions']['Low tail']:>9.3f}{d['regions']['Middle']:>9.3f}"
              f"{d['regions']['High tail']:>9.3f}")
    for k, d in agg.items():
        if k == "quantile_spread_q10_q90":
            continue
        print(f"{k:<24}{d['rmse']['mean']:>9.4f}{d['mae']['mean']:>9.4f}"
              f"{d['r2']['mean']:>9.4f}{d['regions']['Low tail']['mean']:>9.3f}"
              f"{d['regions']['Middle']['mean']:>9.3f}"
              f"{d['regions']['High tail']['mean']:>9.3f}")
    sp = agg["quantile_spread_q10_q90"]
    print(f"\nKhoảng 10–90%: rộng {sp['mean_width']['mean']:.2f} điểm, "
          f"phủ {sp['coverage']['mean']:.3f} (danh nghĩa 0.80)")


if __name__ == "__main__":
    main()
