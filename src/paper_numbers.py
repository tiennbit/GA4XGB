# -*- coding: utf-8 -*-
"""Sinh MỌI con số dùng trong bài, bằng quy ước bin THỐNG NHẤT (src/bins.py).

Mục đích: một nguồn sự thật duy nhất. Mọi con số trong draft phải lấy từ đây,
không tính lại rời rạc ở nơi khác (đó là cách lỗi bin lọt vào bản thảo lần đầu).

Chạy: PYTHONPATH=src python3 src/paper_numbers.py
Xuất: results/paper_numbers.json + bảng in màn hình
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import bins
from preprocess import load_and_preprocess
from ga_xgb import split_params, sample_weights

NAMES = {"ga_rmse": "GA-RMSE", "ga_mae": "GA-MAE", "ga_r2": "GA-R2",
         "ga_tail_a1": "GA4XGB (a=1)", "ga_tail_a0.5": "GA4XGB (a=0.5)",
         "ga_tail_a1_lw": "GA4XGB (a=1, +LW)",
         "random_search": "Random search", "grid_search": "Grid search"}
ORDER = ["Default XGBoost", "Grid search", "Random search", "GA-RMSE", "GA-MAE",
         "GA-R2", "GA4XGB (a=0.5)", "GA4XGB (a=1)", "GA4XGB (a=1, +LW)"]


def main():
    X, y = load_and_preprocess()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    y_np = y_te.values
    out = {}

    # ---------- 1. Thống kê dataset ----------
    cnt = bins.bin_counts(y)
    ds = {"n": int(len(y)), "mean": float(y.mean()), "median": float(y.median()),
          "sd": float(y.std()), "min": float(y.min()), "max": float(y.max()),
          "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
          "bins": {lab: {"n": int(c), "pct": round(100*c/len(y), 2)}
                   for lab, c in zip(bins.LABELS, cnt)},
          "regions": {k: {"n": int(m.sum()), "pct": round(100*m.sum()/len(y), 2)}
                      for k, m in bins.regions(y).items()}}
    out["dataset"] = ds
    print("=" * 78)
    print("1. DATASET (dùng cho IV.A + Introduction)")
    print("=" * 78)
    print(f"n={ds['n']:,} | mean={ds['mean']:.1f} | median={ds['median']:.1f} | "
          f"SD={ds['sd']:.1f} | range {ds['min']:.0f}-{ds['max']:.0f}")
    print(f"train={ds['n_train']:,} / test={ds['n_test']:,}")
    for lab in bins.LABELS:
        b = ds["bins"][lab]
        print(f"  {lab:>10}: {b['n']:>6,} ({b['pct']:>5.2f}%)")
    print("  ---")
    for k in ["Low tail", "Middle", "High tail"]:
        r = ds["regions"][k]
        print(f"  {k:>10}: {r['n']:>6,} ({r['pct']:>5.2f}%)")

    # ---------- 2. Huấn luyện mọi phương pháp ----------
    methods = {"Default XGBoost": ({}, None)}
    for f in sorted(glob.glob("results/*_best.json")):
        tag = os.path.basename(f).replace("_best.json", "")
        if tag not in NAMES:
            continue
        d = json.load(open(f))
        methods[NAMES[tag]] = split_params(d["best_params"])

    preds, overall, perbin, perregion, bias = {}, {}, {}, {}, {}
    for name, (p, beta) in methods.items():
        m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=3, **p)
        sw = sample_weights(y_tr, beta) if beta is not None else None
        m.fit(X_tr, y_tr, sample_weight=sw)
        pr = m.predict(X_te)
        preds[name] = pr
        overall[name] = {"rmse": float(np.sqrt(mean_squared_error(y_te, pr))),
                         "mae": float(mean_absolute_error(y_te, pr)),
                         "r2": float(r2_score(y_te, pr))}
        idx = bins.bin_index(y_np)
        perbin[name] = {lab: float(mean_absolute_error(y_np[idx == i], pr[idx == i]))
                        for i, lab in enumerate(bins.LABELS) if (idx == i).sum()}
        bias[name] = {lab: float((pr[idx == i] - y_np[idx == i]).mean())
                      for i, lab in enumerate(bins.LABELS) if (idx == i).sum()}
        perregion[name] = {k: float(mean_absolute_error(y_np[m_], pr[m_]))
                           for k, m_ in bins.regions(y_np).items()}
    out["overall"] = overall
    out["mae_per_bin"] = perbin
    out["bias_per_bin"] = bias
    out["mae_per_region"] = perregion

    present = [n for n in ORDER if n in overall]

    print("\n" + "=" * 78)
    print("2. TEST OVERALL (Table 4)")
    print("=" * 78)
    print(f"{'Method':<20} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    for n in present:
        o = overall[n]
        print(f"{n:<20} {o['rmse']:>8.4f} {o['mae']:>8.4f} {o['r2']:>8.4f}")

    print("\n" + "=" * 78)
    print("3. MAE THEO BIN (Fig. 6 + bảng dưới hình)")
    print("=" * 78)
    t = pd.DataFrame(perbin).T.reindex(present)[bins.LABELS]
    print(t.round(2).to_string())

    print("\n" + "=" * 78)
    print("4. MAE THEO VÙNG (3 cột tổng hợp của Fig. 6)")
    print("=" * 78)
    r = pd.DataFrame(perregion).T.reindex(present)[["Low tail", "Middle", "High tail", "All"]]
    print(r.round(3).to_string())

    print("\n" + "=" * 78)
    print("5. BIAS THEO BIN của Default (bằng chứng hồi quy về mean — Intro)")
    print("=" * 78)
    for lab, v in bias["Default XGBoost"].items():
        print(f"  {lab:>10}: {v:>+7.2f}")

    with open("results/paper_numbers.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n-> results/paper_numbers.json")


if __name__ == "__main__":
    main()
