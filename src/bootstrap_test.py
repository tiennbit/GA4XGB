# -*- coding: utf-8 -*-
"""Kiểm định ý nghĩa thống kê bằng BOOTSTRAP trên tập test (thay cho multi-seed).

Câu hỏi trả lời: "Chênh lệch MAE ở đuôi giữa 2 phương pháp có THẬT không, hay chỉ
là nhiễu do bin đuôi ít mẫu (vd. 224 học sinh ở bin <50)?"

Phương pháp: bootstrap CẶP (paired) — resample cùng chỉ số học sinh cho cả 2 mô hình,
nên loại được phương sai do thành phần tập test, chỉ còn chênh lệch giữa 2 mô hình.
Báo cáo: chênh lệch trung bình, CI 95% percentile, và p-value 2 phía.

Đây là chuẩn mực khi tập test cố định và chi phí train lại lớn. KHÁC với multi-seed
(đo phương sai do tính ngẫu nhiên của GA) — cần nói rõ trong bài là ta đo cái gì.

Chạy: PYTHONPATH=src python3 src/bootstrap_test.py [--n-boot 10000]
Xuất: results/bootstrap_test.json + bảng in ra màn hình
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from preprocess import load_and_preprocess
from ga_xgb import split_params, sample_weights

BINS = [0, 50, 60, 70, 80, 90, 100, 110, 150]
LAB = ["<50", "50-60", "60-70", "70-80", "80-90", "90-100", "100-110", ">110"]
REGIONS = {"LOW TAIL (<60)": lambda y: y < 60,
           "MIDDLE (60-100)": lambda y: (y >= 60) & (y < 100),
           "HIGH TAIL (>=100)": lambda y: y >= 100,
           "ALL": lambda y: np.ones(len(y), bool)}


def load_methods():
    """Nạp mọi cấu hình đã tìm được từ results/*_best.json."""
    methods = {"Default": ({}, None)}
    for f in sorted(glob.glob("results/*_best.json")):
        tag = os.path.basename(f).replace("_best.json", "")
        d = json.load(open(f))
        if "best_params" not in d:
            continue
        xgb_p, beta = split_params(d["best_params"])
        # Thiếu một mốc ở bảng này thì `.get(tag, tag)` lặng lẽ trả về tên file
        # thô ("ga_tail_a0.25"), và bảng trong bài hiện tên lẫn lộn với các mốc
        # cũ ("GA-tail a=0.5"). Thêm mốc alpha mới thì thêm cả dòng ở đây.
        name = {"ga_rmse": "GA-RMSE", "ga_mae": "GA-MAE", "ga_r2": "GA-R2",
                "ga_tail_a0.25": "GA-tail a=0.25",
                "ga_tail_a0.5": "GA-tail a=0.5",
                "ga_tail_a0.75": "GA-tail a=0.75",
                "ga_tail_a1": "GA-tail a=1",
                "ga_rmse_lw": "GA-RMSE +LW",
                "ga_tail_a1_lw": "GA-tail a=1 +LW",
                "random_search": "RandomSearch", "grid_search": "GridSearch",
                "baseline_xgb_default": None}.get(tag, tag)
        if name:
            methods[name] = (xgb_p, beta)
    return methods


def paired_bootstrap(y, pred_a, pred_b, mask, n_boot, rng):
    """Bootstrap cặp cho hiệu MAE(A) - MAE(B) trên tập con `mask`.

    Trả về (diff quan sát, CI thấp, CI cao, p 2 phía).
    p tính theo cách nghịch đảo CI: tỷ lệ mẫu bootstrap đổi dấu so với diff quan sát."""
    idx = np.flatnonzero(mask)
    ya, a, b = y[idx], pred_a[idx], pred_b[idx]
    obs = np.abs(ya - a).mean() - np.abs(ya - b).mean()
    n = len(idx)
    diffs = np.empty(n_boot)
    err_a, err_b = np.abs(ya - a), np.abs(ya - b)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        diffs[i] = err_a[s].mean() - err_b[s].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # p 2 phía: tỷ lệ bootstrap nằm bên kia 0 so với hiệu quan sát, nhân 2
    p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return obs, lo, hi, min(p, 1.0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--baseline", default="GA-RMSE",
                    help="phương pháp làm mốc so sánh")
    args = ap.parse_args()

    X, y = load_and_preprocess()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    y_np = y_te.values

    methods = load_methods()
    print(f"Huấn luyện {len(methods)} mô hình...")
    preds = {}
    for name, (p, beta) in methods.items():
        m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=args.n_jobs, **p)
        sw = sample_weights(y_tr, beta) if beta is not None else None
        m.fit(X_tr, y_tr, sample_weight=sw)
        preds[name] = m.predict(X_te)
        print(f"  {name:<18} MAE toàn bộ = {mean_absolute_error(y_te, preds[name]):.4f}")

    base = args.baseline
    if base not in preds:
        base = "Default"
    rng = np.random.default_rng(42)
    out = {"baseline": base, "n_boot": args.n_boot, "comparisons": {}}

    print(f"\n{'='*104}")
    print(f"BOOTSTRAP CẶP ({args.n_boot} lần) — hiệu MAE so với {base}")
    print("Số ÂM = tốt hơn baseline. CI không chứa 0 -> khác biệt có ý nghĩa (p<0.05)")
    print("="*104)
    print(f"{'Vùng':<20} {'Phương pháp':<18} {'n':>6} {'ΔMAE':>8} {'CI 95%':>20} {'p':>9}  ")
    print("-"*104)
    for rname, rfn in REGIONS.items():
        mask = rfn(y_np)
        for name in preds:
            if name == base:
                continue
            obs, lo, hi, p, n = paired_bootstrap(y_np, preds[name], preds[base],
                                                 mask, args.n_boot, rng)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"{rname:<20} {name:<18} {n:>6} {obs:>+8.4f} "
                  f"[{lo:>+7.4f},{hi:>+7.4f}] {p:>9.4f} {sig}")
            out["comparisons"].setdefault(rname, {})[name] = {
                "n": int(n), "delta_mae": float(obs), "ci95": [float(lo), float(hi)],
                "p_value": float(p), "significant": bool(p < 0.05)}
        print("-"*104)

    with open("results/bootstrap_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-> results/bootstrap_test.json")


if __name__ == "__main__":
    main()
