# -*- coding: utf-8 -*-
"""Baseline so sánh: RandomSearch và GridSearch cho XGBoost, CÙNG ngân sách
số lần đánh giá với GA (mặc định 622 — số eval thực tế của run GA-RMSE),
cùng protocol: fitness = RMSE trung bình 5-fold CV trên train, test chạm 1 lần.

Chạy:
  python3 src/search_baselines.py --strategy random --budget 622
  python3 src/search_baselines.py --strategy grid   --budget 622
Kết quả: results/{strategy}_search_best.json (+ _log.jsonl từng lần đánh giá)
"""
import argparse
import itertools
import json
import time

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import load_and_preprocess
from ga_xgb import CV_FOLDS

# Cùng miền tìm kiếm với GA (GENES trong ga_xgb.py)
SPACE = {
    "n_estimators":     (100, 800,  int,   False),
    "max_depth":        (3,   12,   int,   False),
    "learning_rate":    (0.01, 0.30, float, True),
    "subsample":        (0.5, 1.0,  float, False),
    "colsample_bytree": (0.5, 1.0,  float, False),
    "min_child_weight": (1,   20,   int,   False),
    "reg_lambda":       (0.0, 10.0, float, False),
}


def random_configs(budget, rng):
    for _ in range(budget):
        p = {}
        for name, (lo, hi, typ, log) in SPACE.items():
            if log:
                v = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                v = rng.uniform(lo, hi)
            p[name] = int(round(v)) if typ is int else float(v)
        yield p


def grid_configs(budget):
    """Lưới ~budget điểm: phân bổ số mức cho từng chiều (ưu tiên tham số nhạy).
    3x3x3x2x2x2x2 = 648 ~ 622."""
    grid = {
        "n_estimators":     [100, 400, 800],
        "max_depth":        [4, 8, 12],
        "learning_rate":    [0.01, 0.05, 0.30],
        "subsample":        [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "min_child_weight": [1, 10],
        "reg_lambda":       [0.0, 5.0],
    }
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))[:budget]
    for c in combos:
        yield dict(zip(keys, c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["random", "grid"], required=True)
    ap.add_argument("--budget", type=int, default=622)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    X, y = load_and_preprocess()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    rng = np.random.default_rng(args.seed)
    configs = (random_configs(args.budget, rng) if args.strategy == "random"
               else grid_configs(args.budget))

    prefix = f"results/{args.strategy}_search"
    if args.seed != 42:
        prefix += f"_seed{args.seed}"
    log_f = open(f"{prefix}_log.jsonl", "w")
    t_start = time.time()
    best_rmse, best_params, n = np.inf, None, 0

    for params in configs:
        rmses = []
        for tr, va in kf.split(X_tr):
            m = XGBRegressor(tree_method="hist", random_state=42,
                             n_jobs=args.n_jobs, verbosity=0, **params)
            m.fit(X_tr.iloc[tr], y_tr.iloc[tr])
            pred = m.predict(X_tr.iloc[va])
            rmses.append(float(np.sqrt(mean_squared_error(y_tr.iloc[va], pred))))
        cv_rmse = float(np.mean(rmses))
        n += 1
        if cv_rmse < best_rmse:
            best_rmse, best_params = cv_rmse, params
        log_f.write(json.dumps({"i": n, "cv_rmse": cv_rmse, "best_rmse": best_rmse,
                                "elapsed_s": round(time.time() - t_start, 1),
                                "params": params}) + "\n")
        log_f.flush()
        if n % 25 == 0:
            print(f"[{args.strategy}] {n}/{args.budget} | best CV-RMSE {best_rmse:.4f} "
                  f"| {round(time.time()-t_start)}s", flush=True)
    log_f.close()

    m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=args.n_jobs, **best_params)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    test = {"rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
            "mae": float(mean_absolute_error(y_te, pred)),
            "r2": float(r2_score(y_te, pred))}
    out = {"strategy": args.strategy, "budget": args.budget, "seed": args.seed,
           "best_cv_rmse": best_rmse, "best_params": best_params, "test": test,
           "total_seconds": round(time.time() - t_start, 1)}
    with open(f"{prefix}_best.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{args.strategy}] Best CV-RMSE {best_rmse:.4f} | "
          f"Test RMSE={test['rmse']:.4f} MAE={test['mae']:.4f} R2={test['r2']:.4f}")


if __name__ == "__main__":
    main()
