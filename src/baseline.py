# -*- coding: utf-8 -*-
"""Baseline: XGBoost tham số mặc định, đánh giá 5-fold CV trên tập train
và trên tập test giữ riêng 20%. Đo thời gian 1 lần fit để lập ngân sách GA."""
import json
import time

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import load_and_preprocess
from ga_xgb import CV_FOLDS

SEED = 42


def evaluate(model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    pred = model.predict(X_te)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "fit_seconds": round(fit_s, 2),
    }


def main():
    X, y = load_and_preprocess()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Train {X_tr.shape}, Test {X_te.shape}")

    params = dict(tree_method="hist", random_state=SEED, n_jobs=-1)

    # 5-fold CV trên train
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = []
    for i, (tr, va) in enumerate(kf.split(X_tr)):
        m = XGBRegressor(**params)
        s = evaluate(m, X_tr.iloc[tr], y_tr.iloc[tr], X_tr.iloc[va], y_tr.iloc[va])
        cv_scores.append(s)
        print(f"  fold {i+1}: RMSE={s['rmse']:.4f} MAE={s['mae']:.4f} R2={s['r2']:.4f} ({s['fit_seconds']}s)")

    cv_mean = {k: float(np.mean([s[k] for s in cv_scores])) for k in cv_scores[0]}
    cv_std = {k: float(np.std([s[k] for s in cv_scores])) for k in cv_scores[0]}

    # Fit toàn bộ train, đánh giá test
    test_score = evaluate(XGBRegressor(**params), X_tr, y_tr, X_te, y_te)

    out = {"cv_mean": cv_mean, "cv_std": cv_std, "test": test_score,
           "default_params": XGBRegressor(**params).get_params()}
    print("\nCV  : RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}".format(**cv_mean))
    print("Test: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} (fit {fit_seconds}s)".format(**test_score))

    with open("results/baseline_xgb_default.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
