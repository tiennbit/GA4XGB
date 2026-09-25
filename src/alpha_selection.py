# -*- coding: utf-8 -*-
"""Chọn α theo chi phí quyết định — TÍNH TRÊN FOLD VALIDATION, không đụng tập test.

VÌ SAO PHẢI CÓ FILE NÀY: bảng "α tối ưu theo K" nếu tính bằng MAE trên tập test
thì chính là chọn siêu tham số theo test — đúng loại rò rỉ mà Section IV-B tự
hào đã kiểm soát. Ở đây α được chấm bằng dự đoán out-of-fold trên phần huấn
luyện; tập test giữ nguyên để báo cáo cuối.

K là tham số CHÍNH SÁCH, không ước lượng từ dữ liệu: một điểm sai ở vùng đuôi
đắt gấp K lần một điểm sai ở vùng giữa. Viện phát biểu K, bài suy ra α.

    Cost(α, K) = Σ_r n_r · c_r · MAE_r(α)  /  Σ_r n_r · c_r
    với c = K ở hai đuôi, c = 1 ở vùng giữa.

  python src/alpha_selection.py
"""
import json
import os

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

import bins as B
from preprocess import load_and_preprocess
from ga_xgb import CV_FOLDS, split_params, sample_weights

# Nhãn α -> file kết quả. α=0 chính là GA-RMSE (hai đầu mút của cùng một họ).
CONFIGS = [("0",    "ga_rmse_best.json"),
           ("0.25", "ga_tail_a0.25_best.json"),
           ("0.5",  "ga_tail_a0.5_best.json"),
           ("0.75", "ga_tail_a0.75_best.json"),
           ("1",    "ga_tail_a1_best.json")]

K_GRID = [1, 1.5, 2, 2.5, 3, 4, 5, 8, 10, 15, 20, 30, 50]


def oof_predictions(X_tr, y_tr, params):
    """Dự đoán out-of-fold trên phần huấn luyện, dùng ĐÚNG cách chia của GA."""
    xgb_p, beta = split_params(params)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(y_tr), dtype=float)
    for tr, va in kf.split(X_tr):
        ytr_fold = y_tr.iloc[tr]
        sw = sample_weights(ytr_fold, beta) if beta is not None else None
        m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=2,
                         verbosity=0, **xgb_p)
        m.fit(X_tr.iloc[tr], ytr_fold, sample_weight=sw)
        oof[va] = m.predict(X_tr.iloc[va])
    return oof


def cost(mae_by_region, n_by_region, K):
    """Chi phí kỳ vọng mỗi học sinh. Vùng giữa hệ số 1, hai đuôi hệ số K."""
    num = den = 0.0
    for r, c in (("Low tail", K), ("Middle", 1.0), ("High tail", K)):
        num += n_by_region[r] * c * mae_by_region[r]
        den += n_by_region[r] * c
    return num / den


def main():
    X, y = load_and_preprocess()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = B.regions(y_tr)
    n_by_region = {r: int(m.sum()) for r, m in reg.items()}
    print(f"Phần huấn luyện: {len(y_tr)} học sinh")
    print("  " + "  ".join(f"{r}={n_by_region[r]}" for r in
                           ["Low tail", "Middle", "High tail"]))

    mae = {}
    for a, fn in CONFIGS:
        p = os.path.join("results", fn)
        if not os.path.exists(p):
            print(f"  BỎ QUA α={a}: thiếu {p}")
            continue
        params = json.load(open(p, encoding="utf-8"))["best_params"]
        oof = oof_predictions(X_tr, y_tr, params)
        mae[a] = {r: float(mean_absolute_error(y_tr[m], oof[m]))
                  for r, m in reg.items()}
        print(f"  α={a:<5} validation MAE  low {mae[a]['Low tail']:.4f}  "
              f"mid {mae[a]['Middle']:.4f}  high {mae[a]['High tail']:.4f}  "
              f"all {mae[a]['All']:.4f}")

    alphas = list(mae.keys())
    rows = []
    print(f"\n{'K':>6} | " + " ".join(f"α={a:>5}" for a in alphas) + " | tối ưu")
    print("-" * (10 + 8 * len(alphas) + 10))
    for K in K_GRID:
        cs = [cost(mae[a], n_by_region, K) for a in alphas]
        best = alphas[int(np.argmin(cs))]
        rows.append({"K": K, "cost": dict(zip(alphas, cs)), "best_alpha": best})
        print(f"{K:>6} | " + " ".join(f"{c:7.4f}" for c in cs) + f" | α={best}")

    # Điểm chuyển: giá trị K nhỏ nhất mà lời giải không còn là α=0.
    # Đây là con số bài báo cần nêu — nó cho biết tiền đề của bài đúng từ đâu.
    lo, hi = 1.0, 200.0
    base = alphas[int(np.argmin([cost(mae[a], n_by_region, 1.0) for a in alphas]))]
    switch = None
    if base == alphas[0]:
        for _ in range(60):
            mid = (lo + hi) / 2
            cs = [cost(mae[a], n_by_region, mid) for a in alphas]
            if alphas[int(np.argmin(cs))] == alphas[0]:
                lo = mid
            else:
                hi = mid
        switch = hi
        cs = [cost(mae[a], n_by_region, switch + 0.01) for a in alphas]
        print(f"\nĐiểm chuyển: K < {switch:.2f} -> α={alphas[0]}; "
              f"K > {switch:.2f} -> α={alphas[int(np.argmin(cs))]}")

    out = {"protocol": "out-of-fold trên phần huấn luyện, tập test không dùng",
           "cv_folds": CV_FOLDS,
           "n_by_region": n_by_region,
           "mae_validation": mae,
           "cost_by_K": rows,
           "switch_K": switch}
    with open("results/alpha_selection.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("-> results/alpha_selection.json")


if __name__ == "__main__":
    main()
