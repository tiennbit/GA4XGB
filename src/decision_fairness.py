# -*- coding: utf-8 -*-
"""Sinh results/c1_c2_analysis.json — hai phân tích được trích dẫn ở Section IV-F và IV-J.

C1 — DECISION UTILITY: quy sai số ra quyết định sàng lọc mà tư vấn viên thực sự
     phải làm, tức gắn cờ học sinh có ĐIỂM DỰ ĐOÁN dưới 60 (vùng đuôi thấp) để
     can thiệp sớm. Báo cáo recall/precision cho cả hai đuôi, cộng tỷ lệ đoán
     đúng bin và sai lệch trong phạm vi 1 bin.

C2 — FAIRNESS THEO TỈNH: mô hình rút ~1/4 sức mạnh dự đoán từ biến tỉnh, nên phải
     kiểm độ chệch có dấu theo từng tỉnh. Chỉ xét tỉnh có >= 100 học sinh trong
     tập test để con số có nghĩa.

Ngưỡng đuôi lấy từ bins.py (nguồn duy nhất) — KHÔNG hardcode 60/100 ở đây.

Chạy: PYTHONPATH=src python3 src/decision_fairness.py
"""
import json

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import bins
from preprocess import load_and_preprocess
from ga_xgb import split_params, sample_weights

MIN_PROV_N = 100          # tỉnh dưới ngưỡng này không đủ mẫu để báo cáo
PROV_PREFIX = "Tỉnh_"

# Bốn cấu hình được trích dẫn trong bài
RUNS = {
    "Default":          None,
    "GA-RMSE":          "results/ga_rmse_best.json",
    "GA4XGB a=1":       "results/ga_tail_a1_best.json",
    "GA4XGB a=1 +LW":   "results/ga_tail_a1_lw_best.json",
}
FAIRNESS_ON = ["GA-RMSE", "GA4XGB a=1 +LW"]


def decision_utility(y_true, pred):
    """Sàng lọc: gắn cờ khi ĐIỂM DỰ ĐOÁN rơi vào vùng đuôi."""
    yt, pr = np.asarray(y_true, float), np.asarray(pred, float)
    out = {}
    for tag, true_m, flag_m in (
            ("low",  yt < bins.LOW_TAIL_MAX,   pr < bins.LOW_TAIL_MAX),
            ("high", yt >= bins.HIGH_TAIL_MIN, pr >= bins.HIGH_TAIL_MIN)):
        hit = (true_m & flag_m).sum()
        out[f"{tag}_recall"] = float(hit / true_m.sum()) if true_m.sum() else float("nan")
        out[f"{tag}_prec"] = float(hit / flag_m.sum()) if flag_m.sum() else float("nan")
    bt, bp = bins.bin_index(yt), bins.bin_index(pr)
    out["exact"] = float((bt == bp).mean())
    out["within1"] = float((np.abs(bt - bp) <= 1).mean())
    out["n_flag_low"] = int((pr < bins.LOW_TAIL_MAX).sum())
    return out


def fairness_province(X_te, y_true, pred):
    """Độ chệch có dấu (pred − true) theo từng tỉnh, chỉ tỉnh n >= MIN_PROV_N."""
    yt, pr = np.asarray(y_true, float), np.asarray(pred, float)
    cols = [c for c in X_te.columns if c.startswith(PROV_PREFIX)]
    rows = []
    for c in cols:
        m = X_te[c].to_numpy().astype(bool)
        if m.sum() < MIN_PROV_N:
            continue
        rows.append({"prov": c[len(PROV_PREFIX):],
                     "n": int(m.sum()),
                     "bias": float((pr[m] - yt[m]).mean()),
                     "mae": float(np.abs(pr[m] - yt[m]).mean())})
    if not rows:
        return {"n_prov": 0}
    bias = np.array([r["bias"] for r in rows])
    mae = np.array([r["mae"] for r in rows])
    return {"bias_min": float(bias.min()), "bias_max": float(bias.max()),
            "bias_sd": float(bias.std(ddof=1)),   # SD mẫu — khớp con số đã báo cáo (0.49 / 0.82)
            "mae_min": float(mae.min()), "mae_max": float(mae.max()),
            "worst_neg": min(rows, key=lambda r: r["bias"]),
            "worst_pos": max(rows, key=lambda r: r["bias"]),
            "n_prov": len(rows)}


def main():
    X, y = load_and_preprocess()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    out = {"decision_utility": {}, "fairness_province": {},
           "config": {"low_tail_max": bins.LOW_TAIL_MAX,
                      "high_tail_min": bins.HIGH_TAIL_MIN,
                      "min_province_n": MIN_PROV_N}}

    for name, path in RUNS.items():
        if path is None:
            params, beta = {}, None
        else:
            params, beta = split_params(json.load(open(path))["best_params"])
        m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1, **params)
        sw = sample_weights(y_tr, beta) if beta is not None else None
        m.fit(X_tr, y_tr, sample_weight=sw)
        pred = m.predict(X_te)

        out["decision_utility"][name] = decision_utility(y_te, pred)
        if name in FAIRNESS_ON:
            out["fairness_province"][name] = fairness_province(X_te, y_te, pred)

        d = out["decision_utility"][name]
        print(f"{name:<18} low recall {d['low_recall']:.3f} (prec {d['low_prec']:.3f})"
              f" | high recall {d['high_recall']:.3f} (prec {d['high_prec']:.3f})"
              f" | flagged {d['n_flag_low']}")

    for name, f in out["fairness_province"].items():
        print(f"{name:<18} bias [{f['bias_min']:+.2f}, {f['bias_max']:+.2f}] "
              f"SD {f['bias_sd']:.2f} over {f['n_prov']} provinces "
              f"(worst− {f['worst_neg']['prov']}, worst+ {f['worst_pos']['prov']})")

    with open("results/c1_c2_analysis.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("-> results/c1_c2_analysis.json")


if __name__ == "__main__":
    main()
