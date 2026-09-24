# -*- coding: utf-8 -*-
"""So sánh chốt: với trung tâm TỐT, quy tắc Bayes còn thắng không?

tail_prior.py chứng minh quy tắc Bayes (trung bình có trọng số trên phân phối sai
số OOF theo bin ŷ) thắng giãn đầu ra và huấn luyện có trọng số LDS/DenseWeight,
nhưng chỉ với XGBoost MẶC ĐỊNH. Một phản biện hợp lý: huấn luyện có trọng số làm
thay đổi tổng hessian mà các ngưỡng min_child_weight, reg_lambda dựa vào, nên với
tham số tốt nó có thể bắt kịp. Lượt này kiểm đúng phản biện đó.

Trung tâm (đặc trưng gốc + mã hoá lệch chuẩn chấm theo trường):
  tuned : cấu hình GA-RMSE seed 42 (dò trên train lần chia 42; ở lần chia khác hơi
          lạc quan, xem center_ablation.py). Huấn luyện có trọng số dùng CÙNG cấu hình.
  bag10 : trung bình 10 XGBoost mặc định, subsample=colsample=0,8. Huấn luyện có
          trọng số cũng là 10 mô hình. Kiểm kết luận có phụ thuộc trung tâm không.

Cách, với mỗi kiểu trọng số (prior λ / relevance φ / bậc thang K) và mỗi tham số:
  raw     : trung tâm, không hiệu chỉnh (một điểm, mốc của đường đánh đổi).
  stretch : giãn hai phía quanh μ, hệ số chọn trên OOF theo đúng trọng số đó.
  wtrain  : huấn luyện lại trung tâm với sample_weight = w(y).
  bayes   : trung bình có trọng số w(y) trên ŷ + phân vị sai số OOF cùng bin ŷ.

Đường đánh đổi đo trên ba thước đo, cùng mức hy sinh vùng giữa Δ so với raw của
chính trung tâm đó: RMSE hai đuôi (y<60 hoặc >=100), macro-RMSE theo 8 bin, SERA.
Ba thước đo ưu tiên ba kiểu trọng số khác nhau, nên thắng trên cả ba mới là thắng thật.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from cost_aware import N_FOLDS
from preprocess import DATA_PATH, load_and_preprocess
from tail_prior import (MID_BUDGETS, PriorDensity, Relevance, bin_samples, point_metrics,
                        stretch_two_sided, weight_families, weighted_mean)
from tail_weights import features

SPLITS = [42, 1, 2, 3, 4]
CENTERS = ["tuned", "bag10"]
FAMILIES = ["prior", "phi", "step"]
METHODS = ["stretch", "wtrain", "bayes"]
METRICS = ["Tails", "macro_bin", "sera"]
VARIANT = "enc_school"
N_BAG = 10
TUNED_FROM = "results_mseed/ga_rmse_best.json"


def fit_center(kind, Xtr, ytr, Xap, tuned, w=None):
    if kind == "tuned":
        m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1, **tuned)
        return m.fit(Xtr, ytr, sample_weight=w).predict(Xap)
    if kind == "bag10":
        return np.mean([XGBRegressor(tree_method="hist", random_state=b, subsample=0.8,
                                     colsample_bytree=0.8, n_jobs=-1)
                        .fit(Xtr, ytr, sample_weight=w).predict(Xap) for b in range(N_BAG)], axis=0)
    raise ValueError(kind)


def along_middle(points, base_mid, metric):
    """Giá trị thước đo khi RMSE vùng giữa tăng Δ so với raw, nội suy dọc tham số."""
    mid = np.array([p["Middle"] for p in points])
    val = np.array([p[metric] for p in points])
    o = np.argsort(mid)
    mid, val = mid[o], val[o]
    return {str(d): (float(np.interp(base_mid + d, mid, val))
                     if mid[0] <= base_mid + d <= mid[-1] else float("nan")) for d in MID_BUDGETS}


def run_split(split, X, y, prov, school, grade_cols, tuned):
    t0 = time.time()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=split)
    y_tr, yt = y[tr], y[te]
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=split).split(tr))
    feats = [features(VARIANT, X, y, prov, school, grade_cols, tr[a], tr[b], split) for a, b in folds]
    Xtr, Xte = features(VARIANT, X, y, prov, school, grade_cols, tr, te, split)
    prior, phi = PriorDensity(y_tr), Relevance(y_tr)
    fams = {k: v for k, v in weight_families(prior, phi).items() if k in FAMILIES}
    phi_te, prov_te, mu = phi(yt), prov[te], float(y_tr.mean())

    res = {}
    for c in CENTERS:
        p_oof = np.zeros(len(tr))
        for (a, b), (A, B) in zip(folds, feats):
            p_oof[b] = fit_center(c, A, y_tr[a], B, tuned)
        p_te = fit_center(c, Xtr, y_tr, Xte, tuned)
        S = bin_samples(p_oof, y_tr, p_te)
        raw = point_metrics(yt, p_te, phi_te, prov_te)
        rc = {"raw": raw, "points": {}, "curves": {}}
        for fam, params in fams.items():
            for meth in METHODS:
                pts = [{"param": params[0][0]} | raw]     # tham số đầu là w ≡ 1: đúng bằng raw
                for prm, wf in params[1:]:
                    if meth == "stretch":
                        pred = stretch_two_sided(p_oof, y_tr, p_te, mu, wf(y_tr))
                    elif meth == "wtrain":
                        pred = fit_center(c, Xtr, y_tr, Xte, tuned, w=wf(y_tr))
                    else:
                        pred = weighted_mean(S, wf)
                    pts.append({"param": prm} | point_metrics(yt, pred, phi_te, prov_te))
                key = f"{meth}|{fam}"
                rc["points"][key] = pts
                rc["curves"][key] = {m: along_middle(pts, raw["Middle"], m) for m in METRICS}
        res[c] = rc
        cv = rc["curves"]
        print(f"  split {split} {c}: RMSE raw={raw['All']:.3f}  đuôi@Δ=0.5 (prior) "
              + " ".join(f"{m}={cv[f'{m}|prior']['Tails']['0.5']:.3f}" for m in METHODS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    return res


def ms(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    return {"mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def summarize(per):
    sp = list(per)
    out = {}
    for c in CENTERS:
        oc = {"raw": {m: ms([per[s][c]["raw"][m] for s in sp])
                      for m in ["Low tail", "Middle", "High tail", "Tails", "All", "macro_bin", "sera"]},
              "curves": {}, "paired": {}, "points": {}}
        for key in per[sp[0]][c]["curves"]:
            oc["curves"][key] = {m: {d: ms([per[s][c]["curves"][key][m][d] for s in sp])
                                     for d in map(str, MID_BUDGETS)} for m in METRICS}
            pts0 = per[sp[0]][c]["points"][key]
            oc["points"][key] = [{"param": p["param"]} |
                                 {m: ms([per[s][c]["points"][key][i][m] for s in sp])
                                  for m in ["Low tail", "Middle", "High tail", "Tails", "All",
                                            "macro_bin", "sera", "audit_bias_sd"]}
                                 for i, p in enumerate(pts0)]
        for fam in FAMILIES:
            for a, b in [("bayes", "wtrain"), ("bayes", "stretch"), ("wtrain", "stretch")]:
                for m in METRICS:
                    for d in map(str, MID_BUDGETS):
                        diffs = [per[s][c]["curves"][f"{a}|{fam}"][m][d] - per[s][c]["curves"][f"{b}|{fam}"][m][d]
                                 for s in sp]
                        v = [x for x in diffs if not np.isnan(x)]
                        oc["paired"][f"{a}-{b}|{fam}|{m}|Δ={d}"] = ms(diffs) | {"n_wins": int(sum(x < 0 for x in v))}
        out[c] = oc
    return out


def print_summary(S):
    for c, oc in S.items():
        r = oc["raw"]
        print(f"\n=== Trung tâm {c}: raw Thấp={r['Low tail']['mean']:.2f} Giữa={r['Middle']['mean']:.2f} "
              f"Cao={r['High tail']['mean']:.2f} Toàn bộ={r['All']['mean']:.3f}")
        for m in METRICS:
            print(f"  [{m}] khi RMSE vùng giữa tăng Δ (thấp hơn là tốt hơn)")
            print("    cách|kiểu            " + "".join(f"  Δ={d:<12}" for d in MID_BUDGETS))
            for fam in FAMILIES:
                for meth in METHODS:
                    v = oc["curves"][f"{meth}|{fam}"][m]
                    print(f"    {meth + '|' + fam:18s}" + "".join(
                        f"  {v[str(d)]['mean']:7.3f}±{v[str(d)]['sd']:.3f}" for d in MID_BUDGETS))
        print("  So từng cặp, số lần thắng / số lần chia (âm = vế trái tốt hơn):")
        for fam in FAMILIES:
            for a, b in [("bayes", "wtrain"), ("bayes", "stretch"), ("wtrain", "stretch")]:
                cells = []
                for m in METRICS:
                    for d in map(str, MID_BUDGETS):
                        p = oc["paired"][f"{a}-{b}|{fam}|{m}|Δ={d}"]
                        cells.append(f"{p['mean']:+.2f}({p['n_wins']}/{p['n']})")
                print(f"    {a}-{b}|{fam:5s} " + " ".join(cells))
        print("    (thứ tự cột: Tails Δ=.25 .5 1 | macro_bin Δ=.25 .5 1 | sera Δ=.25 .5 1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default="results_cost/final_compare.json")
    ap.add_argument("--splits", type=int, nargs="+", default=SPLITS)
    ap.add_argument("--tuned-from", default=TUNED_FROM)
    args = ap.parse_args()

    tuned = json.load(open(args.tuned_from))["best_params"]
    X, y = load_and_preprocess(args.data)
    raw = pd.read_csv(args.data, usecols=["Trường", "Tỉnh"])
    prov = raw["Tỉnh"].astype(str).to_numpy()
    # Khoá trường gồm cả tỉnh: tên trường trùng giữa các tỉnh (988 tên, 1.125 trường)
    school = (raw["Tỉnh"].astype(str) + "||" + raw["Trường"].astype(str)).to_numpy()
    y = y.to_numpy(float)
    grade_cols = [c for c in X.columns if c[:3] in ("10.", "11.", "12.")]
    print(f"n={len(y)}, trung tâm {CENTERS}, cách {METHODS}, kiểu {FAMILIES}", flush=True)

    per = {}
    for s in args.splits:
        per[str(s)] = run_split(s, X, y, prov, school, grade_cols, tuned)
        with open(args.out + ".partial", "w") as fh:
            json.dump(per, fh, ensure_ascii=False)
    summary = summarize(per)
    print_summary(summary)
    meta = {"protocol": "5 lần chia 80/20 như tail_prior; đặc trưng + mã hoá " + VARIANT
                        + "; stretch và phân phối sai số của bayes lấy từ OOF train; wtrain dùng cùng cấu hình trung tâm; "
                        "đường đánh đổi đo tại RMSE vùng giữa = raw + Δ của chính trung tâm",
            "splits": args.splits, "centers": CENTERS, "tuned_params": tuned, "n_bag": N_BAG,
            "families": FAMILIES, "methods": METHODS, "metrics": METRICS, "mid_budgets": MID_BUDGETS}
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "summary": summary, "per_split": per}, fh, indent=1, ensure_ascii=False)
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
