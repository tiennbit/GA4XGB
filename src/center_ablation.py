# -*- coding: utf-8 -*-
"""Vì sao trung bình của 50 phân vị lại là mô hình trung tâm tốt hơn?

tail_prior.py cho thấy: trung bình của 50 phân vị đã hiệu chỉnh (quantile XGBoost,
tham số mặc định) có RMSE 9,56, so với 9,73 của XGBoost thường, thắng 5/5 lần chia,
và dịch cả đường đánh đổi đuôi–giữa khoảng 0,5. Có hai giải thích cạnh tranh:
  (a) ensemble: trung bình 50 mô hình giảm phương sai, loss phân vị không đóng góp gì;
  (b) loss phân vị bền với ngoại lai hơn loss bình phương.
Và một câu hỏi thực dụng: một mô hình thường đã dò siêu tham số có đuổi kịp không?

Các trung tâm, cùng đặc trưng (gốc + mã hoá lệch chuẩn chấm theo trường):
  default : XGBoost mặc định (mốc so sánh, = raw của tail_prior).
  bag10   : trung bình 10 XGBoost mặc định, subsample=colsample=0,8, seed khác nhau.
            Kiểm tra (a): nếu bag10 đuổi kịp qmean thì lợi ích là do ensemble.
  tuned   : cấu hình GA-RMSE seed 42 (results_mseed/ga_rmse_best.json).
            LƯU Ý: cấu hình dò trên train của lần chia 42. Ở các lần chia khác, tập
            test của chúng phần lớn nằm trong train đó nên kết quả tuned hơi lạc quan;
            lần chia 42 là phép so sạch và được in riêng.
  qmean   : trung bình 50 phân vị đã hiệu chỉnh (tầng quant, λ=0 của tail_prior).
  qmedian : trung vị (mức 0,49 và 0,51) của cùng mô hình phân vị. Kiểm tra (b):
            một phân vị, không lấy trung bình qua các mức.

Với mỗi trung tâm: RMSE theo vùng, và đường đánh đổi khi chồng quy tắc Bayes
(tầng bin, trọng số hiệu chỉnh tiên nghiệm λ) lên trên. Mọi thước đo RMSE.
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
from tail_prior import (LAMBDAS, MID_BUDGETS, TAUS, VARIANT, PriorDensity, Relevance,
                        bin_samples, calib_shift, frontier, point_metrics,
                        weighted_mean, xgb_quant)
from tail_weights import features

SPLITS = [42, 1, 2, 3, 4]
CENTERS = ["default", "bag10", "tuned", "qmean", "qmedian"]
N_BAG = 10
TUNED_FROM = "results_mseed/ga_rmse_best.json"


def fit_mean(kind, Xtr, ytr, Xap, tuned):
    if kind == "default":
        return XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1).fit(Xtr, ytr).predict(Xap)
    if kind == "bag10":
        preds = [XGBRegressor(tree_method="hist", random_state=b, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=-1).fit(Xtr, ytr).predict(Xap)
                 for b in range(N_BAG)]
        return np.mean(preds, axis=0)
    if kind == "tuned":
        return XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1, **tuned).fit(Xtr, ytr).predict(Xap)
    raise ValueError(kind)


def run_split(split, X, y, prov, school, grade_cols, tuned):
    t0 = time.time()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=split)
    y_tr, yt = y[tr], y[te]
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=split).split(tr))
    feats = [features(VARIANT, X, y, prov, school, grade_cols, tr[a], tr[b], split) for a, b in folds]
    Xtr, Xte = features(VARIANT, X, y, prov, school, grade_cols, tr, te, split)

    oof, test, secs = {}, {}, {}
    for kind in ["default", "bag10", "tuned"]:
        oof[kind] = np.zeros(len(tr))
        for (a, b), (A, B) in zip(folds, feats):
            oof[kind][b] = fit_mean(kind, A, y_tr[a], B, tuned)
        t1 = time.time()
        test[kind] = fit_mean(kind, Xtr, y_tr, Xte, tuned)
        secs[kind] = time.time() - t1

    Q_oof = np.zeros((len(tr), len(TAUS)))
    for (a, b), (A, B) in zip(folds, feats):
        Q_oof[b] = xgb_quant(A, y_tr[a], B)
    t1 = time.time()
    Q_te = xgb_quant(Xtr, y_tr, Xte)
    secs["qmean"] = secs["qmedian"] = time.time() - t1
    # Hiệu chỉnh độ phủ trên OOF như tail_prior; áp cùng δ cho OOF để trung tâm OOF
    # (dùng cho tầng bin) đi cùng một phép biến đổi với trung tâm test.
    delta = calib_shift(Q_oof, y_tr)
    Qc_oof, Qc_te = np.sort(Q_oof + delta, axis=1), np.sort(Q_te + delta, axis=1)
    jm = [int(np.argmin(abs(TAUS - 0.49))), int(np.argmin(abs(TAUS - 0.51)))]
    oof["qmean"], test["qmean"] = Qc_oof.mean(axis=1), Qc_te.mean(axis=1)
    oof["qmedian"], test["qmedian"] = Qc_oof[:, jm].mean(axis=1), Qc_te[:, jm].mean(axis=1)

    prior, phi = PriorDensity(y_tr), Relevance(y_tr)
    phi_te, prov_te = phi(yt), prov[te]
    base = point_metrics(yt, test["default"], phi_te, prov_te)
    res = {"fit_seconds": secs, "centers": {}}
    for kind in CENTERS:
        pts = []
        S = bin_samples(oof[kind], y_tr, test[kind])
        for lam in LAMBDAS:
            pred = test[kind] if lam == 0 else weighted_mean(S, lambda v, l=lam: prior(v) ** (-l))
            pts.append({"param": lam} | point_metrics(yt, pred, phi_te, prov_te))
        res["centers"][kind] = {
            "point": point_metrics(yt, test[kind], phi_te, prov_te),
            "oof_rmse": float(np.sqrt(np.mean((y_tr - oof[kind]) ** 2))),
            "bin_prior": pts,
            "frontier": frontier(pts, base["Middle"])}
    # Tầng quant đầy đủ (phân phối phân vị, không qua bin) để đối chiếu với tail_prior
    qpts = [{"param": lam} | point_metrics(yt, weighted_mean(Qc_te, lambda v, l=lam: prior(v) ** (-l)),
                                           phi_te, prov_te) for lam in LAMBDAS]
    res["quant_prior"] = {"points": qpts, "frontier": frontier(qpts, base["Middle"])}
    c = res["centers"]
    print(f"  split {split}: RMSE " + " ".join(f"{k}={c[k]['point']['All']:.3f}" for k in CENTERS)
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    return res


def ms(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    return {"mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def summarize(per):
    sp = list(per)
    out = {"point": {}, "frontier": {}, "paired": {}, "fit_seconds": {}}
    for k in CENTERS:
        out["point"][k] = {m: ms([per[s]["centers"][k]["point"][m] for s in sp])
                           for m in ["Low tail", "Middle", "High tail", "Tails", "All", "macro_bin", "sera"]}
        out["point"][k]["oof_rmse"] = ms([per[s]["centers"][k]["oof_rmse"] for s in sp])
        out["frontier"][f"{k}|bin|prior"] = {d: ms([per[s]["centers"][k]["frontier"][d] for s in sp])
                                             for d in map(str, MID_BUDGETS)}
        out["fit_seconds"][k] = ms([per[s]["fit_seconds"][k] for s in sp])
    out["frontier"]["qmean|quant|prior"] = {d: ms([per[s]["quant_prior"]["frontier"][d] for s in sp])
                                            for d in map(str, MID_BUDGETS)}
    for a, b in [("bag10", "default"), ("tuned", "default"), ("qmean", "default"), ("qmedian", "default"),
                 ("qmean", "bag10"), ("qmean", "tuned"), ("qmean", "qmedian"), ("tuned", "bag10")]:
        d = [per[s]["centers"][a]["point"]["All"] - per[s]["centers"][b]["point"]["All"] for s in sp]
        out["paired"][f"RMSE {a} - {b}"] = ms(d) | {"n_wins": int(sum(x < 0 for x in d))}
        for dm in map(str, MID_BUDGETS):
            d = [per[s]["centers"][a]["frontier"][dm] - per[s]["centers"][b]["frontier"][dm] for s in sp]
            v = [x for x in d if not np.isnan(x)]
            out["paired"][f"tails@Δ={dm} {a} - {b}"] = ms(d) | {"n_wins": int(sum(x < 0 for x in v))}
    out["split42"] = {k: {"All": per["42"]["centers"][k]["point"]["All"],
                          "frontier": per["42"]["centers"][k]["frontier"]} for k in CENTERS} if "42" in per else {}
    return out


def print_summary(S):
    f = lambda d: f"{d['mean']:.3f}±{d['sd']:.3f}"
    print("\n[1] Trung tâm, λ=0 (RMSE, mean±sd qua các lần chia)")
    print("  trung tâm  Thấp    Giữa   Cao     Toàn bộ         OOF      thời gian fit (s)")
    for k in CENTERS:
        p = S["point"][k]
        print(f"  {k:9s}  {p['Low tail']['mean']:.2f}   {p['Middle']['mean']:.2f}   {p['High tail']['mean']:.2f}   "
              f"{f(p['All'])}   {p['oof_rmse']['mean']:.3f}    {S['fit_seconds'][k]['mean']:.1f}")
    print("\n[2] RMSE hai đuôi khi RMSE vùng giữa tăng Δ so với default (quy tắc Bayes, tầng bin, prior λ)")
    print("  họ                    " + "".join(f"  Δ={d:<11}" for d in MID_BUDGETS))
    for fam, v in S["frontier"].items():
        print(f"  {fam:22s}" + "".join(f"  {f(v[str(d)]):13s}" for d in MID_BUDGETS))
    print("\n[3] So từng cặp (âm = vế trái tốt hơn):")
    for k, v in S["paired"].items():
        print(f"  {k:32s} {v['mean']:+.3f} (sd {v['sd']:.3f}, thắng {v['n_wins']}/{v['n']})")
    if S["split42"]:
        print("\n[4] Riêng lần chia 42 (phép so sạch cho tuned):")
        for k, v in S["split42"].items():
            print(f"  {k:9s} RMSE={v['All']:.3f}  đuôi@Δ=0.5: {v['frontier']['0.5']:.3f}  đuôi@Δ=1.0: {v['frontier']['1.0']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default="results_cost/center_ablation.json")
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
    print(f"n={len(y)}, trung tâm: {CENTERS}, tuned = {tuned}", flush=True)

    per = {}
    for s in args.splits:
        per[str(s)] = run_split(s, X, y, prov, school, grade_cols, tuned)
        with open(args.out + ".partial", "w") as fh:
            json.dump(per, fh, ensure_ascii=False)
    summary = summarize(per)
    print_summary(summary)
    meta = {"protocol": "5 lần chia 80/20 như tail_prior; đặc trưng + mã hoá " + VARIANT
                        + "; tuned lấy từ " + args.tuned_from + " (dò trên train lần chia 42)",
            "splits": args.splits, "centers": CENTERS, "n_bag": N_BAG, "tuned_params": tuned}
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "summary": summary, "per_split": per}, fh, indent=1, ensure_ascii=False)
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
