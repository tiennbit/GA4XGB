# -*- coding: utf-8 -*-
"""Trọng số riêng cho hai đầu (K_thấp, K_cao) và ablation tỉnh/trường.
XGBoost MẶC ĐỊNH, chưa GA. Mọi thước đo theo RMSE.

Lượt 1, ablation đặc trưng: gốc / + tỉnh / + trường / + cả hai (phân cấp).
  Mã hoá dùng mức "lệch chuẩn chấm" (y − ridge(học bạ)), bản thắng ở cost_aware.py.
  Biến thể dùng cho lượt 2 được chọn theo RMSE OOF trên train, không nhìn test.

Lượt 2, lưới K_thấp × K_cao. Chi phí:
    sqrt( sum(c_i e_i²) / sum(c_i) ),  c_i = K_thấp nếu y<60, K_cao nếu y>=100, 1 nếu không.
  So ba chỗ đặt trọng số:
    stretch : giãn đầu ra quanh μ, hệ số riêng cho ŷ<μ và ŷ>=μ (hai phía tách
              được vì mỗi phía tác động lên một tập mẫu riêng), chọn trên OOF.
    kmean   : trung bình có trọng số của phân phối y | ŷ (sai số OOF theo bin ŷ).
    wtrain  : huấn luyện XGBoost với sample_weight = c(y).
  Vì sao wtrain đáng thử: loss bình phương có trọng số c(y) có nghiệm theo từng
  x là ∫c·y·p / ∫c·p, đúng đại lượng tối ưu của chi phí trên. Mô hình học thẳng
  điểm đó, không qua xấp xỉ bin như kmean. Trọng số β của bài cũ (n_bin^−β) đặt
  theo mật độ bin chứ không theo K, nên không có lý do gì để trúng điểm này.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from bins import HIGH_TAIL_MIN, LOW_TAIL_MAX
from cost_aware import (N_FOLDS, N_PRED_BINS, S_GRID, SEEDS, SPLIT_SEED,
                        HierEncoder, _eb_dev, lenient_target, province_audit,
                        region_rmse)
from preprocess import DATA_PATH, load_and_preprocess

K_GRID = [1, 2, 3, 5, 8]
PAIRS = [(kl, kh) for kl in K_GRID for kh in K_GRID]
VARIANTS = ["base", "enc_prov", "enc_school", "enc_both"]
METHODS = ["raw", "stretch", "kmean", "wtrain"]


def weights(y, kl, kh):
    y = np.asarray(y, dtype=float)
    return np.where(y < LOW_TAIL_MAX, float(kl), np.where(y >= HIGH_TAIL_MIN, float(kh), 1.0))


def cost(y, pred, kl, kh):
    c = weights(y, kl, kh)
    return float(np.sqrt(np.sum(c * (np.asarray(y, dtype=float) - pred) ** 2) / np.sum(c)))


# ---------------------------------------------------------------------------
# Đặc trưng
# ---------------------------------------------------------------------------
class FlatEncoder:
    """Một tầng: trung bình nhóm co về trung bình chung (empirical Bayes)."""

    def fit(self, t, g):
        t = np.asarray(t, dtype=float)
        self.mu = float(t.mean())
        self.dev, self.info = _eb_dev(t - self.mu, g)
        return self

    def transform(self, g):
        return (self.mu + pd.Series(g).map(self.dev).fillna(0.0).to_numpy())[:, None]


def _encoder(variant, prov, school):
    """Trả về (fit(t, rows), transform(enc, rows)) cho từng biến thể."""
    if variant == "enc_prov":
        return (lambda t, r: FlatEncoder().fit(t, prov[r]),
                lambda e, r: e.transform(prov[r]))
    if variant == "enc_school":
        # Trường không lồng trong tỉnh: co thẳng về trung bình chung. So với
        # enc_both để biết tầng tỉnh có thêm gì khi đã có tầng trường.
        return (lambda t, r: FlatEncoder().fit(t, school[r]),
                lambda e, r: e.transform(school[r]))
    return (lambda t, r: HierEncoder().fit(t, prov[r], school[r]),
            lambda e, r: e.transform(prov[r], school[r]))


def features(variant, X, y, prov, school, grade_cols, tr, ap, seed):
    Xtr, Xap = X.iloc[tr].to_numpy(float), X.iloc[ap].to_numpy(float)
    if variant == "base":
        return Xtr, Xap
    fit, tf = _encoder(variant, prov, school)
    t = lenient_target(X, y, tr, grade_cols)
    ncol = 2 if variant == "enc_both" else 1
    etr = np.zeros((len(tr), ncol))
    # Dòng huấn luyện nhận mã OOF để XGBoost không học trên chính nhãn của nó
    for itr, iva in KFold(N_FOLDS, shuffle=True, random_state=seed).split(tr):
        etr[iva] = tf(fit(t[itr], tr[itr]), tr[iva])
    eap = tf(fit(t, tr), ap)
    return np.hstack([Xtr, etr]), np.hstack([Xap, eap])


def xgb(Xtr, ytr, Xap, w=None):
    m = XGBRegressor(tree_method="hist", random_state=SPLIT_SEED, n_jobs=-1)
    m.fit(Xtr, ytr, sample_weight=w)
    return m.predict(Xap)


def oof_and_test(variant, X, y, prov, school, grade_cols, tr, te, seed):
    p_oof = np.zeros(len(tr))
    for otr, ova in KFold(N_FOLDS, shuffle=True, random_state=seed).split(tr):
        a, b = features(variant, X, y, prov, school, grade_cols, tr[otr], tr[ova], seed)
        p_oof[ova] = xgb(a, y[tr[otr]], b)
    Xtr, Xte = features(variant, X, y, prov, school, grade_cols, tr, te, seed)
    return p_oof, xgb(Xtr, y[tr], Xte), Xtr, Xte


# ---------------------------------------------------------------------------
# Quy tắc sau huấn luyện
# ---------------------------------------------------------------------------
def stretch_two_sided(p_oof, y_oof, p_new, mu, kl, kh):
    c = weights(y_oof, kl, kh)
    out, chosen = mu + (p_new - mu), {}
    for side, m_oof, m_new in [("low", p_oof < mu, p_new < mu), ("high", p_oof >= mu, p_new >= mu)]:
        sse = [np.sum(c[m_oof] * (y_oof[m_oof] - (mu + s * (p_oof[m_oof] - mu))) ** 2) for s in S_GRID]
        s = float(S_GRID[int(np.argmin(sse))])
        chosen[side] = s
        out[m_new] = mu + s * (p_new[m_new] - mu)
    return out, chosen


def k_mean(p_oof, y_oof, p_new, kl, kh):
    """Trung bình có trọng số c(y) của phân phối y | ŷ, lấy từ sai số OOF cùng bin ŷ."""
    edges = np.quantile(p_oof, np.linspace(0, 1, N_PRED_BINS + 1))
    b_oof = np.searchsorted(edges[1:-1], p_oof, side="right")
    b_new = np.searchsorted(edges[1:-1], p_new, side="right")
    out = np.empty(len(p_new))
    for b in range(N_PRED_BINS):
        idx = np.where(b_new == b)[0]
        if len(idx) == 0:
            continue
        r = y_oof[b_oof == b] - p_oof[b_oof == b]
        v = p_new[idx, None] + r[None, :]
        w = weights(v, kl, kh)
        out[idx] = (w * v).sum(axis=1) / w.sum(axis=1)
    return out


def evaluate(yt, pred, prov_te, kl, kh):
    return {"cost": cost(yt, pred, kl, kh), "regions": region_rmse(yt, pred),
            "audit": province_audit(yt, pred, prov_te)}


# ---------------------------------------------------------------------------
def ablation(seeds, X, y, prov, school, grade_cols, tr, te):
    res = {}
    for seed in seeds:
        res[str(seed)] = {}
        for v in VARIANTS:
            t0 = time.time()
            p_oof, p_te, _, _ = oof_and_test(v, X, y, prov, school, grade_cols, tr, te, seed)
            yt = y[te]
            res[str(seed)][v] = {
                "oof_rmse": float(np.sqrt(np.mean((y[tr] - p_oof) ** 2))),
                "test_rmse": float(np.sqrt(np.mean((yt - p_te) ** 2))),
                "test_r2": float(1 - np.mean((yt - p_te) ** 2) / np.var(yt)),
                "regions": region_rmse(yt, p_te)}
            r = res[str(seed)][v]
            print(f"  [ablation] seed {seed} {v:10s} OOF={r['oof_rmse']:.3f} test={r['test_rmse']:.3f} "
                  f"R2={r['test_r2']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return res


def k_grid(variant, seeds, X, y, prov, school, grade_cols, tr, te):
    res = {}
    mu = float(y[tr].mean())
    yt, prov_te = y[te], prov[te]
    for seed in seeds:
        t0 = time.time()
        p_oof, p_te, Xtr, Xte = oof_and_test(variant, X, y, prov, school, grade_cols, tr, te, seed)
        rs = {}
        for kl, kh in PAIRS:
            key = f"{kl},{kh}"
            s_pred, s_chosen = stretch_two_sided(p_oof, y[tr], p_te, mu, kl, kh)
            preds = {"raw": p_te, "stretch": s_pred,
                     "kmean": k_mean(p_oof, y[tr], p_te, kl, kh),
                     "wtrain": p_te if (kl, kh) == (1, 1)
                     else xgb(Xtr, y[tr], Xte, weights(y[tr], kl, kh))}
            rs[key] = {m: evaluate(yt, preds[m], prov_te, kl, kh) for m in METHODS}
            rs[key]["stretch_s"] = s_chosen
        res[str(seed)] = rs
        c = rs["3,3"]
        print(f"  [K grid] seed {seed} cost(3,3) " + " ".join(f"{m}={c[m]['cost']:.3f}" for m in METHODS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    return res


def ms(vals):
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals)}


def summarize(abl, grid):
    seeds = list(grid)
    s_abl = {v: {k: ms([abl[s][v][k] for s in abl]) for k in ["oof_rmse", "test_rmse", "test_r2"]}
             | {"regions": {rg: ms([abl[s][v]["regions"][rg] for s in abl])
                            for rg in ["Low tail", "Middle", "High tail", "All"]}}
             for v in VARIANTS}
    s_grid = {}
    for kl, kh in PAIRS:
        key = f"{kl},{kh}"
        e = {m: {"cost": ms([grid[s][key][m]["cost"] for s in seeds]),
                 "regions": {rg: ms([grid[s][key][m]["regions"][rg] for s in seeds])
                             for rg in ["Low tail", "Middle", "High tail", "All"]},
                 "audit_bias_sd": ms([grid[s][key][m]["audit"]["bias_sd"] for s in seeds])}
             for m in METHODS}
        # Chênh từng cặp trên cùng tập test: chính xác hơn so hai mean±sd rời nhau
        for a, b in [("wtrain", "stretch"), ("wtrain", "kmean"), ("kmean", "stretch")]:
            d = [grid[s][key][a]["cost"] - grid[s][key][b]["cost"] for s in seeds]
            e[f"diff_{a}_minus_{b}"] = ms(d) | {"n_wins": int(sum(x < 0 for x in d))}
        s_grid[key] = e
    return {"ablation": s_abl, "grid": s_grid}


def print_summary(summary, variant):
    print("\nAblation (RMSE, mean qua seed):")
    for v, r in summary["ablation"].items():
        rg = r["regions"]
        print(f"  {v:10s} OOF={r['oof_rmse']['mean']:.3f} test={r['test_rmse']['mean']:.3f} "
              f"R2={r['test_r2']['mean']:.4f} "
              + " ".join(f"{k}={rg[k]['mean']:.2f}" for k in ["Low tail", "Middle", "High tail"]))
    print(f"\nLưới K dùng biến thể '{variant}'. Hàng = K_thấp, cột = K_cao.")
    for m in METHODS:
        print(f"\n  {m}: chi phí RMSE")
        print("          " + "".join(f"  K_cao={kh:<3}" for kh in K_GRID))
        for kl in K_GRID:
            print(f"  K_thấp={kl:<2}" + "".join(
                f"  {summary['grid'][f'{kl},{kh}'][m]['cost']['mean']:8.3f}" for kh in K_GRID))
    print("\n  Cách thắng ở từng cặp:")
    for kl in K_GRID:
        row = []
        for kh in K_GRID:
            g = summary["grid"][f"{kl},{kh}"]
            row.append(min(METHODS, key=lambda m: g[m]["cost"]["mean"]))
        print(f"  K_thấp={kl:<2} " + " ".join(f"{b:>8}" for b in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default="results_cost/tail_weights.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    X, y = load_and_preprocess(args.data)
    raw = pd.read_csv(args.data, usecols=["Trường", "Tỉnh"])
    prov = raw["Tỉnh"].astype(str).to_numpy()
    # Khoá trường gồm cả tỉnh: tên trường trùng giữa các tỉnh (988 tên, 1.125 trường)
    school = (raw["Tỉnh"].astype(str) + "||" + raw["Trường"].astype(str)).to_numpy()
    y = y.to_numpy(float)
    grade_cols = [c for c in X.columns if c[:3] in ("10.", "11.", "12.")]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SPLIT_SEED)
    print(f"Train {len(tr)}, Test {len(te)}", flush=True)

    abl = ablation(args.seeds, X, y, prov, school, grade_cols, tr, te)
    oof = {v: np.mean([abl[s][v]["oof_rmse"] for s in abl]) for v in VARIANTS}
    variant = min(oof, key=oof.get)
    print(f"\nChọn '{variant}' cho lưới K (RMSE OOF thấp nhất trên train: {oof[variant]:.3f})", flush=True)

    grid = k_grid(variant, args.seeds, X, y, prov, school, grade_cols, tr, te)
    summary = summarize(abl, grid)
    print_summary(summary, variant)
    meta = {"protocol": "XGBoost mặc định; test 20% random_state=42; seed đổi fold OOF; "
                        "biến thể chọn theo RMSE OOF; tham số quy tắc chọn trên OOF train; mọi thước đo RMSE",
            "seeds": args.seeds, "K_grid": K_GRID, "variant_for_grid": variant,
            "tails": {"low_lt": LOW_TAIL_MAX, "high_ge": HIGH_TAIL_MIN}}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "summary": summary, "ablation": abl, "grid": grid},
                  f, indent=1, ensure_ascii=False)
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
