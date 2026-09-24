# -*- coding: utf-8 -*-
"""Lượt thử khả thi cho hướng cost-aware: XGBoost MẶC ĐỊNH (chưa GA), mã hoá
trường/tỉnh phân cấp, và quy tắc quyết định theo K.

Hai câu hỏi, mỗi câu là một cổng quyết định có đi tiếp hay không:
  1. Thêm thông tin trường/tỉnh có kéo được hai đuôi ra khỏi vùng giữa không?
     Thông tin thật dịch cả đường đánh đổi; giãn đầu ra chỉ trượt dọc theo nó.
  2. Quy tắc trung vị có trọng số theo K có thắng giãn đầu ra và quantile map
     ở mọi K không?

K: một lỗi ở đuôi (y<60 hoặc y>=100) nặng bằng K lỗi ở vùng giữa.
   chi phí_K = sum(c_i |e_i|) / sum(c_i),  c_i = K nếu i ở đuôi, 1 nếu không.
   K=1 chính là MAE.

Vì sao trung vị có trọng số: với mất mát c(y)|ŷ−y|, đạo hàm của kỳ vọng theo ŷ
là ∫_{y<ŷ} c·p − ∫_{y>ŷ} c·p, bằng 0 khi ŷ là trung vị của phân phối ∝ c(y)·p(y|x).
Tức là dự báo tối ưu không còn là trung bình có điều kiện. Fitness α của bài cũ
chỉ tác động qua siêu tham số, mà siêu tham số chỉ đổi cách ước lượng trung bình
có điều kiện, nên nó không thể tới được điểm này. Đây là lý do α bị lấn át.

Tập test: cùng phân hoạch 80/20, random_state=42 như bài cũ, để so được với
results_mseed/. Seed chỉ đổi phân hoạch fold nội bộ (OOF cho mã hoá và cho hiệu
chỉnh); XGBoost mặc định không lấy mẫu con nên bản thân nó tất định.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from bins import HIGH_TAIL_MIN, LOW_TAIL_MAX, regions
from preprocess import DATA_PATH, load_and_preprocess

SEEDS = [42, 1, 2, 3, 4]
KS = [1, 2, 3, 5, 8]
SPLIT_SEED = 42
N_FOLDS = 5
N_PRED_BINS = 20                      # số bin theo ŷ cho phân phối sai số có điều kiện
S_GRID = np.round(np.arange(0.80, 2.505, 0.01), 2)
VARIANTS = ["base", "enc_hsa", "enc_lenient"]
AUDIT_MIN_N = 100                     # tỉnh có ít nhất ngần này em trong test mới audit


def in_tail(y):
    y = np.asarray(y, dtype=float)
    return (y < LOW_TAIL_MAX) | (y >= HIGH_TAIL_MIN)


def cost(y, pred, K):
    c = np.where(in_tail(y), float(K), 1.0)
    return float(np.sum(c * np.abs(np.asarray(y, dtype=float) - pred)) / np.sum(c))


def region_mae(y, pred):
    e = np.abs(np.asarray(y, dtype=float) - pred)
    return {k: float(e[m].mean()) for k, m in regions(y).items()}


# ---------------------------------------------------------------------------
# Mã hoá phân cấp tỉnh -> trường (empirical Bayes)
# ---------------------------------------------------------------------------
def _eb_dev(d, g):
    """Độ lệch trung bình của từng nhóm, co về 0 theo empirical Bayes.

    Ước lượng bằng method of moments: σ² là phương sai trong nhóm (gộp),
    τ² = var(trung bình nhóm) − E[σ²/n]. Nhóm n em được giữ lại tỷ lệ
    n/(n + σ²/τ²) của độ lệch thô. Trường 3 em gần như bị co hết về mức tỉnh,
    nên không cần ngưỡng cỡ nhóm tuỳ tiện.
    """
    s = pd.DataFrame({"g": g, "d": d}).groupby("g")["d"].agg(["size", "mean", "var"])
    ok = s["size"] >= 2
    sigma2 = float(np.average(s.loc[ok, "var"], weights=s.loc[ok, "size"] - 1))
    tau2 = max(float(s["mean"].var() - np.mean(sigma2 / s["size"])), 1e-6)
    dev = s["mean"] * s["size"] / (s["size"] + sigma2 / tau2)
    return dev.to_dict(), {"sigma2": sigma2, "tau2": tau2, "n_groups": int(len(s))}


class HierEncoder:
    """Mức tỉnh co về trung bình chung; mức trường co về mức tỉnh của nó."""

    def fit(self, t, prov, school):
        t = np.asarray(t, dtype=float)
        self.mu = float(t.mean())
        self.prov_dev, self.prov_info = _eb_dev(t - self.mu, prov)
        p = self._prov(prov)
        self.school_dev, self.school_info = _eb_dev(t - p, school)
        return self

    def _prov(self, prov):
        return self.mu + pd.Series(prov).map(self.prov_dev).fillna(0.0).to_numpy()

    def transform(self, prov, school):
        p = self._prov(prov)
        s = p + pd.Series(school).map(self.school_dev).fillna(0.0).to_numpy()
        return np.column_stack([p, s])


def encode(t_tr, prov, school, tr, ap, seed):
    """Dòng huấn luyện nhận mã OOF (5 fold nội bộ) để XGBoost không học trên
    chính nhãn của nó; dòng cần dự báo nhận mã từ toàn bộ tập huấn luyện."""
    enc_tr = np.zeros((len(tr), 2))
    for itr, iva in KFold(N_FOLDS, shuffle=True, random_state=seed).split(tr):
        e = HierEncoder().fit(t_tr[itr], prov[tr[itr]], school[tr[itr]])
        enc_tr[iva] = e.transform(prov[tr[iva]], school[tr[iva]])
    e = HierEncoder().fit(t_tr, prov[tr], school[tr])
    return enc_tr, e.transform(prov[ap], school[ap]), e


def lenient_target(X, y, tr, grade_cols):
    """Phần HSA mà học bạ không giải thích được: y − ridge(học bạ).

    Trung bình của nó theo trường là mức "lệch chuẩn chấm" của trường so với
    học bạ. Ridge fit trên chính tr (phần dư trong mẫu): 90 hệ số trên hàng
    chục nghìn dòng thì độ lạc quan không đáng kể so với τ² cỡ vài điểm.
    """
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    m.fit(X.iloc[tr][grade_cols], y[tr])
    return y[tr] - m.predict(X.iloc[tr][grade_cols])


def fit_predict(variant, X, y, prov, school, grade_cols, tr, ap, seed):
    Xtr, Xap = X.iloc[tr].to_numpy(float), X.iloc[ap].to_numpy(float)
    info = None
    if variant != "base":
        t = y[tr] if variant == "enc_hsa" else lenient_target(X, y, tr, grade_cols)
        etr, eap, enc = encode(t, prov, school, tr, ap, seed)
        Xtr, Xap = np.hstack([Xtr, etr]), np.hstack([Xap, eap])
        info = {"prov": enc.prov_info, "school": enc.school_info}
    m = XGBRegressor(tree_method="hist", random_state=SPLIT_SEED, n_jobs=-1)
    m.fit(Xtr, y[tr])
    return m.predict(Xap), info


# ---------------------------------------------------------------------------
# Quy tắc quyết định
# ---------------------------------------------------------------------------
def k_median(p_oof, y_oof, p_new, K):
    """Trung vị có trọng số của phân phối y | ŷ.

    Phân phối lấy từ sai số OOF trong cùng bin ŷ (bin theo phân vị). Bin theo ŷ
    chứ không dùng một phân phối sai số chung, vì co về giữa làm sai số ở ŷ cao
    lệch âm, ở ŷ thấp lệch dương; phân phối chung sẽ xoá mất đúng hiện tượng
    ta cần sửa.
    """
    edges = np.quantile(p_oof, np.linspace(0, 1, N_PRED_BINS + 1))
    b_oof = np.searchsorted(edges[1:-1], p_oof, side="right")
    b_new = np.searchsorted(edges[1:-1], p_new, side="right")
    out = np.empty(len(p_new))
    for b in range(N_PRED_BINS):
        idx = np.where(b_new == b)[0]
        if len(idx) == 0:
            continue
        r = np.sort(y_oof[b_oof == b] - p_oof[b_oof == b])
        v = p_new[idx, None] + r[None, :]                # ứng viên y, đã sắp tăng dần
        w = np.where(in_tail(v), float(K), 1.0)
        cw = np.cumsum(w, axis=1)
        j = (cw < cw[:, -1:] / 2).sum(axis=1)
        out[idx] = v[np.arange(len(idx)), j]
    return out


def all_rules(p_oof, y_oof, p_te, mu):
    """Trả về {rule: {K: dự báo test}} và tham số đã chọn trên OOF."""
    preds, chosen = {}, {}
    preds["raw"] = {K: p_te for K in KS}

    # Quantile map: ép phân phối dự báo khớp phân phối y (không phụ thuộc K)
    rank = np.searchsorted(np.sort(p_oof), p_te) / len(p_oof)
    qm = np.quantile(y_oof, np.clip(rank, 0.0, 1.0))
    preds["qmap"] = {K: qm for K in KS}

    # Giãn tuyến tính quanh μ, s chọn riêng cho từng K trên OOF
    preds["stretch"], chosen["stretch_s"] = {}, {}
    for K in KS:
        c = [cost(y_oof, mu + s * (p_oof - mu), K) for s in S_GRID]
        s = float(S_GRID[int(np.argmin(c))])
        chosen["stretch_s"][K] = s
        preds["stretch"][K] = mu + s * (p_te - mu)

    preds["kmedian"] = {K: k_median(p_oof, y_oof, p_te, K) for K in KS}
    return preds, chosen


def province_audit(y, pred, prov):
    df = pd.DataFrame({"p": prov, "e": pred - y})
    s = df.groupby("p")["e"].agg(["mean", "size"])
    s = s[s["size"] >= AUDIT_MIN_N]["mean"]
    return {"n_prov": int(len(s)), "bias_sd": float(s.std()),
            "bias_min": float(s.min()), "bias_max": float(s.max())}


# ---------------------------------------------------------------------------
def run_seed(seed, X, y, prov, school, grade_cols, tr, te):
    res = {}
    mu = float(y[tr].mean())
    for v in VARIANTS:
        t0 = time.time()
        p_oof = np.zeros(len(tr))
        for otr, ova in KFold(N_FOLDS, shuffle=True, random_state=seed).split(tr):
            p_oof[ova], _ = fit_predict(v, X, y, prov, school, grade_cols,
                                        tr[otr], tr[ova], seed)
        p_te, info = fit_predict(v, X, y, prov, school, grade_cols, tr, te, seed)
        preds, chosen = all_rules(p_oof, y[tr], p_te, mu)
        yt = y[te]
        r = {"raw_metrics": {"rmse": float(np.sqrt(np.mean((yt - p_te) ** 2))),
                             "mae": float(np.mean(np.abs(yt - p_te))),
                             "r2": float(1 - np.mean((yt - p_te) ** 2) / np.var(yt)),
                             "sd_ratio": float(np.std(p_te) / np.std(yt))},
             "encoder": info, "chosen": chosen, "rules": {}}
        for rule, byK in preds.items():
            r["rules"][rule] = {str(K): {"cost": cost(yt, byK[K], K),
                                         "regions": region_mae(yt, byK[K])} for K in KS}
            r["rules"][rule]["audit_K3"] = province_audit(yt, byK[3], prov[te])
        res[v] = r
        print(f"  seed {seed} {v:12s} R2={r['raw_metrics']['r2']:.4f} "
              f"MAE={r['raw_metrics']['mae']:.3f} "
              f"cost_K3 raw/stretch/qmap/kmed="
              + "/".join(f"{r['rules'][k]['3']['cost']:.3f}"
                         for k in ["raw", "stretch", "qmap", "kmedian"])
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    return res


def summarize(per_seed):
    def ms(vals):
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals)}
    seeds = list(per_seed)
    out = {}
    for v in VARIANTS:
        out[v] = {"raw_metrics": {m: ms([per_seed[s][v]["raw_metrics"][m] for s in seeds])
                                  for m in ["rmse", "mae", "r2", "sd_ratio"]}, "rules": {}}
        for rule in per_seed[seeds[0]][v]["rules"]:
            out[v]["rules"][rule] = {
                str(K): {"cost": ms([per_seed[s][v]["rules"][rule][str(K)]["cost"] for s in seeds]),
                         "regions": {rg: ms([per_seed[s][v]["rules"][rule][str(K)]["regions"][rg]
                                             for s in seeds])
                                     for rg in ["Low tail", "Middle", "High tail", "All"]}}
                for K in KS}
            out[v]["rules"][rule]["audit_K3_bias_sd"] = ms(
                [per_seed[s][v]["rules"][rule]["audit_K3"]["bias_sd"] for s in seeds])
    return out


def print_table(summary):
    print("\nChi phí theo K trên test (mean qua seed). Thấp hơn là tốt hơn.")
    head = "variant      rule     " + "".join(f"   K={K:<5}" for K in KS)
    print(head)
    for v in VARIANTS:
        for rule in ["raw", "stretch", "qmap", "kmedian"]:
            row = summary[v]["rules"][rule]
            print(f"{v:12s} {rule:8s} " + "".join(f"  {row[str(K)]['cost']['mean']:8.3f}" for K in KS))
    print("\nMAE theo vùng, dự báo thô (raw):")
    for v in VARIANTS:
        rg = summary[v]["rules"]["raw"]["1"]["regions"]
        rm = summary[v]["raw_metrics"]
        print(f"{v:12s} R2={rm['r2']['mean']:.4f} "
              + " ".join(f"{k}={rg[k]['mean']:.2f}" for k in ["Low tail", "Middle", "High tail", "All"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default="results_cost/cost_aware.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    X, y = load_and_preprocess(args.data)
    raw = pd.read_csv(args.data, usecols=["Trường", "Tỉnh"])
    prov = raw["Tỉnh"].astype(str).to_numpy()
    # Khoá trường gồm cả tỉnh: tên như "THPT Nguyễn Du" trùng giữa nhiều tỉnh,
    # gộp theo tên trần sẽ trộn các trường khác nhau vào một nhóm.
    school = (raw["Tỉnh"].astype(str) + "||" + raw["Trường"].astype(str)).to_numpy()
    y = y.to_numpy(float)
    grade_cols = [c for c in X.columns if c[:3] in ("10.", "11.", "12.")]

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SPLIT_SEED)
    print(f"Train {len(tr)}, Test {len(te)}, {X.shape[1]} đặc trưng gốc, "
          f"{len(grade_cols)} cột học bạ, {len(set(school))} trường", flush=True)

    per_seed = {}
    for seed in args.seeds:
        per_seed[str(seed)] = run_seed(seed, X, y, prov, school, grade_cols, tr, te)

    summary = summarize(per_seed)
    print_table(summary)
    meta = {"protocol": "XGBoost mặc định (tree_method=hist), test 20% random_state=42; "
                        "seed đổi fold OOF cho mã hoá và hiệu chỉnh; tham số quy tắc chọn trên OOF train",
            "Ks": KS, "seeds": args.seeds, "variants": VARIANTS,
            "tails": {"low_lt": LOW_TAIL_MAX, "high_ge": HIGH_TAIL_MIN},
            "n_pred_bins": N_PRED_BINS}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "summary": summary, "per_seed": per_seed}, f,
                  indent=1, ensure_ascii=False)
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
