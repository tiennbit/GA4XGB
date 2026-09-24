# -*- coding: utf-8 -*-
"""Dự báo đuôi theo hai tầng: THÔNG TIN p(y|x) và TRỌNG SỐ w(y).
XGBoost mặc định, chưa GA. Mọi thước đo sai số theo RMSE.

Vì sao hai tầng. Dự báo thông thường là trung bình của p(y|x) ∝ p(x|y)·p(y).
Học bạ chỉ giải thích khoảng nửa phương sai HSA, nên tiên nghiệm p(y) (dồn quanh
77) kéo dự báo về giữa: đó là hiện tượng co về giữa, và đuôi chịu thiệt vì p(y)
có rất ít khối lượng ở hai đầu. Muốn kéo đuôi ra có hai việc tách bạch:
  1. Tầng thông tin: ước lượng p(y|x) càng sát từng em càng tốt. Hai em cùng ŷ=90,
     một em giỏi Toán ở trường chặt điểm, một em điểm đều ở trường nới điểm, thì
     xác suất đạt >=100 khác nhau. Chỉ nhìn ŷ thì không phân biệt được hai em.
  2. Tầng trọng số: chọn dự báo tối ưu dưới trọng số w(y). Với mất mát bình phương
     có trọng số, nghiệm là trung bình có trọng số  ŷ = E[w(y)·y | x] / E[w(y) | x].

Tầng thông tin, mỗi tầng biểu diễn p(y|x) bằng 50 điểm cách đều xác suất
(các phân vị TAUS) để mọi tầng dùng cùng độ phân giải:
  bin    : ŷ của mô hình trung bình + phân vị sai số OOF trong cùng bin ŷ.
           Chỉ biết ŷ; đây là cách của cost_aware.py / tail_weights.py.
  quant  : quantile XGBoost theo toàn bộ x (học bạ từng môn, trường, tỉnh...).
  qshape : trung bình lấy từ mô hình trung bình, HÌNH DẠNG (độ rộng, độ lệch) lấy
           từ quantile XGBoost. Tách được: lợi ích đến từ hình dạng theo x, hay chỉ
           từ việc đổi mô hình trung bình.
  Mỗi tầng hiệu chỉnh độ phủ trên OOF: dịch phân vị mức τ đi δ_τ = phân vị τ của
  (y − q̂_τ) trên OOF, để P(y <= q̂_τ) = τ (cùng tinh thần CQR, Romano et al. 2019).

Tầng trọng số, mỗi kiểu có một tham số có nghĩa:
  prior(λ): w(y) = p̂(y)^(−λ). Đổi tiên nghiệm p(y) thành p(y)^(1−λ); λ=1 là tiên
            nghiệm đều (mọi mức điểm như nhau trước khi nhìn học bạ). Đây là lập
            luận của Balanced MSE (Ren et al., CVPR 2022); p̂ làm trơn bằng nhân
            Gauss như LDS (Yang et al., ICML 2021).
  phi(K)  : w(y) = 1 + (K−1)·φ(y), φ là hàm relevance tự động theo boxplot
            (Ribeiro 2011; Ribeiro & Moniz 2020): 0 ở trung vị, 1 từ adjacent
            value trở ra, nội suy PCHIP. Cũng là φ dùng để tính SERA.
  step(K) : bậc thang cũ, K ở đuôi (y<60 hoặc >=100), 1 ở giữa. Giữ để so.

Baseline:
  raw      : mô hình trung bình, không hiệu chỉnh.
  stretch  : giãn đầu ra hai phía quanh μ, hệ số chọn trên OOF theo đúng trọng số
             của từng kiểu/tham số.
  lds_train: huấn luyện XGBoost với sample_weight = p̂(y)^(−λ) (LDS / DenseWeight).

Đánh giá: 5 lần chia train/test ngẫu nhiên 80/20 (khác bài cũ vốn cố định test),
nên độ lệch chuẩn phản ánh cả biến động do chia dữ liệu. So từng cặp trên cùng
lần chia. Đường đánh đổi được so ở cùng mức hy sinh vùng giữa: RMSE hai đuôi khi
RMSE vùng giữa tăng Δ so với raw, nội suy dọc theo tham số của từng họ.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from bins import HIGH_TAIL_MIN, LOW_TAIL_MAX, bin_index, regions
from cost_aware import N_FOLDS, N_PRED_BINS, S_GRID, province_audit
from preprocess import DATA_PATH, load_and_preprocess
from tail_weights import features

SPLITS = [42, 1, 2, 3, 4]
VARIANT = "enc_school"          # thắng ablation ở tail_weights.py, chọn theo RMSE OOF
TAUS = (np.arange(50) + 0.5) / 50
TIERS = ["bin", "quant", "qshape"]
LAMBDAS = [round(x, 2) for x in np.arange(0, 1.001, 0.1)]
# K tới 20 để mọi họ với tới mức hy sinh Δ=1,0 ở vùng giữa; lưới hẹp cho NaN ở đường đánh đổi
PHI_KS = [1, 1.5, 2, 3, 5, 8, 12, 20]
STEP_KS = [1, 1.5, 2, 3, 5, 8, 12, 20]
STEP_PAIRS = [(5, 1), (1, 5)]   # một đầu, để thấy kéo riêng từng đuôi
LDS_LAMBDAS = [0.25, 0.5, 0.75, 1.0]
KDE_SIGMA = 2.0                 # điểm HSA; LDS dùng nhân Gauss cỡ vài đơn vị nhãn
DENS_FLOOR = 0.01               # p̂ không xuống dưới 1% đỉnh -> trọng số tối đa 100^λ
MID_BUDGETS = [0.25, 0.5, 1.0]  # mức tăng RMSE vùng giữa chấp nhận được
Y_MAX = 150


def in_tail(y):
    y = np.asarray(y, dtype=float)
    return (y < LOW_TAIL_MAX) | (y >= HIGH_TAIL_MIN)


# ---------------------------------------------------------------------------
# Trọng số
# ---------------------------------------------------------------------------
class PriorDensity:
    """p̂(y) trên lưới điểm nguyên 0..150, làm trơn Gauss, chặn sàn."""

    def __init__(self, y_tr):
        cnt = np.bincount(np.clip(np.rint(y_tr), 0, Y_MAX).astype(int), minlength=Y_MAX + 1)
        k = np.arange(-int(4 * KDE_SIGMA), int(4 * KDE_SIGMA) + 1)
        ker = np.exp(-0.5 * (k / KDE_SIGMA) ** 2)
        d = np.convolve(cnt.astype(float), ker / ker.sum(), mode="same")
        d /= d.max()
        self.grid = np.arange(Y_MAX + 1)
        self.dens = np.maximum(d, DENS_FLOOR)

    def __call__(self, y):
        return np.interp(np.clip(y, 0, Y_MAX), self.grid, self.dens)


class Relevance:
    """φ(y) tự động theo boxplot (Ribeiro 2011): 1 ở adjacent value trở ra, 0 ở trung vị."""

    def __init__(self, y_tr):
        q1, med, q3 = np.quantile(y_tr, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        self.lo = float(y_tr[y_tr >= q1 - 1.5 * iqr].min())
        self.hi = float(y_tr[y_tr <= q3 + 1.5 * iqr].max())
        self.med = float(med)
        self.f = PchipInterpolator([self.lo, self.med, self.hi], [1.0, 0.0, 1.0])

    def __call__(self, y):
        y = np.asarray(y, dtype=float)
        return np.clip(np.where((y <= self.lo) | (y >= self.hi), 1.0,
                                self.f(np.clip(y, self.lo, self.hi))), 0.0, 1.0)


def weight_families(prior, phi):
    """{họ: [(tham số, hàm w)]}. Tham số đầu mỗi họ là w ≡ 1 (không nhấn đuôi)."""
    def step(kl, kh):
        return lambda v: np.where(np.asarray(v) < LOW_TAIL_MAX, float(kl),
                                  np.where(np.asarray(v) >= HIGH_TAIL_MIN, float(kh), 1.0))
    return {
        "prior": [(lam, (lambda v, l=lam: prior(v) ** (-l))) for lam in LAMBDAS],
        "phi": [(K, (lambda v, K=K: 1.0 + (K - 1.0) * phi(v))) for K in PHI_KS],
        "step": [(K, step(K, K)) for K in STEP_KS],
        "step_pair": [(f"{kl},{kh}", step(kl, kh)) for kl, kh in STEP_PAIRS],
    }


# ---------------------------------------------------------------------------
# Mô hình
# ---------------------------------------------------------------------------
def xgb_mean(Xtr, ytr, Xap, w=None):
    m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr, sample_weight=w)
    return m.predict(Xap)


def xgb_quant(Xtr, ytr, Xap):
    # Một mô hình, 50 mức phân vị (mỗi mức một cây mỗi vòng). Thử trên dữ liệu giả:
    # multi_output_tree chậm gấp đôi và pinball tệ hơn (3,56 so với 3,01).
    m = XGBRegressor(tree_method="hist", objective="reg:quantileerror",
                     quantile_alpha=TAUS, random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr)
    return np.sort(m.predict(Xap), axis=1)   # sắp lại: các đường phân vị cắt nhau


def calib_shift(S_oof, y_oof):
    """δ_τ để P(y <= q̂_τ + δ_τ) = τ trên OOF."""
    return np.array([np.quantile(y_oof - S_oof[:, j], TAUS[j]) for j in range(len(TAUS))])


def bin_samples(p_oof, y_oof, p_new):
    """ŷ + phân vị sai số OOF của bin ŷ tương ứng. Bin theo phân vị của ŷ."""
    edges = np.unique(np.quantile(p_oof, np.linspace(0, 1, N_PRED_BINS + 1)))
    b_oof = np.searchsorted(edges[1:-1], p_oof, side="right")
    b_new = np.searchsorted(edges[1:-1], p_new, side="right")
    R = np.vstack([np.quantile(y_oof[b_oof == b] - p_oof[b_oof == b], TAUS)
                   for b in range(len(edges) - 1)])
    return p_new[:, None] + R[b_new]


def build_tiers(p_oof, Q_oof, y_oof, p_te, Q_te):
    """Mỗi tầng: ma trận n×50 điểm cách đều xác suất của p(y|x) trên test."""
    out = {"bin": bin_samples(p_oof, y_oof, p_te)}
    out["quant"] = np.sort(Q_te + calib_shift(Q_oof, y_oof), axis=1)
    qs_oof = p_oof[:, None] + Q_oof - Q_oof.mean(axis=1, keepdims=True)
    qs_te = p_te[:, None] + Q_te - Q_te.mean(axis=1, keepdims=True)
    out["qshape"] = np.sort(qs_te + calib_shift(qs_oof, y_oof), axis=1)
    return out


def weighted_mean(S, wfun):
    w = wfun(S)
    return (w * S).sum(axis=1) / w.sum(axis=1)


def stretch_two_sided(p_oof, y_oof, p_new, mu, c):
    """Giãn quanh μ, hệ số riêng cho ŷ<μ và ŷ>=μ. Hai phía tác động lên hai tập
    mẫu rời nhau nên tối ưu tách được."""
    out = p_new.copy()
    for m_oof, m_new in [(p_oof < mu, p_new < mu), (p_oof >= mu, p_new >= mu)]:
        sse = [np.sum(c[m_oof] * (y_oof[m_oof] - (mu + s * (p_oof[m_oof] - mu))) ** 2) for s in S_GRID]
        s = float(S_GRID[int(np.argmin(sse))])
        out[m_new] = mu + s * (p_new[m_new] - mu)
    return out


# ---------------------------------------------------------------------------
# Thước đo
# ---------------------------------------------------------------------------
def sera(y, pred, phi_y, step=0.001):
    """SERA (Ribeiro & Moniz 2020): ∫_0^1 Σ_{φ(y_i)>=t} e_i² dt. Chia cho n để đọc."""
    e2 = (y - pred) ** 2
    order = np.argsort(-phi_y)
    ph, cs = phi_y[order], np.concatenate([[0.0], np.cumsum(e2[order])])
    t = np.arange(0, 1 + step / 2, step)
    k = np.searchsorted(-ph, -t, side="right")       # số mẫu có φ >= t
    return float(np.trapezoid(cs[k], t) / len(y))


def point_metrics(y, pred, phi_y, prov):
    e2 = (y - pred) ** 2
    r = {k: float(np.sqrt(e2[m].mean())) for k, m in regions(y).items()}
    r["Tails"] = float(np.sqrt(e2[in_tail(y)].mean()))
    b = bin_index(y)
    r["macro_bin"] = float(np.mean([np.sqrt(e2[b == i].mean()) for i in np.unique(b)]))
    r["sera"] = sera(y, pred, phi_y)
    r["audit_bias_sd"] = province_audit(y, pred, prov)["bias_sd"]
    return r


def dist_metrics(y, S):
    """Chất lượng p(y|x) của một tầng: pinball, độ lệch độ phủ, khoảng 90%,
    và xác suất rơi vào đuôi (Brier, AUC) — trả lời thẳng câu 'em này có khả năng
    đạt điểm cao không'."""
    d = y[:, None] - S
    pin = float(np.mean(np.maximum(TAUS * d, (TAUS - 1) * d)))
    cover = (y[:, None] <= S).mean(axis=0)
    j05, j95 = int(np.argmin(abs(TAUS - 0.05))), int(np.argmin(abs(TAUS - 0.95)))
    out = {"pinball": pin, "calib_err": float(np.mean(np.abs(cover - TAUS))),
           "pi90_cover": float(np.mean((y >= S[:, j05]) & (y <= S[:, j95]))),
           "pi90_width": float(np.mean(S[:, j95] - S[:, j05]))}
    for name, thr, low in [("low", LOW_TAIL_MAX, True), ("high", HIGH_TAIL_MIN, False)]:
        cdf = np.array([np.interp(thr, s, TAUS, left=0.0, right=1.0) for s in S])
        p = cdf if low else 1.0 - cdf
        ev = (y < thr) if low else (y >= thr)
        out[f"brier_{name}"] = float(np.mean((p - ev) ** 2))
        out[f"auc_{name}"] = float(roc_auc_score(ev, p)) if 0 < ev.sum() < len(ev) else float("nan")
    return out


def frontier(points, base_mid):
    """RMSE hai đuôi khi RMSE vùng giữa tăng Δ so với raw, nội suy dọc tham số."""
    mid = np.array([p["Middle"] for p in points])
    tails = np.array([p["Tails"] for p in points])
    o = np.argsort(mid)
    mid, tails = mid[o], tails[o]
    res = {}
    for d in MID_BUDGETS:
        tgt = base_mid + d
        res[str(d)] = float(np.interp(tgt, mid, tails)) if mid[0] <= tgt <= mid[-1] else float("nan")
    return res


# ---------------------------------------------------------------------------
def run_split(split, X, y, prov, school, grade_cols):
    t0 = time.time()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=split)
    y_tr, yt = y[tr], y[te]

    p_oof, Q_oof = np.zeros(len(tr)), np.zeros((len(tr), len(TAUS)))
    for otr, ova in KFold(N_FOLDS, shuffle=True, random_state=split).split(tr):
        A, B = features(VARIANT, X, y, prov, school, grade_cols, tr[otr], tr[ova], split)
        p_oof[ova] = xgb_mean(A, y[tr[otr]], B)
        Q_oof[ova] = xgb_quant(A, y[tr[otr]], B)
    Xtr, Xte = features(VARIANT, X, y, prov, school, grade_cols, tr, te, split)
    p_te = xgb_mean(Xtr, y_tr, Xte)
    Q_te = xgb_quant(Xtr, y_tr, Xte)
    print(f"  split {split}: xong mô hình ({time.time() - t0:.0f}s)", flush=True)

    prior, phi = PriorDensity(y_tr), Relevance(y_tr)
    phi_te, prov_te, mu = phi(yt), prov[te], float(y_tr.mean())
    fams = weight_families(prior, phi)
    tiers = build_tiers(p_oof, Q_oof, y_tr, p_te, Q_te)

    res = {"relevance": {"lo": phi.lo, "med": phi.med, "hi": phi.hi},
           "dist": {k: dist_metrics(yt, S) for k, S in tiers.items()},
           "raw": point_metrics(yt, p_te, phi_te, prov_te), "points": {}}
    for fam, params in fams.items():
        for tier, S in tiers.items():
            res["points"][f"{tier}|{fam}"] = [
                {"param": prm} | point_metrics(yt, weighted_mean(S, wf), phi_te, prov_te)
                for prm, wf in params]
        res["points"][f"stretch|{fam}"] = [
            {"param": prm} | point_metrics(yt, stretch_two_sided(p_oof, y_tr, p_te, mu, wf(y_tr)),
                                           phi_te, prov_te)
            for prm, wf in params]
    res["points"]["lds_train|prior"] = [{"param": 0.0} | res["raw"]] + [
        {"param": lam} | point_metrics(yt, xgb_mean(Xtr, y_tr, Xte, prior(y_tr) ** (-lam)),
                                       phi_te, prov_te)
        for lam in LDS_LAMBDAS]
    res["frontier"] = {k: frontier(v, res["raw"]["Middle"])
                       for k, v in res["points"].items() if not k.endswith("step_pair")}
    d = res["dist"]
    print(f"  split {split}: pinball bin/quant/qshape = "
          + "/".join(f"{d[t]['pinball']:.3f}" for t in TIERS)
          + "  AUC cao = " + "/".join(f"{d[t]['auc_high']:.3f}" for t in TIERS)
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    return res


def ms(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def summarize(per):
    sp = list(per)
    first = per[sp[0]]
    out = {"dist": {t: {m: ms([per[s]["dist"][t][m] for s in sp]) for m in first["dist"][t]}
                    for t in TIERS},
           "raw": {m: ms([per[s]["raw"][m] for s in sp]) for m in first["raw"]},
           "points": {}, "frontier": {}, "paired": {}}
    for fam, pts in first["points"].items():
        out["points"][fam] = [{"param": p["param"]} |
                              {m: ms([per[s]["points"][fam][i][m] for s in sp])
                               for m in p if m != "param"}
                              for i, p in enumerate(pts)]
    for fam in first["frontier"]:
        out["frontier"][fam] = {d: ms([per[s]["frontier"][fam][d] for s in sp]) for d in map(str, MID_BUDGETS)}
    # So từng cặp trên cùng lần chia: âm là vế trái tốt hơn
    for a, b in [("qshape|prior", "bin|prior"), ("quant|prior", "bin|prior"),
                 ("quant|prior", "qshape|prior"), ("quant|prior", "stretch|prior"),
                 ("quant|prior", "lds_train|prior"), ("quant|phi", "bin|phi"),
                 ("qshape|prior", "stretch|prior"), ("bin|prior", "stretch|prior"),
                 ("qshape|prior", "lds_train|prior"), ("qshape|phi", "bin|phi"),
                 ("qshape|prior", "qshape|phi"), ("qshape|prior", "qshape|step")]:
        for d in map(str, MID_BUDGETS):
            diffs = [per[s]["frontier"][a][d] - per[s]["frontier"][b][d] for s in sp]
            v = [x for x in diffs if not np.isnan(x)]
            out["paired"][f"{a} - {b} @Δmid={d}"] = ms(diffs) | {"n_wins": int(sum(x < 0 for x in v))}
    for t in ["quant", "qshape"]:
        for m in ["pinball", "brier_low", "brier_high", "auc_low", "auc_high"]:
            diffs = [per[s]["dist"][t][m] - per[s]["dist"]["bin"][m] for s in sp]
            out["paired"][f"{t} - bin: {m}"] = ms(diffs) | {"n_wins": int(sum(x < 0 for x in diffs))}
    return out


def print_summary(S):
    f = lambda d: f"{d['mean']:.3f}±{d['sd']:.3f}"
    print("\n[1] Chất lượng p(y|x) theo tầng thông tin (test, mean±sd qua các lần chia)")
    print("  tầng     pinball        lệch phủ   PI90 phủ  PI90 rộng  Brier thấp  AUC thấp  Brier cao  AUC cao")
    for t in TIERS:
        d = S["dist"][t]
        print(f"  {t:7s}  {f(d['pinball']):13s}  {d['calib_err']['mean']:.3f}     {d['pi90_cover']['mean']:.3f}     "
              f"{d['pi90_width']['mean']:6.2f}     {d['brier_low']['mean']:.4f}     {d['auc_low']['mean']:.3f}    "
              f"{d['brier_high']['mean']:.4f}     {d['auc_high']['mean']:.3f}")
    r = S["raw"]
    print(f"\n[2] raw: Thấp={r['Low tail']['mean']:.2f} Giữa={r['Middle']['mean']:.2f} Cao={r['High tail']['mean']:.2f} "
          f"Đuôi={r['Tails']['mean']:.2f} Toàn bộ={r['All']['mean']:.3f} SERA={r['sera']['mean']:.1f}")
    print("\n[3] RMSE hai đuôi khi chấp nhận RMSE vùng giữa tăng Δ (thấp hơn là tốt hơn)")
    print("  họ                    " + "".join(f"  Δ={d:<11}" for d in MID_BUDGETS))
    for fam, v in sorted(S["frontier"].items()):
        print(f"  {fam:22s}" + "".join(f"  {f(v[str(d)]):13s}" for d in MID_BUDGETS))
    print("\n[4] Trọng số prior(λ) theo tầng: RMSE theo vùng")
    for fam in ["bin|prior", "quant|prior", "qshape|prior", "stretch|prior", "lds_train|prior"]:
      for p in S["points"][fam]:
        if p["param"] in (0.0, 0.25, 0.2, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0):
            print(f"  {fam:16s} λ={p['param']:<4} Thấp={p['Low tail']['mean']:.2f} Giữa={p['Middle']['mean']:.2f} "
                  f"Cao={p['High tail']['mean']:.2f} Toàn bộ={p['All']['mean']:.3f} "
                  f"macro-bin={p['macro_bin']['mean']:.2f} SERA={p['sera']['mean']:.1f} "
                  f"audit_sd={p['audit_bias_sd']['mean']:.2f}")
    print("\n[5] Kéo riêng một đầu (step_pair):")
    for fam in ["bin|step_pair", "quant|step_pair", "qshape|step_pair", "stretch|step_pair"]:
        for p in S["points"][fam]:
            print(f"  {fam:18s} (K_thấp,K_cao)=({p['param']}) Thấp={p['Low tail']['mean']:.2f} "
                  f"Giữa={p['Middle']['mean']:.2f} Cao={p['High tail']['mean']:.2f}")
    print("\n[6] So từng cặp (âm = vế trái tốt hơn):")
    for k, v in S["paired"].items():
        print(f"  {k:52s} {v['mean']:+.3f} (sd {v['sd']:.3f}, thắng {v['n_wins']}/{v['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default="results_cost/tail_prior.json")
    ap.add_argument("--splits", type=int, nargs="+", default=SPLITS)
    args = ap.parse_args()

    X, y = load_and_preprocess(args.data)
    raw = pd.read_csv(args.data, usecols=["Trường", "Tỉnh"])
    prov = raw["Tỉnh"].astype(str).to_numpy()
    # Khoá trường gồm cả tỉnh: tên trường trùng giữa các tỉnh (988 tên, 1.125 trường)
    school = (raw["Tỉnh"].astype(str) + "||" + raw["Trường"].astype(str)).to_numpy()
    y = y.to_numpy(float)
    grade_cols = [c for c in X.columns if c[:3] in ("10.", "11.", "12.")]
    print(f"n={len(y)}, {X.shape[1]} đặc trưng gốc + mã hoá '{VARIANT}', {len(TAUS)} mức phân vị, "
          f"{len(args.splits)} lần chia", flush=True)

    per = {}
    for s in args.splits:
        per[str(s)] = run_split(s, X, y, prov, school, grade_cols)
        # Ghi dần sau mỗi lần chia: job dài, mất kết nối hay bị ngắt vẫn giữ phần đã xong
        with open(args.out + ".partial", "w") as fh:
            json.dump(per, fh, ensure_ascii=False)

    summary = summarize(per)
    print_summary(summary)
    meta = {"protocol": "5 lần chia 80/20 ngẫu nhiên (random_state=split); XGBoost mặc định; "
                        f"đặc trưng + mã hoá {VARIANT}; hiệu chỉnh phân vị và tham số stretch chọn trên OOF train; "
                        "mọi sai số theo RMSE; SERA chia n",
            "splits": args.splits, "taus": TAUS.tolist(), "lambdas": LAMBDAS, "phi_Ks": PHI_KS,
            "step_Ks": STEP_KS, "step_pairs": STEP_PAIRS, "lds_lambdas": LDS_LAMBDAS,
            "kde_sigma": KDE_SIGMA, "dens_floor": DENS_FLOOR, "mid_budgets": MID_BUDGETS,
            "tails": {"low_lt": LOW_TAIL_MAX, "high_ge": HIGH_TAIL_MIN}}
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "summary": summary, "per_split": per}, fh, indent=1, ensure_ascii=False)
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
