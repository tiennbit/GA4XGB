# -*- coding: utf-8 -*-
"""Tầng quyết định: các cách ước lượng μ_K^G = E[w(Y)Y | ŷ] / E[w(Y) | ŷ] (mục 5.4).

Mọi cách dùng CHUNG một giao diện để E2, E4, E5, E6, E9 hoán đổi được:

    rule = make_rule("R2").fit(yhat_oof, y_oof, wfun)   # khớp trên (ŷ OOF, y) của tập huấn luyện
    pred = rule.predict(yhat_test)                       # áp cho ŷ của mô hình khớp lại
    rule.info()                                          # tham số đã chọn, để ghi JSON

wfun là hàm trọng số của y (họ bậc thang, prior, φ ở dưới); None nghĩa là w ≡ 1.

  R0   raw          ŷ.
  R1   Bayes theo bin phần dư: B bin bằng số lượng của ŷ OOF; mỗi ŷ nhận S điểm
                    ŷ + r_(τ_j, b), τ_j = (j - 0,5)/S; g = Σ w(s)s / Σ w(s).
                    Cùng thuật toán với tail_prior.bin_samples + weighted_mean (kiểm
                    trong tests/test_infra.py) nhưng tách fit/predict và cho đổi B, S,
                    kiểu bin (E5). predict nhận wfun khác để dùng lại phân vị đã khớp.
  R1_1 R1 với w ≡ 1: phần hiệu chỉnh ở K = 1, tách khỏi phần nghiêng theo K.
  R2   isotonic có trọng số (PAV, sample_weight = w(y)): ước lượng chuẩn của μ_w^G
                    dưới ràng buộc đơn điệu, không có tham số B, S.
  R3   histogram có trọng số: cùng bin như R1, giá trị bin = Σ w y / Σ w.
  R4   tuyến tính có trọng số: WLS y ~ a + b·ŷ.
  R5   giãn hai phía quanh μ = trung bình y huấn luyện, s riêng cho ŷ < μ và ŷ >= μ,
                    chọn theo SSE có trọng số trên OOF; lưới [0,50; 4,00] bước 0,01
                    (lưới cũ [0,80; 2,50] có thể chặn ở K lớn). info() ghi s có chạm
                    biên lưới không để E5 đếm.
  R6   quantile mapping (như cost_aware.all_rules): không phụ thuộc K, chỉ làm tham
                    chiếu ở K = 1.
  R7   reframing chuẩn (Hernández-Orallo 2014): trong bin b, m_b và s_b là trung bình
                    và SD phần dư; s_j = ŷ + m_b + s_b·Φ⁻¹(τ_j), cắt trong [0; 150],
                    rồi trung bình có trọng số.

R1 KHÔNG có dịch chuyển kiểu CQR và không có tập hiệu chỉnh riêng, nên không mang
bảo đảm conformal nào; bài không gọi nó là CQR hay Mondrian CPD (mục 5.4).

Bin rỗng: với bin bằng bề rộng (E5) hoặc bin bằng số lượng trên mẫu nhỏ, một bin có
thể không có phần dư nào; bin đó được gộp vào bin kề thay vì để np.quantile lỗi.
Khi không có bin rỗng (mọi lượt chính), kết quả trùng tail_prior.bin_samples.
"""
import copy

import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, KFold

import tail_prior
from gates import (DENS_FLOOR, KDE_SIGMA, K_DENSE, LAMBDA_DENSE, N_BINS, N_SAMPLES,
                   RULE_CROSSFIT_FOLDS, STRETCH_S_MAX, STRETCH_S_MIN, STRETCH_S_STEP,
                   TAIL_MASS_HIGH, TAIL_MASS_LOW, Y_MAX, Y_MIN)
from tail_prior import PriorDensity, Relevance, weighted_mean

S_GRID = np.round(np.arange(STRETCH_S_MIN, STRETCH_S_MAX + STRETCH_S_STEP / 2, STRETCH_S_STEP), 2)
K_FREE = {"R0", "R1_1", "R6"}         # quy tắc không phụ thuộc w: tính một lần cho mọi K


# ---------------------------------------------------------------------------
# Đuôi theo khối lượng
# ---------------------------------------------------------------------------
def tail_cutoffs(y_train, mass_low=TAIL_MASS_LOW, mass_high=TAIL_MASS_HIGH):
    """(lo, hi): đuôi thấp y < lo, đuôi cao y >= hi, CHỈ từ y huấn luyện.

    Điểm HSA là số nguyên nên khối lượng nhảy bậc (~1,3% mỗi điểm quanh 60). Chọn
    ngưỡng trong các giá trị quan sát có khối lượng GẦN NHẤT với mục tiêu, không
    chọn "ngưỡng đầu tiên vượt mục tiêu": với P(y < 60) = 9,25% trên toàn khoá,
    tập huấn luyện của một lần chia có thể cho 9,21% và quy tắc "vượt" sẽ nhảy sang
    61. Quy tắc gần nhất cho 60/100 ổn định qua các lần chia và vẫn đúng với y liên
    tục (E11)."""
    y = np.sort(np.asarray(y_train, dtype=float))
    u = np.unique(y)
    below = np.searchsorted(y, u, side="left") / len(y)            # P(y < u)
    lo = float(u[int(np.argmin(np.abs(below - mass_low)))])
    hi = float(u[int(np.argmin(np.abs((1.0 - below) - mass_high)))])
    return lo, hi


def tail_masses(y, lo, hi):
    y = np.asarray(y, dtype=float)
    return float(np.mean(y < lo)), float(np.mean(y >= hi))


# ---------------------------------------------------------------------------
# Họ trọng số: hàm của y, nhận mảng mọi chiều (cả ma trận n×S điểm mẫu)
# ---------------------------------------------------------------------------
def step(K_low, K_high, lo, hi):
    """w(y) = K_L nếu y < lo, K_H nếu y >= hi, 1 nếu không."""
    def w(v):
        v = np.asarray(v, dtype=float)
        return np.where(v < lo, float(K_low), np.where(v >= hi, float(K_high), 1.0))
    w.family, w.param = "step", (K_low, K_high)
    return w


class _Density:
    """Bản có tham số của tail_prior.PriorDensity (E5 đổi σ và sàn). Cùng thuật toán;
    ở σ = 2, sàn = 0,01 trùng khít PriorDensity (kiểm trong tests/test_infra.py)."""

    def __init__(self, y_tr, sigma, floor, y_max=tail_prior.Y_MAX):
        cnt = np.bincount(np.clip(np.rint(y_tr), 0, y_max).astype(int), minlength=y_max + 1)
        k = np.arange(-int(4 * sigma), int(4 * sigma) + 1)
        ker = np.exp(-0.5 * (k / sigma) ** 2)
        d = np.convolve(cnt.astype(float), ker / ker.sum(), mode="same")
        d /= d.max()
        self.y_max = y_max
        self.grid = np.arange(y_max + 1)
        self.dens = np.maximum(d, floor)

    def __call__(self, y):
        return np.interp(np.clip(y, 0, self.y_max), self.grid, self.dens)


def density(y_train, sigma=KDE_SIGMA, floor=DENS_FLOOR):
    """p̂(y) kiểu LDS trên y huấn luyện. Mặc định dùng thẳng tail_prior.PriorDensity."""
    y_train = np.asarray(y_train, dtype=float)
    if sigma == tail_prior.KDE_SIGMA and floor == tail_prior.DENS_FLOOR:
        return PriorDensity(y_train)
    return _Density(y_train, sigma, floor)


def prior(lam, dens):
    """w(y) = p̂(y)^(-λ). `dens` là mật độ đã khớp (density()) hoặc mảng y huấn luyện."""
    if not callable(dens):
        dens = density(dens)

    def w(v):
        return dens(v) ** (-float(lam))
    w.family, w.param = "prior", lam
    return w


def phi(K, rel):
    """w(y) = 1 + (K - 1)·φ(y). `rel` là tail_prior.Relevance đã khớp hoặc mảng y huấn luyện."""
    if not callable(rel):
        rel = Relevance(np.asarray(rel, dtype=float))

    def w(v):
        return 1.0 + (float(K) - 1.0) * rel(v)
    w.family, w.param = "phi", K
    return w


def weight_grid(y_train, lo, hi, step_Ks=K_DENSE, lambdas=LAMBDA_DENSE, phi_Ks=K_DENSE,
                asym_pairs=()):
    """{họ: [(tham số, wfun)]} trên lưới dày của mục 5.6, mật độ và φ khớp trên y huấn luyện."""
    dens, rel = density(y_train), Relevance(np.asarray(y_train, dtype=float))
    return {"step": [(K, step(K, K, lo, hi)) for K in step_Ks],
            "prior": [(lam, prior(lam, dens)) for lam in lambdas],
            "phi": [(K, phi(K, rel)) for K in phi_Ks],
            "step_pair": [(f"{a},{b}", step(a, b, lo, hi)) for a, b in asym_pairs]}


def weights_of(wfun, y):
    y = np.asarray(y, dtype=float)
    return np.ones_like(y) if wfun is None else np.asarray(wfun(y), dtype=float)


def normalize_weights(w):
    """Chuẩn hoá về trung bình 1 cho huấn luyện có trọng số (E2b): thang trọng số làm
    đổi tổng hessian so với min_child_weight và reg_lambda (Nhận xét 4)."""
    w = np.asarray(w, dtype=float)
    return w / w.mean()


# ---------------------------------------------------------------------------
# Bin theo ŷ OOF
# ---------------------------------------------------------------------------
def taus(S):
    return (np.arange(int(S)) + 0.5) / int(S)


def bin_assign(edges, x):
    """Chỉ số bin, giống tail_prior.bin_samples: ŷ ngoài miền OOF rơi vào bin đầu/cuối."""
    return np.searchsorted(edges[1:-1], np.asarray(x, dtype=float), side="right")


def _merge_empty(edges, x):
    edges = np.asarray(edges, dtype=float)
    while len(edges) > 2:
        cnt = np.bincount(bin_assign(edges, x), minlength=len(edges) - 1)
        empty = np.where(cnt == 0)[0]
        if len(empty) == 0:
            break
        j = int(empty[0])
        # Bỏ mép giữa bin rỗng và bin kề (bên phải nếu có, không thì bên trái)
        edges = np.delete(edges, j + 1 if j < len(edges) - 2 else j)
    return edges


def bin_edges(x, n_bins=N_BINS, binning="count"):
    x = np.asarray(x, dtype=float)
    if binning == "count":
        e = np.unique(np.quantile(x, np.linspace(0, 1, int(n_bins) + 1)))
    elif binning == "width":
        e = np.linspace(x.min(), x.max(), int(n_bins) + 1)
    else:
        raise ValueError(f"binning lạ: {binning}")
    if len(e) < 2:                      # mọi ŷ bằng nhau: một bin duy nhất
        e = np.array([x.min(), x.max()])
    return _merge_empty(e, x)


# ---------------------------------------------------------------------------
# Các cách ước lượng
# ---------------------------------------------------------------------------
class Rule:
    code = "R?"

    def fit(self, yhat_oof, y_oof, wfun=None):
        return self

    def predict(self, yhat):
        raise NotImplementedError

    def info(self):
        return {}


class Raw(Rule):
    code = "R0"

    def predict(self, yhat):
        return np.asarray(yhat, dtype=float).copy()


class ResidualBinBayes(Rule):
    """R1 (uniform=False) và R1₁ (uniform=True)."""

    def __init__(self, n_bins=N_BINS, n_samples=N_SAMPLES, binning="count", uniform=False):
        self.n_bins, self.n_samples, self.binning = n_bins, n_samples, binning
        self.uniform = uniform
        self.code = "R1_1" if uniform else "R1"

    def fit(self, yhat_oof, y_oof, wfun=None):
        p, y = np.asarray(yhat_oof, dtype=float), np.asarray(y_oof, dtype=float)
        self.edges = bin_edges(p, self.n_bins, self.binning)
        b = bin_assign(self.edges, p)
        self.taus = taus(self.n_samples)
        self.R = np.vstack([np.quantile(y[b == k] - p[b == k], self.taus)
                            for k in range(len(self.edges) - 1)])
        self.wfun = None if self.uniform else wfun
        return self

    def samples(self, yhat):
        """Ma trận n×S điểm cách đều xác suất của p(y | ŷ) (dùng cho E9: P(Y >= c), PI90)."""
        p = np.asarray(yhat, dtype=float)
        return p[:, None] + self.R[bin_assign(self.edges, p)]

    def predict(self, yhat, wfun=None):
        S = self.samples(yhat)
        wf = None if self.uniform else (wfun if wfun is not None else self.wfun)
        return S.mean(axis=1) if wf is None else weighted_mean(S, wf)

    def info(self):
        return {"n_bins": int(len(self.edges) - 1), "n_samples": int(self.n_samples),
                "binning": self.binning}


class WeightedIsotonic(Rule):
    code = "R2"

    def fit(self, yhat_oof, y_oof, wfun=None):
        p, y = np.asarray(yhat_oof, dtype=float), np.asarray(y_oof, dtype=float)
        self.iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self.iso.fit(p, y, sample_weight=weights_of(wfun, y))
        return self

    def predict(self, yhat):
        return self.iso.predict(np.asarray(yhat, dtype=float))

    def info(self):
        return {"n_steps": int(len(self.iso.X_thresholds_))}


class WeightedHistogram(Rule):
    code = "R3"

    def __init__(self, n_bins=N_BINS, binning="count"):
        self.n_bins, self.binning = n_bins, binning

    def fit(self, yhat_oof, y_oof, wfun=None):
        p, y = np.asarray(yhat_oof, dtype=float), np.asarray(y_oof, dtype=float)
        w = weights_of(wfun, y)
        self.edges = bin_edges(p, self.n_bins, self.binning)
        b = bin_assign(self.edges, p)
        nb = len(self.edges) - 1
        self.value = np.bincount(b, weights=w * y, minlength=nb) / np.bincount(b, weights=w, minlength=nb)
        return self

    def predict(self, yhat):
        return self.value[bin_assign(self.edges, yhat)]

    def info(self):
        return {"n_bins": int(len(self.edges) - 1), "binning": self.binning}


class WeightedLinear(Rule):
    code = "R4"

    def fit(self, yhat_oof, y_oof, wfun=None):
        p, y = np.asarray(yhat_oof, dtype=float), np.asarray(y_oof, dtype=float)
        w = weights_of(wfun, y)
        pm, ym = np.average(p, weights=w), np.average(y, weights=w)
        self.b = float(np.sum(w * (p - pm) * (y - ym)) / np.sum(w * (p - pm) ** 2))
        self.a = float(ym - self.b * pm)
        return self

    def predict(self, yhat):
        return self.a + self.b * np.asarray(yhat, dtype=float)

    def info(self):
        return {"a": self.a, "b": self.b}


class TwoSidedStretch(Rule):
    code = "R5"

    def __init__(self, grid=None, mu=None):
        self.grid = S_GRID if grid is None else np.asarray(grid, dtype=float)
        self.mu = mu

    def fit(self, yhat_oof, y_oof, wfun=None):
        p, y = np.asarray(yhat_oof, dtype=float), np.asarray(y_oof, dtype=float)
        c = weights_of(wfun, y)
        self.mu_ = float(y.mean()) if self.mu is None else float(self.mu)
        self.s, self.edge = {}, {}
        g = self.grid
        for side, m in (("low", p < self.mu_), ("high", p >= self.mu_)):
            if not m.any():
                self.s[side], self.edge[side] = 1.0, False
                continue
            x, r, cm = p[m] - self.mu_, y[m] - self.mu_, c[m]
            # SSE(s) = Σc r² - 2s Σc r x + s² Σc x²: cùng argmin với vòng lặp của
            # tail_prior.stretch_two_sided, nhưng không cần ma trận lưới × mẫu.
            sse = np.sum(cm * r * r) - 2 * g * np.sum(cm * r * x) + g ** 2 * np.sum(cm * x * x)
            k = int(np.argmin(sse))
            self.s[side], self.edge[side] = float(g[k]), k in (0, len(g) - 1)
        return self

    def predict(self, yhat):
        p = np.asarray(yhat, dtype=float)
        out = p.copy()
        lo = p < self.mu_
        out[lo] = self.mu_ + self.s["low"] * (p[lo] - self.mu_)
        out[~lo] = self.mu_ + self.s["high"] * (p[~lo] - self.mu_)
        return out

    def info(self):
        return {"mu": self.mu_, "s_low": self.s["low"], "s_high": self.s["high"],
                "edge_low": bool(self.edge["low"]), "edge_high": bool(self.edge["high"]),
                "grid": [float(self.grid[0]), float(self.grid[-1])]}


class QuantileMap(Rule):
    code = "R6"

    def fit(self, yhat_oof, y_oof, wfun=None):
        self.p_sorted = np.sort(np.asarray(yhat_oof, dtype=float))
        self.y = np.asarray(y_oof, dtype=float)
        return self

    def predict(self, yhat):
        rank = np.searchsorted(self.p_sorted, np.asarray(yhat, dtype=float)) / len(self.p_sorted)
        return np.quantile(self.y, np.clip(rank, 0.0, 1.0))


class NormalReframe(Rule):
    code = "R7"

    def __init__(self, n_bins=N_BINS, n_samples=N_SAMPLES, binning="count", clip=(Y_MIN, Y_MAX)):
        self.n_bins, self.n_samples, self.binning, self.clip = n_bins, n_samples, binning, clip

    def fit(self, yhat_oof, y_oof, wfun=None):
        p, y = np.asarray(yhat_oof, dtype=float), np.asarray(y_oof, dtype=float)
        self.edges = bin_edges(p, self.n_bins, self.binning)
        b = bin_assign(self.edges, p)
        nb = len(self.edges) - 1
        r = y - p
        self.m = np.array([r[b == k].mean() for k in range(nb)])
        self.sd = np.array([r[b == k].std(ddof=1) if (b == k).sum() > 1 else 0.0 for k in range(nb)])
        self.z = norm.ppf(taus(self.n_samples))
        self.wfun = wfun
        return self

    def samples(self, yhat):
        p = np.asarray(yhat, dtype=float)
        b = bin_assign(self.edges, p)
        S = p[:, None] + self.m[b][:, None] + self.sd[b][:, None] * self.z[None, :]
        return S if self.clip is None else np.clip(S, self.clip[0], self.clip[1])

    def predict(self, yhat, wfun=None):
        S = self.samples(yhat)
        wf = wfun if wfun is not None else self.wfun
        return S.mean(axis=1) if wf is None else weighted_mean(S, wf)

    def info(self):
        return {"n_bins": int(len(self.edges) - 1), "n_samples": int(self.n_samples)}


RULES = {"R0": Raw, "R1": ResidualBinBayes, "R1_1": ResidualBinBayes, "R2": WeightedIsotonic,
         "R3": WeightedHistogram, "R4": WeightedLinear, "R5": TwoSidedStretch,
         "R6": QuantileMap, "R7": NormalReframe}


def make_rule(code, **kw):
    """Quy tắc chưa khớp theo mã R0..R7 (R1_1 là R1 với w ≡ 1)."""
    if code == "R1_1":
        return ResidualBinBayes(uniform=True, **kw)
    if code not in RULES:
        raise ValueError(f"mã quy tắc lạ: {code}; có {sorted(RULES)}")
    return RULES[code](**kw)


def fit_apply(code, yhat_oof, y_oof, yhat_new, wfun=None, **kw):
    """(dự đoán cho yhat_new, info) trong một lời gọi."""
    r = make_rule(code, **kw).fit(yhat_oof, y_oof, wfun)
    return r.predict(yhat_new), r.info()


def _fresh(estimator):
    if isinstance(estimator, str):
        return make_rule(estimator)
    if isinstance(estimator, Rule):
        return copy.deepcopy(estimator)
    if callable(estimator):
        return estimator()
    raise TypeError("estimator phải là mã quy tắc, Rule, hoặc hàm tạo Rule")


def crossfit_rule_oof(yhat_oof, y_oof, estimator, wfun=None, n_folds=RULE_CROSSFIT_FOLDS, seed=0,
                      groups=None):
    """Dự đoán của quy tắc trên chính OOF, khớp chéo: khớp trên n_folds - 1 phần, áp
    cho phần còn lại. Dùng cho E4 (B_j(K)) để không chấm điểm quy tắc trên chính dữ
    liệu đã khớp nó; một quy tắc linh hoạt (R1, R2) khớp trong mẫu sẽ trông tốt hơn
    thật và làm lệch việc chọn cấu hình.

    `groups` (tuỳ chọn, dài len(yhat_oof), thẳng hàng với OOF, tức groups[idx_tr]
    theo quy ước của splits.py): có thì chia bằng GroupKFold(shuffle=True, cùng seed)
    để một cặp (ŷ, y) trùng khoá bản ghi không nằm cả ở phần khớp lẫn phần chấm
    điểm của quy tắc. Không có nhóm thì giữ nguyên KFold như trước."""
    p, y = np.asarray(yhat_oof, dtype=float), np.asarray(y_oof, dtype=float)
    out = np.empty(len(p))
    if groups is None:
        parts = KFold(n_folds, shuffle=True, random_state=seed).split(p)
    else:
        groups = np.asarray(groups)
        assert len(groups) == len(p), "groups phải thẳng hàng với yhat_oof (dài n_tr)"
        parts = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(p, groups=groups)
    for a, b in parts:
        out[b] = _fresh(estimator).fit(p[a], y[a], wfun).predict(p[b])
    return out
