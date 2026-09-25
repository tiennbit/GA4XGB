# -*- coding: utf-8 -*-
"""E6 (khung bài 24/9, mục 6.10): mô phỏng với p(y|x) biết trước.

Vì sao cần: trên dữ liệu HSA không biết μ_K^{σ(ŷ)}, nên "khoảng hụt quyết định" của
một quy tắc chỉ ước lượng được qua một quy tắc khác (R1) đứng thay cho μ_K^{σ(ŷ)}
(Mệnh đề 1). Nếu chính R1 còn xa μ_K^{σ(ŷ)} thì mọi khoảng hụt ước lượng đều lệch
cùng một lượng: ước lượng(Rx) − thật(Rx) = cost(μ^G) − cost(R1), với MỌI Rx. Mô
phỏng có đáp án đo đúng độ lệch đó, và dựng ba tình huống bài phải trả lời:
  S2  thông tin ngoài ŷ (phương sai phụ thuộc x3 − x4): huấn luyện có trọng số (R8),
      dùng toàn bộ x, có thể thắng mọi quy tắc chỉ nhìn ŷ (Nhận xét 4).
  S3  phân phối lệch, hiệu ứng nhóm và sai số đo như điểm thi thật.
  S4  2% dòng là đoán mò, độc lập với x.

Thiết kế (đúng mục 6.10, lựa chọn riêng ghi ở dưới):
  X ~ N(0, Σ) trong R^20, Σ_ij = 0,3^|i-j|; a(x) = Σ_j x_j / j + 0,5·sin(x1·x2);
  m(x) = 77,2 + 13,6·0,7·a(x)/sd(a), tức R² của m bằng 0,49 như HSA.
  S1  Y = m + ε, ε ~ N(0, σ²), σ² = 13,6²·0,51.
  S2  như S1 nhưng sd(ε | x) = σ₀·exp(0,4·(x3 − x4)), σ₀ chọn để E[sd²] = σ².
  S3  Y = m + u_g + ε + e, u_g ~ N(0, 2²) trên 1.000 nhóm, ε skew-normal hình dạng 3
      (trung bình 0), e ~ N(0; 4,3²) là sai số đo; Var(ε) = σ² − 4 − 4,3² để R² vẫn 0,49.
  S4  như S1 nhưng 2% dòng là Binomial(112; 0,25), độc lập với x.
  Mọi kịch bản: làm tròn về số nguyên, cắt trong [0; 150].
  Đuôi theo khối lượng 9,25% / 5,72% của Y trên tập kiểm tra cố định (xấp xỉ quần thể).
  n huấn luyện ∈ {10.000; 57.000}; tập kiểm tra cố định 200.000 dòng; 50 lần lặp.
  Trung tâm default và bag10; quy tắc R0, R1, R2, R4, R5 của decision_layer, và R8 =
  huấn luyện có trọng số chuẩn hoá về trung bình 1, DÙNG LẠI siêu tham số của trung
  tâm, KHÔNG dò lại (khác E2b; bài phải ghi rõ).

Đáp án:
  μ_K^{σ(X)} = A_K(x)/B_K(x), A_K = Σ_k w_K(k)·k·P(Y=k|x), B_K = Σ_k w_K(k)·P(Y=k|x),
      tính đúng trên lưới k = 0..150 (Y nguyên sau làm tròn; khung bài nói lưới bước 0,1
      cho Y liên tục, ở đây Y rời rạc nên tổng trên số nguyên là chính xác).
  μ_K^{σ(ŷ)}: 2.000.000 dòng Monte Carlo độc lập, 400 bin phân vị của ŷ, giá trị bin
      = Σ A / Σ B trong bin. Dùng A, B đã tích phân thay cho y ngẫu nhiên nên nhiễu
      Monte Carlo chỉ còn do bin và do X, không do ε.

Lựa chọn khi khung bài chưa rõ (ghi cả vào meta.choices):
 1. S3: 1.000 giá trị u_g rút MỘT lần cho mỗi kịch bản (quần thể cố định; huấn luyện,
    kiểm tra và Monte Carlo dùng chung), chuẩn hoá về trung bình 0, SD 2. Đáp án coi
    u ~ N(0, 2²); sai khác với hỗn hợp 1.000 điểm là rất nhỏ và ghi ở đây.
 2. S3: ε + u + e là skew-normal (tổng của skew-normal với chuẩn độc lập vẫn là
    skew-normal: ω' = sqrt(ω² + s²), δ' = δ·ω/ω'), nên CDF có dạng đóng.
 3. S2: x3 − x4 có tương quan rất nhỏ với a(x) qua β3, β4; khung bài gọi là "độc lập
    với a(x)", ở đây chỉ gần độc lập.
 4. Sai số tương đối của khoảng hụt chỉ tính khi |khoảng hụt thật| ≥ GAP_MIN (0,05):
    dưới mức đó tỉ số mất ổn định; số tuyệt đối vẫn được ghi.
 5. Cổng "định lượng" xét ở K = 3 cho mọi quy tắc đủ lớn, cả hai trung tâm.
 6. Mô hình dùng random_state = 0 (default) và 0..9 (bag10) như E1; seed chỉ đổi dữ liệu.

Chạy (server): PYTHONPATH=src .venv/bin/python -W ignore src/sim_decomp.py --workers 8
Khói (Mac): PYTHONPATH=src python3 src/sim_decomp.py --smoke --out <scratch>/sim.json
"""
import argparse
import os
import time
from functools import lru_cache

import numpy as np
from joblib import Parallel, delayed
from scipy.special import ndtr
from scipy.stats import binom, skewnorm
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

import decision_layer as dl
import preds_io
import provenance
import splits
import stats_paired as sp
from gates import BAG_SUBSAMPLE, K_GRID, PRIMARY_K, SESOI

EXPERIMENT = "E6"
DEFAULT_OUT = "results_cost/sim_decomp.json"
SCENARIOS = ["S1", "S2", "S3", "S4"]
CENTERS = ["default", "bag10"]
RULES = ["R0", "R1", "R2", "R4", "R5"]          # R8 thêm riêng (huấn luyện có trọng số)
KS = [1] + list(K_GRID)                          # K = 1 cho phần hiệu chỉnh
FRONTIER_KS = [1, 1.5, 2, 3, 5, 8, 12, 20]
GAP_MIN = 0.05
QUANT_GATE_REL = 0.10

P = 20
RHO = 0.3
MU_Y, SD_Y, R2 = 77.2, 13.6, 0.49
SIGMA2 = SD_Y ** 2 * (1 - R2)
HET = 0.4
VAR_D = 2 - 2 * RHO                                   # Var(x3 − x4)
SIGMA0 = np.sqrt(SIGMA2 / np.exp(0.5 * (2 * HET) ** 2 * VAR_D))
GROUP_SD, N_GROUPS, MEAS_SD, SKEW_ALPHA = 2.0, 1000, 4.3, 3.0
GUESS_FRAC, GUESS_N, GUESS_P = 0.02, 112, 0.25
Y_MAX = 150
K_GRID_Y = np.arange(Y_MAX + 1, dtype=float)
FOLDS = 5

_IDX = np.arange(P)
CHOL = np.linalg.cholesky(RHO ** np.abs(_IDX[:, None] - _IDX[None, :]))
BETA = 1.0 / np.arange(1, P + 1)


class Size:
    """Cỡ của một lượt: thật hoặc --smoke."""

    def __init__(self, smoke):
        self.smoke = smoke
        self.n_trains = [2000] if smoke else [10_000, 57_000]
        self.n_test = 20_000 if smoke else 200_000
        self.n_mc = 100_000 if smoke else 2_000_000
        self.mc_bins = 100 if smoke else 400
        self.reps = 2 if smoke else 50
        self.n_bag = 2 if smoke else 10
        self.trees = 30 if smoke else 100             # n_estimators mặc định của XGBoost

    def as_dict(self):
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Sinh dữ liệu
# ---------------------------------------------------------------------------
def draw_x(n, rng):
    return (rng.standard_normal((n, P)) @ CHOL.T).astype(np.float32)


def index_a(x):
    x = np.asarray(x, dtype=float)
    return x @ BETA + 0.5 * np.sin(x[:, 0] * x[:, 1])


@lru_cache(maxsize=1)
def sd_a():
    """sd(a(X)) trên 2 triệu dòng, seed cố định: một hằng số của thiết kế."""
    return float(np.std(index_a(draw_x(2_000_000, np.random.default_rng(12345)))))


def mean_fn(x):
    return MU_Y + SD_Y * np.sqrt(R2) * index_a(x) / sd_a()


def sd_fn(scen, x):
    """sd của phần nhiễu chuẩn theo x (chỉ S2 thay đổi theo x)."""
    if scen == "S2":
        x = np.asarray(x, dtype=float)
        return SIGMA0 * np.exp(HET * (x[:, 2] - x[:, 3]))
    return np.full(len(x), np.sqrt(SIGMA2))


def _skew_params():
    """(xi_e, omega_e) của ε và (omega_T, alpha_T) của T = ε + u + e trong S3."""
    delta = SKEW_ALPHA / np.sqrt(1 + SKEW_ALPHA ** 2)
    var_eps = SIGMA2 - GROUP_SD ** 2 - MEAS_SD ** 2
    omega_e = np.sqrt(var_eps / (1 - 2 * delta ** 2 / np.pi))
    xi_e = -omega_e * delta * np.sqrt(2 / np.pi)                 # trung bình ε = 0
    omega_t = np.sqrt(omega_e ** 2 + GROUP_SD ** 2 + MEAS_SD ** 2)
    delta_t = delta * omega_e / omega_t
    alpha_t = delta_t / np.sqrt(1 - delta_t ** 2)
    return xi_e, omega_e, omega_t, alpha_t


XI_E, OMEGA_E, OMEGA_T, ALPHA_T = _skew_params()


@lru_cache(maxsize=4)
def group_effects(scen_seed):
    u = np.random.default_rng(scen_seed).standard_normal(N_GROUPS)
    return (u - u.mean()) / u.std() * GROUP_SD


def draw_y(scen, x, rng, scen_seed):
    m = mean_fn(x)
    n = len(m)
    if scen == "S3":
        g = rng.integers(0, N_GROUPS, n)
        eps = XI_E + skewnorm.rvs(SKEW_ALPHA, scale=OMEGA_E, size=n, random_state=rng)
        ystar = m + group_effects(scen_seed)[g] + eps + rng.normal(0, MEAS_SD, n)
    else:
        ystar = m + sd_fn(scen, x) * rng.standard_normal(n)
    y = np.clip(np.rint(ystar), 0, Y_MAX)
    if scen == "S4":
        guess = rng.random(n) < GUESS_FRAC
        y[guess] = rng.binomial(GUESS_N, GUESS_P, guess.sum())
    return y


# ---------------------------------------------------------------------------
# Đáp án: P(Y = k | x) trên k = 0..150, rồi A_K, B_K
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _s3_cdf_grid():
    """CDF của T = ε + u + e trên lưới z bước 0,005. skewnorm.cdf đi qua hàm Owen's T,
    chậm: gọi trực tiếp trên 2,9 triệu điểm của bảng theo m mất khoảng 5 phút. Nhiễu S3
    không phụ thuộc x nên CDF chỉ là hàm của z; lập một lần rồi nội suy tuyến tính
    (sai số nội suy < 1e-6, dưới mọi số được báo)."""
    z = np.arange(-250.0, 250.0 + 1e-9, 0.005)
    return z, skewnorm.cdf(z, ALPHA_T, loc=XI_E, scale=OMEGA_T)


def _noise_cdf(scen, z, sd):
    if scen == "S3":
        zg, Fg = _s3_cdf_grid()
        return np.interp(z, zg, Fg, left=0.0, right=1.0)
    return ndtr(z / sd)


_GUESS_PMF = binom.pmf(K_GRID_Y, GUESS_N, GUESS_P)


def pmf_rows(scen, m, sd):
    """Ma trận n×151 của P(Y = k | x) (làm tròn, cắt biên, trộn đoán mò ở S4)."""
    edges = np.concatenate([[-np.inf], K_GRID_Y[:-1] + 0.5, [np.inf]])
    z = edges[None, :] - m[:, None]
    F = _noise_cdf(scen, z, sd[:, None])
    F[:, 0], F[:, -1] = 0.0, 1.0
    p = np.diff(F, axis=1)
    if scen == "S4":
        p = (1 - GUESS_FRAC) * p + GUESS_FRAC * _GUESS_PMF[None, :]
    return p


@lru_cache(maxsize=4)
def _s3_table():
    """S3 không có nhiễu phụ thuộc x: P(Y=k|x) chỉ phụ thuộc m, nên lập bảng theo m
    (bước 0,01) một lần thay cho gọi skewnorm.cdf (chậm) trên hàng triệu dòng."""
    mg = np.arange(-20.0, Y_MAX + 20.0 + 1e-9, 0.01)
    return mg, pmf_rows("S3", mg, np.ones_like(mg))


def moments(scen, x, wks, chunk=50_000):
    """{K: (A, B)} với A = E[w_K(Y)·Y | x], B = E[w_K(Y) | x], float64."""
    m, sd = mean_fn(x), sd_fn(scen, x)
    n = len(m)
    out = {K: (np.empty(n), np.empty(n)) for K in wks}
    table = _s3_table() if scen == "S3" else None
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        if table is not None:
            mg, pg = table
            j = np.clip(np.rint((m[a:b] - mg[0]) / 0.01).astype(int), 0, len(mg) - 1)
            p = pg[j]
        else:
            p = pmf_rows(scen, m[a:b], sd[a:b])
        for K, wv in wks.items():
            out[K][0][a:b] = p @ (wv * K_GRID_Y)
            out[K][1][a:b] = p @ wv
    return out


# ---------------------------------------------------------------------------
# Tập cố định của một kịch bản (kiểm tra 200k, Monte Carlo 2M), giữ một bộ mỗi tiến trình
# ---------------------------------------------------------------------------
_FIXED = {}


def fixed_sets(scen, size):
    key = (scen, size.n_test, size.n_mc)
    if key in _FIXED:
        return _FIXED[key]
    _FIXED.clear()                                   # một kịch bản mỗi lúc: giữ RAM thấp
    sseed = 7000 + SCENARIOS.index(scen)
    rng = np.random.default_rng(sseed)
    x_te = draw_x(size.n_test, rng)
    y_te = draw_y(scen, x_te, rng, sseed)
    lo, hi = dl.tail_cutoffs(y_te)
    wks = {K: dl.step(K, K, lo, hi)(K_GRID_Y) for K in sorted(set(KS) | set(FRONTIER_KS))}
    x_mc = draw_x(size.n_mc, np.random.default_rng(sseed + 100))
    mom_te = moments(scen, x_te, wks)
    mom_mc = {K: (a.astype(np.float32), b.astype(np.float32))
              for K, (a, b) in moments(scen, x_mc, wks).items()}
    _FIXED[key] = dict(sseed=sseed, x_te=x_te, y_te=y_te, lo=lo, hi=hi, x_mc=x_mc,
                       mom_te=mom_te, mom_mc=mom_mc)
    return _FIXED[key]


# ---------------------------------------------------------------------------
# Trung tâm và một lần lặp
# ---------------------------------------------------------------------------
def fit_center(kind, X, y, size, n_jobs, w=None):
    if kind == "default":
        specs = [dict(random_state=0)]
    else:
        specs = [dict(random_state=b, subsample=BAG_SUBSAMPLE, colsample_bytree=BAG_SUBSAMPLE)
                 for b in range(size.n_bag)]
    return [XGBRegressor(tree_method="hist", n_estimators=size.trees, n_jobs=n_jobs, **s)
            .fit(X, y, sample_weight=w) for s in specs]


def predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def mu_g_truth(yhat_mc, mom_mc, yhat_te, n_bins):
    """μ_K^{σ(ŷ)} tại ŷ kiểm tra: Σ A / Σ B trong bin phân vị của ŷ Monte Carlo."""
    edges = np.unique(np.quantile(yhat_mc, np.linspace(0, 1, n_bins + 1)))
    b_mc = np.clip(np.searchsorted(edges[1:-1], yhat_mc, side="right"), 0, len(edges) - 2)
    b_te = np.clip(np.searchsorted(edges[1:-1], yhat_te, side="right"), 0, len(edges) - 2)
    nb = len(edges) - 1
    out = {}
    for K, (A, B) in mom_mc.items():
        sa = np.bincount(b_mc, weights=A, minlength=nb)
        sb = np.bincount(b_mc, weights=B, minlength=nb)
        out[K] = (sa / np.maximum(sb, 1e-300))[b_te]
    return out


def kish_ratio(w):
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (len(w) * np.sum(w ** 2)))


def run_task(scen, n_train, rep, size, n_jobs):
    t0 = time.time()
    fx = fixed_sets(scen, size)
    lo, hi, y_te = fx["lo"], fx["hi"], fx["y_te"]
    seed = int(np.random.SeedSequence([SCENARIOS.index(scen), n_train, rep]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    x_tr = draw_x(n_train, rng)
    y_tr = draw_y(scen, x_tr, rng, fx["sseed"])
    mu_x = {K: a / b for K, (a, b) in fx["mom_te"].items()}
    folds = list(KFold(FOLDS, shuffle=True, random_state=seed % (2 ** 31)).split(x_tr))
    res = {"seed": seed, "kish": {str(K): kish_ratio(dl.step(K, K, lo, hi)(y_tr)) for K in KS},
           "centers": {}}
    for c in CENTERS:
        oof = np.zeros(n_train)
        for a, b in folds:
            oof[b] = predict(fit_center(c, x_tr[a], y_tr[a], size, n_jobs), x_tr[b])
        models = fit_center(c, x_tr, y_tr, size, n_jobs)
        yh_te = predict(models, fx["x_te"])
        mu_g = mu_g_truth(predict(models, fx["x_mc"]), fx["mom_mc"], yh_te, size.mc_bins)
        rc = {"cost": {}, "frontier": {}}
        for K in KS:
            wf = dl.step(K, K, lo, hi)
            preds = {"R0": yh_te, "muG": mu_g[K], "muX": mu_x[K]}
            for r in RULES[1:]:
                preds[r] = dl.make_rule(r).fit(oof, y_tr, wf).predict(yh_te)
            if K != 1:
                w8 = dl.normalize_weights(wf(y_tr))
                preds["R8"] = predict(fit_center(c, x_tr, y_tr, size, n_jobs, w=w8), fx["x_te"])
            rc["cost"][str(K)] = {k: sp.cost_k(y_te, v, K, K, lo, hi) for k, v in preds.items()}
        r1 = dl.make_rule("R1").fit(oof, y_tr, None)
        for K in FRONTIER_KS:
            wf = dl.step(K, K, lo, hi)
            fr = {}
            for k, v in (("R1", r1.predict(yh_te, wf)), ("muG", mu_g[K])):
                reg = sp.region_rmse(y_te, v, lo, hi)
                fr[k] = {"mid": reg["Middle"], "tails": reg["Tails"]}
            rc["frontier"][str(K)] = fr
        res["centers"][c] = rc
    res["seconds"] = time.time() - t0
    return (scen, n_train, rep), res


# ---------------------------------------------------------------------------
# Tổng hợp và cổng
# ---------------------------------------------------------------------------
def _ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def summarize(per, size):
    out = {}
    for scen in SCENARIOS:
        for n in size.n_trains:
            reps = [per[k] for k in per if k.startswith(f"{scen}|{n}|")]
            if not reps:
                continue
            cell = {"n_reps": len(reps), "kish": {str(K): _ms([r["kish"][str(K)] for r in reps]) for K in KS},
                    "centers": {}}
            for c in CENTERS:
                cc = {}
                for K in KS:
                    costs = [r["centers"][c]["cost"][str(K)] for r in reps]
                    names = list(costs[0])
                    e = {"cost": {k: _ms([x[k] for x in costs]) for k in names}}
                    r1_gap = [x["R1"] - x["muG"] for x in costs]
                    e["R1_gap_to_muG"] = _ms(r1_gap)
                    e["center_gap"] = _ms([x["muG"] - x["muX"] for x in costs])
                    gaps = {}
                    for k in names:
                        if k in ("R1", "muG", "muX"):
                            continue
                        est = float(np.mean([x[k] - x["R1"] for x in costs]))
                        tru = float(np.mean([x[k] - x["muG"] for x in costs]))
                        gaps[k] = {"est": est, "true": tru,
                                   "rel_err": (est - tru) / tru if abs(tru) >= GAP_MIN else None}
                    e["decision_gap"] = gaps
                    if "R8" in names:
                        v8 = np.var([x["R8"] for x in costs], ddof=1) if len(costs) > 1 else np.nan
                        v1 = np.var([x["R1"] for x in costs], ddof=1) if len(costs) > 1 else np.nan
                        e["R8_minus_R1"] = _ms([x["R8"] - x["R1"] for x in costs])
                        e["var_ratio_R8_R1"] = float(v8 / v1) if v1 and np.isfinite(v1) and v1 > 0 else None
                        e["inv_kish"] = 1.0 / cell["kish"][str(K)]["mean"]
                    e["R5_minus_R1"] = _ms([x["R5"] - x["R1"] for x in costs])
                    cc[str(K)] = e
                fr = {}
                for K in FRONTIER_KS:
                    fr[str(K)] = {k: {"mid": _ms([r["centers"][c]["frontier"][str(K)][k]["mid"] for r in reps]),
                                      "tails": _ms([r["centers"][c]["frontier"][str(K)][k]["tails"] for r in reps])}
                                  for k in ("R1", "muG")}
                cell["centers"][c] = {"by_K": cc, "frontier": fr}
            out[f"{scen}|{n}"] = cell
    return out


def gates(S, size):
    n = max(size.n_trains)
    k = str(PRIMARY_K)
    g = {"quantitative_fig5": None, "single_index_needed": {}, "guessing_flips_R5_R1": {}}
    worst = []
    for scen in ("S1", "S3"):
        cell = S.get(f"{scen}|{n}")
        if not cell:
            continue
        for c in CENTERS:
            for r, d in cell["centers"][c]["by_K"][k]["decision_gap"].items():
                if d["rel_err"] is not None:
                    worst.append({"scen": scen, "center": c, "rule": r, "rel_err": d["rel_err"]})
    if worst:
        mx = max(worst, key=lambda d: abs(d["rel_err"]))
        g["quantitative_fig5"] = {"pass": bool(abs(mx["rel_err"]) <= QUANT_GATE_REL), "worst": mx,
                                  "threshold": QUANT_GATE_REL}
    for c in CENTERS:
        s2 = S.get(f"S2|{n}")
        if s2 and "R8_minus_R1" in s2["centers"][c]["by_K"][k]:
            d = s2["centers"][c]["by_K"][k]["R8_minus_R1"]["mean"]
            g["single_index_needed"][c] = {"R8_minus_R1": d, "pass": bool(d is not None and d <= -SESOI)}
        s1, s4 = S.get(f"S1|{n}"), S.get(f"S4|{n}")
        if s1 and s4:
            a = s1["centers"][c]["by_K"][k]["R5_minus_R1"]["mean"]
            b = s4["centers"][c]["by_K"][k]["R5_minus_R1"]["mean"]
            g["guessing_flips_R5_R1"][c] = {"S1": a, "S4": b,
                                           "flip": bool(a is not None and b is not None and np.sign(a) != np.sign(b))}
    return g


def print_summary(S, G, size):
    k = str(PRIMARY_K)
    print(f"\n[E6] K = {k}; khoảng hụt quyết định: ước lượng (qua R1) / thật (qua μ^G) / sai số tương đối")
    for key, cell in S.items():
        for c in CENTERS:
            e = cell["centers"][c]["by_K"][k]
            parts = [f"{r}: {d['est']:+.3f}/{d['true']:+.3f}/"
                     + ("—" if d["rel_err"] is None else f"{d['rel_err']:+.0%}")
                     for r, d in e["decision_gap"].items()]
            extra = ""
            if "R8_minus_R1" in e:
                extra = (f"  R8−R1={e['R8_minus_R1']['mean']:+.3f}"
                         f"  var R8/R1={e['var_ratio_R8_R1'] if e['var_ratio_R8_R1'] is None else round(e['var_ratio_R8_R1'], 2)}"
                         f" (1/Kish {e['inv_kish']:.2f})")
            print(f"  {key:9s} {c:7s} R1−μG={e['R1_gap_to_muG']['mean']:+.3f} μG−μX={e['center_gap']['mean']:+.3f}  "
                  + "  ".join(parts) + extra)
    print(f"\nCổng: {G}")


# ---------------------------------------------------------------------------
def self_check(size):
    """Kiểm nhanh các đại lượng thiết kế trên một mẫu lớn (in ra, ghi vào JSON)."""
    rng = np.random.default_rng(99)
    out = {}
    for scen in SCENARIOS:
        x = draw_x(200_000, rng)
        y = draw_y(scen, x, rng, 7000 + SCENARIOS.index(scen))
        m = mean_fn(x)
        wks = {1: np.ones_like(K_GRID_Y)}
        a, b = moments(scen, x[:20_000], wks)[1]
        out[scen] = {"y_mean": float(y.mean()), "y_sd": float(y.std()), "r2_m": float(np.corrcoef(m, y)[0, 1] ** 2),
                     "mean_y_minus_EYx": float(np.mean(y[:20_000] - a / b)),
                     "frac_le_37": float(np.mean(y <= 37))}
    print("[E6] kiểm thiết kế: " + "; ".join(f"{s} ȳ={d['y_mean']:.1f} sd={d['y_sd']:.1f} R²={d['r2_m']:.3f} "
                                             f"E[y−E(y|x)]={d['mean_y_minus_EYx']:+.3f}" for s, d in out.items()),
          flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="E6: mô phỏng với p(y|x) biết trước")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    ap.add_argument("--on-mismatch", default="raise", choices=["raise", "recompute"])
    args = ap.parse_args(argv)
    provenance.print_versions()
    size = Size(args.smoke)
    n_jobs = splits.xgb_threads(args.workers)
    if args.smoke:
        n_jobs = min(2, n_jobs)
    cfg = {"script": "sim_decomp", "size": size.as_dict(), "rules": RULES, "ks": KS,
           "frontier_ks": FRONTIER_KS, "centers": CENTERS, "gap_min": GAP_MIN}
    fp = preds_io.fingerprint(cfg)
    partial = args.out + ".partial"
    res = preds_io.load_partial(partial, fp, {"per_task": {}}, on_mismatch=args.on_mismatch)
    checks = self_check(size)
    # Theo kịch bản rồi mới tới n, lặp: mỗi tiến trình giữ tập cố định của MỘT kịch bản
    tasks = [(s, n, r) for s in args.scenarios for n in size.n_trains for r in range(size.reps)
             if f"{s}|{n}|{r}" not in res["per_task"]]
    print(f"[E6] {len(tasks)} lần lặp còn lại; workers={args.workers} x n_jobs={n_jobs}; dấu {fp[:12]}", flush=True)
    t0 = time.time()
    gen = Parallel(n_jobs=args.workers, return_as="generator_unordered")(
        delayed(run_task)(s, n, r, size, n_jobs) for s, n, r in tasks)
    for i, ((s, n, r), out) in enumerate(gen, 1):
        res["per_task"][f"{s}|{n}|{r}"] = out
        preds_io.dump_json_atomic(preds_io.to_jsonable(res), partial)
        if i % 10 == 0 or i == len(tasks):
            print(f"  {i}/{len(tasks)} xong ({time.time() - t0:.0f}s)", flush=True)
    S = summarize(res["per_task"], size)
    G = gates(S, size)
    print_summary(S, G, size)
    res["summary"], res["gates"] = S, G
    res["meta"] |= {"experiment": EXPERIMENT, "size": size.as_dict(), "design_checks": checks,
                    "constants": {"sigma2": SIGMA2, "sigma0": float(SIGMA0), "xi_e": float(XI_E),
                                  "omega_e": float(OMEGA_E), "omega_t": float(OMEGA_T),
                                  "alpha_t": float(ALPHA_T), "sd_a": sd_a()},
                    "choices": [l.strip() for l in __doc__.split("Lựa chọn khi khung bài chưa rõ")[1]
                                .split("Chạy (server)")[0].strip().splitlines() if l.strip()],
                    "provenance": provenance.stamp()}
    preds_io.dump_json_atomic(preds_io.to_jsonable(res), args.out)
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    # Gọi qua module đã import, không qua __main__: joblib (loky) gửi hàm sang tiến trình
    # con theo tên module; hàm của __main__ có lru_cache (sd_a) thì tiến trình con
    # không tìm lại được ("Can't get attribute 'sd_a'").
    import sim_decomp
    sim_decomp.main()
