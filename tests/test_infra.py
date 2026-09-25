# -*- coding: utf-8 -*-
"""Kiểm thử hạ tầng E0: splits, stats_paired, gates, provenance, features,
decision_layer, preds_io. Script thuần với assert (không cần pytest).

Chạy: cd <repo> && PYTHONPATH=src python3 tests/test_infra.py
Chỉ dùng dữ liệu giả: tests/fixtures/fake_hsa.csv (tự sinh nếu chưa có) và mảng
tổng hợp trong bộ nhớ. Không đọc data/data_final.csv.
"""
import json
import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import cost_aware  # noqa: E402
import decision_layer as dl  # noqa: E402
import features  # noqa: E402
import gates  # noqa: E402
import preds_io  # noqa: E402
import provenance  # noqa: E402
import splits  # noqa: E402
import stats_paired as sp  # noqa: E402
import tail_prior  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

FAKE = os.path.join(ROOT, "tests", "fixtures", "fake_hsa.csv")
FAKE_PREDS = os.path.join(ROOT, "tests", "fixtures", "preds", "decomp")
TESTS = []


def test(f):
    TESTS.append(f)
    return f


def fake_npz():
    """npz giả đã sinh; bỏ qua .tmp còn sót khi make_fake_preds bị ngắt giữa chừng."""
    if not os.path.isdir(FAKE_PREDS):
        return []
    return [os.path.join(FAKE_PREDS, f) for f in sorted(os.listdir(FAKE_PREDS)) if f.endswith(".npz")]


def fake_frame():
    if not os.path.exists(FAKE):
        import make_fake_data
        make_fake_data.main(["--out", FAKE])
    return features.load_frame(FAKE)


def realistic(n=6000, seed=0):
    """(ŷ OOF, y) có co về giữa như thật: ŷ = trung tâm co, y = ŷ-thật + nhiễu, điểm nguyên."""
    rng = np.random.default_rng(seed)
    m = 77 + 9.5 * rng.normal(size=n)
    y = np.clip(np.round(m + rng.normal(0, 9.7, n)), 0, 150)
    yhat = 77 + 0.85 * (m - 77) + rng.normal(0, 1.5, n)
    return yhat, y


# ---------------------------------------------------------------------------
# Thước đo
# ---------------------------------------------------------------------------
@test
def cost_k1_is_rmse():
    rng = np.random.default_rng(1)
    y = np.round(rng.normal(77, 13.6, 5000))
    p = y + rng.normal(0, 9, 5000)
    rmse = np.sqrt(np.mean((y - p) ** 2))
    assert abs(sp.cost_k(y, p, 1, 1, 60, 100) - rmse) < 1e-12
    assert abs(sp.cost_w(y, p, np.ones(len(y))) - rmse) < 1e-12
    # cost_K là RMSE có trọng số: tính tay
    w = np.where((y < 60) | (y >= 100), 3.0, 1.0)
    assert abs(sp.cost_k(y, p, 3, 3, 60, 100) - np.sqrt(np.sum(w * (y - p) ** 2) / w.sum())) < 1e-12
    # cost_from_regions suy đúng cost_K từ RMSE toàn bộ, giữa, đuôi gộp
    r = sp.region_rmse(y, p, 60, 100)
    assert abs(r["All"] - rmse) < 1e-12
    for K in (1, 2, 3, 8):
        got = sp.cost_from_regions(r["All"], r["Middle"], r["Tails"], K)
        assert abs(got - sp.cost_k(y, p, K, K, 60, 100)) < 1e-9, (K, got)
    # họ trọng số ở tham số trung hoà đều là w ≡ 1
    v = np.linspace(20, 140, 50)
    for w in (dl.step(1, 1, 60, 100), dl.prior(0.0, y), dl.phi(1.0, y)):
        assert np.allclose(w(v), 1.0)


@test
def weighted_bayes_equals_brute_force():
    # Phân phối rời rạc biết trước: hành động Bayes dưới w(y)(y - f)² là Σ p w y / Σ p w
    s = np.array([30.0, 55, 70, 85, 105, 120])
    p = np.array([0.05, 0.15, 0.30, 0.30, 0.15, 0.05])
    w = dl.step(3, 3, 60, 100)
    mu = np.sum(p * w(s) * s) / np.sum(p * w(s))
    grid = np.arange(0, 150, 0.001)
    risk = ((p * w(s))[None, :] * (s[None, :] - grid[:, None]) ** 2).sum(axis=1)
    assert abs(grid[np.argmin(risk)] - mu) < 1e-3
    # weighted_mean trên S điểm cách đều xác suất = cực tiểu của rủi ro rời rạc
    rng = np.random.default_rng(2)
    S = np.sort(rng.normal(80, 14, (4, 50)), axis=1)
    g = tail_prior.weighted_mean(S, w)
    for i in range(4):
        risk = (w(S[i])[None, :] * (S[i][None, :] - grid[:, None]) ** 2).sum(axis=1)
        assert abs(grid[np.argmin(risk)] - g[i]) < 1e-3
    # R1 với một bin: điểm mẫu là ŷ + phân vị phần dư; dự đoán = cực tiểu brute force
    yhat = np.full(2000, 80.0) + rng.normal(0, 1e-6, 2000)
    y = 80.0 + rng.choice([-30.0, -12.0, 0.0, 9.0, 25.0], size=2000)
    r1 = dl.make_rule("R1", n_bins=1).fit(yhat, y, w)
    Sq = r1.samples(np.array([80.0]))[0]
    risk = (w(Sq)[None, :] * (Sq[None, :] - grid[:, None]) ** 2).sum(axis=1)
    assert abs(r1.predict(np.array([80.0]))[0] - grid[np.argmin(risk)]) < 1e-3


@test
def r1_matches_tail_prior():
    # Ở B = 20, S = 50, R1 phải trùng khít cách cũ tail_prior.bin_samples + weighted_mean
    p_oof, y_oof = realistic(8000, 3)
    p_te, _ = realistic(2000, 4)
    lo, hi = 60, 100
    for K in (1, 3, 8):
        w = dl.step(K, K, lo, hi)
        old = tail_prior.weighted_mean(tail_prior.bin_samples(p_oof, y_oof, p_te), w)
        new = dl.make_rule("R1").fit(p_oof, y_oof, w).predict(p_te)
        assert np.allclose(old, new, rtol=0, atol=1e-10), K
    # R1_1 = R1 với w ≡ 1; và R1 ở K = 1 trùng R1_1
    a = dl.make_rule("R1_1").fit(p_oof, y_oof, dl.step(5, 5, lo, hi)).predict(p_te)
    b = dl.make_rule("R1").fit(p_oof, y_oof, dl.step(1, 1, lo, hi)).predict(p_te)
    assert np.allclose(a, b, atol=1e-10)
    # predict(wfun=...) dùng lại phân vị đã khớp cho K khác
    r = dl.make_rule("R1").fit(p_oof, y_oof, None)
    assert np.allclose(r.predict(p_te, wfun=dl.step(3, 3, lo, hi)),
                       dl.make_rule("R1").fit(p_oof, y_oof, dl.step(3, 3, lo, hi)).predict(p_te))
    # bin bằng bề rộng và B lớn: không lỗi dù có bin rỗng ở đuôi ŷ
    for B in (10, 80):
        out = dl.make_rule("R1", n_bins=B, binning="width").fit(p_oof, y_oof, dl.step(3, 3, lo, hi)).predict(p_te)
        assert np.all(np.isfinite(out))


@test
def calibration_rules_reduce_at_w1():
    from calibration_table import calib_stats
    p_oof, y_oof = realistic(6000, 5)
    p_te, _ = realistic(1500, 6)
    # R4 ở w = 1 là hồi quy hiệu chỉnh y ~ a + b·ŷ (calib_slope/intercept của calibration_table)
    r4 = dl.make_rule("R4").fit(p_oof, y_oof, None)
    cs = calib_stats(y_oof, p_oof)
    assert abs(r4.b - cs["calib_slope"]) < 1e-9 and abs(r4.a - cs["calib_intercept"]) < 1e-7
    # R2 ở w = 1 là isotonic không trọng số
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(p_oof, y_oof)
    assert np.allclose(dl.make_rule("R2").fit(p_oof, y_oof).predict(p_te), iso.predict(p_te))
    # R3 ở w = 1 là trung bình y trong bin ŷ (cùng bin với R1)
    r3 = dl.make_rule("R3").fit(p_oof, y_oof)
    b = dl.bin_assign(r3.edges, p_oof)
    assert np.allclose(r3.value, [y_oof[b == k].mean() for k in range(len(r3.edges) - 1)])
    # Hiệu chỉnh trong lớn có trọng số: Σ w (y - g(ŷ)) = 0 trên chính dữ liệu khớp, mọi w
    for wf in (None, dl.step(3, 3, 60, 100), dl.step(8, 2, 60, 100)):
        wv = dl.weights_of(wf, y_oof)
        for code in ("R2", "R3", "R4"):
            g = dl.make_rule(code).fit(p_oof, y_oof, wf).predict(p_oof)
            assert abs(np.sum(wv * (y_oof - g)) / wv.sum()) < 1e-6, (code, wf)
    # Trung tâm đã hiệu chỉnh (y = ŷ + nhiễu đối xứng): R4 ở w = 1 cho b ≈ 1, a ≈ 0
    rng = np.random.default_rng(7)
    ph = rng.normal(77, 10, 20000)
    yy = ph + rng.normal(0, 8, 20000)
    r4 = dl.make_rule("R4").fit(ph, yy)
    assert abs(r4.b - 1) < 0.03 and abs(r4.a) < 2.5
    # R7 không cắt, w = 1: trung bình điểm mẫu = ŷ + m_b (Φ⁻¹(τ) đối xứng quanh 0)
    r7 = dl.make_rule("R7", clip=None).fit(p_oof, y_oof)
    bb = dl.bin_assign(r7.edges, p_te)
    assert np.allclose(r7.predict(p_te), p_te + r7.m[bb], atol=1e-9)
    assert np.all(np.isfinite(dl.make_rule("R7").fit(p_oof, y_oof, dl.step(5, 5, 60, 100)).predict(p_te)))


@test
def stretch_and_qmap_match_old_code():
    p_oof, y_oof = realistic(6000, 8)
    p_te, _ = realistic(1500, 9)
    mu = float(y_oof.mean())
    for K in (1, 3, 8):
        w = dl.step(K, K, 60, 100)
        old = tail_prior.stretch_two_sided(p_oof, y_oof, p_te, mu, w(y_oof))
        new = dl.TwoSidedStretch(grid=cost_aware.S_GRID).fit(p_oof, y_oof, w).predict(p_te)
        assert np.allclose(old, new, atol=1e-10), K
    # lưới mới [0,50; 4,00] và cờ chạm biên
    assert abs(dl.S_GRID[0] - 0.5) < 1e-12 and abs(dl.S_GRID[-1] - 4.0) < 1e-12 and len(dl.S_GRID) == 351
    info = dl.TwoSidedStretch().fit(p_oof, y_oof, dl.step(50, 50, 60, 100)).info()
    assert set(info) >= {"s_low", "s_high", "edge_low", "edge_high", "mu"}
    narrow = dl.TwoSidedStretch(grid=[1.0, 1.01]).fit(p_oof, y_oof, dl.step(50, 50, 60, 100)).info()
    assert narrow["edge_low"] and narrow["edge_high"]
    # R6 = quantile map của cost_aware.all_rules
    preds, _ = cost_aware.all_rules(p_oof, y_oof, p_te, mu)
    assert np.allclose(preds["qmap"][1], dl.make_rule("R6").fit(p_oof, y_oof).predict(p_te))
    # Mật độ có tham số trùng PriorDensity ở mặc định
    a, b = tail_prior.PriorDensity(y_oof), dl._Density(y_oof, 2.0, 0.01)
    v = np.linspace(0, 150, 1001)
    assert np.allclose(a(v), b(v)) and isinstance(dl.density(y_oof), tail_prior.PriorDensity)


@test
def crossfit_never_scores_on_fitted_rows():
    seen = []

    class Spy(dl.Rule):
        def fit(self, yhat_oof, y_oof, wfun=None):
            self.ids = set(np.asarray(yhat_oof).astype(int).tolist())
            return self

        def predict(self, yhat):
            ids = set(np.asarray(yhat).astype(int).tolist())
            seen.append((self.ids, ids))
            return np.asarray(yhat, dtype=float)

    n = 1000
    out = dl.crossfit_rule_oof(np.arange(n, dtype=float), np.zeros(n), Spy, n_folds=5, seed=0)
    assert len(seen) == 5 and np.array_equal(out, np.arange(n))
    assert all(not (a & b) for a, b in seen)
    assert set().union(*[b for _, b in seen]) == set(range(n))
    p_oof, y_oof = realistic(3000, 10)
    cf = dl.crossfit_rule_oof(p_oof, y_oof, "R1", dl.step(3, 3, 60, 100), seed=1)
    assert cf.shape == p_oof.shape and np.all(np.isfinite(cf))
    # Có nhóm: không nhóm nào vừa ở phần khớp vừa ở phần chấm điểm của quy tắc
    seen.clear()
    g = np.arange(n) // 2                                  # cặp dòng trùng khoá
    out = dl.crossfit_rule_oof(np.arange(n, dtype=float), np.zeros(n), Spy, n_folds=5, seed=0, groups=g)
    assert len(seen) == 5 and np.array_equal(out, np.arange(n))
    for a, b in seen:
        assert not (set(g[sorted(a)]) & set(g[sorted(b)]))
    assert set().union(*[b for _, b in seen]) == set(range(n))
    # Không có nhóm: giữ nguyên KFold cũ từng bit
    assert np.array_equal(cf, dl.crossfit_rule_oof(p_oof, y_oof, "R1", dl.step(3, 3, 60, 100), seed=1,
                                                   groups=None))


# ---------------------------------------------------------------------------
# Thống kê
# ---------------------------------------------------------------------------
@test
def nb_ttest_by_hand():
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = sp.nb_ttest(d, ratio=0.25)
    # tay: mean 3, s² = 2,5, var = (1/5 + 0,25)·2,5 = 1,125, t = 3/sqrt(1,125), df 4
    se = np.sqrt(1.125)
    t = 3 / se
    # CDF Student df = 4 dạng đóng (không qua scipy)
    F = 0.5 + (3 / 8) * (t / np.sqrt(1 + t * t / 4)) * (1 - (t * t) / (12 * (1 + t * t / 4)))
    assert r["n"] == 5 and r["df"] == 4 and r["wins"] == 0
    assert abs(r["mean"] - 3) < 1e-12 and abs(r["se"] - se) < 1e-12 and abs(r["t"] - t) < 1e-12
    assert abs(r["p"] - 2 * (1 - F)) < 1e-10, (r["p"], 2 * (1 - F))
    assert abs(r["t"] - 2.8284271247) < 1e-9 and abs(r["p"] - 0.0474206556) < 1e-8
    q = 2.7764451052          # t_{0,975; 4}
    assert abs(r["ci_lo"] - (3 - q * se)) < 1e-8 and abs(r["ci_hi"] - (3 + q * se)) < 1e-8
    # Nadeau-Bengio rộng hơn t thường với J = 10
    rng = np.random.default_rng(0)
    d10 = rng.normal(-0.05, 0.02, 10)
    nb = sp.nb_ttest(d10)
    assert abs(nb["se"] ** 2 - (0.1 + 0.25) * np.var(d10, ddof=1)) < 1e-15 and nb["wins"] == int((d10 < 0).sum())
    # NaN bị bỏ, mọi chênh bằng nhau không chia cho 0
    assert sp.nb_ttest([0.1, np.nan, 0.2, 0.3])["n"] == 3
    assert sp.nb_ttest([0.1] * 5)["p"] == 0.0


@test
def tost_and_holm():
    # Holm, ví dụ biết đáp án
    got = sp.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(got, [0.03, 0.06, 0.06, 0.02]), got
    got = sp.holm([0.01, float("nan"), 0.04])
    assert np.isnan(got[1]) and np.allclose([got[0], got[2]], [0.02, 0.04])
    assert np.allclose(sp.holm([0.5, 0.9]), [1.0, 1.0])      # 2 x 0,5 bị chặn ở 1
    assert np.allclose(sp.holm([0.02, 0.9]), [0.04, 0.9])
    # Họ cố định m: phép so thiếu coi như p = 1, không co họ lại
    got = sp.holm([0.03, float("nan"), 0.02], m=3)
    assert np.isnan(got[1]) and np.allclose([got[0], got[2]], [0.06, 0.06]), got
    assert np.allclose(sp.holm([0.03, 0.02], m=3), sp.holm([0.03, 0.02, 1.0])[:2])
    assert np.allclose(sp.holm([0.01, 0.04, 0.03], m=3), sp.holm([0.01, 0.04, 0.03]))
    try:
        sp.holm([0.01, 0.02, 0.03], m=2)
        raise AssertionError("m nhỏ hơn số p phải báo lỗi")
    except ValueError:
        pass
    # Cổng chính: C3 một mình ở p = 0,03 KHÔNG qua (0,09), dù holm co họ sẽ cho 0,03
    ph = sp.primary_holm({"C3": 0.03})
    assert set(ph) == set(gates.PRIMARY_CONTRASTS) and all(v["m"] == 3 for v in ph.values())
    assert abs(ph["C3"]["p_holm"] - 0.09) < 1e-12 and ph["C1"]["missing"] and ph["C1"]["p_holm"] == 1.0
    full = sp.primary_holm({"C1": 0.01, "C2": 0.04, "C3": 0.03})
    assert np.allclose([full[c]["p_holm"] for c in ("C1", "C2", "C3")], sp.holm([0.01, 0.04, 0.03]))
    assert sp.primary_holm({"C1": float("nan"), "C2": None})["C1"]["missing"]
    try:
        sp.primary_holm({"C9": 0.01})
        raise AssertionError("phép so ngoài họ phải báo lỗi")
    except ValueError:
        pass
    # TOST: chênh nhỏ, đều quanh 0 -> tương đương; lệch 0,3 -> không
    rng = np.random.default_rng(3)
    eq = sp.tost_nb(rng.normal(0.0, 0.01, 10), margin=0.10)
    assert eq["passed"] and eq["ci90"][0] > -0.1 and eq["ci90"][1] < 0.1
    ne = sp.tost_nb(rng.normal(0.3, 0.01, 10), margin=0.10)
    assert not ne["passed"]
    # qua TOST <=> CI 90% nằm trong biên
    for s in range(20):
        d = rng.normal(rng.uniform(-0.12, 0.12), 0.03, 10)
        t = sp.tost_nb(d, 0.10)
        assert t["passed"] == (t["ci90"][0] > -0.10 and t["ci90"][1] < 0.10)


@test
def cluster_bootstrap():
    rng = np.random.default_rng(4)
    n, G = 3000, 60
    cl = rng.integers(0, G, n)
    y = np.round(rng.normal(77, 13.6, n) + rng.normal(0, 3, G)[cl])
    p1 = y + rng.normal(0, 8, n)
    r0 = sp.cluster_boot_diff(y, p1, p1, cl, 3, 3, 60, 100, B=300, seed=0)
    assert r0["est"] == 0 and r0["ci_lo"] == 0 and r0["ci_hi"] == 0 and not r0["excludes_zero"]
    p2 = p1 + rng.normal(0, 6, n)
    r = sp.cluster_boot_diff(y, p1, p2, cl, 3, 3, 60, 100, B=500, seed=0)
    assert abs(r["est"] - (sp.cost_k(y, p1, 3, 3, 60, 100) - sp.cost_k(y, p2, 3, 3, 60, 100))) < 1e-10
    assert r["est"] < 0 and r["ci_hi"] < 0 and r["excludes_zero"] and r["n_clusters"] == G
    assert r == sp.cluster_boot_diff(y, p1, p2, cl, 3, 3, 60, 100, B=500, seed=0)    # tái lập


# ---------------------------------------------------------------------------
# Lần chia và ngưỡng đuôi
# ---------------------------------------------------------------------------
@test
def splits_disjoint():
    n = 3000
    tr, te = splits.outer_split(n, 100)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == n and len(te) == 600
    assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(n))
    tr2, _ = splits.outer_split(n, 100)
    assert np.array_equal(tr, tr2)
    plan = splits.fold_plan(tr, 100)
    assert len(plan) == splits.CV_FOLDS
    allva = np.concatenate([f["va"] for f in plan])
    assert np.array_equal(np.sort(allva), np.arange(len(tr)))          # OOF phủ đúng một lần
    for f in plan:
        assert not np.intersect1d(f["es"], f["va"]).size                # dừng sớm không bao giờ là OOF
        assert not np.intersect1d(f["fit"], f["va"]).size
        assert not np.intersect1d(f["fit"], f["es"]).size
        assert np.array_equal(np.sort(np.concatenate([f["fit"], f["es"]])), np.sort(f["train"]))
        assert abs(len(f["es"]) - 0.1 * len(f["train"])) <= 1
        fit2, es2 = splits.es_split(f["train"], 100, f["fold"])
        assert np.array_equal(es2, f["es"])                             # rng riêng (seed, fold)
    fo = splits.fold_of(plan, len(tr))
    assert all((fo[f["va"]] == f["fold"]).all() for f in plan)
    # theo nhóm: không nhóm nào nằm hai phía
    rng = np.random.default_rng(0)
    g = rng.integers(0, 700, n)
    trg, teg = splits.outer_split(n, 101, groups=g)
    assert not np.intersect1d(g[trg], g[teg]).size and len(trg) + len(teg) == n
    for a, b in splits.inner_folds(trg, 101, groups=g):
        assert not np.intersect1d(g[trg][a], g[trg][b]).size
    for f in splits.fold_plan(trg, 101, groups=g):
        gt = g[trg]
        assert not np.intersect1d(gt[f["fit"]], gt[f["es"]]).size         # dừng sớm theo nhóm
        assert not np.intersect1d(gt[f["es"]], gt[f["va"]]).size
        assert np.array_equal(np.sort(np.concatenate([f["fit"], f["es"]])), np.sort(f["train"]))
        n_es, big = round(0.1 * len(f["train"])), np.bincount(gt).max()
        assert n_es <= len(f["es"]) < n_es + big                           # tiền tố ngắn nhất
    # cặp trùng khoá (như E0b tìm thấy): không cặp nào xẻ đôi fit/es
    gp = np.arange(n) // 2
    trp, _ = splits.outer_split(n, 102, groups=gp)
    for f in splits.fold_plan(trp, 102, groups=gp):
        assert not np.intersect1d(gp[trp][f["fit"]], gp[trp][f["es"]]).size
    # không nhóm: es_split giữ nguyên từng bit thuật toán cũ
    for f in plan:
        rng_old = np.random.default_rng(1000 * 100 + f["fold"])
        pick = np.zeros(len(f["train"]), dtype=bool)
        pick[rng_old.choice(len(f["train"]), size=int(round(0.1 * len(f["train"]))), replace=False)] = True
        assert np.array_equal(f["es"], f["train"][pick]) and np.array_equal(f["fit"], f["train"][~pick])
    # cấu hình: đúng miền của ga_xgb.BASE_GENES, không có n_estimators, tái lập theo seed
    cf = splits.sample_configs(60, 100)
    assert cf == splits.sample_configs(60, 100) and cf != splits.sample_configs(60, 101)
    for c in cf:
        assert "n_estimators" not in c and 3 <= c["max_depth"] <= 12 and 0.01 <= c["learning_rate"] <= 0.3
        assert 0.5 <= c["subsample"] <= 1 and 1 <= c["min_child_weight"] <= 20 and 0 <= c["reg_lambda"] <= 10


@test
def cutoffs_from_train_only():
    # Phân phối nguyên dựng sẵn với P(y < 60) = 9,25% và P(y >= 100) = 5,72% -> 60/100
    n = 100000
    lowv = np.repeat(np.arange(40, 60), [463] * 10 + [462] * 10)          # 9.250 dòng dưới 60
    highv = np.repeat(np.arange(100, 110), 572)                         # 5.720 dòng từ 100
    mid = np.resize(np.arange(60, 100), n - len(lowv) - len(highv))
    y = np.concatenate([lowv, mid, highv]).astype(float)
    assert len(lowv) == 9250 and np.allclose(dl.tail_masses(y, 60, 100), (0.0925, 0.0572))
    assert dl.tail_cutoffs(y) == (60.0, 100.0)
    # Nhiễu lấy mẫu của một tập huấn luyện 80% không làm ngưỡng nhảy (quy tắc gần nhất)
    for seed in gates.SEEDS[:5]:
        tr, te = splits.outer_split(n, seed)
        assert dl.tail_cutoffs(y[tr]) == (60.0, 100.0)
    # Ngưỡng chỉ phụ thuộc y huấn luyện: đổi y kiểm tra không đổi ngưỡng; dùng cả y thì đổi
    tr, te = splits.outer_split(n, 100)
    y2 = y.copy()
    y2[te] = 30.0
    assert dl.tail_cutoffs(y2[tr]) == dl.tail_cutoffs(y[tr])
    assert dl.tail_cutoffs(y2) != dl.tail_cutoffs(y[tr])
    # npz giả: lo/hi ghi trong meta tính từ y_tr
    for f in fake_npz():
        d = preds_io.load_split(f)
        assert (d["meta"]["lo"], d["meta"]["hi"]) == dl.tail_cutoffs(d["y_tr"])


# ---------------------------------------------------------------------------
# Đặc trưng, IO, nguồn gốc, hằng số
# ---------------------------------------------------------------------------
@test
def feature_sets_and_school_features():
    F = fake_frame()
    full, dt = F.columns("F_full"), F.columns("F_dt")
    cn, bc = F.columns("F_dt-cn"), F.columns("F_dt-cn-bc")
    assert len(full) - len(dt) == 24 and set(full) - set(dt) == set(features.G12_LATE)
    assert set(dt) - set(cn) == {"gioiTinh", "thangSinh"}
    assert all(c.startswith(("khuVuc_", "Tỉnh_")) for c in set(cn) - set(bc))
    assert not any(c.startswith(("khuVuc_", "Tỉnh_")) for c in bc)
    assert "12.Môn ngoại ngữ_Tiếng Anh" in dt and "12.Toán HK I" in dt and "12.Toán CN" not in dt
    assert "truong_freq" not in F.X.columns and full[-2:] == features.SCHOOL_BASIC_COLS
    # codes không mang tên; khoá trường gồm tỉnh (60 trường, tên trùng giữa tỉnh)
    assert F.school_code.dtype.kind == "i" and F.n_schools == 60 and F.n_provs == 20
    assert set(F.groups) == set(features.GROUP_COLS)
    assert all(v.dtype == np.int8 and len(v) == F.n for v in F.groups.values())
    # Tần suất trường chỉ từ hàng huấn luyện
    tr, te = splits.outer_split(F.n, 100)
    Xtr, Xte = F.design("F_dt", tr, te)
    ftr = Xtr[:, -2]
    _, first = np.unique(F.school_code[tr], return_index=True)
    assert abs(ftr[first].sum() - 1.0) < 1e-12
    sc2 = F.school_code.copy()
    sc2[te] = 10 ** 6                                   # đổi trường của mọi hàng kiểm tra
    a = features.school_basic_features(F.prov_code, sc2, tr, te, chuyen=F.chuyen)
    b = features.school_basic_features(F.prov_code, sc2, tr, tr, chuyen=F.chuyen)
    assert np.all(a[:, 0] == 0.0)                       # trường chưa thấy -> tần suất 0
    assert np.allclose(b[:, 0], ftr)                    # hàng huấn luyện không bị ảnh hưởng
    # design (bincount theo mã) trùng school_basic_features (khoá chuỗi) cho cả hàng áp dụng
    ref = features.school_basic_features(F.prov_code, F.school_code, tr, te, chuyen=F.chuyen)
    assert np.array_equal(Xte[:, -2:], ref)
    names = np.array(["THPT Chuyên A", "THPT B", "THPT Chuyên A", "THPT B"])
    prov = np.array(["P1", "P1", "P2", "P2"])
    sb = features.school_basic_features(prov, names, [0, 1, 1], [0, 1, 2, 3])
    assert np.allclose(sb, [[1 / 3, 1], [2 / 3, 0], [0, 1], [0, 0]])    # khoá (tỉnh, tên)
    assert Xtr.shape[1] == len(dt) and np.all(np.isfinite(Xtr))


@test
def preds_io_roundtrip():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        p = preds_io.split_path(tmp, 100)
        oof = {"default": rng.normal(size=8), "bag10": rng.normal(size=8)}
        test = {"default": rng.normal(size=2), "bag10": rng.normal(size=2)}
        preds_io.save_split(p, idx_tr=np.arange(8), idx_te=np.array([8, 9]), y_tr=np.arange(8.0),
                            y_te=np.array([1.0, 2.0]), school_code=np.arange(10), prov_code=np.zeros(10, int),
                            oof=oof, test=test, seed=100, meta={"lo": 60.0, "note": "giả"})
        d = preds_io.load_split(p)
        assert set(d["oof"]) == {"default", "bag10"} and np.array_equal(d["oof"]["bag10"], oof["bag10"])
        assert d["meta"] == {"lo": 60.0, "note": "giả"} and d["seed"] == 100
        assert preds_io.validate_split(d) == []
        assert np.array_equal(preds_io.rows(d, "te", "school_code"), [8, 9])
        with np.load(p, allow_pickle=False) as z:
            assert "oof__keys" in z.files
        try:
            preds_io.save_split(p, bad=np.array([{"a": 1}], dtype=object))
            raise AssertionError("mảng object phải bị từ chối")
        except TypeError:
            pass
        j = os.path.join(tmp, "r.json")
        preds_io.dump_json_atomic({"a": [1, 2]}, j)
        assert preds_io.load_json(j) == {"a": [1, 2]} and preds_io.load_json(j + "x", {}) == {}
    for f in fake_npz():
        d = preds_io.load_split(f)
        assert preds_io.validate_split(d) == [], f
        assert set(d["oof"]) == {"default", "bag10", "rs_tuned"}
        assert d["trace_oof"].shape[0] == len(d["meta"]["trace_params"])
        # Lược đồ trung bình mô hình fold (E5) và số cây khớp lại
        assert set(d["test_foldavg"]) == set(d["test"]), f
        assert d["trace_test_foldavg"].shape == d["trace_test"].shape
        j = d["meta"]["rs_tuned_index"]
        assert np.array_equal(d["test_foldavg"]["rs_tuned"], d["trace_test_foldavg"][j])
        assert np.array_equal(d["trace_n_trees"], np.median(d["trace_best_iter"] + 1, axis=1).astype(int))
        assert len(d["meta"]["fingerprint"]) == 64
        assert not set(preds_io.GROUP_ARRAYS) & set(d)          # biến nhóm E10 không lưu theo lần chia


@test
def preds_io_json_numpy_and_validation():
    with tempfile.TemporaryDirectory() as tmp:
        # số numpy (np.argmin, np.median trên số nguyên) ở giá trị lẫn khoá dict
        obj = {"j": np.argmin([3, 1, 2]), "med": np.median(np.array([1, 2, 3], dtype=np.int64)),
               "ok": np.bool_(True), "arr": np.arange(3), "f32": np.float32(0.5),
               "per_split": {np.int64(100): {"n": np.int64(7)}}, "t": (np.int32(1), 2.0)}
        j = os.path.join(tmp, "r.json")
        preds_io.dump_json_atomic(obj, j)
        back = preds_io.load_json(j)
        assert back == {"j": 1, "med": 2.0, "ok": True, "arr": [0, 1, 2], "f32": 0.5,
                        "per_split": {"100": {"n": 7}}, "t": [1, 2.0]}, back
        assert not os.path.exists(j + ".tmp")
        p = preds_io.split_path(tmp, 100)
        preds_io.save_split(p, idx_tr=np.arange(4), idx_te=np.array([4]), y_tr=np.arange(4.0),
                            y_te=np.array([1.0]), meta={"jb": np.int64(3), "lo": np.float64(60)})
        assert preds_io.load_split(p)["meta"] == {"jb": 3, "lo": 60.0}
        assert preds_io.split_path(tmp, 100, tag="F_dt-cn").endswith("split100_F_dt-cn.npz")
        # validate_split bắt NaN trong vết, fold_of sai độ dài, test_foldavg sai
        n_tr, n_te = 10, 4
        base = dict(idx_tr=np.arange(n_tr), idx_te=np.arange(n_tr, n_tr + n_te), y_tr=np.zeros(n_tr),
                    y_te=np.zeros(n_te), school_code=np.zeros(n_tr + n_te, int),
                    prov_code=np.zeros(n_tr + n_te, int), oof={"a": np.zeros(n_tr)},
                    test={"a": np.zeros(n_te)}, test_foldavg={"a": np.zeros(n_te)},
                    fold_of=np.arange(n_tr) % 5, trace_oof=np.zeros((3, n_tr)),
                    trace_test=np.zeros((3, n_te)), trace_test_foldavg=np.zeros((3, n_te)),
                    trace_best_iter=np.array([[9, 10, 11, 12, 13]] * 3), trace_n_trees=np.array([12] * 3),
                    trace_oof_rmse=np.ones(3))
        assert preds_io.validate_split(base) == []

        def bad(**kw):
            d = dict(base)
            d.update(kw)
            return preds_io.validate_split(d)
        t = np.zeros((3, n_tr))
        t[1, 0] = np.nan
        assert any("trace_oof" in e for e in bad(trace_oof=t))
        assert any("trace_test" in e for e in bad(trace_test=np.full((3, n_te), np.inf)))
        assert any("fold_of" in e for e in bad(fold_of=np.zeros(n_tr - 1, int)))
        assert any("test_foldavg" in e for e in bad(test_foldavg={"a": np.zeros(n_te + 1)}))
        assert any("test_foldavg" in e for e in bad(test_foldavg={"a": np.full(n_te, np.nan)}))
        assert any("test_foldavg" in e for e in bad(test_foldavg={"zz": np.zeros(n_te)}))
        assert any("trace_test_foldavg" in e for e in bad(trace_test_foldavg=np.zeros((2, n_te))))
        assert any("trace_n_trees" in e for e in bad(trace_n_trees=np.array([11] * 3)))    # 0-based nhầm
        assert any("gender" in e for e in bad(gender=np.zeros(n_tr + n_te, np.int8)))      # không lưu nhân khẩu


@test
def fingerprint_resume():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"feature_set": "F_dt", "n_cfg": 60, "max_trees": 3000}
        fp = preds_io.fingerprint(cfg)
        assert fp == preds_io.fingerprint(dict(reversed(list(cfg.items()))))    # không phụ thuộc thứ tự khoá
        assert fp != preds_io.fingerprint(cfg | {"n_cfg": 40})
        code2 = dict(provenance.code_hashes(), **{"splits.py": "0" * 64})
        assert fp != preds_io.fingerprint(cfg, code=code2)                   # đổi mã -> đổi dấu
        p = preds_io.split_path(tmp, 100)
        assert preds_io.load_or_none(p, fp) is None                           # chưa có
        preds_io.save_split(p, idx_tr=np.arange(3), meta={"fingerprint": fp})
        assert preds_io.load_or_none(p, fp)["meta"]["fingerprint"] == fp      # cùng dấu -> dùng lại
        fp2 = preds_io.fingerprint(cfg | {"stage": 2})
        try:
            preds_io.load_or_none(p, fp2)
            raise AssertionError("khác dấu phải bị từ chối")
        except preds_io.FingerprintMismatch:
            pass
        assert preds_io.load_or_none(p, fp2, on_mismatch="recompute") is None
        preds_io.save_split(p, idx_tr=np.arange(3))                           # npz cũ, không dấu
        try:
            preds_io.load_or_none(p, fp)
            raise AssertionError("npz không dấu phải bị từ chối")
        except preds_io.FingerprintMismatch:
            pass
        # JSON .partial
        part = os.path.join(tmp, "out.json.partial")
        res = preds_io.load_partial(part, fp, {"per_split": {}})
        assert res == {"per_split": {}, "meta": {"fingerprint": fp}}
        res["per_split"]["100"] = {"x": np.int64(1)}
        preds_io.dump_json_atomic(res, part)
        assert preds_io.load_partial(part, fp, {"per_split": {}})["per_split"] == {"100": {"x": 1}}
        try:
            preds_io.load_partial(part, fp2, {"per_split": {}})
            raise AssertionError(".partial khác dấu phải bị từ chối")
        except preds_io.FingerprintMismatch:
            pass
        fresh = preds_io.load_partial(part, fp2, {"per_split": {}}, on_mismatch="recompute")
        assert fresh == {"per_split": {}, "meta": {"fingerprint": fp2}}
        assert len(preds_io.file_sha256(p)) == 64


@test
def rows_from_frame():
    F = fake_frame()
    files = fake_npz()
    if not files:
        return
    d = preds_io.load_split(files[0])
    assert d["meta"]["feature_set"] in features.FEATURE_SETS
    for part in ("tr", "te"):
        idx = d["idx_" + part]
        assert np.array_equal(preds_io.rows(d, part, "school_code"), F.school_code[idx])
        assert np.array_equal(preds_io.rows(d, part, "school_code", frame=F), F.school_code[idx])
        for g in preds_io.GROUP_ARRAYS:
            assert np.array_equal(preds_io.rows(d, part, g, frame=F), F.groups[g][idx])
    try:
        preds_io.rows(d, "te", "gender")
        raise AssertionError("thiếu frame phải báo KeyError")
    except KeyError:
        pass

    class Other:
        y = F.y[::-1].copy()
        groups = F.groups
    try:
        preds_io.rows(d, "te", "gender", frame=Other)
        raise AssertionError("frame khác dữ liệu phải bị phát hiện")
    except ValueError:
        pass


@test
def provenance_stamp():
    s = provenance.stamp({"exp": "test"})
    need = {"commit", "dirty", "diff_sha256", "lock_sha256", "code_sha256", "launched", "host",
            "argv", "lib_versions", "stamp_source"}
    assert need <= set(s) and s["exp"] == "test"
    assert {"decision_layer.py", "splits.py", "stats_paired.py", "provenance.py"} <= set(s["code_sha256"])
    assert s["lib_versions"]["xgboost"] and len(s["lock_sha256"]) == 64
    json.dumps(s)
    with tempfile.TemporaryDirectory() as tmp:
        rs = os.path.join(tmp, "stamp.json")
        provenance.emit_run_stamp(rs, label="t")
        with open(rs, "w") as fh:
            json.dump({"commit": "abc123", "dirty": False, "diff_sha256": ""}, fh)
        os.environ["RUN_STAMP"] = rs
        try:
            s2 = provenance.stamp()
            assert s2["commit"] == "abc123" and s2["dirty"] is False and s2["stamp_source"].startswith("RUN_STAMP")
            with open(rs, "w") as fh:
                fh.write("commit=086ce8e dirty=yes launched=2026-09-24T11:16:22+00:00\n")
            s3 = provenance.stamp()
            assert s3["commit"] == "086ce8e" and s3["dirty"] is True
            assert s3["code_matches_stamp"] is None                     # dạng dòng: không biết
            # stamp do emit_run_stamp ghi ở cùng cây src/: khớp
            provenance.emit_run_stamp(rs, label="t")
            s4 = provenance.stamp()
            assert s4["code_matches_stamp"] is True and s4["stamp_mismatch"] == [], s4["stamp_mismatch"]
            # stamp cũ/sai: commit deadbeef, băm splits.py khác -> bị gắn cờ
            st = json.load(open(rs))
            st["commit"], st["src_sha256"]["splits.py"] = "deadbeef", "0" * 64
            json.dump(st, open(rs, "w"))
            s5 = provenance.stamp()
            assert s5["commit"] == "deadbeef" and s5["code_matches_stamp"] is False
            assert s5["stamp_mismatch"] == ["splits.py"]
        finally:
            del os.environ["RUN_STAMP"]
    # Ảnh chụp lúc khởi động: code_sha256 là mã đã nạp, không phải đĩa lúc ghi JSON
    assert s["launched"] == provenance._BOOT_TIME and "stamped" in s
    assert s["code_changed_since_start"] is False and s["code_changed_files"] == []
    real = provenance._BOOT_DISK["splits.py"]
    try:
        provenance._BOOT_DISK["splits.py"] = "f" * 64          # giả lập: sync đã thay splits.py giữa chừng
        s6 = provenance.stamp()
        assert s6["code_sha256"]["splits.py"] == "f" * 64
        assert s6["code_changed_since_start"] is True and s6["code_changed_files"] == ["splits.py"]
    finally:
        provenance._BOOT_DISK["splits.py"] = real
    # File chưa theo dõi được băm riêng (git diff không thấy chúng)
    g = provenance.git_state()
    if g["commit"] != provenance.UNKNOWN:
        assert isinstance(g["untracked_sha256"], dict)
        assert all(k.startswith("src/") and k.endswith(".py") for k in g["untracked_sha256"])
        assert "untracked_sha256" in s


@test
def gates_constants():
    assert gates.PRIMARY_K == 3 and gates.K_GRID == [2, 3, 5, 8] and gates.SESOI == 0.10
    assert gates.SEEDS == list(range(100, 110)) and gates.B_STAR_TOL == 0.02
    assert (gates.TAIL_MASS_LOW, gates.TAIL_MASS_HIGH) == (0.0925, 0.0572)
    assert len(gates.K_DENSE) == 17 and gates.K_DENSE[0] == 1 and gates.K_DENSE[-1] == 50
    assert len(gates.LAMBDA_DENSE) == 21 and gates.LAMBDA_DENSE[-1] == 1.0
    assert set(gates.CONTRASTS) == set(gates.PRIMARY_CONTRASTS) == {"C1", "C2", "C3"}
    assert all(c["K"] == 3 for c in gates.CONTRASTS.values())
    assert gates.r8_star("bag10") == "R8_bag5" and gates.r8_star("rs_tuned_bag5") == "R8_bag5"
    assert gates.r8_star("rs_tuned") == "R8"
    assert gates.E10_MARGINS == {"a_g": 1.0, "b_g": 0.05, "tau": 1.0, "dFNR": 0.05, "dFPR": 0.02}
    assert splits.CV_FOLDS == 5
    # splits chặn ngay lúc import nếu ga_xgb.CV_FOLDS bị quay về 3 (HEAD cũ)
    code = "import ga_xgb; ga_xgb.CV_FOLDS = 3\ntry:\n import splits\nexcept ImportError: print('chan')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src")))
    assert r.stdout.strip() == "chan", (r.stdout, r.stderr)


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"ok    {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} kiểm thử qua")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
