# -*- coding: utf-8 -*-
"""Kiểm thử E3 (src/proper_scores.py) và E5 (src/sensitivity.py). Script thuần với
assert (không cần pytest).

Chạy: cd <repo> && PYTHONPATH=src python3 tests/test_post_hoc.py [--keep DIR]
Chỉ dùng dữ liệu giả: tests/fixtures/fake_hsa.csv và npz giả của
tests/make_fake_preds.py (tự sinh nếu chưa có; không ghi đè). npz E2b giả và mọi
đầu ra ghi vào thư mục tạm (hoặc --keep DIR), không bao giờ vào results_cost/.
Không đọc data/data_final.csv.

Phần toán kiểm các đẳng thức mà hai script dựa vào:
  - (x - y)² = 2∫S_θ dθ (Ehm et al. 2016): điểm sơ cấp đúng thang.
  - crps_points bằng tích phân số ∫(F - 1{y <= z})² dz, cả khi xác suất không đều.
  - twCRPS qua hàm xích bằng tích phân số ∫(F - 1{y <= z})² u(z) dz.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import decision_layer as dl  # noqa: E402
import preds_io  # noqa: E402
import proper_scores as ps  # noqa: E402
import sensitivity as sens  # noqa: E402
from gates import E5_TAIL_DEFS, K_GRID  # noqa: E402

FAKE = os.path.join(ROOT, "tests", "fixtures", "fake_hsa.csv")
FAKE_PREDS = os.path.join(ROOT, "tests", "fixtures", "preds", "decomp")
TESTS = []


def test(f):
    TESTS.append(f)
    return f


# ---------------------------------------------------------------------------
# Toán
# ---------------------------------------------------------------------------
@test
def elementary_integrates_to_half_squared_error():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(40, 120, 50), rng.integers(30, 130, 50).astype(float)
    th = np.linspace(0, 150, 150 * 200 + 1)
    S = ps.elementary_scores(x, y, th)
    integ = np.trapezoid(S, th, axis=1)
    # S_θ nhảy một bước cao |x - y| tại θ = x, nên quy tắc hình thang sai tối đa h·|x - y|/2
    # ở tích phân, tức h·|x - y| ở 2∫. Dung sai cố định 0,02 cũ chặt hơn cận này (~0,4).
    h = th[1] - th[0]
    err = np.abs(2 * integ - (x - y) ** 2)
    assert np.all(err <= h * np.abs(x - y) + 1e-9), np.max(err - h * np.abs(x - y))
    # Dự đoán đúng bằng y: điểm bằng 0 ở mọi θ
    assert np.all(ps.elementary_scores(y, y, th[::500]) == 0)


def _crps_numeric(X, P, y, u=None, z=np.linspace(-20, 200, 220001)):
    out = []
    for xs, ps_, yy in zip(X, P, y):
        o = np.argsort(xs)
        cdf = np.concatenate([[0.0], np.cumsum(ps_[o])])[np.searchsorted(xs[o], z, side="right")]
        f = (cdf - (yy <= z)) ** 2
        out.append(np.trapezoid(f if u is None else f * u(z), z))
    return np.array(out)


@test
def crps_matches_numeric_integral():
    rng = np.random.default_rng(1)
    n, S = 6, 50
    X = rng.normal(80, 12, (n, S))
    y = rng.normal(80, 14, n)
    P = rng.uniform(0.1, 3, (n, S))
    P /= P.sum(axis=1, keepdims=True)
    for PP in (np.full((n, S), 1 / S), P):
        got = ps.crps_points(X, y, PP)
        want = _crps_numeric(X, PP, y)
        assert np.allclose(got, want, atol=1e-3), (got, want)


@test
def twcrps_chain_matches_weighted_integral():
    rng = np.random.default_rng(2)
    n, S, lo, hi = 6, 50, 60.0, 100.0
    X = rng.normal(80, 18, (n, S))
    y = np.array([45.0, 58.0, 75.0, 99.5, 104.0, 130.0])
    P = rng.uniform(0.1, 3, (n, S))
    P /= P.sum(axis=1, keepdims=True)
    us = {"tails": lambda z: ((z < lo) | (z >= hi)).astype(float),
          "low": lambda z: (z < lo).astype(float), "high": lambda z: (z >= hi).astype(float)}
    for kind, u in us.items():
        v = ps.chain(kind, lo, hi)
        got = ps.crps_points(v(X), v(y), P)
        want = _crps_numeric(X, P, y, u)
        assert np.allclose(got, want, atol=1e-3), (kind, got, want)
    # Trọng số u ≡ 1 ở mọi nơi thì twCRPS = CRPS: lo = +inf, hi = -inf không dùng được,
    # nên kiểm tính cộng: tails = low + high khi hai miền rời nhau
    for PP in (None, P):
        t = ps.crps_points(ps.chain("tails", lo, hi)(X), ps.chain("tails", lo, hi)(y), PP)
        lsum = (ps.crps_points(ps.chain("low", lo, hi)(X), ps.chain("low", lo, hi)(y), PP)
                + ps.crps_points(ps.chain("high", lo, hi)(X), ps.chain("high", lo, hi)(y), PP))
        assert np.allclose(t, lsum, atol=1e-9)


@test
def taggart_thetas_split_by_cutoffs():
    lo_t, hi_t = ps.taggart_thetas(60, 100, (29, 129))
    assert lo_t[0] == 29 and lo_t[-1] == 59 and len(lo_t) == 31
    assert hi_t[0] == 100 and hi_t[-1] == 129 and len(hi_t) == 30


@test
def tilted_r1_is_worse_on_consistent_scores_in_expectation():
    """Trên dữ liệu có p(y|ŷ) đúng như R1 ước lượng, nghiêng theo K làm điểm Taggart
    (nhất quán cho trung bình) tệ hơn R1₁: đúng điều cổng E3 kiểm."""
    rng = np.random.default_rng(3)
    n = 40000
    m = 77 + 9.5 * rng.normal(size=n)
    y = np.round(m + rng.normal(0, 9.7, n))
    half = n // 2
    r = dl.ResidualBinBayes().fit(m[:half], y[:half])
    lo, hi = dl.tail_cutoffs(y[:half])
    tl, th = ps.taggart_thetas(lo, hi, (int(y.min()), int(y.max())))
    t_all = np.concatenate([tl, th])
    s1 = ps.elementary_scores(r.predict(m[half:]), y[half:], t_all).mean()
    sK = ps.elementary_scores(r.predict(m[half:], dl.step(5, 5, lo, hi)), y[half:], t_all).mean()
    assert sK > s1, (sK, s1)


@test
def cluster_boot_mean():
    rng = np.random.default_rng(4)
    d = rng.normal(0.3, 1.0, 3000)
    b = ps.cluster_boot_mean_diff(d, np.arange(3000), B=4000, seed=0)
    assert abs(b["est"] - d.mean()) < 1e-12
    assert abs(b["se"] - d.std(ddof=1) / np.sqrt(3000)) < 0.003
    assert b["excludes_zero"] and b["ci_lo"] > 0


@test
def r8_key_matching():
    keys = ["R8_K2", "R8_K3", "R8_bag5_K3", "R8_bag5_K2", "R8_K20"]
    assert sens.match_r8_key(keys, "R8", 2) == "R8_K2"
    assert sens.match_r8_key(keys, "R8_bag5", 3) == "R8_bag5_K3"
    assert sens.match_r8_key(keys, "R8", 5) is None
    assert sens.match_r8_key(["K=1.5", "K=15"], "R8", 1.5) == "K=1.5"
    try:
        sens.match_r8_key(["R8_K3", "wtrain_K3"], "R8", 3)
        raise AssertionError("khoá mơ hồ phải báo lỗi")
    except ValueError:
        pass


@test
def settings_grid_matches_gates():
    S = sens.build_settings(refit=False)
    names = [s["name"] for s in S]
    assert names[0] == "baseline" and "stretch=old[0.80,2.50]" in names
    assert sum(s["group"] == "bins" for s in S) == 7
    assert sum(s["group"] == "prior" for s in S) == 9
    assert [s["tail"] for s in S if s["group"] == "tail"] == [list(t) for t in E5_TAIL_DEFS[1:]]
    assert not any(s["needs_refit"] for s in S)
    R = sens.build_settings(refit=True)
    assert [s["name"] for s in R if s["needs_refit"]] == ["tree_factor=1", "tree_factor=1.2"]
    assert sens.S_GRID_OLD[0] == 0.80 and sens.S_GRID_OLD[-1] == 2.50


@test
def guard_refuses_real_data_on_mac():
    if sys.platform != "darwin":
        return
    try:
        sens.guard_real_data(sens.REAL_DATA)
        raise AssertionError("phải từ chối data_final.csv trên Mac")
    except SystemExit:
        pass


# ---------------------------------------------------------------------------
# Chạy thử đầu cuối trên dữ liệu giả
# ---------------------------------------------------------------------------
def ensure_fake_e1():
    if not os.path.exists(FAKE):
        import make_fake_data
        make_fake_data.main(["--out", FAKE])
    if not (os.path.isdir(FAKE_PREDS) and any(f.endswith(".npz") for f in os.listdir(FAKE_PREDS))):
        import make_fake_preds
        make_fake_preds.main(["--n-jobs", "2"])
    return sorted(int(f[5:-4]) for f in os.listdir(FAKE_PREDS)
                  if f.startswith("split") and f.endswith(".npz") and "_" not in f)


def make_fake_e2b(out_dir, seeds, base="bag10"):
    """npz E2b GIẢ: R8 và R8_bag5 ở từng K là trung tâm `base` giãn theo K. Chỉ để chạy
    qua đường mã của C1 (khoá, căn hàng), không mô phỏng huấn luyện có trọng số."""
    for s in seeds:
        d = preds_io.load_split(preds_io.split_path(FAKE_PREDS, s))
        mu = float(np.mean(d["y_tr"]))
        p = np.asarray(d["test"][base], dtype=float)
        test = {}
        for K in K_GRID:
            test[f"R8_K{K}"] = mu + (1 + 0.05 * (K - 1)) * (p - mu)
            test[f"R8_bag5_K{K}"] = mu + (1 + 0.04 * (K - 1)) * (p - mu)
        preds_io.save_split(preds_io.split_path(out_dir, s), idx_te=d["idx_te"], y_te=d["y_te"],
                            test=test, meta={"fake": True, "seed": int(s)})


def _finite(x):
    return x is not None and np.isfinite(x)


def run_end_to_end(work):
    seeds = ensure_fake_e1()
    assert seeds, "không có npz E1 giả"
    e2b = os.path.join(work, "preds", "wtrain")
    make_fake_e2b(e2b, seeds)
    common = ["--smoke", "--seeds", *map(str, seeds), "--e1-dir", FAKE_PREDS, "--data", FAKE,
              "--centers-json", os.path.join(work, "khong_co.json"), "--primary", "bag10",
              "--bstar", "bag10", "--workers", "1"]

    out3 = os.path.join(work, "results", "proper_scores.json")
    ps.main(common + ["--out", out3, "--preds-dir", os.path.join(work, "preds", "proper_scores")])
    r3 = json.load(open(out3, encoding="utf-8"))
    assert r3["meta"]["fingerprint"] and r3["meta"]["provenance"]["code_sha256"]
    e = r3["per_split"][str(seeds[0])]
    assert len(e["forecasts"]["bag10|R0"]["murphy"]) == 101
    assert _finite(e["forecasts"]["bag10|R1@K3"]["taggart"])
    assert {"crps", "twcrps", "pinball"} <= set(e["dist"]["bag10"])
    S3 = r3["summary"]
    assert "center_forecast_gain" in S3["gates"] and "tilt_check" in S3["gates"]
    assert "bag10|R0 - default|R0" in S3["pairs"]

    out5 = os.path.join(work, "results", "sensitivity.json")
    pd5 = os.path.join(work, "preds", "sensitivity")
    sens.main(common + ["--out", out5, "--preds-dir", pd5, "--e2b-dir", e2b, "--refit"])
    r5 = json.load(open(out5, encoding="utf-8"))
    e = r5["per_split"][str(seeds[0])]
    assert e["r8"]["status"] == "ok" and e["r8"]["keys"]["3"] == "R8_bag5_K3", e["r8"]
    assert e["refit"]["status"] == "fitted", e["refit"]
    assert os.path.exists(preds_io.split_path(pd5, seeds[0], "refit"))
    base = r5["summary"]["settings"]["baseline"]["contrasts"]["step:3"]
    for cn in sens.CONTRASTS:
        assert _finite(base[cn]["mean"]) and base[cn]["valid"], (cn, base[cn])
    tail = r5["summary"]["settings"]["tail=fixed50/110"]["contrasts"]["step:3"]["C1"]
    assert not tail["valid"] and "đuôi" in tail["note"]
    pri = r5["summary"]["settings"]["prior:sigma=2,floor=0.01"]["contrasts"]
    assert set(pri) == {"prior:0.5", "prior:1"} and not pri["prior:1"]["C1"]["valid"]
    tf = r5["summary"]["settings"]["tree_factor=1.2"]
    assert tf["needs_refit"] and "contrasts" in tf, tf
    sk = r5["summary"]["sign_keep"]["step:3"]["C2"]
    assert sk["all"]["n_settings"] >= sk["posthoc_only"]["n_settings"]
    assert r5["summary"]["stretch_edges"]["old"]["K_DENSE"]["n_cells"] == len(seeds[:1]) * 17
    fi = r5["summary"]["frontier_interp"]["rules"]["R1"]
    assert set(fi) == {"0.25", "0.5", "1.0"}
    g = r5["summary"]["gates"]
    assert {"unconditional", "c2_tail_definition", "stretch_old_grid"} <= set(g)

    # Chạy lại: lần chia đã có trong .partial được bỏ qua, bộ đệm refit được dùng
    os.remove(out5)
    sens.main(common + ["--out", out5, "--preds-dir", pd5, "--e2b-dir", e2b, "--refit"])
    r5b = json.load(open(out5, encoding="utf-8"))
    assert r5b["summary"]["settings"]["baseline"] == r5["summary"]["settings"]["baseline"]
    # Xoá .partial để tính lại lần chia: refit phải đọc từ bộ đệm, cùng số
    os.remove(out5 + ".partial")
    sens.main(common + ["--out", out5, "--preds-dir", pd5, "--e2b-dir", e2b, "--refit"])
    r5c = json.load(open(out5, encoding="utf-8"))
    assert r5c["per_split"][str(seeds[0])]["refit"]["status"] == "cached"
    assert (r5c["summary"]["settings"]["tree_factor=1.2"]["contrasts"]
            == r5["summary"]["settings"]["tree_factor=1.2"]["contrasts"])
    # Thiếu E2b: C1 = NaN và không hợp lệ, không lỗi
    out5n = os.path.join(work, "results", "sensitivity_no_e2b.json")
    sens.main(common + ["--out", out5n, "--preds-dir", pd5, "--e2b-dir", os.path.join(work, "khong_co")])
    r5n = json.load(open(out5n, encoding="utf-8"))
    c1 = r5n["summary"]["settings"]["baseline"]["contrasts"]["step:3"]["C1"]
    assert not c1["valid"] and c1["n"] == 0
    return out3, out5


@test
def end_to_end_smoke():
    keep = ARGS.keep
    work = keep or tempfile.mkdtemp(prefix="post_hoc_")
    os.makedirs(work, exist_ok=True)
    try:
        run_end_to_end(work)
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def main():
    failed = 0
    for f in TESTS:
        try:
            f()
            print(f"ok   {f.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {f.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} qua")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=None, help="giữ đầu ra ở thư mục này thay vì thư mục tạm")
    ARGS = ap.parse_args()
    main()
