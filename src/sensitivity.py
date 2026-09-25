# -*- coding: utf-8 -*-
"""E5: độ nhạy của từng tham số trong Bảng II (khung bài 24/9, mục 6.9).

Vì sao cần E5: mỗi tham số của tầng quyết định (B, S, kiểu bin, định nghĩa đuôi,
lưới giãn, σ và sàn của mật độ, số cây khi khớp lại, nguồn ŷ test) hoặc phải có lý
do, hoặc phải được chứng minh là không đổi kết luận trong một dải. Kết luận ở đây là
DẤU của ba phép so chính và của R2 - R1:

  C1    = cost_K(R1) - cost_K(R8*)           trên trung tâm chính (R8* từ E2b)
  C2    = cost_K(R1) - cost_K(R5)            trên trung tâm chính
  C3    = cost_K(R1 trên rs_tuned) - cost_K(R1 trên bag B*)
  R2-R1 = cost_K(R2) - cost_K(R1)            trên trung tâm chính
(âm = vế trái tốt hơn). Với mỗi thiết lập: dấu, trung bình và p Nadeau-Bengio qua
các lần chia, ở mọi K của lưới báo cáo (K = 3 là chính). Cổng: một tuyên bố được
viết không điều kiện khi dấu giữ ở >= 90% thiết lập; thiết lập nào đổi dấu thì được
nêu tên, không chỉnh thiết lập để lấy lại tiêu đề.

Hai phần TÁCH BẠCH:
  1. Hậu kỳ thuần (mặc định, vài phút): chỉ khớp lại tầng quyết định trên npz của E1
     (preds/decomp) và dùng dự đoán R8* của E2b (preds/wtrain).
  2. Cần khớp lại mô hình (--refit, tuỳ chọn): hệ số số cây khi khớp lại {1,0; 1,2}.
     Khớp lại cấu hình rs_tuned đã chọn (và rs_tuned_bag5 nếu có trong npz) trên toàn
     tập huấn luyện, một lần cho mỗi hệ số và mỗi lần chia. Cần --data (chạy ở server)
     và XGBoost. Dự đoán lưu đệm ở <preds-dir>/split<seed>_refit.npz. Hệ số 1,0 cũng
     khớp lại (không lấy từ npz) để hai hệ số đi cùng một đường mã; max|Δ| so với npz
     được ghi làm phép kiểm tái lập. Nhóm này mang cờ needs_refit trong JSON và
     cổng dấu được báo cả khi có và khi không có nó.

Thiết lập (đổi MỘT thứ so với mốc, trừ B × kiểu bin đi cặp như khung bài):
  baseline         B = 20 bằng số lượng, S = 50, đuôi 9,25%/5,72% theo khối lượng,
                   lưới giãn [0,50; 4,00], ŷ test của mô hình khớp lại, mọi bản ghi.
  bins             B ∈ {10; 20; 40; 80} × {bằng số lượng; bằng bề rộng}
  samples          S ∈ {25; 100}
  tail             5%/5% theo khối lượng; cố định 50/110; relevance boxplot 41/113
  stretch          lưới cũ [0,80; 2,50]
  prior            KDE σ ∈ {1; 2; 4} × sàn ∈ {0,001; 0,01; 0,05}, họ prior ở λ ∈ {0,5; 1}
  test_src         ŷ test = trung bình 5 mô hình fold (test_foldavg của npz E1)
  near_dup         bỏ các nhóm trùng gần (mọi dòng thuộc nhóm có >= 2 dòng)
  refit            hệ số số cây 1,0 và 1,2 (chỉ với --refit)
Thêm hai phần mô tả, không kiểm định (mục 5.6: "không kiểm định trên biên"):
  stretch_edges    số ô (lần chia × K) mà s của giãn chạm biên lưới, lưới cũ và mới,
                   trên K_GRID và K_DENSE
  frontier_interp  RMSE hai đuôi khi RMSE giữa tăng Δ ∈ {0,25; 0,5; 1,0} so với raw,
                   nội suy trên lưới K dày (17 điểm) so với 8 điểm cũ, cho R1, R2, R5

Chọn lựa khi khung bài để ngỏ (ghi lại theo yêu cầu):
  - C1 chỉ HỢP LỆ khi R8* được huấn luyện cho đúng chi phí đang chấm: không hợp lệ ở
    nhóm tail (R8* học với đuôi chính, E5 không huấn luyện lại), ở họ prior (E2b chỉ
    dò R8 cho họ bậc thang) và ở nhóm refit nếu trung tâm chính là trung tâm được
    khớp lại (R8* không được khớp lại theo hệ số). Số vẫn được ghi, kèm valid = false
    và lý do; cổng dấu chỉ đếm thiết lập hợp lệ.
  - Họ prior không có C1..C3 theo định nghĩa (chúng ở họ bậc thang), nên nhóm prior
    báo các phép so tương tự dưới cost của chính họ prior, ở λ ∈ {0,5; 1} (λ = 1 là
    Balanced MSE hậu kỳ, nơi sàn mật độ tác động mạnh nhất). Cổng dấu của nhóm này
    lấy mốc riêng là σ = 2, sàn 0,01.
  - Đuôi theo khối lượng tính trên y huấn luyện (sau khi bỏ trùng gần nếu có);
    đuôi cố định dùng thẳng ngưỡng.
  - Bỏ trùng gần là xấp xỉ HẬU KỲ: bỏ khỏi cặp (ŷ OOF, y) để khớp quy tắc và khỏi tập
    kiểm tra; mô hình trung tâm và R8* vẫn là mô hình đã học cả các dòng đó. Khoá gần
    dựng lại tại đây theo định nghĩa (2) của E0b: (tỉnh, trường, mọi cột học bạ bắt
    đầu bằng "10.", "11.", "12.", chuỗi đã strip), không có ngày sinh. Header thật có
    111 cột như vậy (khung bài ghi 129); dùng đúng các cột có trong file. Vì cần đọc
    CSV, phần này chạy ở server (hoặc trên CSV giả khi kiểm thử).
  - Nguồn ŷ test chỉ đổi ŷ mà QUY TẮC được áp; R8* là dự đoán trực tiếp, không đổi.
  - Hệ số cây: cấu hình được chọn = argmin trace_oof_rmse (đúng quy tắc chọn của E1);
    đối chiếu với meta.rs_tuned_index nếu npz có. random_state 0 cho rs_tuned và
    0..4 cho rs_tuned_bag5 (khung bài mục 6.4).
  - Tên khoá R8* trong npz E2b: --r8-key (mẫu format với {r8}, {K}); mặc định "auto"
    tìm khoá trong d["test"] có token K (ví dụ "R8_bag5_K3", "K3") và có/không có
    "bag" đúng với R8* (gates.r8_star). Không thấy thì C1 = NaN, ghi rõ. npz E2b phải
    cùng lần chia với E1 (so idx_te nếu có).
  - Tỉ lệ "dấu giữ" tính trên mọi thiết lập hợp lệ CÓ tính mốc; tỉ lệ không tính mốc
    cũng được ghi.

Chạy (server): PYTHONPATH=src .venv/bin/python src/sensitivity.py --workers 5 [--refit]
Đầu ra: results_cost/sensitivity.json.
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

import decision_layer as dl
# features import ở đầu file, không import muộn trong _frame: dấu vân tay băm các module src/
# đang import lúc main() bắt đầu, import muộn thì features.py nằm ngoài dấu, và hai lượt gọi
# trong cùng tiến trình (lượt sau đã có features) ra hai dấu khác nhau.
import features
import preds_io
import provenance
import splits
import stats_paired as sp
from gates import (DENS_FLOOR, E5_BINNING, E5_DENS_FLOOR, E5_EDGE_HIT_MAX, E5_KDE_SIGMA,
                   E5_N_BINS, E5_N_SAMPLES, E5_SIGN_KEEP, E5_TAIL_DEFS, E5_TREE_FACTOR, K_DENSE,
                   K_GRID, KDE_SIGMA, N_BINS, N_SAMPLES, PRIMARY_K, R8_BAG, STRETCH_S_OLD,
                   STRETCH_S_STEP, TAIL_MASS_HIGH, TAIL_MASS_LOW, r8_star)
from proper_scores import (add_common_args, drive_splits, e1_paths, fmt_k, input_shas, load_e1,
                           ms, parse_tags, resolve_centers)
from tail_prior import MID_BUDGETS, STEP_KS as OLD_FRONTIER_KS, frontier

CONTRASTS = ["C1", "C2", "C3", "R2-R1"]
PRIOR_LAMBDAS = [0.5, 1.0]
S_GRID_OLD = np.round(np.arange(STRETCH_S_OLD[0], STRETCH_S_OLD[1] + STRETCH_S_STEP / 2,
                                STRETCH_S_STEP), 2)
REFIT_CENTERS = ("rs_tuned", "rs_tuned_bag5")
REFIT_VERSION = 1                 # tăng khi đổi mã khớp lại: làm mất hiệu lực bộ đệm refit
SMOKE_TREE_CAP = 50
GRADE_PREFIXES = ("10.", "11.", "12.")
BASELINE = "baseline"
PRIOR_BASELINE = f"prior:sigma={KDE_SIGMA:g},floor={DENS_FLOOR:g}"
REAL_DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                         "data", "data_final.csv"))

assert tuple(E5_TAIL_DEFS[0]) == ("mass", TAIL_MASS_LOW, TAIL_MASS_HIGH), \
    "E5_TAIL_DEFS[0] phải là định nghĩa đuôi chính"


# ---------------------------------------------------------------------------
# Thiết lập
# ---------------------------------------------------------------------------
BASE = {"n_bins": N_BINS, "binning": "count", "n_samples": N_SAMPLES,
        "tail": list(E5_TAIL_DEFS[0]), "grid": "new", "family": "step", "sigma": KDE_SIGMA,
        "floor": DENS_FLOOR, "test_src": "refit", "drop_near_dup": False, "tree_factor": None,
        "needs_refit": False}


def tail_name(td):
    kind, a, b = td
    return f"tail=mass{a * 100:g}%/{b * 100:g}%" if kind == "mass" else f"tail=fixed{a:g}/{b:g}"


def build_settings(refit):
    S = [dict(BASE, name=BASELINE, group="baseline")]
    for B in E5_N_BINS:
        for bn in E5_BINNING:
            if (B, bn) != (N_BINS, "count"):
                S.append(dict(BASE, name=f"B={B},{bn}", group="bins", n_bins=B, binning=bn))
    for s in E5_N_SAMPLES:
        if s != N_SAMPLES:
            S.append(dict(BASE, name=f"S={s}", group="samples", n_samples=s))
    for td in E5_TAIL_DEFS[1:]:
        S.append(dict(BASE, name=tail_name(td), group="tail", tail=list(td)))
    S.append(dict(BASE, name=f"stretch=old[{STRETCH_S_OLD[0]:.2f},{STRETCH_S_OLD[1]:.2f}]",
                  group="stretch", grid="old"))
    for sg in E5_KDE_SIGMA:
        for fl in E5_DENS_FLOOR:
            S.append(dict(BASE, name=f"prior:sigma={sg:g},floor={fl:g}", group="prior",
                          family="prior", sigma=sg, floor=fl))
    S.append(dict(BASE, name="test_src=foldavg", group="test_src", test_src="foldavg"))
    S.append(dict(BASE, name="drop_near_dup", group="near_dup", drop_near_dup=True))
    if refit:
        for f in E5_TREE_FACTOR:
            S.append(dict(BASE, name=f"tree_factor={f:g}", group="refit", tree_factor=float(f),
                          needs_refit=True))
    names = [s["name"] for s in S]
    assert len(set(names)) == len(names), names
    assert PRIOR_BASELINE in names
    return S


# ---------------------------------------------------------------------------
# Khoá trùng gần (E0b, định nghĩa 2) và khớp lại mô hình
# ---------------------------------------------------------------------------
def guard_real_data(path):
    """Không đọc data_final.csv trên Mac (AGENTS.md, ràng buộc 1): server chạy Linux."""
    if sys.platform == "darwin" and os.path.abspath(path) == REAL_DATA:
        sys.exit("Không đọc data/data_final.csv trên máy này (AGENTS.md, ràng buộc 1). "
                 "Kiểm thử bằng --data tests/fixtures/fake_hsa.csv; chạy thật ở server.")


def near_dup_rows(path):
    """(mask dài n: dòng thuộc nhóm trùng gần có >= 2 dòng, y theo CSV, thông tin đếm).

    Khoá: (tỉnh, trường, mọi cột học bạ), chuỗi đã strip. Nhóm hoá bằng factorize
    trên chuỗi ghép thay vì SHA-256: cùng phân nhóm, không cần lưu khoá."""
    head = list(pd.read_csv(path, nrows=0).columns)
    grade = [c for c in head if c.startswith(GRADE_PREFIXES)]
    cols = ["Tỉnh", "Trường"] + grade
    raw = pd.read_csv(path, usecols=cols + ["Điểm HSA"], dtype=str, keep_default_na=False)
    parts = [raw[c].astype(str).str.strip() for c in cols]
    key = parts[0].str.cat(parts[1:], sep="\x1f")
    codes, _ = pd.factorize(key)
    cnt = np.bincount(codes)
    mask = cnt[codes] >= 2
    y = pd.to_numeric(raw["Điểm HSA"], errors="coerce").to_numpy(float)
    info = {"n_rows": int(len(mask)), "n_grade_cols": len(grade), "n_groups_ge2": int((cnt >= 2).sum()),
            "n_rows_in_groups": int(mask.sum())}
    return mask, y, info


_FRAME = {}


def _frame(path):
    """features.Frame, nạp một lần mỗi tiến trình (preprocess mất vài giây trên 57k dòng)."""
    if path not in _FRAME:
        _FRAME[path] = features.load_frame(path)
    return _FRAME[path]


def refit_predictions(d, seed, cfg):
    """({hệ số: {trung tâm: ŷ test}}, info): khớp lại cấu hình rs_tuned đã chọn với
    số cây = round(hệ số × trace_n_trees[j*]) trên toàn tập huấn luyện."""
    cents = [c for c in REFIT_CENTERS if c in d["test"]]
    need = [k for k in ("trace_oof_rmse", "trace_n_trees") if k not in d]
    params_all = (d.get("meta") or {}).get("trace_params")
    if not cents or need or not params_all:
        return {}, {"status": "skipped", "reason": f"thiếu trung tâm {REFIT_CENTERS} hoặc vết cấu "
                    f"hình (thiếu {need}, trace_params={'có' if params_all else 'không'})"}
    jb = int(np.argmin(np.asarray(d["trace_oof_rmse"], dtype=float)))
    meta_jb = d["meta"].get("rs_tuned_index")
    params = dict(params_all[jb])
    n0 = int(np.asarray(d["trace_n_trees"])[jb])
    cap = SMOKE_TREE_CAP if cfg["smoke"] else None
    n_of = {f: min(max(1, int(round(f * n0))), cap or 10 ** 9) for f in cfg["tree_factors"]}
    info = {"cfg_index": jb, "meta_rs_tuned_index": meta_jb, "params": params, "n_trees_base": n0,
            "n_trees": {fmt_k(f): n for f, n in n_of.items()}, "centers": cents,
            "tree_cap": cap, "random_state": {c: list(range(R8_BAG)) if c.endswith("bag5") else [0]
                                              for c in cents}}
    code = {k: v for k, v in provenance.code_hashes().items() if k in ("features.py", "preprocess.py")}
    fpc = preds_io.fingerprint({"refit_version": REFIT_VERSION, "e1": cfg["inputs_of"][seed]["e1"],
                                "data_sha256": cfg["data_sha256"], "n_trees": info["n_trees"],
                                "centers": cents, "params": params}, code=code)
    cache = preds_io.split_path(cfg["preds_dir"], seed, "refit")
    hit = preds_io.load_or_none(cache, fpc, on_mismatch="recompute")
    if hit is not None:
        preds = hit["refit"]
        info["status"] = "cached"
    else:
        from xgboost import XGBRegressor
        F = _frame(cfg["data"])
        tr, te = np.asarray(d["idx_tr"]), np.asarray(d["idx_te"])
        if not np.array_equal(np.asarray(F.y, dtype=float)[tr], np.asarray(d["y_tr"], dtype=float)):
            raise ValueError(f"lần chia {seed}: y của --data không khớp y_tr trong npz E1")
        Xtr, Xte = F.design(d["meta"]["feature_set"], tr, te)
        preds, t0 = {}, time.time()
        for f, n in n_of.items():
            for c in cents:
                ms_ = [XGBRegressor(tree_method="hist", random_state=r, n_estimators=n,
                                    n_jobs=cfg["threads"], verbosity=0, **params)
                       .fit(Xtr, d["y_tr"]).predict(Xte) for r in info["random_state"][c]]
                preds[f"{f:g}|{c}"] = np.mean(ms_, axis=0)
        info["status"] = "fitted"
        info["fit_s"] = round(time.time() - t0, 1)
        preds_io.save_split(cache, idx_te=te, refit=preds,
                            meta={"fingerprint": fpc, "seed": int(seed), "info": info,
                                  "provenance": provenance.stamp()})
    out = {}
    for k, a in preds.items():
        f, c = k.split("|", 1)
        out.setdefault(float(f), {})[c] = np.asarray(a, dtype=float)
    if 1.0 in out:
        info["repro_max_abs_vs_npz"] = {c: float(np.max(np.abs(out[1.0][c] - np.asarray(d["test"][c]))))
                                        for c in cents}
    return out, info


# ---------------------------------------------------------------------------
# R8* từ npz của E2b
# ---------------------------------------------------------------------------
def match_r8_key(keys, r8name, K):
    """Khoá của R8* ở K trong d["test"] của npz E2b: có token K (K3, k=3, K_3) và có
    'bag' đúng như r8name. Nhiều khoá khớp thì báo lỗi thay vì chọn bừa."""
    want_bag = "bag" in r8name.lower()
    pat = re.compile(rf"k[=_]?{re.escape(fmt_k(K))}(?![0-9.])")
    c = [k for k in keys if ("bag" in k.lower()) == want_bag and pat.search(k.lower())]
    if len(c) > 1:
        raise ValueError(f"nhiều khoá R8* khớp K = {fmt_k(K)}: {c}; đặt --r8-key")
    return c[0] if c else None


def load_r8(cfg, seed, d_e1):
    p = preds_io.split_path(cfg["e2b_dir"], seed, cfg["e2b_tag"])
    info = {"path": p, "r8_star": cfg["r8_star"]}
    if not os.path.exists(p):
        return {}, info | {"status": "missing"}
    d = preds_io.load_split(p)
    for k in ("idx_te", "y_te"):
        if k in d and not np.array_equal(np.asarray(d[k], dtype=float), np.asarray(d_e1[k], dtype=float)):
            raise ValueError(f"{p}: '{k}' khác npz E1 của lần chia {seed}; E1 và E2b không cùng lần chia")
    tests = d.get("test", {})
    out, keys = {}, {}
    for K in cfg["ks"]:
        if cfg["r8_key"] == "auto":
            k = match_r8_key(list(tests), cfg["r8_star"], K)
        else:
            k = cfg["r8_key"].format(r8=cfg["r8_star"], K=fmt_k(K))
        if k is None or k not in tests:
            continue
        a = np.asarray(tests[k], dtype=float)
        if len(a) != len(d_e1["y_te"]):
            raise ValueError(f"{p}: test[{k}] dài {len(a)}, cần {len(d_e1['y_te'])}")
        out[K], keys[fmt_k(K)] = a, k
    info |= {"status": "ok" if out else "no_matching_keys", "keys": keys,
             "aligned_by": "idx_te" if "idx_te" in d else "length", "available_keys": sorted(tests)}
    return out, info


# ---------------------------------------------------------------------------
# Một thiết lập trên một lần chia
# ---------------------------------------------------------------------------
def skip_reason(st, ctx):
    need = ctx["need"]
    if st["test_src"] == "foldavg" and not all(c in ctx["foldavg"] for c in need):
        return f"npz E1 thiếu test_foldavg cho {[c for c in need if c not in ctx['foldavg']]}"
    if st["drop_near_dup"] and ctx.get("keep_tr") is None:
        return "không có khoá trùng gần (thiếu --data)"
    if st["tree_factor"] is not None and st["tree_factor"] not in ctx.get("refit", {}):
        return "chưa khớp lại (--refit tắt hoặc npz thiếu vết cấu hình)"
    return None


def eval_setting(st, ctx):
    if st["drop_near_dup"]:
        mtr, mte = ctx["keep_tr"], ctx["keep_te"]
    else:
        mtr = mte = slice(None)
    y_tr, y_te = ctx["y_tr"][mtr], ctx["y_te"][mte]
    kind, a, b = st["tail"]
    lo, hi = dl.tail_cutoffs(y_tr, a, b) if kind == "mass" else (float(a), float(b))
    src = dict(ctx["foldavg"] if st["test_src"] == "foldavg" else ctx["test"])
    if st["tree_factor"] is not None:
        src |= ctx["refit"][st["tree_factor"]]
    P, BS, RS = ctx["primary"], ctx["bstar"], ctx["rs"]
    oof = {c: np.asarray(ctx["oof"][c], dtype=float)[mtr] for c in ctx["need"]}
    pte = {c: np.asarray(src[c], dtype=float)[mte] for c in ctx["need"]}
    r1 = {c: dl.ResidualBinBayes(st["n_bins"], st["n_samples"], st["binning"]).fit(oof[c], y_tr)
          for c in ctx["need"]}
    grid = S_GRID_OLD if st["grid"] == "old" else dl.S_GRID
    if st["family"] == "step":
        params = [(f"step:{fmt_k(K)}", K, dl.step(K, K, lo, hi)) for K in ctx["ks"]]
    else:
        dens = dl.density(y_tr, st["sigma"], st["floor"])
        params = [(f"prior:{lam:g}", lam, dl.prior(lam, dens)) for lam in ctx["lambdas"]]
    ml, mh = dl.tail_masses(y_tr, lo, hi)
    out = {"lo": lo, "hi": hi, "mass_low": ml, "mass_high": mh, "n_tr": int(len(y_tr)),
           "n_te": int(len(y_te))}
    nan = float("nan")
    for key, prm, wf in params:
        wte = wf(y_te)

        def cost(p):
            return sp.cost_w(y_te, p, wte)
        R1 = {c: r1[c].predict(pte[c], wf) for c in ctx["need"]}
        R2 = dl.WeightedIsotonic().fit(oof[P], y_tr, wf).predict(pte[P])
        r5 = dl.TwoSidedStretch(grid=grid).fit(oof[P], y_tr, wf)
        r8 = ctx["r8"].get(prm) if st["family"] == "step" else None
        c = {"R0": cost(pte[P]), "R1": cost(R1[P]), "R2": cost(R2), "R5": cost(r5.predict(pte[P])),
             "R1@rs_tuned": cost(R1[RS]) if RS else nan, "R1@bstar": cost(R1[BS]),
             "R8*": cost(np.asarray(r8)[mte]) if r8 is not None else nan}
        out[key] = {"C1": c["R1"] - c["R8*"], "C2": c["R1"] - c["R5"],
                    "C3": c["R1@rs_tuned"] - c["R1@bstar"], "R2-R1": c["R2"] - c["R1"],
                    "cost": c, "r5": r5.info()}
    return out


def frontier_and_edges(ctx):
    """Biên nội suy (lưới dày so với 8 điểm cũ) và số lần s của giãn chạm biên lưới,
    trên trung tâm chính với định nghĩa đuôi chính."""
    P, y_tr, y_te, lo, hi = ctx["primary"], ctx["y_tr"], ctx["y_te"], ctx["lo"], ctx["hi"]
    oofP, teP = np.asarray(ctx["oof"][P], dtype=float), np.asarray(ctx["test"][P], dtype=float)
    r1 = dl.ResidualBinBayes().fit(oofP, y_tr)
    base_mid = sp.region_rmse(y_te, teP, lo, hi)["Middle"]
    pts = {"R1": [], "R2": [], "R5": []}
    edges = {"new": {}, "old": {}}
    for K in K_DENSE:
        wf = dl.step(K, K, lo, hi)
        r5 = {g: dl.TwoSidedStretch(grid=grid).fit(oofP, y_tr, wf)
              for g, grid in (("new", dl.S_GRID), ("old", S_GRID_OLD))}
        preds = {"R1": r1.predict(teP, wf), "R2": dl.WeightedIsotonic().fit(oofP, y_tr, wf).predict(teP),
                 "R5": r5["new"].predict(teP)}
        for r, p in preds.items():
            rr = sp.region_rmse(y_te, p, lo, hi)
            pts[r].append({"K": K, "Middle": rr["Middle"], "Tails": rr["Tails"]})
        for g in r5:
            i = r5[g].info()
            edges[g][fmt_k(K)] = {k: i[k] for k in ("s_low", "s_high", "edge_low", "edge_high")}
    old = {float(k) for k in OLD_FRONTIER_KS}
    fr = {r: {"dense": frontier(p, base_mid),
              "old": frontier([q for q in p if float(q["K"]) in old], base_mid), "points": p}
          for r, p in pts.items()}
    return {"base_mid": base_mid, "rules": fr}, edges


def run_split(seed, cfg):
    t0 = time.time()
    d = load_e1(cfg["e1_dir"], seed, cfg["e1_tags"])
    y_tr = np.asarray(d["y_tr"], dtype=float)
    y_te = np.asarray(d["y_te"], dtype=float)
    lo, hi = dl.tail_cutoffs(y_tr)
    P, BS = cfg["primary"], cfg["bstar"]
    RS = "rs_tuned" if "rs_tuned" in d["test"] else None
    need = list(dict.fromkeys(c for c in (P, BS, RS) if c is not None))
    r8, r8info = load_r8(cfg, seed, d)
    ctx = {"y_tr": y_tr, "y_te": y_te, "lo": lo, "hi": hi, "primary": P, "bstar": BS, "rs": RS,
           "need": need, "oof": d["oof"], "test": d["test"], "foldavg": d.get("test_foldavg", {}),
           "r8": r8, "ks": cfg["ks"], "lambdas": cfg["lambdas"]}
    out = {"lo": lo, "hi": hi, "n_tr": int(len(y_tr)), "n_te": int(len(y_te)), "r8": r8info,
           "centers": {"primary": P, "bstar": BS, "rs_tuned": RS}, "settings": {}}
    if cfg["near_dup"] is not None:
        mask, y_csv = np.asarray(cfg["near_dup"]), np.asarray(cfg["near_dup_y"])
        n = len(np.asarray(d["school_code"]))
        if len(mask) != n or not np.allclose(y_csv[d["idx_te"]], y_te):
            raise ValueError(f"lần chia {seed}: khoá trùng gần dựng từ --data không thẳng hàng npz E1")
        ctx["keep_tr"], ctx["keep_te"] = ~mask[d["idx_tr"]], ~mask[d["idx_te"]]
        out["near_dup"] = {"n_drop_tr": int((~ctx["keep_tr"]).sum()), "n_drop_te": int((~ctx["keep_te"]).sum())}
    if cfg["refit"]:
        ctx["refit"], out["refit"] = refit_predictions(d, seed, cfg)
    for st in cfg["settings"]:
        why = skip_reason(st, ctx)
        out["settings"][st["name"]] = {"skipped": why} if why else eval_setting(st, ctx)
    out["frontier"], out["stretch_edges"] = frontier_and_edges(ctx)
    b = out["settings"][BASELINE][f"step:{fmt_k(PRIMARY_K)}"]
    print(f"  split {seed}: K={PRIMARY_K} C1={b['C1']:+.3f} C2={b['C2']:+.3f} C3={b['C3']:+.3f} "
          f"R2-R1={b['R2-R1']:+.3f}; R8* {r8info['status']} ({time.time() - t0:.1f}s)", flush=True)
    return seed, out


# ---------------------------------------------------------------------------
# Tổng hợp
# ---------------------------------------------------------------------------
def _sign(m):
    return int(np.sign(m)) if np.isfinite(m) else None


def validity(st, cname, primary, refit_centers, has_rs):
    if cname == "C1":
        if st["family"] != "step":
            return False, "R8* chỉ có cho họ bậc thang (E2b)"
        if st["group"] == "tail":
            return False, "R8* học theo đuôi chính; E5 không huấn luyện lại theo định nghĩa đuôi này"
        if st["group"] == "refit" and primary in refit_centers:
            return False, "trung tâm chính được khớp lại theo hệ số số cây, R8* thì không"
    if cname == "C3" and not has_rs:
        return False, "npz E1 không có rs_tuned"
    return True, ""


def summarize(per, seeds, cfg):
    ents_all = [per[str(s)] for s in seeds if str(s) in per]
    J = len(ents_all)
    has_rs = all(e["centers"]["rs_tuned"] for e in ents_all)
    refit_centers = set()
    for e in ents_all:
        refit_centers |= set((e.get("refit") or {}).get("centers", []))
    S = {"n_splits": J, "settings": {}, "sign_keep": {}, "stretch_edges": {}, "frontier_interp": {},
         "gates": {}}
    for st in cfg["settings"]:
        ents = [e["settings"][st["name"]] for e in ents_all]
        meta = {k: st[k] for k in ("group", "family", "needs_refit")} | {
            "params": {k: st[k] for k in ("n_bins", "binning", "n_samples", "tail", "grid", "sigma",
                                          "floor", "test_src", "drop_near_dup", "tree_factor")}}
        skipped = [e["skipped"] for e in ents if "skipped" in e]
        if skipped:
            S["settings"][st["name"]] = meta | {"skipped": sorted(set(skipped))}
            continue
        con, costs = {}, {}
        for pk in [k for k in ents[0] if k.startswith(("step:", "prior:"))]:
            con[pk] = {}
            for cn in CONTRASTS:
                r = sp.nb_ttest([e[pk][cn] for e in ents])
                ok, note = validity(st, cn, cfg["primary"], refit_centers, has_rs)
                if r["n"] == 0:
                    ok, note = False, note or "không có số (thiếu npz E2b hoặc khoá R8*?)"
                con[pk][cn] = {"mean": r["mean"], "p": r["p"], "sign": _sign(r["mean"]), "wins": r["wins"],
                               "n": r["n"], "se": r["se"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                               "valid": bool(ok), "note": note}
            costs[pk] = {rule: ms([e[pk]["cost"][rule] for e in ents]) for rule in ents[0][pk]["cost"]}
        S["settings"][st["name"]] = meta | {
            "contrasts": con, "cost": costs,
            "cutoffs": {"lo": ms([e["lo"] for e in ents]), "hi": ms([e["hi"] for e in ents]),
                        "mass_low": ms([e["mass_low"] for e in ents]),
                        "mass_high": ms([e["mass_high"] for e in ents])}}

    # Cổng dấu: mỗi họ so với mốc của họ đó
    for fam, base in (("step", BASELINE), ("prior", PRIOR_BASELINE)):
        b = S["settings"].get(base, {})
        for pk, cons in (b.get("contrasts") or {}).items():
            for cn, e0 in cons.items():
                s0 = e0["sign"]
                rec = {"baseline": base, "baseline_mean": e0["mean"], "baseline_sign": s0}
                if not e0["valid"] or s0 is None:
                    S["sign_keep"].setdefault(pk, {})[cn] = rec | {"status": "mốc không hợp lệ: " + e0["note"]}
                    continue
                tallies = {}
                for scope in ("all", "posthoc_only"):
                    inc, keep, flips = [base], 1, []
                    for st in cfg["settings"]:
                        if st["family"] != fam or st["name"] == base:
                            continue
                        if scope == "posthoc_only" and st["needs_refit"]:
                            continue
                        e = ((S["settings"][st["name"]].get("contrasts") or {}).get(pk) or {}).get(cn)
                        if not e or not e["valid"] or e["sign"] is None:
                            continue
                        inc.append(st["name"])
                        if e["sign"] == s0:
                            keep += 1
                        else:
                            flips.append({"setting": st["name"], "mean": e["mean"], "p": e["p"]})
                    frac = keep / len(inc)
                    tallies[scope] = {"n_settings": len(inc), "n_keep": keep, "frac": frac,
                                      "frac_excl_baseline": ((keep - 1) / (len(inc) - 1)
                                                             if len(inc) > 1 else float("nan")),
                                      "unconditional": bool(frac >= E5_SIGN_KEEP), "flips": flips}
                S["sign_keep"].setdefault(pk, {})[cn] = rec | tallies

    # Giãn chạm biên lưới
    for g in ("new", "old"):
        S["stretch_edges"][g] = {}
        for label, Ks in (("K_GRID", K_GRID), ("K_DENSE", K_DENSE)):
            cells = [e["stretch_edges"][g][fmt_k(K)] for e in ents_all for K in Ks]
            n = len(cells)
            hit = sum(c["edge_low"] or c["edge_high"] for c in cells)
            S["stretch_edges"][g][label] = {
                "n_cells": n, "n_hit": int(hit), "frac_hit": hit / n if n else float("nan"),
                "n_hit_low": int(sum(c["edge_low"] for c in cells)),
                "n_hit_high": int(sum(c["edge_high"] for c in cells)),
                "K_hit": sorted({float(K) for e in ents_all for K in Ks
                                 if e["stretch_edges"][g][fmt_k(K)]["edge_low"]
                                 or e["stretch_edges"][g][fmt_k(K)]["edge_high"]})}

    # Sai số nội suy biên (mô tả)
    dkeys = [str(x) for x in MID_BUDGETS]
    fi = {"rules": {}, "pairs": {}, "old_Ks": [float(k) for k in OLD_FRONTIER_KS],
          "dense_Ks": [float(k) for k in K_DENSE]}
    for r in ("R1", "R2", "R5"):
        fi["rules"][r] = {}
        for dk in dkeys:
            dn = np.array([e["frontier"]["rules"][r]["dense"][dk] for e in ents_all], dtype=float)
            od = np.array([e["frontier"]["rules"][r]["old"][dk] for e in ents_all], dtype=float)
            err = od - dn
            fi["rules"][r][dk] = {"dense": ms(dn), "old": ms(od), "old_minus_dense": ms(err),
                                  "max_abs_err": float(np.nanmax(np.abs(err))) if np.isfinite(err).any()
                                  else float("nan")}
    for a, b in (("R1", "R5"), ("R2", "R1")):
        fi["pairs"][f"{a}-{b}"] = {}
        for dk in dkeys:
            out = {}
            for grid in ("dense", "old"):
                v = np.array([e["frontier"]["rules"][a][grid][dk] - e["frontier"]["rules"][b][grid][dk]
                              for e in ents_all], dtype=float)
                out[grid] = ms(v) | {"wins": int(np.sum(v < 0))}
            fi["pairs"][f"{a}-{b}"][dk] = out
    S["frontier_interp"] = fi

    # Cổng E5
    pk3 = f"step:{fmt_k(PRIMARY_K)}"
    g = S["gates"]
    g["unconditional"] = {pk: {cn: (v.get("all") or {}).get("unconditional") for cn, v in cons.items()}
                          for pk, cons in S["sign_keep"].items()}
    base_c2 = S["sign_keep"].get(pk3, {}).get("C2", {}).get("baseline_sign")
    flip = []
    for td in (("mass", 0.05, 0.05), ("fixed", 50, 110)):
        nm = tail_name(td)
        e = ((S["settings"].get(nm, {}).get("contrasts") or {}).get(pk3) or {}).get("C2")
        if e and base_c2 is not None and e["sign"] is not None and e["sign"] != base_c2:
            flip.append(nm)
    g["c2_tail_definition"] = {"flipped_in": flip, "limit_claim": bool(flip),
                               "rule": "5%/5% hoặc 50/110 đổi dấu C2 thì giới hạn tuyên bố vào "
                                       "'đuôi chiếm khoảng 10% đến 15% khối lượng'"}
    fo = S["stretch_edges"]["old"]["K_GRID"]["frac_hit"]
    g["stretch_old_grid"] = {"frac_hit_K_GRID": fo, "threshold": E5_EDGE_HIT_MAX,
                             "blocked": bool(np.isfinite(fo) and fo > E5_EDGE_HIT_MAX),
                             "frac_hit_K_DENSE": S["stretch_edges"]["old"]["K_DENSE"]["frac_hit"]}
    return S


def print_summary(S):
    pk3 = f"step:{fmt_k(PRIMARY_K)}"
    print(f"\n[E5] {S['n_splits']} lần chia; K = {PRIMARY_K} (âm = vế trái tốt hơn; * = không hợp lệ)")
    print(f"  {'thiết lập':30s}" + "".join(f"{c:>22s}" for c in CONTRASTS))
    for name, v in S["settings"].items():
        if "skipped" in v:
            print(f"  {name:30s} BỎ QUA: {v['skipped']}")
            continue
        pk = pk3 if pk3 in v["contrasts"] else sorted(v["contrasts"])[-1]
        cells = []
        for cn in CONTRASTS:
            e = v["contrasts"][pk][cn]
            cells.append(f"{e['mean']:+8.3f} (p {e['p']:.2g}){'' if e['valid'] else '*'}")
        print(f"  {name:30s}" + "".join(f"{c:>22s}" for c in cells) + ("" if pk == pk3 else f"  [{pk}]"))
    print("  Dấu giữ (mọi thiết lập hợp lệ, có mốc):")
    for pk, cons in S["sign_keep"].items():
        print(f"    {pk:10s} " + "  ".join(
            f"{cn}: {v['all']['n_keep']}/{v['all']['n_settings']}" if "all" in v else f"{cn}: -"
            for cn, v in cons.items()))
    se = S["stretch_edges"]
    print(f"  Giãn chạm biên (K_GRID): lưới cũ {se['old']['K_GRID']['n_hit']}/{se['old']['K_GRID']['n_cells']}, "
          f"lưới mới {se['new']['K_GRID']['n_hit']}/{se['new']['K_GRID']['n_cells']}")
    g = S["gates"]
    print(f"  Cổng: C2 đổi dấu theo đuôi ở {g['c2_tail_definition']['flipped_in'] or 'không'}; "
          f"lưới giãn cũ bị chặn: {g['stretch_old_grid']['blocked']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="E5: độ nhạy của tầng quyết định (hậu kỳ, --refit tuỳ chọn)")
    add_common_args(ap, "sensitivity")
    ap.add_argument("--e2b-dir", default=os.path.join("preds", "wtrain"), help="npz của E2b (R8, R8_bag5)")
    ap.add_argument("--e2b-tag", default=None)
    ap.add_argument("--r8-key", default="auto",
                    help='khoá R8* trong d["test"] của npz E2b, mẫu format với {r8} và {K}, '
                         'ví dụ "{r8}_K{K}"; "auto" tự tìm')
    ap.add_argument("--ks", type=float, nargs="+", default=K_GRID)
    ap.add_argument("--lambdas", type=float, nargs="+", default=PRIOR_LAMBDAS)
    ap.add_argument("--refit", action="store_true",
                    help="bật mục cần khớp lại mô hình (hệ số số cây 1,0/1,2); cần --data")
    args = ap.parse_args(argv)
    seeds = args.seeds[:1] if args.smoke else args.seeds
    ks = [int(k) if float(k).is_integer() else float(k) for k in args.ks]
    if PRIMARY_K not in ks:
        ks = sorted(set(ks) | {PRIMARY_K})
    provenance.print_versions()
    guard_real_data(args.data)

    first = load_e1(args.e1_dir, seeds[0], args.e1_tags)
    primary, bstar, csrc = resolve_centers(args.primary, args.bstar, args.centers_json, list(first["test"]))
    r8s = r8_star(primary)
    data_ok = os.path.exists(args.data)
    if args.refit and not data_ok:
        sys.exit(f"--refit cần dữ liệu: không thấy {args.data}")
    if data_ok:
        nd_mask, nd_y, nd_info = near_dup_rows(args.data)
        data_sha = preds_io.file_sha256(args.data)
    else:
        nd_mask, nd_y, nd_info, data_sha = None, None, {"status": f"không thấy {args.data}"}, "missing"
        print(f"[E5] CẢNH BÁO: không thấy {args.data}; bỏ qua thiết lập drop_near_dup", flush=True)
    settings = build_settings(args.refit)
    threads = splits.xgb_threads(args.workers)
    if args.smoke:
        threads = min(2, threads)
    cfg = {"script": "sensitivity", "e1_dir": args.e1_dir, "e1_tags": parse_tags(args.e1_tags),
           "e2b_dir": args.e2b_dir, "e2b_tag": args.e2b_tag, "r8_key": args.r8_key,
           "primary": primary, "bstar": bstar, "r8_star": r8s, "ks": ks, "lambdas": list(args.lambdas),
           "settings": settings, "refit": bool(args.refit), "tree_factors": [float(f) for f in E5_TREE_FACTOR],
           "smoke": bool(args.smoke), "data": args.data, "data_sha256": data_sha,
           "preds_dir": args.preds_dir, "threads": threads,
           "near_dup": nd_mask, "near_dup_y": nd_y}
    fp_cfg = {k: v for k, v in cfg.items()
              if k not in ("e1_dir", "e2b_dir", "data", "preds_dir", "threads", "near_dup", "near_dup_y")}
    fp = preds_io.fingerprint(fp_cfg)
    print(f"E5: {len(seeds)} lần chia, {len(settings)} thiết lập ({'có' if args.refit else 'không'} "
          f"khớp lại); chính = {primary}, bag B* = {bstar}, R8* = {r8s} ({csrc}); "
          f"trùng gần: {nd_info}; dấu {fp[:12]}", flush=True)

    partial = args.out + ".partial"
    res = preds_io.load_partial(partial, fp, {"per_split": {}}, on_mismatch=args.on_mismatch)
    inputs_of = {s: {"e1": input_shas(e1_paths(args.e1_dir, s, args.e1_tags)),
                     "e2b": input_shas([preds_io.split_path(args.e2b_dir, s, args.e2b_tag)])}
                 for s in seeds}
    cfg["inputs_of"] = inputs_of
    drive_splits(run_split, seeds, cfg, args.workers, res, partial, inputs_of)

    S = summarize(res["per_split"], seeds, cfg)
    print_summary(S)
    res["summary"] = S
    res["meta"] |= {
        "experiment": "E5", "seeds": seeds, "primary": primary, "bstar": bstar, "r8_star": r8s,
        "centers_source": csrc, "ks": ks, "lambdas": list(args.lambdas), "primary_k": PRIMARY_K,
        "settings": settings, "refit": bool(args.refit), "smoke": bool(args.smoke),
        "near_dup": nd_info, "data_sha256": data_sha, "contrasts": {
            "C1": "cost(R1) - cost(R8*) trên trung tâm chính", "C2": "cost(R1) - cost(R5) trên trung tâm chính",
            "C3": "cost(R1 trên rs_tuned) - cost(R1 trên bag B*)", "R2-R1": "cost(R2) - cost(R1) trên trung tâm chính"},
        "sign_keep_threshold": E5_SIGN_KEEP, "mid_budgets": list(MID_BUDGETS),
        "provenance": provenance.stamp()}
    preds_io.dump_json_atomic(res, args.out)
    print(f"\nĐã ghi {args.out}")
    return res


if __name__ == "__main__":
    main()
