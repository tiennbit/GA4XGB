# -*- coding: utf-8 -*-
"""E3: điểm hợp thức cho trung tâm và tầng quyết định (khung bài 24/9, mục 6.7).

Vì sao cần E3: cost_K ở K cố định nhất quán cho μ_K (Bổ đề 1), nên xếp hạng theo nó
đúng với MỤC TIÊU CỦA NÓ, nhưng không tách được hai nguồn lợi: trung tâm dự báo tốt
hơn (lợi ích dự báo), hay quy tắc chỉ dời dự đoán về phía chi phí đòi (lợi ích do chi
phí). Đó là forecaster's dilemma (Lerch et al. 2017). Điểm hợp thức cho TRUNG BÌNH
không có kẽ hở đó, nên E3 dùng chúng để tách hai nguồn (Nhận xét 5):

  - Điểm sơ cấp (Ehm et al. 2016): S_θ(x, y) = |1{y < θ} - 1{x < θ}|·|y - θ|.
    Mọi hàm điểm nhất quán cho trung bình là hỗn hợp không âm của S_θ (với sai số
    bình phương: (x - y)² = 2∫S_θ dθ, kiểm trong tests/test_post_hoc.py). Trung tâm
    A trội B ở mọi θ nghĩa là A tốt hơn dưới MỌI hàm điểm nhất quán cho trung bình,
    không riêng RMSE. Biểu đồ Murphy: trung bình S_θ trên tập kiểm tra, θ = 30..130
    (gates.E3_THETA).
  - Điểm nhấn đuôi kiểu Taggart (2022): trung bình S_θ trên θ nguyên thuộc
    [29; lo) ∪ [hi; 129]. Vẫn là hỗn hợp không âm nên vẫn nhất quán cho trung bình:
    dự đoán nghiêng theo K KHÔNG được thắng R1₁ một cách hệ thống. Nếu R1 ở K > 1
    thắng R1₁ ở >= 9/10 lần chia thì cài đặt có lỗi (cổng E3).
    29 và 129 là y nhỏ nhất và lớn nhất của khoá (mục 5.1), cố định cho mọi lần chia
    để điểm Taggart so được giữa các lần chia; đổi bằng --taggart-range (E11).
  - Với R1: CRPS và twCRPS (Gneiting và Ranjan 2011; trọng số u(z) = 1 ở đuôi) của
    phân phối S điểm cách đều xác suất. twCRPS tính qua hàm xích v(z) = ∫u (Allen,
    Ginsbourger và Ziegel 2023): twCRPS(F, y) = CRPS(F∘v⁻¹, v(y)), nên dùng lại đúng
    công thức nhân của CRPS trên v(điểm mẫu). Kiểm bằng tích phân số trong test.
    Thứ cấp: CRPS của phân phối R1 NGHIÊNG theo K (xác suất ∝ w_K(s_j)), trung bình
    của nó chính là g_K. CRPS là hợp thức nên nghiêng không được làm tốt hơn; con số
    cho thấy cái giá của việc nghiêng trên thang phân phối.

Chọn lựa khi khung bài để ngỏ (ghi lại theo yêu cầu):
  - Đầu vào chỉ là npz của E1. R1, R1₁ khớp lại tại chỗ bằng decision_layer (rẻ,
    tất định), không cần npz của E2 (E2 chỉ ghi JSON).
  - Biểu đồ Murphy vẽ cho MỌI trung tâm có trong npz (R0) và R1₁ của từng trung tâm,
    cộng R1 ở K ∈ K_GRID trên trung tâm chính. Rẻ, và phép so "trung tâm còn lợi
    sau khi hiệu chỉnh lại không" cần R1₁ của cả hai phía.
  - "Trội ở θ" dùng bất đẳng thức KHÔNG chặt (≤): ở θ ngoài miền của mọi dự đoán hai
    trung tâm bằng nhau đúng từng bit, và trội theo nghĩa Murphy là ≤ ở mọi θ. Tỉ lệ
    theo < chặt và số θ hoà vẫn được ghi.
    CỔNG tính tỉ lệ trên các θ CÓ THÔNG TIN (không hoà ở mọi lần chia), không trên cả
    101 θ. Ở θ mà mọi dự đoán của cả hai trung tâm cùng một phía, S_θ chỉ còn phụ thuộc
    y nên hai trung tâm hoà đúng từng bit; với ŷ trong khoảng 45 đến 110 đó là gần một
    phần ba lưới 30..130. Đếm các θ đó là "trội" thì cổng 90% gần như tự qua. Tỉ lệ
    trên cả lưới vẫn được ghi (frac_theta_weak) để đối chiếu.
  - "Ít nhất 9/10 lần chia" co theo J: ceil(0,9·J) (J = 1 khi --smoke).
  - Bootstrap cụm theo trường (quy ước chung mục 6) cho chênh điểm Taggart của từng
    cặp: lấy lại trường có hoàn lại, thống kê là trung bình chênh từng em.
  - tail_prior.dist_metrics (pinball, lệch phủ, PI90, Brier và AUC đuôi) chỉ báo khi
    S = 50 vì hàm đó cố định 50 mức τ và ngưỡng 60/100 của bins.py (trên khoá này
    trùng đuôi theo khối lượng).
  - Trung tâm chính và bag B*: --primary/--bstar; không có thì đọc cổng G1 trong
    results_cost/decomp_centers.json ở summary.G1 (primary_center, bag_Bstar), đúng
    chỗ E1 ghi và decomp_rules (E2) đọc; các vị trí cũ gates.G1 vẫn được thử sau.
    Không có nữa thì bag10 kèm cảnh báo. Nguồn được ghi vào meta.centers_source.
    (Bản trước chỉ tìm gates.G1, nơi E1 không ghi gì, nên luôn rơi về bag10.)
  - Cổng chỉ có giá trị kết luận khi đủ lần chia: summary.complete = (J == số seed
    mong đợi: 10 của gates.SEEDS, hoặc số seed của --smoke).
  - --preds-dir: E3 không ghi dự đoán từng bản ghi nào; cờ giữ cho giao diện chung.
  - --data: không đọc (E3 chỉ cần npz); giữ cho giao diện chung.

File này cũng chứa phần dùng chung với sensitivity.py (E5): nạp npz E1 có gộp nhiều
tag, chọn trung tâm, và vòng chạy theo lần chia có tiếp tục (.partial).

Chạy (server): PYTHONPATH=src .venv/bin/python src/proper_scores.py --workers 5
Kiểm thử khói (Mac, dữ liệu giả): xem tests/test_post_hoc.py.
Đầu ra: results_cost/proper_scores.json.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

import decision_layer as dl
import preds_io
import provenance
import stats_paired as sp
from calibration_table import calib_stats
from gates import (ALPHA, CLUSTER_BOOT_B, E3_DOMINANCE, E3_SPLITS_MIN, E3_THETA, K_GRID,
                   N_BINS, N_SAMPLES, PRIMARY_K, SEEDS)
from preprocess import DATA_PATH
from tail_prior import dist_metrics

# y nhỏ nhất và lớn nhất của khoá HSA 2024 (mục 5.1): miền θ của điểm Taggart.
TAGGART_Y_RANGE = (29, 129)
TIE_TOL = 1e-12
SMOKE_BOOT_B = 200
DEFAULT_CENTERS_JSON = os.path.join("results_cost", "decomp_centers.json")


# ---------------------------------------------------------------------------
# Hàm điểm
# ---------------------------------------------------------------------------
def elementary_scores(x, y, thetas):
    """Ma trận n×T: S_θ(x_i, y_i) = |1{y < θ} - 1{x < θ}|·|y - θ| (Ehm et al. 2016)."""
    x = np.asarray(x, dtype=float)[:, None]
    y = np.asarray(y, dtype=float)[:, None]
    t = np.asarray(thetas, dtype=float)[None, :]
    return np.abs((y < t).astype(float) - (x < t).astype(float)) * np.abs(y - t)


def taggart_thetas(lo, hi, y_range=TAGGART_Y_RANGE):
    """θ nguyên của điểm Taggart: (thấp, cao) = [ymin; lo) và [hi; ymax]."""
    t = np.arange(int(y_range[0]), int(y_range[1]) + 1, dtype=float)
    return t[t < lo], t[t >= hi]


def crps_points(X, y, P=None):
    """CRPS của phân phối rời rạc: điểm X (n×S), xác suất P (n×S, tổng 1; None = đều).

    CRPS = Σ p_j|x_j - y| - ½ΣΣ p_j p_k |x_j - x_k|. Sau khi sắp x tăng dần, nửa tổng
    đôi bằng Σ_k p_k (x_k·W_k - M_k), W_k = Σ_{j<k} p_j, M_k = Σ_{j<k} p_j x_j: O(S)
    mỗi dòng thay vì O(S²). Đúng bằng tích phân ∫(F(z) - 1{y <= z})² dz của hàm phân
    phối bậc thang."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, S = X.shape
    P = np.full((n, S), 1.0 / S) if P is None else np.asarray(P, dtype=float)
    o = np.argsort(X, axis=1, kind="stable")
    X = np.take_along_axis(X, o, axis=1)
    P = np.take_along_axis(P, o, axis=1)
    t1 = np.sum(P * np.abs(X - y[:, None]), axis=1)
    W = np.cumsum(P, axis=1) - P
    M = np.cumsum(P * X, axis=1) - P * X
    return t1 - np.sum(P * (X * W - M), axis=1)


def chain(kind, lo, hi):
    """Hàm xích v(z) = ∫u của trọng số ngưỡng u: đuôi gộp, chỉ đuôi thấp, chỉ đuôi cao."""
    if kind == "tails":
        return lambda z: np.minimum(z, lo) + np.maximum(np.asarray(z, dtype=float) - hi, 0.0)
    if kind == "low":
        return lambda z: np.minimum(z, lo)
    if kind == "high":
        return lambda z: np.maximum(z, hi)
    raise ValueError(kind)


def dist_scores(y, X, lo, hi, P=None):
    """CRPS và twCRPS (đuôi gộp, thấp, cao) trung bình trên tập kiểm tra."""
    out = {"crps": float(np.mean(crps_points(X, y, P)))}
    for kind in ("tails", "low", "high"):
        v = chain(kind, lo, hi)
        key = "twcrps" if kind == "tails" else f"twcrps_{kind}"
        out[key] = float(np.mean(crps_points(v(X), v(np.asarray(y, dtype=float)), P)))
    return out


def cluster_boot_mean_diff(d, clusters, B=CLUSTER_BOOT_B, seed=0, alpha=ALPHA):
    """CI phân vị của trung bình chênh từng em d_i, lấy lại TRƯỜNG có hoàn lại.

    stats_paired.cluster_boot_diff chỉ nhận cost_K (căn của trung bình có trọng số);
    điểm Taggart là trung bình thường nên cần bản này. Cùng cách tính theo tổng cụm:
    số lần bốc (B×G) nhân tổng (G×2)."""
    d = np.asarray(d, dtype=float)
    _, cl = np.unique(np.asarray(clusters), return_inverse=True)
    G = int(cl.max()) + 1
    S = np.column_stack([np.bincount(cl, weights=d, minlength=G),
                         np.bincount(cl, minlength=G).astype(float)])
    rng = np.random.default_rng(seed)
    tot = rng.multinomial(G, np.full(G, 1.0 / G), size=int(B)) @ S
    boot = tot[:, 0] / tot[:, 1]
    lo_ci, hi_ci = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"est": float(d.mean()), "ci_lo": float(lo_ci), "ci_hi": float(hi_ci),
            "se": float(boot.std(ddof=1)), "excludes_zero": bool(lo_ci > 0 or hi_ci < 0),
            "B": int(B), "n_clusters": G}


def ms(vals):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "n": int(len(v))}


def fmt_k(K):
    return f"{float(K):g}"


# ---------------------------------------------------------------------------
# Phần dùng chung với sensitivity.py: nạp npz, chọn trung tâm, vòng lần chia
# ---------------------------------------------------------------------------
def parse_tags(tags):
    """--e1-tags: danh sách rỗng hoặc 'none' nghĩa là split<seed>.npz không tag."""
    tags = list(tags or [])
    return [None if (t is None or str(t).lower() == "none") else str(t) for t in tags] or [None]


def e1_paths(e1_dir, seed, tags):
    return [preds_io.split_path(e1_dir, seed, t) for t in parse_tags(tags)]


MERGE_DICTS = ("oof", "test", "test_foldavg")
TRACE_KEYS = ("trace_oof", "trace_test", "trace_test_foldavg", "trace_best_iter",
              "trace_n_trees", "trace_oof_rmse")


def load_e1(e1_dir, seed, tags):
    """npz E1 của một lần chia; nhiều tag thì GỘP (E1 có thể tách giai đoạn 1 và 2
    ra hai file). Các file phải cùng lần chia (idx, y giống hệt); một trung tâm có ở
    hai file thì hai mảng phải bằng nhau, nếu không báo lỗi thay vì chọn bừa."""
    paths = e1_paths(e1_dir, seed, tags)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"thiếu npz E1: {missing}")
    ds = [preds_io.load_split(p) for p in paths]
    out = ds[0]
    out.setdefault("meta", {})
    for p, d in zip(paths[1:], ds[1:]):
        for k in ("idx_tr", "idx_te", "y_tr", "y_te"):
            if not np.array_equal(np.asarray(out[k]), np.asarray(d[k])):
                raise ValueError(f"{p}: '{k}' khác {paths[0]}; không cùng lần chia")
        for part in MERGE_DICTS:
            for c, a in d.get(part, {}).items():
                have = out.setdefault(part, {})
                if c in have and not np.array_equal(have[c], a):
                    raise ValueError(f"trung tâm '{c}' ({part}) có ở hai file E1 với giá trị khác nhau")
                have[c] = a
        for k in TRACE_KEYS:
            if k in d and k not in out:
                out[k] = d[k]
        for k, v in (d.get("meta") or {}).items():
            out["meta"].setdefault(k, v)
    err = preds_io.validate_split(out)
    if err:
        raise ValueError(f"npz E1 lần chia {seed} không hợp lệ: {err}")
    return out


def input_shas(paths):
    """{tên file: sha256} của các npz đầu vào (thiếu file: 'missing')."""
    return {os.path.basename(p): (preds_io.file_sha256(p) if os.path.exists(p) else "missing")
            for p in paths}


def _g1_from_json(path):
    """(primary, bstar, nguồn) từ cổng G1 của E1, hoặc (None, None, None)."""
    obj = preds_io.load_json(path) if path and os.path.exists(path) else None
    if not isinstance(obj, dict):
        return None, None, None
    summ = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
    # summary.G1 là nơi decomp_centers (E1) ghi cổng; hai vị trí sau giữ cho JSON cũ
    for label, where in (("summary.G1", summ), ("gates.G1", obj.get("gates")),
                         ("summary.gates.G1", summ.get("gates"))):
        g1 = where.get("G1") if isinstance(where, dict) else None
        if not isinstance(g1, dict):
            continue
        prim = g1.get("primary_center") or g1.get("primary")
        bst = g1.get("bag_Bstar") or g1.get("b_star_center") or g1.get("bstar_center")
        if bst is None and isinstance(g1.get("B_star"), int):
            bst = f"bag{int(g1['B_star'])}"
        if prim or bst:
            return prim, bst, f"{path}: {label}"
    return None, None, None


def resolve_centers(primary, bstar, centers_json, available):
    """(primary, bstar, nguồn): CLI > cổng G1 trong decomp_centers.json > bag10."""
    src = []
    if primary is not None:
        src.append("primary: --primary")
    if bstar is not None:
        src.append("bstar: --bstar")
    if primary is None or bstar is None:
        jp, jb, jsrc = _g1_from_json(centers_json)
        if primary is None and jp is not None:
            primary = jp
            src.append(f"primary: {jsrc}")
        if bstar is None and jb is not None:
            bstar = jb
            src.append(f"bstar: {jsrc}")
    for name, val in (("primary", primary), ("bstar", bstar)):
        if val is None:
            fb = "bag_Bstar" if (name == "bstar" and "bag_Bstar" in available) else "bag10"
            print(f"[post_hoc] CẢNH BÁO: chưa có cổng G1 cho {name}; dùng '{fb}'. "
                  "Đặt --primary/--bstar khi E1 đã chốt.", file=sys.stderr, flush=True)
            if name == "primary":
                primary = fb
            else:
                bstar = fb
            src.append(f"{name}: mặc định {fb} (chưa có G1)")
    for name, val in (("primary", primary), ("bstar", bstar)):
        if val not in available:
            raise ValueError(f"trung tâm {name} = '{val}' không có trong npz E1; có {sorted(available)}")
    return primary, bstar, "; ".join(src)


def drive_splits(worker, seeds, cfg, workers, res, partial_path, inputs_of):
    """Chạy worker(seed, cfg) -> (seed, dict) cho các lần chia chưa xong, ghi .partial
    sau MỖI lần chia (job bị ngắt không mất phần đã xong).

    Lần chia đã có trong .partial được bỏ qua chỉ khi sha256 npz đầu vào còn khớp; E1
    hay E2b chạy lại thì lần chia đó được tính lại (hậu kỳ rẻ) thay vì trộn số của hai
    bản dự đoán."""
    todo = []
    for s in seeds:
        e = res["per_split"].get(str(s))
        if e is not None and e.get("inputs") == inputs_of[s]:
            print(f"  split {s}: đã có trong .partial, cùng npz -> bỏ qua", flush=True)
            continue
        if e is not None:
            print(f"  split {s}: npz đầu vào đã đổi -> tính lại", flush=True)
        todo.append(s)
    if not todo:
        return
    nw = max(1, min(int(workers), len(todo)))
    if nw == 1:
        gen = (worker(s, cfg) for s in todo)
    else:
        gen = Parallel(n_jobs=nw, return_as="generator_unordered")(delayed(worker)(s, cfg) for s in todo)
    for s, r in gen:
        r["inputs"] = inputs_of[s]
        res["per_split"][str(s)] = r
        preds_io.dump_json_atomic(res, partial_path)


def add_common_args(ap, exp):
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default=os.path.join("results_cost", f"{exp}.json"))
    ap.add_argument("--preds-dir", default=os.path.join("preds", exp))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1 lần chia, ngân sách nhỏ")
    ap.add_argument("--workers", type=int, default=2, help="số tiến trình joblib (theo lần chia)")
    ap.add_argument("--e1-dir", default=os.path.join("preds", "decomp"), help="npz của E1")
    ap.add_argument("--e1-tags", nargs="*", default=[],
                    help="tag của npz E1 (split<seed>_<tag>.npz); nhiều tag thì gộp; rỗng = không tag")
    ap.add_argument("--primary", default=None, help="tên trung tâm chính trong npz (cổng G1)")
    ap.add_argument("--bstar", default=None, help="tên trung tâm bag B* trong npz (cổng G1)")
    ap.add_argument("--centers-json", default=DEFAULT_CENTERS_JSON,
                    help="JSON của E1 để đọc cổng G1 khi không có --primary/--bstar")
    ap.add_argument("--on-mismatch", default="raise", choices=["raise", "recompute"],
                    help=".partial khác dấu lượt chạy: báo lỗi (mặc định) hay tính lại từ đầu")


# ---------------------------------------------------------------------------
# Một lần chia
# ---------------------------------------------------------------------------
def forecast_pairs(centers, primary, bstar, ks):
    """Các cặp (A, B) cho điểm Taggart và độ trội Murphy; âm = A tốt hơn."""
    P = primary
    pairs = [(f"{P}|R0", f"{c}|R0") for c in centers if c != P]
    if "default" in centers:
        if bstar not in (P, "default"):
            pairs.append((f"{bstar}|R0", "default|R0"))
        if "sub1" in centers:
            pairs.append(("sub1|R0", "default|R0"))
        if P != "default":
            pairs.append((f"{P}|R1_1", "default|R1_1"))
    if bstar != P:
        pairs.append((f"{P}|R1_1", f"{bstar}|R1_1"))
    pairs.append((f"{P}|R1_1", f"{P}|R0"))
    pairs += [(f"{P}|R1@K{fmt_k(K)}", f"{P}|R1_1") for K in ks]
    out = []
    for p in pairs:
        if p not in out:
            out.append(p)
    return out


def run_split(seed, cfg):
    t0 = time.time()
    d = load_e1(cfg["e1_dir"], seed, cfg["e1_tags"])
    y_tr = np.asarray(d["y_tr"], dtype=float)
    y_te = np.asarray(d["y_te"], dtype=float)
    lo, hi = dl.tail_cutoffs(y_tr)
    school = preds_io.rows(d, "te", "school_code")
    centers = [c for c in (cfg["centers"] or list(d["test"])) if c in d["test"]]
    P = cfg["primary"]
    thetas = np.asarray(cfg["theta"], dtype=float)
    t_low, t_high = taggart_thetas(lo, hi, cfg["taggart_range"])
    t_all = np.concatenate([t_low, t_high])

    fc, r1 = {}, {}
    for c in centers:
        fc[f"{c}|R0"] = np.asarray(d["test"][c], dtype=float)
        r1[c] = dl.ResidualBinBayes(cfg["n_bins"], cfg["n_samples"]).fit(d["oof"][c], y_tr)
        fc[f"{c}|R1_1"] = r1[c].predict(d["test"][c])          # w ≡ 1: trung bình S điểm
    for K in cfg["ks"]:
        fc[f"{P}|R1@K{fmt_k(K)}"] = r1[P].predict(d["test"][P], dl.step(K, K, lo, hi))

    out = {"lo": lo, "hi": hi, "n_te": int(len(y_te)), "n_clusters": int(len(np.unique(school))),
           "taggart_theta": {"low": [float(t_low[0]), float(t_low[-1])] if len(t_low) else [],
                             "high": [float(t_high[0]), float(t_high[-1])] if len(t_high) else []},
           "forecasts": {}, "dist": {}, "boot": {}}
    row_tag = {}
    for name, x in fc.items():
        St = elementary_scores(x, y_te, t_all)
        row_tag[name] = St.mean(axis=1)
        nl = len(t_low)
        out["forecasts"][name] = {
            "murphy": elementary_scores(x, y_te, thetas).mean(axis=0),
            "taggart": float(row_tag[name].mean()),
            "taggart_low": float(St[:, :nl].mean()) if nl else float("nan"),
            "taggart_high": float(St[:, nl:].mean()) if St.shape[1] > nl else float("nan"),
            "rmse": float(np.sqrt(np.mean((y_te - x) ** 2))),
            "calib": calib_stats(y_te, x)}
    for i, (a, b) in enumerate(cfg["pairs"]):
        if a in row_tag and b in row_tag:
            out["boot"][f"{a} - {b}"] = cluster_boot_mean_diff(
                row_tag[a] - row_tag[b], school, B=cfg["boot_B"], seed=1000 * int(seed) + i)

    # Phân phối S điểm của R1 (không phụ thuộc K) trên từng trung tâm, và bản nghiêng
    # theo K trên trung tâm chính.
    for c in centers:
        X = r1[c].samples(d["test"][c])
        e = dist_scores(y_te, X, lo, hi)
        if X.shape[1] == 50:
            e |= dist_metrics(y_te, X)
        out["dist"][c] = e
    Xp = r1[P].samples(d["test"][P])
    for K in cfg["ks"]:
        w = dl.step(K, K, lo, hi)(Xp)
        out["dist"][f"{P}@K{fmt_k(K)}"] = dist_scores(y_te, Xp, lo, hi, w / w.sum(axis=1, keepdims=True))
    f0 = out["forecasts"]
    print(f"  split {seed}: Taggart {P}|R0 = {f0[f'{P}|R0']['taggart']:.4f}, "
          f"CRPS R1 = {out['dist'][P]['crps']:.4f}; cutoffs {lo:g}/{hi:g} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return seed, out


# ---------------------------------------------------------------------------
# Tổng hợp qua các lần chia
# ---------------------------------------------------------------------------
def summarize(per, seeds, cfg, n_expected=len(SEEDS)):
    ents = [per[str(s)] for s in seeds if str(s) in per]
    J = len(ents)
    min_splits = max(1, math.ceil(E3_SPLITS_MIN / 10 * J))
    names = [f for f in ents[0]["forecasts"] if all(f in e["forecasts"] for e in ents)]
    S = {"n_splits": J, "n_expected": int(n_expected), "complete": bool(J == n_expected),
         "min_splits": min_splits, "murphy_mean": {}, "scores": {}, "pairs": {},
         "dist": {}, "dist_pairs": {}, "gates": {}}
    for f in names:
        S["murphy_mean"][f] = np.mean([e["forecasts"][f]["murphy"] for e in ents], axis=0).tolist()
        S["scores"][f] = {m: ms([e["forecasts"][f][m] for e in ents])
                          for m in ("taggart", "taggart_low", "taggart_high", "rmse")}
        S["scores"][f]["calib_slope"] = ms([e["forecasts"][f]["calib"]["calib_slope"] for e in ents])
        S["scores"][f]["sd_ratio"] = ms([e["forecasts"][f]["calib"]["sd_ratio"] for e in ents])

    keys = []
    for a, b in cfg["pairs"]:
        if a not in names or b not in names:
            continue
        key = f"{a} - {b}"
        keys.append(key)
        Ma = np.array([e["forecasts"][a]["murphy"] for e in ents], dtype=float)
        Mb = np.array([e["forecasts"][b]["murphy"] for e in ents], dtype=float)
        weak = ((Ma <= Mb + TIE_TOL).sum(axis=0) >= min_splits)
        strict = ((Ma < Mb - TIE_TOL).sum(axis=0) >= min_splits)
        tied = np.all(np.abs(Ma - Mb) <= TIE_TOL, axis=0)
        info = ~tied                    # θ có thông tin: không hoà ở ít nhất một lần chia
        boots = [e["boot"].get(key) for e in ents if e["boot"].get(key)]
        S["pairs"][key] = {
            "taggart": sp.nb_ttest([e["forecasts"][a]["taggart"] - e["forecasts"][b]["taggart"]
                                    for e in ents]),
            "taggart_low": ms([e["forecasts"][a]["taggart_low"] - e["forecasts"][b]["taggart_low"]
                               for e in ents]),
            "taggart_high": ms([e["forecasts"][a]["taggart_high"] - e["forecasts"][b]["taggart_high"]
                                for e in ents]),
            "dominance": {"frac_theta_weak": float(weak.mean()), "frac_theta_strict": float(strict.mean()),
                          "frac_theta_informative": (float(weak[info].mean()) if info.any()
                                                     else float("nan")),
                          "n_theta_informative": int(info.sum()),
                          "n_theta_tied": int(tied.sum()), "n_theta": int(len(weak)),
                          "theta_not_dominated": [float(t) for t, ok in zip(cfg["theta"], weak) if not ok]},
            "boot_excludes_zero": {"n": int(sum(b_["excludes_zero"] for b_ in boots)),
                                   "n_favor_left": int(sum(b_["ci_hi"] < 0 for b_ in boots)),
                                   "of": len(boots)}}
    adj = sp.holm([S["pairs"][k]["taggart"]["p"] for k in keys])
    for k, p in zip(keys, adj):
        S["pairs"][k]["taggart"]["p_holm"] = p

    dnames = [k for k in ents[0]["dist"] if all(k in e["dist"] for e in ents)]
    for k in dnames:
        S["dist"][k] = {m: ms([e["dist"][k][m] for e in ents]) for m in ents[0]["dist"][k]}
    P, bst = cfg["primary"], cfg["bstar"]
    dpairs = [(P, c) for c in ("default", bst) if c != P and c in dnames]
    dpairs += [(f"{P}@K{fmt_k(K)}", P) for K in cfg["ks"] if f"{P}@K{fmt_k(K)}" in dnames]
    for a, b in dpairs:
        S["dist_pairs"][f"{a} - {b}"] = {
            m: sp.nb_ttest([e["dist"][a][m] - e["dist"][b][m] for e in ents])
            for m in ("crps", "twcrps", "twcrps_low", "twcrps_high")}

    # Cổng E3
    g = S["gates"]
    key = f"{P}|R0 - default|R0"
    if key in S["pairs"]:
        dom = S["pairs"][key]["dominance"]
        frac = dom["frac_theta_informative"]
        g["center_forecast_gain"] = {
            "frac_theta": frac, "frac_theta_all_grid": dom["frac_theta_weak"],
            "n_theta_informative": dom["n_theta_informative"], "threshold": E3_DOMINANCE,
            "passed": bool(np.isfinite(frac) and frac >= E3_DOMINANCE),
            "rule": "trung tâm chính trội default (<=) ở >= 90% θ CÓ THÔNG TIN (bỏ θ hoà ở mọi lần "
                    "chia), mỗi θ trong >= ceil(0,9·J) lần chia; không qua thì chỉ báo lợi ích "
                    "trung tâm bằng cost_K"}
    tilt = {}
    for K in cfg["ks"]:
        k2 = f"{P}|R1@K{fmt_k(K)} - {P}|R1_1"
        if float(K) > 1 and k2 in S["pairs"]:
            w = S["pairs"][k2]["taggart"]["wins"]
            tilt[fmt_k(K)] = {"wins": w, "alarm": bool(w >= min_splits)}
    g["tilt_check"] = {"per_K": tilt, "alarm": bool(any(v["alarm"] for v in tilt.values())),
                       "rule": "R1 ở K > 1 thắng R1₁ trên điểm Taggart ở >= ceil(0,9·J) lần chia "
                               "thì kiểm lại cài đặt (điểm nhất quán cho trung bình)"}
    g["complete"] = S["complete"]
    return S


def print_summary(S, cfg):
    P = cfg["primary"]
    print(f"\n[E3] {S['n_splits']}/{S['n_expected']} lần chia ({'đủ' if S['complete'] else 'CHƯA đủ, cổng tạm'}); "
          f"trội khi đúng ở >= {S['min_splits']} lần chia")
    print("  cặp (âm = trái tốt hơn)                              Taggart     p_NB    p_Holm  trội θ")
    for k, v in S["pairs"].items():
        t = v["taggart"]
        print(f"  {k:52s} {t['mean']:+.4f}  {t['p']:.3g}  {t.get('p_holm', float('nan')):.3g}  "
              f"{v['dominance']['frac_theta_weak']:.2f}")
    print("  CRPS / twCRPS của R1 (trung bình qua lần chia):")
    for k, v in S["dist"].items():
        print(f"    {k:22s} CRPS={v['crps']['mean']:.4f} twCRPS={v['twcrps']['mean']:.4f}")
    g = S["gates"]
    if "center_forecast_gain" in g:
        c = g["center_forecast_gain"]
        print(f"  Cổng: {P} trội default ở {c['frac_theta']:.0%} θ -> "
              f"{'gọi được là lợi ích dự báo' if c['passed'] else 'chỉ báo bằng cost_K'}")
    print(f"  Cổng nghiêng: {'CẢNH BÁO, kiểm lại cài đặt' if g['tilt_check']['alarm'] else 'ổn'} "
          f"{ {k: v['wins'] for k, v in g['tilt_check']['per_K'].items()} }")


def main(argv=None):
    ap = argparse.ArgumentParser(description="E3: điểm hợp thức (Murphy, Taggart, CRPS/twCRPS)")
    add_common_args(ap, "proper_scores")
    ap.add_argument("--centers", nargs="*", default=None,
                    help="trung tâm cho biểu đồ Murphy (mặc định: mọi trung tâm trong npz)")
    ap.add_argument("--ks", type=float, nargs="+", default=K_GRID)
    ap.add_argument("--taggart-range", type=int, nargs=2, default=list(TAGGART_Y_RANGE))
    ap.add_argument("--boot-B", type=int, default=CLUSTER_BOOT_B)
    args = ap.parse_args(argv)
    seeds = args.seeds[:1] if args.smoke else args.seeds
    boot_B = min(args.boot_B, SMOKE_BOOT_B) if args.smoke else args.boot_B
    ks = [int(k) if float(k).is_integer() else float(k) for k in args.ks]
    provenance.print_versions()

    first = load_e1(args.e1_dir, seeds[0], args.e1_tags)
    available = list(first["test"])
    primary, bstar, csrc = resolve_centers(args.primary, args.bstar, args.centers_json, available)
    centers = [c for c in (args.centers or available) if c in available]
    for c in (primary, bstar):
        if c not in centers:
            centers.append(c)
    cfg = {"script": "proper_scores", "e1_dir": args.e1_dir, "e1_tags": parse_tags(args.e1_tags),
           "centers": centers, "primary": primary, "bstar": bstar, "ks": ks,
           "theta": list(E3_THETA), "taggart_range": list(args.taggart_range),
           "n_bins": N_BINS, "n_samples": N_SAMPLES, "boot_B": boot_B, "smoke": bool(args.smoke)}
    cfg["pairs"] = forecast_pairs(centers, primary, bstar, ks)
    fp_cfg = {k: v for k, v in cfg.items() if k not in ("e1_dir",)}
    fp = preds_io.fingerprint(fp_cfg)
    print(f"E3: {len(seeds)} lần chia, trung tâm {centers}; chính = {primary}, bag B* = {bstar} "
          f"({csrc}); dấu {fp[:12]}", flush=True)

    partial = args.out + ".partial"
    res = preds_io.load_partial(partial, fp, {"per_split": {}}, on_mismatch=args.on_mismatch)
    inputs_of = {s: {"e1": input_shas(e1_paths(args.e1_dir, s, args.e1_tags))} for s in seeds}
    drive_splits(run_split, seeds, cfg, args.workers, res, partial, inputs_of)

    S = summarize(res["per_split"], seeds, cfg, n_expected=len(seeds) if args.smoke else len(SEEDS))
    print_summary(S, cfg)
    res["summary"] = S
    res["meta"] |= {
        "experiment": "E3", "protocol": "hậu kỳ trên npz E1; R1, R1₁ khớp trên (ŷ OOF, y) của tập "
        "huấn luyện, áp cho ŷ test của mô hình khớp lại; đuôi theo khối lượng của y huấn luyện",
        "seeds": seeds, "primary": primary, "bstar": bstar, "centers_source": csrc,
        "centers": centers, "ks": ks, "theta": list(E3_THETA),
        "taggart_range": list(args.taggart_range), "n_bins": N_BINS, "n_samples": N_SAMPLES,
        "boot_B": boot_B, "primary_k": PRIMARY_K, "smoke": bool(args.smoke),
        "pairs": [f"{a} - {b}" for a, b in cfg["pairs"]],
        "notes": ["điểm Taggart: trung bình S_θ trên θ nguyên thuộc [ymin; lo) ∪ [hi; ymax]",
                  "twCRPS: u(z) = 1{z < lo} + 1{z >= hi} qua hàm xích (Allen et al. 2023)",
                  "dist_metrics của tail_prior dùng ngưỡng cố định 60/100 và S = 50"],
        "provenance": provenance.stamp()}
    preds_io.dump_json_atomic(res, args.out)
    print(f"\nĐã ghi {args.out}")
    return res


if __name__ == "__main__":
    main()
