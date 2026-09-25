# -*- coding: utf-8 -*-
"""E4 (khung bài 24/9, mục 6.8): HPO nên tối ưu gì?

Câu hỏi (RQ3): khi đã có tầng quyết định, việc chọn siêu tham số theo hàm mục tiêu
đuôi có mua thêm được gì so với một lần dò theo RMSE không? Thiết kế tách hai thứ
mà GA4XGB cũ trộn lẫn: hàm mục tiêu dùng để CHỌN cấu hình, và việc CÓ hay KHÔNG có
tầng quyết định sau khi chọn. Nhận xét 2 nói định tính rằng hàm thích nghi đuôi
không thay được tầng quyết định khi mọi ứng viên cùng nhắm E[Y|x]; E4 là bằng chứng
thực nghiệm ở mẫu hữu hạn, và được thiết kế để có thể thất bại.

Hậu kỳ thuần trên npz của E1 (không khớp XGBoost nào): với mỗi lần chia, vết 60
cấu hình của E1 đã có OOF 5-fold (trace_oof) và dự đoán test của mô hình khớp lại
trên toàn tập huấn luyện (trace_test). Với cấu hình j và mỗi K trong gates.E4_KS:

  A_j    = RMSE OOF thô (không phụ thuộc K).
  B_j(K) = cost_K OOF SAU quy tắc, quy tắc khớp chéo 5-fold ngay trong OOF
           (decision_layer.crossfit_rule_oof). Khớp quy tắc trên toàn OOF rồi chấm
           điểm trên chính OOF đó sẽ thưởng cho quy tắc linh hoạt (R1 có 20×50 phân
           vị) và làm lệch việc chọn cấu hình.
  C_j(K) = cost_K OOF thô.
  T_j(K) = cost_K test sau quy tắc khớp trên TOÀN OOF của j (w = w_K), áp cho
           trace_test[j]. Mọi chính sách có quy tắc đều báo T tại j* của mình.

Năm chính sách (mỗi chính sách báo cost_K trên test):
  (a)  j* = argmin A, rồi quy tắc.            Một lần dò theo RMSE, tầng quyết định lo K.
  (b)  j* = argmin B(K), rồi quy tắc.         Dò theo đúng chi phí đã qua quy tắc.
  (c)  j* = argmin C(K), không quy tắc.       Cách GA4XGB: hàm thích nghi đuôi, không tầng quyết định.
  (c+) j* = argmin C(K), rồi quy tắc.         Hàm thích nghi đuôi cộng tầng quyết định.
  (o)  j* = argmin T(K), rồi quy tắc.         Oracle nhìn test: trần của việc chọn cấu hình.

Thống kê (Bảng VI): (b)-(a), (c+)-(a), (c)-(a), (o)-(a) ở từng K; t Nadeau-Bengio
theo lần chia, Holm trên MỌI ô của bảng (4 phép so × 8 K), TOST ±SESOI cho (b)-(a)
và (c+)-(a); kèm bootstrap cụm theo trường trong từng tập kiểm tra (đếm số lần chia
có CI loại 0). Âm nghĩa là chính sách bên trái tốt hơn (a).

Giả thuyết 7 (Blackwell): Kendall τ giữa thứ hạng theo RMSE và thứ hạng theo cost_K
sau quy tắc, (i) trên 60 cấu hình (A với B(K)), kèm Spearman, bản trên 20 cấu hình
tốt nhất theo A và độ trùng top-10; (ii) trên tập trung tâm có trong npz (default,
sub1, bag B, rs_tuned, rs_tuned_bag5, ...), đo cả trên OOF (A với B) và trên test
(RMSE thô với cost_K sau quy tắc).

Lựa chọn khi khung bài chưa nói rõ (ghi cả vào meta.choices của JSON):
  1. Phân hoạch khớp chéo quy tắc: KFold(5, shuffle, random_state = seed lần chia)
     trên vị trí OOF, như crossfit_rule_oof làm sẵn; DÙNG CHUNG cho mọi cấu hình,
     mọi K và mọi trung tâm, nên so B giữa các cấu hình là so cặp. Không bám fold
     của E1 (fold_of): đơn giản hơn và không đổi kỳ vọng của B. Có mảng
     `group_code` (dài n) trong npz thì chia theo nhóm (GroupKFold).
  2. Holm trên cả Bảng VI (32 ô), không riêng từng K: khung bài viết "Holm trong
     Bảng VI". (c)-(a) và (o)-(a) nằm trong họ, nên họ bảo thủ hơn cần thiết.
  3. "(b) hoặc (c+) thắng (a) quá 0,10 với Holm p < 0,05" hiểu là trung bình chênh
     <= -SESOI và p Holm < ALPHA, cùng kiểu với cổng C1..C3.
  4. Cổng Giả thuyết 7: trung bình τ theo lần chia >= E4_KENDALL_MIN ở MỌI K cho cả
     ba thước đo (60 cấu hình OOF, trung tâm OOF, trung tâm test). Bảo thủ: một
     thước đo trượt ở một K là cổng trượt, và JSON liệt kê (thước đo, K) trượt.
  5. Hoà khi chọn: np.argmin (chỉ số nhỏ nhất). Top-20 và top-10 bị chặn bởi n_cfg.
  6. B* của Bảng chi phí: --bstar (tên trung tâm hoặc số B) nếu có; không thì đọc
     decomp_centers.json (E1) theo ĐÚNG thứ tự của decomp_rules.resolve_roles (E2):
     summary.G1.bag_Bstar trước, rồi tìm theo chiều sâu các khoá BSTAR_KEYS (cùng
     thứ tự với E2); số nguyên là bag<B>, chuỗi là tên trung tâm. Bản trước tìm theo
     chiều sâu ngay từ gốc với thứ tự khoá khác, nên gặp summary.b_star.B_star trước
     G1: trên lược đồ E1 hiện tại hai đường cho cùng B*, nhưng một lần E1 đổi lược đồ
     là đủ để E2 và E4 lặng lẽ dùng hai B* khác nhau;
     không thì trung tâm tên bag_Bstar nếu npz có; không nữa thì tính lại ở đây
     theo định nghĩa của E1 (B nhỏ nhất có trung bình |cost_3(bag B) - cost_3(bag
     tham chiếu)| <= 0,02, tham chiếu là bag lớn nhất có mặt, lẽ ra là 40) với quy
     tắc của E4. B* tính lại luôn được ghi kèm để đối chiếu.
  7. Thời gian dò và thời gian fit mô hình cuối lấy từ decomp_centers.json (E4
     không fit gì); E4 ghi thêm thời gian của chính nó (chi phí tính B(K) cho mọi
     cấu hình, tức phần thêm của chính sách (b)).
  8. --data giữ cho giao diện chung nhưng KHÔNG được đọc: E4 chỉ cần npz.
  9. NaN/inf ghi thành null để JSON hợp lệ chặt (jq đọc được).
 10. Ngoài JSON, mỗi lần chia ghi npz nhỏ ở --preds-dir: dự đoán test của năm chính
     sách ở mọi K (idx_te, y_te, policy_test["<K>|<chính sách>"]), để bootstrap lại
     hay E9 dùng mà không tính lại. Như mọi npz khác, chỉ ở server.

Chạy (server): PYTHONPATH=src .venv/bin/python -W ignore src/select_policy.py
Kiểm thử khói (Mac, dữ liệu giả):
  PYTHONPATH=src python3 src/select_policy.py --smoke --e1-dir tests/fixtures/preds/decomp \\
      --preds-dir <scratch>/preds --out <scratch>/select_policy.json --workers 1
"""
import argparse
import math
import os
import re
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import stats

import decision_layer as dl
import preds_io
import provenance
import stats_paired as sp
from gates import (ALPHA, B_STAR_TOL, CLUSTER_BOOT_B, E4_KENDALL_MIN, E4_KS, E4_TOP, K_GRID,
                   N_BINS, N_SAMPLES, PRIMARY_K, RULE_CROSSFIT_FOLDS, SEEDS, SESOI)
from preprocess import DATA_PATH

POLICIES = ["a", "b", "c", "c+", "o"]
CONTRASTS = [("b", "a"), ("c+", "a"), ("c", "a"), ("o", "a")]
TOST_CONTRASTS = ("b-a", "c+-a")          # cổng "một lần dò là đủ"
TOP_SUB = 20                              # tương quan hạng trên 20 cấu hình tốt nhất theo A
RULE_CHOICES = ["R1", "R2", "R3", "R4", "R5", "R7"]   # R0 và R6 không phụ thuộc K
COMPUTE_CENTERS = ["default", "sub1", "bag_Bstar", "rs_tuned", "rs_tuned_bag5"]
TAU_MEASURES = ["kendall_all", "kendall_centers_oof", "kendall_centers_test"]
SMOKE_BOOT_B = 200
SMOKE_MAX_CFG = 10
# Khoá tìm B* trong decomp_centers.json: chép từ decomp_rules.resolve_roles (E2),
# cùng thứ tự, và chỉ dùng SAU summary.G1.bag_Bstar như E2, để hai thí nghiệm không
# bao giờ chọn hai B* khác nhau. Đổi bên E2 thì đổi ở đây cùng lúc.
BSTAR_KEYS = ["bag_Bstar", "bstar_center", "B_star_center", "B_star", "Bstar"]
TRACE_KEYS = ("trace_oof", "trace_test", "trace_test_foldavg", "trace_best_iter",
              "trace_n_trees", "trace_oof_rmse")


def kkey(K):
    """Khoá JSON của K: '1', '1.5', '2', ... (không có '.0')."""
    return f"{float(K):g}"


# ---------------------------------------------------------------------------
# Đọc npz của E1
# ---------------------------------------------------------------------------
def parse_tags(tags):
    """--e1-tags như proper_scores (E3): rỗng hoặc 'none' là split<seed>.npz không tag."""
    tags = list(tags or [])
    return [None if (t is None or str(t).lower() == "none") else str(t) for t in tags] or [None]


def load_inputs(e1_dir, seed, tags=None):
    """(d, inputs): npz của E1 cho một lần chia, gộp nhiều file nếu E1 tách theo tag.

    Vì sao cho gộp: preds_io.split_path có tag (ví dụ một file cho giai đoạn 1 theo
    tập đặc trưng, một file cho vết 60 cấu hình của giai đoạn 2). E4 cần cả trung
    tâm lẫn vết. Các file gộp phải cùng lần chia (idx, y trùng từng phần tử); trung
    tâm trùng tên phải trùng dự đoán, nếu không là hai lượt E1 khác nhau."""
    paths = [preds_io.split_path(e1_dir, seed, t) for t in parse_tags(tags)]
    d, inputs, metas, trace_at = None, [], [], 0
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"thiếu npz của E1: {p}")
        x = preds_io.load_split(p)
        meta = x.get("meta") or {}
        inputs.append({"path": p, "sha256": preds_io.file_sha256(p),
                       "e1_fingerprint": meta.get("fingerprint")})
        metas.append(meta)
        if "trace_oof" in x:
            trace_at = len(metas) - 1
        if d is None:
            d = x
            continue
        for k in ("idx_tr", "idx_te", "y_tr", "y_te"):
            if not np.array_equal(np.asarray(d[k]), np.asarray(x[k])):
                raise ValueError(f"{p}: '{k}' khác file đầu, không cùng lần chia")
        for part in ("oof", "test", "test_foldavg"):
            for c, a in x.get(part, {}).items():
                have = d.setdefault(part, {})
                if c in have and not np.array_equal(have[c], a):
                    raise ValueError(f"{p}: {part}[{c}] khác file trước (hai lượt E1 khác nhau?)")
                have[c] = a
        for k in TRACE_KEYS:
            if k in x:
                if k in d:
                    raise ValueError(f"{p}: '{k}' có ở hai file; chỉ một file mang vết cấu hình")
                d[k] = x[k]
        for k in ("school_code", "prov_code", "group_code", "fold_of"):
            if k in x and k not in d:
                d[k] = x[k]
    err = preds_io.validate_split(d)
    if err:
        raise ValueError(f"npz lần chia {seed} không hợp lệ: {err}")
    if "trace_oof" not in d or "trace_test" not in d:
        raise ValueError(f"npz lần chia {seed} không có trace_oof/trace_test (vết cấu hình của E1)")
    # meta của file mang vết (rs_tuned_index, trace_params) được ưu tiên, rồi tới
    # các file khác theo thứ tự --in-tags
    merged = {}
    for m in [metas[trace_at]] + metas[:trace_at] + metas[trace_at + 1:]:
        for k, v in m.items():
            merged.setdefault(k, v)
    d["meta"] = merged
    return d, inputs


# ---------------------------------------------------------------------------
# Thước đo
# ---------------------------------------------------------------------------
def wcost_rows(y, M, w):
    """cost_K cho từng hàng của ma trận dự đoán M (n_cfg × n): cùng công thức với
    stats_paired.cost_w, gộp thành một phép nhân ma trận cho 60 cấu hình."""
    y, M, w = (np.asarray(a, dtype=float) for a in (y, M, w))
    return np.sqrt(((y[None, :] - M) ** 2) @ w / w.sum())


def _corr(fn, a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(fn(a, b)[0])


def spearman(a, b):
    return _corr(stats.spearmanr, a, b)


def kendall(a, b):
    """Kendall τ-b (hiệu chỉnh hoà), mặc định của scipy."""
    return _corr(stats.kendalltau, a, b)


def rank_stats(A, B, top_sub=TOP_SUB, top_k=E4_TOP):
    """Tương quan hạng giữa A (RMSE OOF) và B(K) trên mọi cấu hình, trên top_sub
    cấu hình tốt nhất theo A, và độ trùng top_k. Hạng tính bằng argsort ổn định."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    oa, ob = np.argsort(A, kind="stable"), np.argsort(B, kind="stable")
    sub = oa[:min(int(top_sub), len(A))]
    k = min(int(top_k), len(A))
    return {"spearman_all": spearman(A, B), "kendall_all": kendall(A, B),
            "spearman_top": spearman(A[sub], B[sub]), "kendall_top": kendall(A[sub], B[sub]),
            "top_overlap": int(len(set(oa[:k].tolist()) & set(ob[:k].tolist()))),
            "top_k": int(k), "top_sub": int(len(sub))}


def rule_block(p_oof, y_tr, p_te, rule, wfun, seed, n_folds, groups):
    """(ŷ OOF sau quy tắc khớp chéo, ŷ test sau quy tắc khớp trên toàn OOF)."""
    cf = dl.crossfit_rule_oof(p_oof, y_tr, rule, wfun, n_folds=n_folds, seed=seed, groups=groups)
    te = dl.make_rule(rule).fit(p_oof, y_tr, wfun).predict(p_te)
    return cf, te


# ---------------------------------------------------------------------------
# Một lần chia
# ---------------------------------------------------------------------------
def run_split(seed, e1_dir, tags, rule, Ks, boot_B, max_cfg, n_folds, preds_dir, fp):
    """Mọi số của một lần chia: (seed, per_split[seed], detail[seed])."""
    t0 = time.time()
    d, inputs = load_inputs(e1_dir, seed, tags)
    meta = d["meta"]
    y_tr, y_te = np.asarray(d["y_tr"], dtype=float), np.asarray(d["y_te"], dtype=float)
    P, Q = np.asarray(d["trace_oof"], dtype=float), np.asarray(d["trace_test"], dtype=float)
    truncated = max_cfg is not None and max_cfg < P.shape[0]
    if truncated:
        P, Q = P[:max_cfg], Q[:max_cfg]
    n_cfg = P.shape[0]
    lo, hi = dl.tail_cutoffs(y_tr)
    groups = preds_io.rows(d, "tr", "group_code") if "group_code" in d else None
    school_te = preds_io.rows(d, "te", "school_code")

    A = np.sqrt(np.mean((y_tr[None, :] - P) ** 2, axis=1))
    checks = {"lo_hi_e1": [meta.get("lo"), meta.get("hi")],
              "lo_hi_match_e1": (meta.get("lo") is None
                                 or (float(meta["lo"]) == lo and float(meta["hi"]) == hi))}
    if "trace_oof_rmse" in d:
        ref = np.asarray(d["trace_oof_rmse"], dtype=float)[:n_cfg]
        checks["A_vs_e1_trace_oof_rmse_maxabs"] = float(np.max(np.abs(A - ref)))
    ja = int(np.argmin(A))
    # (a) phải chọn đúng cấu hình rs_tuned của E1 (cùng tiêu chí RMSE OOF)
    if meta.get("rs_tuned_index") is not None and not truncated:
        checks["a_is_e1_rs_tuned"] = bool(int(meta["rs_tuned_index"]) == ja)
    # Lệch với E1 không dừng job (số của E4 vẫn tự nhất quán) nhưng phải thấy được
    # trong log: nó nghĩa là E4 và E1 không đọc cùng một thứ.
    bad = [k for k in ("lo_hi_match_e1", "a_is_e1_rs_tuned") if checks.get(k) is False]
    if checks.get("A_vs_e1_trace_oof_rmse_maxabs", 0.0) > 1e-6:
        bad.append("A_vs_e1_trace_oof_rmse_maxabs")
    if bad:
        print(f"  split {seed}: CẢNH BÁO lệch với E1 ở {bad}: {checks}", flush=True)

    centers = [c for c in (meta.get("centers") or sorted(d["oof"])) if c in d["oof"] and c in d["test"]]
    centers += [c for c in sorted(d["oof"]) if c in d["test"] and c not in centers]
    Pc = np.vstack([np.asarray(d["oof"][c], dtype=float) for c in centers]) if centers else None
    Qc = np.vstack([np.asarray(d["test"][c], dtype=float) for c in centers]) if centers else None
    cen = {"names": centers}
    if centers:
        cen["oof_rmse"] = dict(zip(centers, np.sqrt(np.mean((y_tr[None, :] - Pc) ** 2, axis=1)).tolist()))
        cen["test_rmse"] = dict(zip(centers, np.sqrt(np.mean((y_te[None, :] - Qc) ** 2, axis=1)).tolist()))
        cen["by_K"] = {}

    out, by_K, policy_test, chosen = {}, {}, {}, {}
    rule_s = {}
    for K in Ks:
        kk = kkey(K)
        t1 = time.time()
        wfun = dl.step(K, K, lo, hi)
        w_tr, w_te = dl.weights_of(wfun, y_tr), dl.weights_of(wfun, y_te)
        C = wcost_rows(y_tr, P, w_tr)
        Traw = wcost_rows(y_te, Q, w_te)
        CF, G = np.empty_like(P), np.empty_like(Q)
        for j in range(n_cfg):
            CF[j], G[j] = rule_block(P[j], y_tr, Q[j], rule, wfun, seed, n_folds, groups)
        B = wcost_rows(y_tr, CF, w_tr)
        T = wcost_rows(y_te, G, w_te)
        rule_s[kk] = time.time() - t1

        crit = {"a": A, "b": B, "c": C, "c+": C, "o": T}
        res, preds = {}, {}
        for pol in POLICIES:
            j = int(np.argmin(crit[pol]))
            pred = Q[j] if pol == "c" else G[j]
            preds[pol] = pred
            res[pol] = {"j": j, "test_cost": sp.cost_w(y_te, pred, w_te),
                        "criterion": float(crit[pol][j]),
                        "region_rmse": sp.region_rmse(y_te, pred, lo, hi)}
            policy_test[f"{kk}|{pol}"] = pred
        out[kk] = res
        chosen[kk] = {pol: res[pol]["j"] for pol in POLICIES}

        boot = {}
        for x, a in CONTRASTS:
            b = sp.cluster_boot_diff(y_te, preds[x], preds[a], school_te, K, K, lo, hi,
                                     B=boot_B, seed=seed)
            boot[f"{x}-{a}"] = {k: b[k] for k in ("est", "ci_lo", "ci_hi", "excludes_zero")}
        rank = rank_stats(A, B)

        if centers:
            CFc, Gc = np.empty_like(Pc), np.empty_like(Qc)
            for i in range(len(centers)):
                CFc[i], Gc[i] = rule_block(Pc[i], y_tr, Qc[i], rule, wfun, seed, n_folds, groups)
            Bc, Tc = wcost_rows(y_tr, CFc, w_tr), wcost_rows(y_te, Gc, w_te)
            Trc = wcost_rows(y_te, Qc, w_te)
            cen["by_K"][kk] = {"B": dict(zip(centers, Bc.tolist())), "T": dict(zip(centers, Tc.tolist())),
                               "Traw": dict(zip(centers, Trc.tolist()))}
            rank["kendall_centers_oof"] = kendall([cen["oof_rmse"][c] for c in centers], Bc)
            rank["kendall_centers_test"] = kendall([cen["test_rmse"][c] for c in centers], Tc)
        else:
            rank["kendall_centers_oof"] = rank["kendall_centers_test"] = float("nan")
        by_K[kk] = {"B": B, "C": C, "T": T, "Traw": Traw, "rank": rank, "boot": boot}

    elapsed = time.time() - t0
    detail = {"inputs": inputs, "feature_set": meta.get("feature_set"), "lo": lo, "hi": hi,
              "tail_mass_train": list(dl.tail_masses(y_tr, lo, hi)),
              "n_tr": int(len(y_tr)), "n_te": int(len(y_te)), "n_cfg": int(n_cfg),
              "grouped_crossfit": groups is not None, "checks": checks, "A": A,
              "by_K": by_K, "centers": cen,
              "timing": {"split_s": elapsed, "rule_s_per_K": rule_s,
                         "rule_s_total": float(sum(rule_s.values()))}}
    if preds_dir:
        preds_io.save_split(preds_io.split_path(preds_dir, seed),
                            idx_te=np.asarray(d["idx_te"]), y_te=y_te, policy_test=policy_test,
                            meta={"seed": int(seed), "rule": rule, "lo": lo, "hi": hi,
                                  "chosen": chosen, "fingerprint": fp, "inputs": inputs})
    a3 = out.get(kkey(PRIMARY_K), out[kkey(Ks[0])])
    print(f"  split {seed}: n_cfg={n_cfg}, cutoffs {lo:g}/{hi:g}, j(a)={ja}; K={kkey(PRIMARY_K)} "
          + " ".join(f"{p}={a3[p]['test_cost']:.3f}" for p in POLICIES)
          + f" ({elapsed:.0f}s)", flush=True)
    return seed, out, detail


# ---------------------------------------------------------------------------
# Tổng hợp qua các lần chia
# ---------------------------------------------------------------------------
def _num(v):
    return float("nan") if v is None else float(v)


def _desc(vals):
    v = np.asarray([_num(x) for x in vals], dtype=float)
    ok = v[np.isfinite(v)]
    return {"mean": float(ok.mean()) if len(ok) else float("nan"),
            "min": float(ok.min()) if len(ok) else float("nan"),
            "max": float(ok.max()) if len(ok) else float("nan"),
            "n": int(len(ok)), "per_split": v.tolist()}


def summarize(per, detail, Ks):
    seeds = sorted(per, key=int)
    policies, contrasts, rank_corr = {}, {}, {}
    cells = []
    for K in Ks:
        kk = kkey(K)
        pk = {}
        for pol in POLICIES:
            cost = [per[s][kk][pol]["test_cost"] for s in seeds]
            same = sum(per[s][kk][pol]["j"] == per[s][kk]["a"]["j"] for s in seeds)
            pk[pol] = _desc(cost) | {"same_j_as_a": int(same), "j": [per[s][kk][pol]["j"] for s in seeds]}
        policies[kk] = pk
        ck = {}
        for x, a in CONTRASTS:
            name = f"{x}-{a}"
            # _num: sau khi chạy tiếp, .partial có thể mang null thay cho NaN (lựa chọn 9)
            diffs = [_num(per[s][kk][x]["test_cost"]) - _num(per[s][kk][a]["test_cost"]) for s in seeds]
            r = sp.paired(diffs)
            boot = [detail[s]["by_K"][kk]["boot"][name] for s in seeds]
            r |= {"diffs": diffs, "tost_gate": name in TOST_CONTRASTS,
                  "boot_splits_excluding_zero": int(sum(bool(b["excludes_zero"]) for b in boot)),
                  "boot_n": len(boot)}
            ck[name] = r
            cells.append((kk, name))
        contrasts[kk] = ck
        rk = {}
        for m in ["spearman_all", "kendall_all", "spearman_top", "kendall_top", "top_overlap",
                  "kendall_centers_oof", "kendall_centers_test"]:
            rk[m] = _desc([detail[s]["by_K"][kk]["rank"][m] for s in seeds])
        rk["top_k"] = detail[seeds[0]]["by_K"][kk]["rank"]["top_k"]
        rk["top_sub"] = detail[seeds[0]]["by_K"][kk]["rank"]["top_sub"]
        rank_corr[kk] = rk
    # Holm trên cả Bảng VI (lựa chọn 2 ở docstring)
    adj = sp.holm([contrasts[kk][n]["nb"]["p"] for kk, n in cells])
    for (kk, n), p in zip(cells, adj):
        contrasts[kk][n]["p_holm"] = p
    return {"n_splits": len(seeds), "seeds": [int(s) for s in seeds], "policies": policies,
            "contrasts": contrasts, "holm_family_size": int(np.isfinite(np.asarray(adj, float)).sum())}, rank_corr


def gates_e4(contrasts, rank_corr, n_splits):
    """Cổng của E4 (mục 6.8). Chỉ có nghĩa khi đủ 10 lần chia; n_splits ghi kèm."""
    kg = [kkey(K) for K in K_GRID]
    missing = [k for k in kg if k not in contrasts]
    tost_ok = {f"{k}|{c}": bool(contrasts[k][c]["tost"]["passed"])
               for k in kg if k in contrasts for c in TOST_CONTRASTS}
    one = not missing and all(tost_ok.values())
    oracle = {k: contrasts[k]["o-a"]["nb"]["mean"] for k in kg if k in contrasts}
    nothing = one and all(np.isfinite(v) and v > -SESOI for v in oracle.values())
    helps = []
    for k, ck in contrasts.items():
        for c in TOST_CONTRASTS:
            m, ph = ck[c]["nb"]["mean"], ck[c].get("p_holm")
            if np.isfinite(m) and m <= -SESOI and ph is not None and np.isfinite(ph) and ph < ALPHA:
                helps.append({"K": k, "contrast": c, "mean": m, "p_holm": ph})
    fails = []
    for m in TAU_MEASURES:
        for k, rk in rank_corr.items():
            v = rk[m]["mean"]
            if not (np.isfinite(v) and v >= E4_KENDALL_MIN):
                fails.append({"measure": m, "K": k, "mean_tau": v})
    if helps:
        verdict = ("Dò theo K có lợi ở " + ", ".join(sorted({h['K'] for h in helps}, key=float))
                   + ": bỏ câu 'dò một lần', báo dải K và chi phí tính toán; loại tiêu đề 1")
    elif nothing:
        verdict = "Một lần dò theo RMSE là đủ; việc chọn cấu hình không mua được gì đáng kể"
    elif one:
        verdict = "Một lần dò theo RMSE là đủ (oracle còn khoảng cách > SESOI ở ít nhất một K)"
    else:
        verdict = "Chưa kết luận: TOST của (b)-(a) hoặc (c+)-(a) chưa qua ở mọi K trong K_GRID"
    return {"n_splits": n_splits, "complete": n_splits >= len(SEEDS),
            "provisional_note": None if n_splits >= len(SEEDS) else
            f"chỉ {n_splits}/{len(SEEDS)} lần chia: cổng chưa có giá trị kết luận",
            "K_gate": kg, "missing_K": missing,
            "one_tuning_suffices": bool(one), "tost_passed": tost_ok,
            "selection_buys_nothing": bool(nothing), "oracle_minus_a_mean": oracle,
            "per_K_tuning_helps": helps, "verdict": verdict,
            "h7_kendall_min": E4_KENDALL_MIN, "h7_holds": not fails, "h7_failures": fails,
            "note": "(c)-(a) chỉ minh hoạ cách làm cũ, không dùng làm bằng chứng cho Nhận xét 2"}


def _find_key(obj, names):
    """Giá trị đầu tiên (duyệt theo chiều sâu) có khoá thuộc names và là chuỗi hoặc số.
    Chép từ decomp_rules._find_key (E2) để E4 đọc B* đúng như E2; không import
    decomp_rules vì nó là mã của thí nghiệm khác (dấu mã của E4 sẽ đổi theo nó)."""
    if isinstance(obj, dict):
        for k in names:
            v = obj.get(k)
            if isinstance(v, (str, int)) and not isinstance(v, bool):
                return v
        for v in obj.values():
            r = _find_key(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, names)
            if r is not None:
                return r
    return None


def _center_name(v, cli=False):
    """Số nguyên B thành 'bag<B>'; chuỗi là tên trung tâm, như decomp_rules (E2).
    Riêng --bstar (cli=True) nhận thêm chuỗi toàn chữ số ("5" -> bag5) cho tiện gõ;
    giá trị đọc từ JSON thì theo đúng quy ước của E2 để hai bên không lệch."""
    if isinstance(v, int) and not isinstance(v, bool):
        return f"bag{v}"
    if cli and isinstance(v, str) and v.isdigit():
        return f"bag{int(v)}"
    return str(v)


def resolve_bstar(cli, e1, centers_json, available, local):
    """(tên trung tâm B*, nguồn): --bstar > decomp_centers.json (summary.G1.bag_Bstar,
    rồi BSTAR_KEYS theo chiều sâu, như E2) > bag_Bstar trong npz > tính lại ở E4."""
    if cli is not None:
        return _center_name(cli, cli=True), "--bstar"
    if e1 is not None:
        g1 = (e1.get("summary") or {}).get("G1") or {}
        v = g1.get("bag_Bstar")
        if isinstance(v, (str, int)) and not isinstance(v, bool):
            return _center_name(v), f"{centers_json}:summary.G1.bag_Bstar={v}"
        v = _find_key(e1, BSTAR_KEYS)
        if v is not None:
            return _center_name(v), f"{centers_json}:tìm theo khoá={v}"
    if "bag_Bstar" in available:
        return "bag_Bstar", "npz có trung tâm bag_Bstar"
    if local is not None:
        return f"bag{local}", "tính lại ở E4"
    return None, "không có (npz không có trung tâm bag<B>)"


def bstar_local(detail, seeds):
    """B* tính lại theo định nghĩa của E1 từ cost_3 test (sau quy tắc của E4) của các
    trung tâm tên bag<B>. Tham chiếu là bag lớn nhất có mặt."""
    k3 = kkey(PRIMARY_K)
    bags = {}
    for s in seeds:
        T = ((detail[s]["centers"].get("by_K") or {}).get(k3) or {}).get("T") or {}
        for name, v in T.items():
            m = re.fullmatch(r"bag(\d+)", name)
            if m and v is not None:
                bags.setdefault(int(m.group(1)), {})[s] = float(v)
    if not bags:
        return None, None, {}
    ref = max(bags)
    dev = {}
    for B in sorted(bags):
        common = [s for s in bags[B] if s in bags[ref]]
        dev[B] = float(np.mean([abs(bags[B][s] - bags[ref][s]) for s in common])) if common else float("nan")
    best = next((B for B in sorted(dev) if np.isfinite(dev[B]) and dev[B] <= B_STAR_TOL), None)
    return best, ref, dev


def compute_table(per, detail, summary, args, centers_json):
    """Bảng chi phí: thời gian dò, thời gian fit mô hình cuối (từ E1) và cost_3 test
    của các trung tâm (thô và sau quy tắc), cộng thời gian của chính E4."""
    seeds = sorted(per, key=int)
    k3 = kkey(PRIMARY_K)
    fset = detail[seeds[0]].get("feature_set")
    e1 = preds_io.load_json(centers_json) if centers_json and os.path.exists(centers_json) else None
    # Luôn tính B* tại chỗ để đối chiếu, kể cả khi lấy B* từ nguồn khác
    b_loc, ref, dev = bstar_local(detail, seeds)
    available = sorted({c for s in seeds for c in detail[s]["centers"]["names"]})
    bstar, src = resolve_bstar(args.bstar, e1, centers_json, available, b_loc)
    names = {c: (bstar if c == "bag_Bstar" and bstar is not None else c) for c in COMPUTE_CENTERS}
    rows = {}
    for c, name in names.items():
        cost_raw, cost_rule, fit, tune = [], [], [], []
        for s in seeds:
            cen = detail[s]["centers"]
            byk = (cen.get("by_K") or {}).get(k3) or {}
            cost_raw.append((byk.get("Traw") or {}).get(name))
            cost_rule.append((byk.get("T") or {}).get(name))
            # Lược đồ E1 (mục 6.4): per_split[seed][tập đặc trưng][trung tâm]; thiếu tầng
            # tập đặc trưng thì thử per_split[seed][trung tâm]
            ps = (((e1 or {}).get("per_split") or {}).get(str(s))) or {}
            rec = ps.get(fset) if isinstance(ps.get(fset), dict) else ps
            rec = rec.get(name) if isinstance(rec.get(name), dict) else {}
            fit.append(rec.get("fit_s"))
            tune.append(rec.get("tune_s"))
        present = any(v is not None for v in cost_raw)
        rows[c] = {"center": name, "present": present,
                   "cost3_raw": _desc(cost_raw), f"cost3_{args.rule}": _desc(cost_rule),
                   "fit_s": _desc(fit), "tune_s": _desc(tune)}
    pol = {p: summary["policies"][k3][p]["mean"] for p in POLICIES} if k3 in summary["policies"] else {}
    t_rule = _desc([detail[s]["timing"]["rule_s_total"] for s in seeds])
    t_perK = {kk: _desc([detail[s]["timing"]["rule_s_per_K"][kk] for s in seeds])["mean"]
              for kk in detail[seeds[0]]["timing"]["rule_s_per_K"]}
    return {"centers": rows, "bstar": bstar, "bstar_source": src,
            "bstar_local": None if b_loc is None else f"bag{b_loc}",
            "bstar_local_rule": args.rule, "bstar_reference": None if ref is None else f"bag{ref}",
            "bstar_local_dev": {str(k): v for k, v in dev.items()},
            "policy_cost3_mean": pol,
            "e4_seconds": {"rule_all_cfg_all_K_per_split": t_rule, "rule_all_cfg_per_K_mean": t_perK,
                           "split_total": _desc([detail[s]["timing"]["split_s"] for s in seeds])},
            "centers_json": centers_json if e1 is not None else None,
            "centers_json_sha256": preds_io.file_sha256(centers_json) if e1 is not None else None,
            "notes": ["(a) cần một lần dò (tune_s của rs_tuned) và một lần khớp quy tắc cho mỗi K.",
                      "(b) cần thêm khớp chéo quy tắc cho mọi cấu hình ở mỗi K (e4_seconds.rule_*), "
                      "không fit thêm mô hình vì cả 60 cấu hình đã được khớp lại ở E1.",
                      "(c)/(c+) trong thực hành cần một lượt dò riêng cho mỗi K (|K| × tune_s); ở đây "
                      "dùng chung vết của E1 nên không tốn thêm fit.",
                      "Cấu hình GA cũ chỉ xuất hiện ở chú thích, ghi là dò trong pipeline khác."]}


# ---------------------------------------------------------------------------
# JSON hợp lệ chặt
# ---------------------------------------------------------------------------
def strict_json(obj):
    """Bản sao với NaN/inf thành None (lựa chọn 9): JSON chuẩn không có NaN."""
    obj = preds_io.to_jsonable(obj)

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    return walk(obj)


def print_summary(res):
    S, G = res["summary"], res["gates"]
    names = [f"{x}-{a}" for x, a in CONTRASTS]
    print(f"\nE4 trên {S['n_splits']} lần chia, quy tắc {res['meta']['rule']}. cost_K test trung bình;"
          " chênh so với (a): trung bình, p Holm, T = qua TOST ±SESOI")
    print("   K " + "".join(f"{p:>8s}" for p in POLICIES) + "  " + "".join(f"{n:>21s}" for n in names))
    for kk, pk in S["policies"].items():
        line = f"{kk:>4s}" + "".join(f"{_num(pk[p]['mean']):8.3f}" for p in POLICIES) + "  "
        for n in names:
            c = S["contrasts"][kk][n]
            line += (f"{_num(c['nb']['mean']):+9.3f} pH={_num(c.get('p_holm')):5.3f}"
                     f"{'T' if c['tost']['passed'] else ' '}")
        print(line)
    print("\nTương quan hạng A và B(K) trên các cấu hình, τ trên tập trung tâm (trung bình theo lần chia):")
    for kk, rk in res["rank_corr"].items():
        print(f"  K={kk:>4s}: ρ_all={_num(rk['spearman_all']['mean']):.3f} τ_all={_num(rk['kendall_all']['mean']):.3f} "
              f"τ_top{rk['top_sub']}={_num(rk['kendall_top']['mean']):.3f} "
              f"trùng top{rk['top_k']}={_num(rk['top_overlap']['mean']):.1f} "
              f"τ_tâm_oof={_num(rk['kendall_centers_oof']['mean']):.3f} "
              f"τ_tâm_test={_num(rk['kendall_centers_test']['mean']):.3f}")
    ct = res["compute_table"]
    print(f"\nBảng chi phí: B* = {ct['bstar']} ({ct['bstar_source']}); E4 khớp quy tắc "
          f"{_num(ct['e4_seconds']['rule_all_cfg_all_K_per_split']['mean']):.1f}s mỗi lần chia")
    print(f"Cổng: {G['verdict']}. Giả thuyết 7 (τ >= {G['h7_kendall_min']}): "
          + ("giữ" if G["h7_holds"] else f"không giữ ở {len(G['h7_failures'])} ô (thước đo × K)"))


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="E4: HPO nên tối ưu gì (hậu kỳ trên npz của E1)")
    ap.add_argument("--data", default=DATA_PATH,
                    help="giữ cho giao diện chung; E4 KHÔNG đọc file này (chỉ cần npz)")
    ap.add_argument("--out", default="results_cost/select_policy.json")
    ap.add_argument("--e1-dir", default=os.path.join("preds", "decomp"), help="thư mục npz của E1 (đầu vào)")
    ap.add_argument("--e1-tags", nargs="*", default=[],
                    help="tag của các npz E1 cần gộp cho mỗi lần chia; rỗng/none: split<seed>.npz")
    ap.add_argument("--preds-dir", default="preds/select_policy",
                    help="nơi ghi npz dự đoán test của năm chính sách")
    ap.add_argument("--centers-json", default="results_cost/decomp_centers.json",
                    help="JSON của E1: thời gian dò/fit và B* cho Bảng chi phí (không có thì bỏ qua)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--rule", default="R1", choices=RULE_CHOICES,
                    help="quy tắc tầng quyết định; R2 nếu cổng G2 chọn R2 làm mặc định")
    ap.add_argument("--bstar", default=None,
                    help="bag B* của E1: tên trung tâm trong npz (bag5) hoặc số B (ghi đè nguồn khác)")
    ap.add_argument("--boot-B", type=int, default=CLUSTER_BOOT_B)
    ap.add_argument("--max-cfg", type=int, default=None, help="chỉ dùng n cấu hình đầu của vết")
    ap.add_argument("--smoke", action="store_true",
                    help=f"1 lần chia, bootstrap {SMOKE_BOOT_B} vòng, tối đa {SMOKE_MAX_CFG} cấu hình")
    ap.add_argument("--workers", type=int, default=1, help="số tiến trình joblib (mỗi tiến trình một lần chia)")
    ap.add_argument("--on-mismatch", default="raise", choices=["raise", "recompute"],
                    help=".partial hoặc npz đầu vào khác dấu: báo lỗi (mặc định) hay tính lại")
    args = ap.parse_args(argv)

    provenance.print_versions()
    seeds = args.seeds[:1] if args.smoke else list(args.seeds)
    boot_B = min(args.boot_B, SMOKE_BOOT_B) if args.smoke else args.boot_B
    max_cfg = args.max_cfg if args.max_cfg is not None else (SMOKE_MAX_CFG if args.smoke else None)
    Ks = list(E4_KS)
    config = {"script": "select_policy", "rule": args.rule, "Ks": Ks, "crossfit_folds": RULE_CROSSFIT_FOLDS,
              "n_bins": N_BINS, "n_samples": N_SAMPLES, "top_sub": TOP_SUB, "top_k": E4_TOP,
              "boot_B": boot_B, "max_cfg": max_cfg, "smoke": bool(args.smoke),
              "e1_tags": parse_tags(args.e1_tags)}
    fp = preds_io.fingerprint(config)
    res = preds_io.load_partial(args.out + ".partial", fp, {"per_split": {}, "per_split_detail": {}},
                                on_mismatch=args.on_mismatch)
    res.setdefault("per_split_detail", {})

    # Lần chia đã xong chỉ được bỏ qua khi npz đầu vào còn đúng bản đã đọc: chạy lại
    # E1 giữa chừng sẽ đổi sha256, và trộn hai bản dự đoán trong một bảng là sai.
    todo = []
    for s in seeds:
        k = str(s)
        if k in res["per_split"] and k in res["per_split_detail"]:
            old = res["per_split_detail"][k]["inputs"]
            now = [preds_io.file_sha256(i["path"]) if os.path.exists(i["path"]) else None for i in old]
            if now == [i["sha256"] for i in old]:
                print(f"  split {s}: đã có trong .partial, npz không đổi -> bỏ qua", flush=True)
                continue
            msg = f"split {s}: npz của E1 đã đổi từ lúc tính .partial"
            if args.on_mismatch == "raise":
                raise preds_io.FingerprintMismatch(msg + "; xoá .partial hoặc dùng --on-mismatch recompute")
            print(f"  {msg} -> tính lại", flush=True)
        res["per_split"].pop(k, None)
        res["per_split_detail"].pop(k, None)
        todo.append(s)

    print(f"E4: {len(seeds)} lần chia ({len(todo)} cần tính), quy tắc {args.rule}, K = {Ks}, "
          f"bootstrap {boot_B}, workers {args.workers}; đầu vào {args.e1_dir}", flush=True)
    jobs = (delayed(run_split)(s, args.e1_dir, args.e1_tags, args.rule, Ks, boot_B, max_cfg,
                               RULE_CROSSFIT_FOLDS, args.preds_dir, fp) for s in todo)
    for s, out, det in Parallel(n_jobs=max(1, args.workers), return_as="generator_unordered")(jobs):
        res["per_split"][str(s)] = out
        res["per_split_detail"][str(s)] = det
        preds_io.dump_json_atomic(strict_json(res), args.out + ".partial")

    per = {k: res["per_split"][k] for k in map(str, seeds)}
    det = {k: res["per_split_detail"][k] for k in map(str, seeds)}
    summary, rank_corr = summarize(per, det, Ks)
    final = {"per_split": per, "per_split_detail": det, "summary": summary, "rank_corr": rank_corr,
             "gates": gates_e4(summary["contrasts"], rank_corr, summary["n_splits"]),
             "compute_table": compute_table(per, det, summary, args, args.centers_json)}
    final["meta"] = {
        "experiment": "E4 select_policy (khung bài 24/9, mục 6.8)", "fingerprint": fp, "config": config,
        "rule": args.rule, "Ks": Ks, "K_gate": list(K_GRID), "sesoi": SESOI, "alpha": ALPHA,
        "seeds": [int(s) for s in seeds], "e1_dir": args.e1_dir, "e1_tags": parse_tags(args.e1_tags),
        "preds_dir": args.preds_dir, "data_arg_not_read": args.data,
        "policies": {"a": "argmin RMSE OOF, rồi quy tắc", "b": "argmin cost_K OOF sau quy tắc khớp chéo, rồi quy tắc",
                     "c": "argmin cost_K OOF thô, không quy tắc (GA4XGB)",
                     "c+": "argmin cost_K OOF thô, rồi quy tắc", "o": "oracle: argmin cost_K test sau quy tắc"},
        "choices": [
            "Khớp chéo quy tắc: KFold(5, shuffle, random_state=seed lần chia) trên vị trí OOF, chung cho mọi cấu hình, K và trung tâm; GroupKFold khi npz có group_code.",
            "Holm trên cả Bảng VI (4 phép so × 8 K).",
            "'Thắng quá 0,10' = trung bình chênh <= -SESOI và p Holm < ALPHA.",
            "Giả thuyết 7: trung bình τ >= 0,9 ở mọi K cho cả ba thước đo (60 cấu hình OOF, trung tâm OOF, trung tâm test).",
            "Hoà khi chọn: chỉ số nhỏ nhất (np.argmin).",
            "B*: --bstar, rồi decomp_centers.json (summary.G1.bag_Bstar rồi khoá như decomp_rules), rồi bag_Bstar trong npz, rồi tính lại ở E4.",
            "Thời gian dò/fit lấy từ decomp_centers.json; E4 chỉ đo thời gian khớp quy tắc.",
            "NaN/inf ghi thành null.",
        ],
        "provenance": provenance.stamp(),
    }
    final = strict_json(final)
    preds_io.dump_json_atomic(final, args.out)
    print_summary(final)
    print(f"\nĐã ghi {args.out}", flush=True)
    return final


if __name__ == "__main__":
    main()
