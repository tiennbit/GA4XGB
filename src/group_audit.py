# -*- coding: utf-8 -*-
"""E10 (khung bài 24/9, mục 6.12): kiểm toán nhóm với biên tương đương khai trước.

Vì sao: kiểm toán tỉnh của lượt thăm dò (F18) chỉ gồm 17/49 tỉnh và đọc "p > 0,05"
như "không có thiên lệch". Ở đây mọi tỉnh được dùng, mọi kết luận đi qua biên
tương đương của Bảng II (gates.E10_MARGINS), và câu "trọng số đuôi làm tăng thiên
lệch tỉnh" chỉ được viết khi chênh τ̂ vượt phân phối hoán vị.

Dữ liệu dự đoán: cross-fit 2 × 5-fold trên TOÀN BỘ 57.174 bản ghi (gates.E10_CROSSFIT;
E0b không thấy dòng trùng nên chia theo dòng). Mỗi fold ngoài: khớp trung tâm bằng
use_validity.fit_center (cùng giao thức E1: fold trong, tập dừng sớm, số cây trung
vị), R1 khớp trên (ŷ OOF, y) của phần huấn luyện, áp cho fold giữ lại:
  g_K   = R1 ở K ∈ gates.E9_KS (K = 1 là R1₁, phần hiệu chỉnh);
  p̂_c   = P(Y thuộc biến cố c | x): tỉ lệ 50 điểm của R1 thuộc biến cố, hiệu chỉnh
          isotonic trên OOF khớp chéo (đúng P2 của E9, dùng chung hàm của use_validity).
Hai lần lặp được lấy trung bình theo dòng (ŷ, g_K, p̂_c) trước mọi phân tích.

Mô hình được kiểm toán: trung tâm chính (cổng G1 của E1) trên tập đặc trưng chính,
cấu hình dò lấy ở lần chia gates.E10_CONFIG_SEED; và (d) bag10 trên F_dt, F_dt-cn,
F_dt-cn-bc để thấy tác động của việc bỏ thuộc tính nhạy cảm.

Phân tích (mỗi mô hình):
 (a) Cleary theo từng biến nhóm (gates.E10_GROUPS): trong mỗi mức g, OLS của y theo
     ŷ đã trừ trung bình; a_g, b_g = chênh hệ số chặn, hệ số góc so với đường gộp.
     CI phân vị từ bootstrap cụm theo trường.
 (b) Tỉnh: phần dư y − pred với pred ∈ {ŷ, g_K}; mô hình hiệu ứng ngẫu nhiên trên
     trung bình tỉnh (phương sai trong tỉnh gộp); τ̂ bằng REML và DerSimonian–Laird,
     CI của τ bằng Q-profile (Viechtbauer 2007); đủ 49 tỉnh. Nền hoán vị: hoán vị
     nhãn tỉnh giữa các dòng trong từng khuVuc; cổng so τ̂(K) − τ̂(1) với phân vị 95%.
     RMSE theo tam phân vị cỡ tỉnh.
 (c) Tỉ lệ cờ, FNR, FPR, PPV theo nhóm cho P0 (ŷ), P1 (g_K) và P2 (p̂ ≥ 1/(1+K)) ở
     mỗi ngưỡng và K; chênh so với toàn bộ, CI bootstrap cụm theo trường.

Lựa chọn khi khung bài chưa rõ (ghi cả vào meta.choices):
 1. Không dùng statsmodels MixedLM: gói không có trong requirements.lock.txt và thêm
    gói là đổi môi trường đã khoá. Mô hình tỉnh là hiệu ứng ngẫu nhiên một chiều, nên
    REML trên trung bình tỉnh với phương sai trong tỉnh gộp (dạng meta-analysis chuẩn)
    cho cùng đại lượng τ; DerSimonian–Laird và Q-profile cài bằng numpy/scipy.
 2. JSON KHÔNG có giá trị theo từng tỉnh: mã tỉnh sắp theo tên nên dịch ngược được
    ra tên. Chỉ ghi τ̂, CI, phân vị và số tỉnh vượt biên. Việc nêu tên tỉnh chưa được
    chủ dữ liệu cho phép.
 3. Mức nhóm có dưới MIN_LEVEL_N dòng bị bỏ khỏi (a) và (c) (ghi số dòng bị bỏ).
 4. Verdict theo CI 95%: toàn bộ CI trong biên -> "trong biên"; CI chứa 0 nhưng vượt
    biên -> "chưa xác định"; CI không chứa 0 -> "có chênh" (kèm vượt biên hay không).
    p bootstrap hai phía và Holm trong từng họ (mỗi mô hình × phân tích) ghi kèm.
 5. Cleary và (c) dùng dự đoán trung bình của hai lần lặp; bootstrap không lấy lại
    phần khớp mô hình (bất định do khớp không nằm trong CI).
 6. Q-profile: nếu Q(0) nhỏ hơn phân vị cần, cận tương ứng là 0.

Chạy (server): PYTHONPATH=src .venv/bin/python -W ignore src/group_audit.py --workers 8
Khói (Mac): PYTHONPATH=src python3 src/group_audit.py --data tests/fixtures/fake_hsa.csv \
    --smoke --center bag10 --fset F_dt --out <scratch>/ga.json --preds-dir <scratch>/preds
"""
import argparse
import os
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2
from sklearn.model_selection import KFold

import data_audit
import decision_layer as dl
import gates
import preds_io
import provenance
import splits
import stats_paired as sp
import use_validity as uv
from preprocess import DATA_PATH

EXPERIMENT = "E10"
DEFAULT_OUT = "results_cost/group_audit.json"
DEFAULT_PREDS = os.path.join("preds", "group_audit")
E1_JSON = os.path.join("results_cost", "decomp_centers.json")
E1_PREDS = os.path.join("preds", "decomp")
D_MODELS = [("bag10", "F_dt"), ("bag10", "F_dt-cn"), ("bag10", "F_dt-cn-bc")]
MIN_LEVEL_N = 30
TAU_MAX = 400.0                      # cận trên của τ² khi tìm (20 điểm, dư sức)


# ---------------------------------------------------------------------------
# Trung tâm chính và cấu hình
# ---------------------------------------------------------------------------
def resolve_primary(center, fset, e1_json):
    """(trung tâm, tập đặc trưng, nguồn). 'auto' đọc summary.G1 của E1."""
    src = "cli"
    if "auto" in (center, fset):
        S = (preds_io.load_json(e1_json) or {}).get("summary", {})
        g1 = S.get("G1") or {}
        if center == "auto":
            center = g1.get("primary_center")
        if fset == "auto":
            fset = g1.get("feature_set") or (S.get("feature_gate") or {}).get("primary")
        src = f"{e1_json}:summary.G1"
        if not center or not fset:
            raise SystemExit(f"Không đọc được trung tâm/tập đặc trưng chính từ {e1_json}; "
                             "truyền --center và --fset.")
    return center, fset, src


def center_config(center, e1_preds, smoke):
    """Cấu hình dò (cho rs_tuned*) từ npz E1 của lần chia gates.E10_CONFIG_SEED."""
    if not center.startswith("rs_tuned"):
        return None, "không cần"
    path = preds_io.split_path(e1_preds, gates.E10_CONFIG_SEED)
    if not os.path.exists(path):
        raise SystemExit(f"{center} cần cấu hình từ {path} (E1 lần chia {gates.E10_CONFIG_SEED})")
    cfg, j = uv.rs_tuned_config(preds_io.load_split(path))
    if cfg is None:
        raise SystemExit(f"{path}: {j}")
    return cfg, f"{path}:cấu hình {j}"


# ---------------------------------------------------------------------------
# Cross-fit: một (mô hình, lần lặp, fold ngoài) mỗi tác vụ
# ---------------------------------------------------------------------------
def crossfit_folds(n, rep):
    return list(KFold(gates.E10_CROSSFIT[1], shuffle=True, random_state=10_000 + rep).split(np.arange(n)))


def run_fold(model, rep, k, data, cfg, thresholds, smoke, n_jobs, preds_dir, fp):
    center, fset = model
    path = os.path.join(preds_dir, f"{center}_{fset}_r{rep}_f{k}.npz")
    have = preds_io.load_or_none(path, fp, on_mismatch="recompute")
    if have is not None:
        return model, rep, k, path, "cached"
    F = uv.load_frame_cached(data)
    tr, te = crossfit_folds(F.n, rep)[k]
    seed = 10_000 + 10 * rep + k
    plan = splits.fold_plan(tr, seed)
    members, es = uv.center_members(center, cfg, smoke=smoke)
    kw = {}
    if smoke:
        kw = dict(max_trees=uv.SMOKE_TREES, es_rounds=uv.SMOKE_ES)
    oof, p_te, info = uv.fit_center(F, fset, tr, te, plan, members, es, n_jobs, **kw)
    y_tr = F.y[tr]
    lo, hi = dl.tail_cutoffs(y_tr)
    r1 = dl.make_rule("R1").fit(oof, y_tr)
    arrays = {"idx_te": te, "yhat": p_te}
    for K in gates.E9_KS:
        arrays[f"g_{uv.kkey(K)}"] = r1.predict(p_te, None if K == 1 else dl.step(K, K, lo, hi))
    S_te = r1.samples(p_te)
    S_cf = uv.crossfit_r1_samples(oof, y_tr, seed)
    for d, c in thresholds:
        praw_oof = uv.exceed_prob(S_cf, d, c)
        praw_te = uv.exceed_prob(S_te, d, c)
        arrays[f"p2_{uv.thr_key(d, c)}"] = uv.iso_calibrate(praw_oof, uv.event(y_tr, d, c), praw_te)
    preds_io.save_split(path, **arrays, meta={"fingerprint": fp, "model": list(model), "rep": rep,
                                              "fold": k, "info": info, "cutoffs": [lo, hi]})
    return model, rep, k, path, "fitted"


def assemble(model, preds_dir, n, reps, thresholds):
    """Trung bình theo dòng qua các lần lặp: {'yhat', 'g_K'..., 'p2_...'}."""
    keys = ["yhat"] + [f"g_{uv.kkey(K)}" for K in gates.E9_KS] + \
           [f"p2_{uv.thr_key(d, c)}" for d, c in thresholds]
    acc = {k: np.zeros(n) for k in keys}
    cnt = np.zeros(n)
    for rep in range(reps):
        for k in range(gates.E10_CROSSFIT[1]):
            d = preds_io.load_split(os.path.join(preds_dir, f"{model[0]}_{model[1]}_r{rep}_f{k}.npz"))
            idx = np.asarray(d["idx_te"])
            for key in keys:
                acc[key][idx] += np.asarray(d[key], dtype=float)
            cnt[idx] += 1
    assert np.all(cnt == reps), "cross-fit thiếu dòng: mỗi dòng phải có đúng một dự đoán mỗi lần lặp"
    return {k: v / cnt for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Bootstrap cụm theo trường, vector hoá
# ---------------------------------------------------------------------------
def boot_weights(n_clusters, B, seed):
    """Ma trận B × n_clusters số lần mỗi trường được rút (lấy lại có hoàn lại)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=B).astype(np.float64)


def _ci(boot, est):
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return {"est": float(est), "ci_lo": None, "ci_hi": None, "p": None}
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
    return {"est": float(est), "ci_lo": float(lo), "ci_hi": float(hi), "p": float(max(p, 1.0 / len(boot)))}


def verdict(ci, margin):
    lo, hi = ci["ci_lo"], ci["ci_hi"]
    if lo is None:
        return "không đủ dữ liệu"
    if -margin <= lo and hi <= margin:
        return "trong biên"
    if lo <= 0 <= hi:
        return "chưa xác định"
    return "có chênh, vượt biên" if (hi > margin or lo < -margin) else "có chênh, trong biên"


def _holm_into(items):
    """Gắn p_holm vào các dict có khoá 'p' (một họ)."""
    ps = [it["p"] for it in items]
    if not ps:
        return
    adj = sp.holm([p if p is not None else 1.0 for p in ps], m=len(ps))
    adj = adj.get("p_adj", adj) if isinstance(adj, dict) else adj
    for it, a in zip(items, adj):
        it["p_holm"] = float(a)


def _levels(g, min_n=MIN_LEVEL_N):
    vals, cnt = np.unique(g, return_counts=True)
    keep = vals[cnt >= min_n]
    return keep, int(cnt[cnt < min_n].sum())


# ---------------------------------------------------------------------------
# (a) Cleary
# ---------------------------------------------------------------------------
def _ols_from_sums(s):
    n, sx, sy, sxx, sxy = (s[..., i] for i in range(5))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (sxy - sx * sy / n) / (sxx - sx ** 2 / n)
        a = sy / n - b * sx / n
    return a, b


def cleary(y, yhat, groups, school, W):
    x = yhat - yhat.mean()
    ns = W.shape[1]
    out, family = {}, []
    for gname in gates.E10_GROUPS:
        g = groups[gname]
        levels, dropped = _levels(g)
        S = np.zeros((ns, len(levels), 5))
        for j, lv in enumerate(levels):
            m = g == lv
            for i, v in enumerate((np.ones(m.sum()), x[m], y[m], x[m] ** 2, x[m] * y[m])):
                S[:, j, i] = np.bincount(school[m], weights=v, minlength=ns)
        full = S.sum(axis=0)
        a_l, b_l = _ols_from_sums(full)
        a_p, b_p = _ols_from_sums(full.sum(axis=0))
        Sb = np.einsum("bs,slk->blk", W, S)
        ab_l, bb_l = _ols_from_sums(Sb)
        ab_p, bb_p = _ols_from_sums(Sb.sum(axis=1))
        lv_out = {}
        for j, lv in enumerate(levels):
            ca = _ci(ab_l[:, j] - ab_p, a_l[j] - a_p)
            cb = _ci(bb_l[:, j] - bb_p, b_l[j] - b_p)
            ca["verdict"] = verdict(ca, gates.E10_MARGINS["a_g"])
            cb["verdict"] = verdict(cb, gates.E10_MARGINS["b_g"])
            family += [ca, cb]
            lv_out[str(int(lv))] = {"n": int(full[j, 0]), "a_g": ca, "b_g": cb}
        out[gname] = {"levels": lv_out, "n_dropped_small_levels": dropped,
                      "pooled": {"a": float(a_p), "b": float(b_p)}}
    _holm_into(family)
    return out


# ---------------------------------------------------------------------------
# (b) Tỉnh: τ bằng REML, DL, Q-profile; nền hoán vị
# ---------------------------------------------------------------------------
def prov_stats(r, prov, n_prov):
    n = np.bincount(prov, minlength=n_prov).astype(float)
    s1 = np.bincount(prov, weights=r, minlength=n_prov)
    s2 = np.bincount(prov, weights=r * r, minlength=n_prov)
    ok = n > 0
    n, s1, s2 = n[ok], s1[ok], s2[ok]
    m = s1 / n
    within = (s2 - n * m ** 2).sum() / (n.sum() - len(n))       # phương sai trong tỉnh gộp
    return m, within / n


def _mu_w(m, v, t2):
    w = 1.0 / (v + t2)
    return w, (w * m).sum() / w.sum()


def tau2_reml(m, v):
    def nll(t2):
        w, mu = _mu_w(m, v, t2)
        return 0.5 * (np.log(v + t2).sum() + np.log(w.sum()) + (w * (m - mu) ** 2).sum())
    r = minimize_scalar(nll, bounds=(0.0, TAU_MAX), method="bounded", options={"xatol": 1e-6})
    return float(r.x) if nll(r.x) <= nll(0.0) else 0.0


def tau2_dl(m, v):
    w = 1.0 / v
    mu = (w * m).sum() / w.sum()
    Q = (w * (m - mu) ** 2).sum()
    return float(max(0.0, (Q - (len(m) - 1)) / (w.sum() - (w ** 2).sum() / w.sum())))


def q_profile(m, v, level=0.95):
    df = len(m) - 1

    def Q(t2):
        w, mu = _mu_w(m, v, t2)
        return (w * (m - mu) ** 2).sum()
    hi_q, lo_q = chi2.ppf((1 + level) / 2, df), chi2.ppf((1 - level) / 2, df)

    def root(target):
        if Q(0.0) <= target:
            return 0.0
        if Q(TAU_MAX) > target:
            return TAU_MAX
        return brentq(lambda t: Q(t) - target, 0.0, TAU_MAX)
    return root(hi_q), root(lo_q)


def permute_within(prov, region, rng):
    out = prov.copy()
    for rg in np.unique(region):
        idx = np.flatnonzero(region == rg)
        out[idx] = prov[rng.permutation(idx)]
    return out


def province_block(y, P, prov, region, n_perm, seed):
    n_prov = int(prov.max()) + 1
    out = {"n_provinces": int(len(np.unique(prov))), "by_pred": {}}
    tau = {}
    for name, pred in P.items():
        m, v = prov_stats(y - pred, prov, n_prov)
        t2 = tau2_reml(m, v)
        lo, hi = q_profile(m, v)
        tau[name] = np.sqrt(t2)
        ci = {"est": float(np.sqrt(t2)), "ci_lo": float(np.sqrt(lo)), "ci_hi": float(np.sqrt(hi))}
        ci["verdict"] = ("trong biên" if ci["ci_hi"] <= gates.E10_MARGINS["tau"]
                         else "chưa xác định" if ci["ci_lo"] <= gates.E10_MARGINS["tau"] else "vượt biên")
        out["by_pred"][name] = {"tau_reml": ci, "tau_dl": float(np.sqrt(tau2_dl(m, v))),
                                "mean_resid_quantiles": {q: float(np.quantile(m, q / 100)) for q in (5, 25, 50, 75, 95)},
                                "n_prov_abs_mean_gt_margin": int(np.sum(np.abs(m) > gates.E10_MARGINS["tau"]))}
    rng = np.random.default_rng(seed)
    base = "g_1"
    null = {k: [] for k in P if k.startswith("g_") and k != base}
    null_tau = {k: [] for k in P}
    for _ in range(n_perm):
        pp = permute_within(prov, region, rng)
        tp = {}
        for name, pred in P.items():
            m, v = prov_stats(y - pred, pp, n_prov)
            tp[name] = np.sqrt(tau2_reml(m, v))
            null_tau[name].append(tp[name])
        for k in null:
            null[k].append(tp[k] - tp[base])
    out["permutation"] = {"n_perm": n_perm, "within": "khuVuc",
                          "tau_null_q95": {k: float(np.quantile(v, 0.95)) for k, v in null_tau.items()}}
    gate = {}
    for k, v in null.items():
        obs = float(tau[k] - tau[base])
        q95 = float(np.quantile(v, 0.95))
        gate[k] = {"delta_tau_vs_K1": obs, "null_q95": q95, "increase": bool(obs > q95)}
    out["gate_tail_weight_increases_province_bias"] = gate
    tert = uv.province_size_tertile(prov)
    out["rmse_by_province_size_tertile"] = {
        name: {str(t): float(np.sqrt(np.mean((y[tert == t] - pred[tert == t]) ** 2))) for t in np.unique(tert)}
        for name, pred in P.items()}
    return out


# ---------------------------------------------------------------------------
# (c) Tỉ lệ cờ theo nhóm
# ---------------------------------------------------------------------------
def _rates(c):
    tp, fp, fn, tn = (c[..., i] for i in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"flag_rate": (tp + fp) / (tp + fp + fn + tn), "FNR": fn / (tp + fn),
                "FPR": fp / (fp + tn), "PPV": tp / (tp + fp)}


def flags_block(y, A, groups, school, W, thresholds):
    ns = W.shape[1]
    out, family = {}, []
    for d, c in thresholds:
        ev = uv.event(y, d, c)
        tk = uv.thr_key(d, c)
        pols = {"P0": uv.flag_point(A["yhat"], d, c)}
        for K in gates.E9_KS:
            kk = uv.kkey(K)
            pols[f"P1@K{kk}"] = uv.flag_point(A[f"g_{kk}"], d, c)
            pols[f"P2@K{kk}"] = A[f"p2_{tk}"] >= 1.0 / (1.0 + K)
        out[tk] = {}
        for pname, flag in pols.items():
            cell = {"overall": {k: float(v) for k, v in _rates(np.array(
                [np.sum(flag & ev), np.sum(flag & ~ev), np.sum(~flag & ev), np.sum(~flag & ~ev)], float)).items()},
                    "groups": {}}
            for gname in gates.E10_GROUPS:
                g = groups[gname]
                levels, dropped = _levels(g)
                C = np.zeros((ns, len(levels), 4))
                for j, lv in enumerate(levels):
                    m = g == lv
                    for i, v in enumerate((flag & ev, flag & ~ev, ~flag & ev, ~flag & ~ev)):
                        C[:, j, i] = np.bincount(school[m], weights=v[m].astype(float), minlength=ns)
                full = C.sum(axis=0)
                Cb = np.einsum("bs,slk->blk", W, C)
                r_l, r_all = _rates(full), _rates(full.sum(axis=0))
                rb_l, rb_all = _rates(Cb), _rates(Cb.sum(axis=1))
                lv_out = {}
                for j, lv in enumerate(levels):
                    e = {"n": int(full[j].sum()), "flag_rate": float(r_l["flag_rate"][j])}
                    for met, mk in (("FNR", "dFNR"), ("FPR", "dFPR")):
                        ci = _ci(rb_l[met][:, j] - rb_all[met], r_l[met][j] - r_all[met])
                        ci["verdict"] = verdict(ci, gates.E10_MARGINS[mk])
                        family.append(ci)
                        e[mk] = ci
                    e["dPPV"] = _ci(rb_l["PPV"][:, j] - rb_all["PPV"], r_l["PPV"][j] - r_all["PPV"])
                    e["dflag_rate"] = _ci(rb_l["flag_rate"][:, j] - rb_all["flag_rate"],
                                          r_l["flag_rate"][j] - r_all["flag_rate"])
                    lv_out[str(int(lv))] = e
                cell["groups"][gname] = {"levels": lv_out, "n_dropped_small_levels": dropped}
            out[tk][pname] = cell
    _holm_into(family)
    return out


# ---------------------------------------------------------------------------
def audit_model(F, A, thresholds, B, n_perm, seed):
    W = boot_weights(F.n_schools, B, seed)
    P = {"yhat": A["yhat"]} | {f"g_{uv.kkey(K)}": A[f"g_{uv.kkey(K)}"] for K in gates.E9_KS}
    return {"cleary": cleary(F.y, A["yhat"], F.groups, F.school_code, W),
            "province": province_block(F.y, P, F.prov_code, F.groups["region"], n_perm, seed + 1),
            "flags": flags_block(F.y, A, F.groups, F.school_code, W, thresholds),
            "rmse": float(np.sqrt(np.mean((F.y - A["yhat"]) ** 2)))}


def print_summary(res):
    for key, r in res["models"].items():
        pr = r["province"]["by_pred"]
        print(f"\n[E10] {key}: RMSE={r['rmse']:.3f}; τ tỉnh (REML, CI Q-profile):", flush=True)
        for name, d in pr.items():
            t = d["tau_reml"]
            print(f"    {name:6s} τ={t['est']:.3f} [{t['ci_lo']:.3f}; {t['ci_hi']:.3f}] {t['verdict']}; "
                  f"DL {d['tau_dl']:.3f}")
        print(f"    Tăng thiên lệch tỉnh theo K (so nền hoán vị 95%): "
              + ", ".join(f"{k}: {v['delta_tau_vs_K1']:+.3f} vs {v['null_q95']:.3f}"
                          + (" TĂNG" if v["increase"] else "")
                          for k, v in r["province"]["gate_tail_weight_increases_province_bias"].items()))
        bad = [(g, lv, k) for g, gd in r["cleary"].items() for lv, e in gd["levels"].items()
               for k in ("a_g", "b_g") if e[k]["verdict"] != "trong biên"]
        print(f"    Cleary: {sum(len(g['levels']) for g in r['cleary'].values())} mức, "
              f"ngoài 'trong biên': {bad if bad else 'không có'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="E10: kiểm toán nhóm")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--preds-dir", default=DEFAULT_PREDS)
    ap.add_argument("--e1-json", default=E1_JSON)
    ap.add_argument("--e1-preds", default=E1_PREDS)
    ap.add_argument("--center", default="auto")
    ap.add_argument("--fset", default="auto")
    ap.add_argument("--cutoffs", nargs="*", default=[], help="ngưỡng sàn thêm, dạng 85 hoặc ge:85")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    provenance.print_versions()
    uv.guard_real_data_on_mac(args.data)
    center, fset, csrc = resolve_primary(args.center, args.fset, args.e1_json)
    cfg, cfg_src = center_config(center, args.e1_preds, args.smoke)
    thresholds = uv.all_thresholds(args.cutoffs)
    models = [(center, fset)] + [m for m in D_MODELS if m != (center, fset)]
    reps = 1 if args.smoke else gates.E10_CROSSFIT[0]
    B = 200 if args.smoke else gates.CLUSTER_BOOT_B
    n_perm = 50 if args.smoke else gates.E10_PERMUTATIONS
    n_jobs = splits.xgb_threads(args.workers)
    if args.smoke:
        n_jobs = min(2, n_jobs)
    data_sha = preds_io.file_sha256(args.data)
    cfgs = {m: (cfg if m == (center, fset) else None) for m in models}
    fp = preds_io.fingerprint({"script": "group_audit", "data_sha256": data_sha, "smoke": args.smoke,
                               "thresholds": thresholds, "Ks": list(gates.E9_KS), "cfg": cfg,
                               "crossfit": list(gates.E10_CROSSFIT)})
    os.makedirs(args.preds_dir, exist_ok=True)
    print(f"[E10] trung tâm chính {center} trên {fset} ({csrc}); cấu hình: {cfg_src}; "
          f"{len(models)} mô hình × {reps} lần lặp × {gates.E10_CROSSFIT[1]} fold; dấu {fp[:12]}", flush=True)
    t0 = time.time()
    tasks = [(m, r, k) for m in models for r in range(reps) for k in range(gates.E10_CROSSFIT[1])]
    done = Parallel(n_jobs=args.workers, return_as="generator_unordered")(
        delayed(run_fold)(m, r, k, args.data, cfgs[m], thresholds, args.smoke, n_jobs, args.preds_dir, fp)
        for m, r, k in tasks)
    for i, (m, r, k, _, how) in enumerate(done, 1):
        print(f"  {m[0]}/{m[1]} lặp {r} fold {k}: {how} ({i}/{len(tasks)}, {time.time() - t0:.0f}s)", flush=True)
    F = uv.load_frame_cached(args.data)
    res = {"models": {}, "meta": {}}
    for m in models:
        A = assemble(m, args.preds_dir, F.n, reps, thresholds)
        res["models"][f"{m[0]}|{m[1]}"] = audit_model(F, A, thresholds, B, n_perm, seed=2026)
    print_summary(res)
    res["meta"] = {"experiment": EXPERIMENT, "fingerprint": fp, "primary": {"center": center, "fset": fset,
                   "source": csrc, "config_source": cfg_src, "config": cfg}, "d_models": D_MODELS,
                   "thresholds": thresholds, "Ks": list(gates.E9_KS), "reps": reps, "boot_B": B,
                   "n_perm": n_perm, "margins": gates.E10_MARGINS, "min_level_n": MIN_LEVEL_N,
                   "choices": [l.strip() for l in __doc__.split("Lựa chọn khi khung bài chưa rõ")[1]
                               .split("Chạy (server)")[0].strip().splitlines() if l.strip()],
                   "provenance": provenance.stamp()}
    res = uv.sanitize(preds_io.to_jsonable(res))
    # Chốt cuối trước khi ghi: JSON được kéo về Mac, không được chứa tên trường, tên
    # tỉnh hay ngày sinh (cùng chốt với data_audit.py).
    data_audit._assert_no_pii(res, data_audit.forbidden_strings(data_audit.read_raw(args.data)))
    preds_io.dump_json_atomic(res, args.out)
    print(f"\nĐã ghi {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    # Gọi qua module đã import để joblib gửi hàm sang tiến trình con theo tên module.
    import group_audit
    group_audit.main()
