# -*- coding: utf-8 -*-
"""E9: ngưỡng cố định, khoảng dự đoán, gần đoán mò (khung bài 24/9, mục 6.11).

Vì sao có thí nghiệm này: cost_K phạt theo VÙNG CỦA y THẬT, không theo việc dự
đoán có vượt một ngưỡng hay không (Nhận xét 3). Một em vùng giữa bị đoán thành 100
tốn 1, một em đạt 100 bị đoán thành 95 tốn K. Vì vậy "g_K(ŷ) cộng ngưỡng" nói chung
không phải hành động Bayes cho quyết định gắn cờ, và E9 đo điều đó bằng đơn vị người
đọc hiểu được: số em trên 1.000.

Đầu vào: npz của E1 (preds_io, lược đồ E1) cho trung tâm chính; không khớp lại mô
hình, trừ phần (d). Mỗi lần chia:

(a) Với mỗi ngưỡng c và K ∈ gates.E9_KS:
    P0     gắn cờ khi ŷ >= c (đuôi thấp: ŷ < c).
    P1(K)  gắn cờ khi g_K(ŷ) >= c, g_K là R1 ở K (bậc thang đối xứng).
    P2(K)  p̂(x) = tỉ lệ trong 50 điểm của R1 vượt c, hiệu chỉnh isotonic của
           1{biến cố} theo p̂; gắn cờ khi p̂ >= 1/(1 + K).
    L_K = 1000·(K·FN + FP)/n; kèm tỉ lệ cờ, TPR, FNR, FPR, PPV, FP/TP và số em
    trên 1.000 đổi trạng thái cờ giữa hai chính sách.
(b) Spearman(ŷ, g_K) và kiểm g_K đơn điệu trên lưới ŷ; sàng lọc theo ngân sách
    (screening_curves.at_budget) bằng ŷ và bằng g_K.
(c) PI90 từ R1: độ phủ theo vùng thật, thập phân vị ŷ, giới, khuVuc, cờ chuyên,
    tam phân vị cỡ tỉnh; bề rộng.
(d) (tuỳ chọn, --near-guess, vì phải khớp lại) bỏ y <= 37 khỏi huấn luyện và kiểm
    tra, khớp lại trung tâm chính, bag B*, rs_tuned và R8*; tính lại C1, C2, C3 và
    bảng phân rã.
(e) P(biến cố | x): Brier và biểu đồ tin cậy 10 bin, cho p̂ thô và p̂ đã hiệu chỉnh.

Các lựa chọn khi khung bài chưa nói rõ (ghi cả vào meta.choices của JSON):
  1. Hiệu chỉnh P2 "khớp chéo 5-fold trong OOF": p̂ của OOF lấy bằng R1 khớp chéo
     (khớp R1 trên 4 phần OOF, lấy điểm mẫu cho phần còn lại), để p̂ OOF có cùng tính
     ngoài mẫu như p̂ test. Isotonic khớp trên toàn bộ cặp (p̂ OOF, biến cố OOF) rồi
     áp cho p̂ test, p̂ test lấy từ R1 khớp trên toàn OOF. Khớp R1 rồi chấm trên chính
     OOF sẽ cho p̂ OOF sắc hơn thật và isotonic học một ánh xạ quá tự tin.
  2. y là số nguyên nên P(Y >= c) = P(Y > c - 0,5): p̂ thô so điểm mẫu với c - 0,5
     (hiệu chỉnh liên tục). Chỉ ảnh hưởng p̂ thô; p̂ hiệu chỉnh bất biến với phép
     biến đổi đơn điệu này. P0, P1 so ŷ, g_K với c đúng như khung bài.
  3. Ngưỡng mặc định cố định 60 (biến cố y < 60) và 100 (y >= 100), gates.
     E9_THRESHOLDS; --cutoffs THÊM ngưỡng người dùng (mặc định chiều "ge", viết
     "lt:55" cho chiều thấp), --cutoffs-source ghi URL và ngày truy cập.
  4. Họ Holm cho L(P2) - L(P1) là cả lưới c × K, kể cả K = 1 (bảo thủ hơn lưới K > 1).
  5. PI90 là [điểm mẫu τ = 0,05; τ = 0,95] của R1 (điểm thứ 3 và 48 trong 50).
     Thập phân vị ŷ cắt trên ŷ test của chính lần chia.
  6. Tam phân vị cỡ tỉnh: mỗi TỈNH là một đơn vị, xếp theo số bản ghi trên toàn bộ
     dữ liệu (hoà thì theo mã), chia ba nhóm số tỉnh gần bằng nhau. Chia theo số em
     thì Hà Nội một mình chiếm gần một nhóm và nhóm "tỉnh nhỏ" gom quá nhiều tỉnh.
  7. Cổng Spearman dùng giá trị NHỎ NHẤT qua lần chia và K; cổng độ phủ dùng trung
     bình qua lần chia của từng thập phân vị và từng đuôi.
  8. (d): ngưỡng đuôi lo/hi giữ như trên tập huấn luyện ĐẦY ĐỦ của lần chia, để hàm
     chi phí không đổi giữa hai nhánh; kế hoạch fold là kế hoạch của E1 (splits.
     fold_plan, kiểm khớp với fold_of trong npz) bỏ đi các vị trí y <= 37. Cả nhánh
     đầy đủ lẫn nhánh bỏ đều KHỚP LẠI bằng cùng mã ở đây, nên chênh giữa hai nhánh chỉ
     do dữ liệu; C2, C3 tính từ npz của E1 được ghi kèm (npz_full) để thấy mã khớp lại
     có trùng E1 không. Số cây của trung tâm dò (rs_tuned, R8) = int(trung vị(best
     _iteration + 1)) của 5 mô hình fold dừng sớm, như E1. Trung tâm bag có dừng sớm
     (rs_tuned_bag5, R8_bag5): mỗi thành viên random_state b dừng sớm riêng trong
     từng fold. R8* chỉ khớp ở K = 3 (cấu hình E2b ở K = 3); ô (iii) của R8* ở K khác
     để trống. Cấu hình rs_tuned lấy từ meta npz (trace_params, rs_tuned_index hoặc
     argmin trace_oof_rmse); cấu hình R8 từ results_cost/wtrain_tuned.json.
  9. Trung tâm chính và bag B* là tham số (--center, --bstar) vì chỉ biết sau cổng
     G1 của E1; run_use_validity.sh đọc CENTER, BSTAR từ môi trường.

g_K không bao giờ được gọi là điểm dự đoán và không khuyến nghị cho học sinh xem
(cổng E9); JSON chỉ có số tổng hợp, không có dòng dữ liệu.

Chạy (server): PYTHONPATH=src .venv/bin/python src/use_validity.py --near-guess
Kiểm thử khói (Mac, dữ liệu giả):
  PYTHONPATH=src python3 src/use_validity.py --smoke --data tests/fixtures/fake_hsa.csv \
      --in-preds <npz giả> --preds-dir <scratch> --out <scratch>/use_validity.json --near-guess
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

import decision_layer as dl
import features
import gates
import preds_io
import provenance
import splits
import stats_paired as sp
from calibration_table import calib_stats
from ga_xgb import BASE_GENES
from preprocess import DATA_PATH
from screening_curves import FLAG_RATES, at_budget

EXP = "use_validity"
OUT = os.path.join("results_cost", f"{EXP}.json")
IN_PREDS = os.path.join("preds", "decomp")
PREDS_DIR = os.path.join("preds", EXP)
WTRAIN_JSON = os.path.join("results_cost", "wtrain_tuned.json")
REL_BINS = 10
DECOMP_RULES = ["R2", "R3", "R4", "R5", "R7"]
GROUPS_PI = ["gender", "region", "chuyen"]
XGB_GENES = [g[0] for g in BASE_GENES if g[0] != "n_estimators"]
# --smoke: ngân sách tí hon, chỉ để mọi nhánh mã chạy qua trên CSV giả.
SMOKE_BAG, SMOKE_TREES, SMOKE_ES = 2, 50, 10
_BAG_RE = re.compile(r"bag(\d+)")


# ---------------------------------------------------------------------------
# Tiện ích chung (group_audit.py dùng lại)
# ---------------------------------------------------------------------------
def kkey(K):
    """Khoá JSON cho K: 3 -> "3", 1.5 -> "1.5"."""
    return str(int(K)) if float(K).is_integer() else f"{K:g}"


def guard_real_data_on_mac(path):
    """Dữ liệu thật chỉ chạy trên server (AGENTS.md, quy ước phân vai): trên Mac
    (darwin) từ chối file data_final.csv để một lệnh gõ nhầm không đọc PII."""
    if sys.platform == "darwin" and os.path.basename(os.path.abspath(path)) == "data_final.csv":
        sys.exit("Không chạy thí nghiệm trên dữ liệu thật ở Mac: dùng CSV giả "
                 "(tests/make_fake_data.py) hoặc chạy trên server.")


_FRAMES = {}
_GROUPS = {}


def load_frame_cached(path):
    """features.load_frame, nhớ theo tiến trình: mỗi tiến trình joblib đọc CSV một lần."""
    key = os.path.abspath(path)
    if key not in _FRAMES:
        _FRAMES[key] = features.load_frame(path)
    return _FRAMES[key]


def record_groups(path):
    """Mã nhóm bản ghi trùng CHÍNH XÁC theo khoá của E0b (mục 6.3): ngày sinh, giới
    tính, tỉnh, trường và mọi cột học bạ lớp 10 đến 12, chuẩn hoá chuỗi (strip).

    Chỉ cần lớp tương đương, nên dùng băm 64 bit của pandas thay cho SHA-256 (xác
    suất va chạm cỡ n²/2⁶⁴, không đáng kể). Nếu data_audit.py của E0b đổi định nghĩa
    khoá, sửa hàm này theo, vì nhóm ở đây phải trùng nhóm E1 dùng để chia (phần (d)
    kiểm fold_of nên lệch sẽ báo lỗi)."""
    key = os.path.abspath(path)
    if key in _GROUPS:
        return _GROUPS[key]
    head = pd.read_csv(path, nrows=0).columns
    cols = ["Ngày sinh", "Giới tính", "Tỉnh", "Trường"] + [c for c in head if c[:3] in ("10.", "11.", "12.")]
    raw = pd.read_csv(path, usecols=cols, dtype=str, keep_default_na=False)[cols]
    raw = raw.apply(lambda s: s.str.strip())
    h = pd.util.hash_pandas_object(raw, index=False).to_numpy()
    codes = pd.factorize(h, sort=True)[0].astype(np.int64)
    _GROUPS[key] = codes
    return codes


def default_thresholds():
    lo_c, hi_c = gates.E9_THRESHOLDS
    return [("lt", float(lo_c)), ("ge", float(hi_c))]


def parse_cutoffs(tokens):
    """['85', 'ge:90', 'lt:55'] -> [('ge', 85.0), ('ge', 90.0), ('lt', 55.0)].
    Ngưỡng sàn xét tuyển là biến cố y >= c nên chiều mặc định là 'ge'."""
    out = []
    for t in tokens or []:
        m = re.fullmatch(r"(?:(lt|ge):)?(\d+(?:\.\d+)?)", str(t).strip())
        if not m:
            raise ValueError(f"ngưỡng không hợp lệ: {t!r} (dạng 85, ge:85 hoặc lt:55)")
        out.append((m.group(1) or "ge", float(m.group(2))))
    return out


def all_thresholds(tokens):
    th = default_thresholds()
    for t in parse_cutoffs(tokens):
        if t not in th:
            th.append(t)
    return th


def thr_key(direction, c):
    return f"{direction}{c:g}"


def event(y, direction, c):
    y = np.asarray(y, dtype=float)
    return y < c if direction == "lt" else y >= c


def flag_point(pred, direction, c):
    """P0 và P1: so điểm (ŷ hoặc g_K) với c, đúng chiều của biến cố."""
    return event(pred, direction, c)


def exceed_prob(S, direction, c, integer_y=True):
    """p̂ thô = tỉ lệ điểm mẫu của R1 thuộc biến cố. y nguyên: cắt ở c - 0,5 (lựa chọn 2)."""
    cut = c - 0.5 if integer_y else c
    S = np.asarray(S, dtype=float)
    return (S < cut).mean(axis=1) if direction == "lt" else (S >= cut).mean(axis=1)


def crossfit_r1_samples(p_oof, y_oof, seed, groups=None, n_folds=gates.RULE_CROSSFIT_FOLDS):
    """Điểm mẫu R1 cho chính OOF, khớp chéo (lựa chọn 1). Cùng cách chia với
    decision_layer.crossfit_rule_oof: KFold(shuffle, seed), GroupKFold khi có nhóm."""
    p, y = np.asarray(p_oof, dtype=float), np.asarray(y_oof, dtype=float)
    S = np.empty((len(p), gates.N_SAMPLES))
    if groups is None:
        parts = KFold(n_folds, shuffle=True, random_state=int(seed)).split(p)
    else:
        parts = GroupKFold(n_splits=n_folds, shuffle=True, random_state=int(seed)).split(
            p, groups=np.asarray(groups))
    for a, b in parts:
        S[b] = dl.make_rule("R1").fit(p[a], y[a]).samples(p[b])
    return S


def iso_calibrate(p_fit, ev_fit, p_new):
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(p_fit, dtype=float), np.asarray(ev_fit, dtype=float))
    return iso.predict(np.asarray(p_new, dtype=float))


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def flag_rates(flag, ev, K):
    """Tỉ lệ cờ, TPR, FNR, FPR, PPV, FP/TP và L_K = 1000·(K·FN + FP)/n."""
    flag, ev = np.asarray(flag, dtype=bool), np.asarray(ev, dtype=bool)
    n = len(flag)
    TP = int(np.sum(flag & ev))
    FP = int(np.sum(flag & ~ev))
    FN = int(np.sum(~flag & ev))
    P = TP + FN
    return {"n_flag": TP + FP, "TP": TP, "FP": FP, "FN": FN,
            "flag_rate": _div(TP + FP, n), "TPR": _div(TP, P), "FNR": _div(FN, P),
            "FPR": _div(FP, n - P), "PPV": _div(TP, TP + FP), "FP_per_TP": _div(FP, TP),
            "L": 1000.0 * (float(K) * FN + FP) / n}


def reliability(p, ev, n_bins=REL_BINS):
    """Brier, ECE và bảng tin cậy n_bins bin bằng bề rộng trên [0; 1]."""
    p, ev = np.asarray(p, dtype=float), np.asarray(ev, dtype=float)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        m = b == i
        k = int(m.sum())
        mp = float(p[m].mean()) if k else float("nan")
        ob = float(ev[m].mean()) if k else float("nan")
        if k:
            ece += k / len(p) * abs(mp - ob)
        rows.append({"lo": i / n_bins, "n": k, "mean_pred": mp, "obs_rate": ob})
    return {"brier": float(np.mean((p - ev) ** 2)), "ece": float(ece), "reliability": rows}


def pi_bounds(S, level=gates.E9_PI_LEVEL):
    """Cận của khoảng dự đoán mức `level` từ ma trận điểm mẫu cách đều xác suất."""
    tau = dl.taus(S.shape[1])
    a = (1.0 - level) / 2.0
    j_lo, j_hi = int(np.argmin(np.abs(tau - a))), int(np.argmin(np.abs(tau - (1.0 - a))))
    return S[:, j_lo], S[:, j_hi]


def cover(y, lo_b, hi_b, mask=None):
    y = np.asarray(y, dtype=float)
    m = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    k = int(m.sum())
    if not k:
        return {"n": 0, "cover": float("nan"), "width": float("nan")}
    return {"n": k, "cover": float(np.mean((y[m] >= lo_b[m]) & (y[m] <= hi_b[m]))),
            "width": float(np.mean(hi_b[m] - lo_b[m]))}


def province_size_tertile(prov_code, n_groups=3):
    """Nhóm cỡ tỉnh (0 = nhỏ) cho từng dòng; mỗi tỉnh là một đơn vị (lựa chọn 6)."""
    prov_code = np.asarray(prov_code)
    size = np.bincount(prov_code)
    provs = np.flatnonzero(size > 0)
    order = provs[np.lexsort((provs, size[provs]))]
    t = np.zeros(len(size), dtype=np.int8)
    for i, part in enumerate(np.array_split(order, n_groups)):
        t[part] = i
    return t[prov_code]


def sanitize(obj):
    """NaN/inf thành null để JSON đúng chuẩn (json của Python ghi NaN, parser chặt từ chối)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _num(v):
    return v is None or (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool))


def aggregate(items):
    """Gộp danh sách cấu trúc cùng dạng (một phần tử mỗi lần chia) thành cùng cấu trúc
    với lá {"mean", "sd", "n"} (bỏ NaN/None); bool thành tỉ lệ True; chuỗi giữ phần tử đầu."""
    first = next((x for x in items if x is not None), None)
    if isinstance(first, dict):
        keys = list(dict.fromkeys(k for x in items if isinstance(x, dict) for k in x))
        return {k: aggregate([x.get(k) if isinstance(x, dict) else None for x in items]) for k in keys}
    if isinstance(first, list):
        L = max(len(x) for x in items if isinstance(x, list))
        return [aggregate([x[i] if isinstance(x, list) and i < len(x) else None for x in items])
                for i in range(L)]
    if isinstance(first, bool):
        v = [bool(x) for x in items if isinstance(x, bool)]
        return {"frac_true": float(np.mean(v)), "n": len(v)}
    if all(_num(x) for x in items):
        v = np.array([np.nan if x is None else float(x) for x in items])
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()) if len(v) else float("nan"),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": int(len(v))}
    return first


# ---------------------------------------------------------------------------
# Khớp trung tâm (phần (d) của E9 và cross-fit của E10)
# ---------------------------------------------------------------------------
def xgb_params(cfg):
    """Chỉ giữ gene XGBoost của không gian dò (bỏ n_estimators: số cây do dừng sớm)."""
    return {k: (int(v) if k in ("max_depth", "min_child_weight") else float(v))
            for k, v in dict(cfg).items() if k in XGB_GENES}


def center_members(name, cfg=None, smoke=False):
    """(thành viên, có dừng sớm?) của một trung tâm theo tên dùng trong E1/E2b.

    default: XGBRegressor mặc định, random_state 0. sub1: một mô hình subsample =
    colsample_bytree = 0,8. bagB: B mô hình như sub1, random_state 0..B-1. rs_tuned
    và R8: cấu hình dò, dừng sớm. rs_tuned_bag5 và R8_bag5: cấu hình dò với
    random_state 0..4. --smoke cắt còn SMOKE_BAG thành viên và SMOKE_TREES cây."""
    bs = gates.BAG_SUBSAMPLE
    m = _BAG_RE.fullmatch(name)
    if name == "default":
        members, es = [{"random_state": 0}], False
    elif name == "sub1":
        members, es = [{"random_state": 0, "subsample": bs, "colsample_bytree": bs}], False
    elif m:
        members = [{"random_state": b, "subsample": bs, "colsample_bytree": bs}
                   for b in range(int(m.group(1)))]
        es = False
    elif name in ("rs_tuned", "R8", "rs_tuned_bag5", "R8_bag5"):
        if cfg is None:
            raise ValueError(f"trung tâm {name} cần cấu hình dò")
        nb = gates.R8_BAG if name.endswith("_bag5") else 1
        members, es = [dict(xgb_params(cfg), random_state=b) for b in range(nb)], True
    else:
        raise ValueError(f"trung tâm lạ: {name}")
    if smoke:
        members = members[:SMOKE_BAG]
        if not es:
            members = [dict(kw, n_estimators=SMOKE_TREES) for kw in members]
    return members, es


def _xgb(n_jobs, **kw):
    return XGBRegressor(tree_method="hist", n_jobs=n_jobs, verbosity=0, **kw)


def fit_center(F, fset, tr, te, plan, members, es, n_jobs, wfun=None,
               max_trees=gates.MAX_TREES, es_rounds=gates.EARLY_STOP_ROUNDS):
    """(OOF dài len(tr), dự đoán test, info) theo giao thức E1.

    Trung tâm không dừng sớm khớp trên cả fold huấn luyện (fit ∪ es). Trung tâm dừng
    sớm khớp trên fit, dừng trên es, số cây khi khớp lại = int(median(best_iteration
    + 1)) qua các fold (splits.py, preds_io). Cột trường của mọi ma trận tính CHỈ từ
    hàng huấn luyện của mô hình đó (features.Frame.design).
    wfun (R8): sample_weight = w(y)/mean(w(y_fold_train)); dừng sớm theo RMSE có trọng
    số w của tập dừng sớm, đúng cost_K (E2b)."""
    t0 = time.time()
    y = F.y
    tr, te = np.asarray(tr), np.asarray(te)
    oof = np.zeros(len(tr))
    best = np.zeros((len(members), len(plan)), dtype=np.int64)
    for f in plan:
        r_train, r_fit, r_es, r_va = (tr[f[k]] for k in ("train", "fit", "es", "va"))
        X_train, X_fit, X_es, X_va = F.design(fset, r_train, r_fit, r_es, r_va)
        preds = []
        for i, kw in enumerate(members):
            if es:
                sw = swe = None
                if wfun is not None:
                    sw = wfun(y[r_fit]) / np.mean(wfun(y[r_train]))
                    swe = [wfun(y[r_es])]
                m = _xgb(n_jobs, n_estimators=max_trees, early_stopping_rounds=es_rounds,
                         eval_metric="rmse", **kw)
                m.fit(X_fit, y[r_fit], sample_weight=sw, eval_set=[(X_es, y[r_es])],
                      sample_weight_eval_set=swe, verbose=False)
                best[i, f["fold"]] = m.best_iteration          # đánh số từ 0
            else:
                sw = None if wfun is None else dl.normalize_weights(wfun(y[r_train]))
                m = _xgb(n_jobs, **kw).fit(X_train, y[r_train], sample_weight=sw)
            preds.append(m.predict(X_va))
        oof[f["va"]] = np.mean(preds, axis=0)
    X_tr, X_te = F.design(fset, tr, te)
    sw = None if wfun is None else dl.normalize_weights(wfun(y[tr]))
    tests, n_trees = [], []
    for i, kw in enumerate(members):
        if es:
            kw = dict(kw, n_estimators=int(np.median(best[i] + 1)))
        m = _xgb(n_jobs, **kw).fit(X_tr, y[tr], sample_weight=sw)
        tests.append(m.predict(X_te))
        n_trees.append(int(kw.get("n_estimators", 100)))
    return oof, np.mean(tests, axis=0), {"n_trees": n_trees, "fit_s": round(time.time() - t0, 2)}


def subset_plan(plan, keep):
    """Kế hoạch fold của E1 thu về các vị trí `keep` (bool dài n_tr); vị trí mới đánh
    trong tr[keep]. Giữ nguyên fold và tập dừng sớm, chỉ bỏ dòng (lựa chọn 8)."""
    keep = np.asarray(keep, dtype=bool)
    new_pos = np.cumsum(keep) - 1
    out = []
    for f in plan:
        g = {"fold": f["fold"]}
        for k in ("train", "fit", "es", "va"):
            p = np.asarray(f[k])
            g[k] = new_pos[p[keep[p]]]
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# Cấu hình dò từ E1 (npz) và E2b (JSON)
# ---------------------------------------------------------------------------
def rs_tuned_config(d):
    """(cấu hình, chỉ số) của rs_tuned từ meta npz E1, hoặc (None, lý do)."""
    meta = d.get("meta") or {}
    params = meta.get("trace_params")
    if not params:
        return None, "npz không có meta.trace_params"
    j = meta.get("rs_tuned_index")
    if j is None and "trace_oof_rmse" in d:
        j = int(np.argmin(np.asarray(d["trace_oof_rmse"])))
    if j is None:
        return None, "không xác định được cấu hình rs_tuned"
    return xgb_params(params[int(j)]), int(j)


def r8_config(path, seed, K=gates.PRIMARY_K):
    """(cấu hình R8 đã chọn ở E2b cho lần chia seed và K, nguồn) hoặc (None, lý do).

    Lược đồ theo mục 6.6: per_split[seed][K] = {best_params, ...}. Khoá K có thể là
    "3", "3.0", "K3"; cả file và mục đều có thể chưa có khi E2b chưa chạy xong."""
    if not path or not os.path.exists(path):
        return None, f"không có {path}"
    obj = preds_io.load_json(path) or {}
    per = obj.get("per_split", obj)
    row = per.get(str(seed)) if isinstance(per, dict) else None
    if not isinstance(row, dict):
        return None, f"{path} chưa có lần chia {seed}"
    for k in (kkey(K), f"{float(K)}", f"K{kkey(K)}", f"K={kkey(K)}"):
        cell = row.get(k)
        if isinstance(cell, dict) and isinstance(cell.get("best_params"), dict):
            return xgb_params(cell["best_params"]), f"{path}:per_split.{seed}.{k}"
    return None, f"{path}: lần chia {seed} không có best_params ở K = {kkey(K)}"


# ---------------------------------------------------------------------------
# (a), (b), (c), (e): hậu kỳ trên npz của E1
# ---------------------------------------------------------------------------
def threshold_block(direction, c, Ks, y_tr, y_te, p_te, g, S_te, S_cf, integer_y):
    ev_tr, ev_te = event(y_tr, direction, c), event(y_te, direction, c)
    praw_oof = exceed_prob(S_cf, direction, c, integer_y)
    praw_te = exceed_prob(S_te, direction, c, integer_y)
    pcal_te = iso_calibrate(praw_oof, ev_tr, praw_te)
    f0 = flag_point(p_te, direction, c)
    pol = {"P0": {}, "P1": {}, "P2": {}}
    flips = {"P1_vs_P0": {}, "P2_vs_P1": {}, "P2_vs_P0": {}}
    for K in Ks:
        kk = kkey(K)
        f1 = flag_point(g[kk], direction, c)
        f2 = pcal_te >= 1.0 / (1.0 + float(K))
        pol["P0"][kk] = flag_rates(f0, ev_te, K)
        pol["P1"][kk] = flag_rates(f1, ev_te, K)
        pol["P2"][kk] = flag_rates(f2, ev_te, K)
        flips["P1_vs_P0"][kk] = 1000.0 * float(np.mean(f1 != f0))
        flips["P2_vs_P1"][kk] = 1000.0 * float(np.mean(f2 != f1))
        flips["P2_vs_P0"][kk] = 1000.0 * float(np.mean(f2 != f0))
    # Sàng lọc theo ngân sách: cùng số em được gắn cờ, xếp theo ŷ hoặc theo g_K.
    sign = -1.0 if direction == "lt" else 1.0
    budget = {}
    for rate in FLAG_RATES:
        nf = int(round(rate * len(y_te)))
        base = at_budget(sign * p_te, ev_te, nf) if ev_te.any() else {"recall": float("nan")}
        row = {"recall_yhat": base["recall"], "recall_g": {}}
        for K in Ks:
            kk = kkey(K)
            row["recall_g"][kk] = (at_budget(sign * g[kk], ev_te, nf)["recall"] if ev_te.any()
                                   else float("nan"))
        budget[f"{rate:.0%}"] = row
    base_rate = float(ev_tr.mean())
    prob = {"raw": reliability(praw_te, ev_te), "cal": reliability(pcal_te, ev_te),
            "climatology": {"p": base_rate, "brier": float(np.mean((base_rate - ev_te) ** 2))}}
    for k in ("raw", "cal"):
        cb = prob["climatology"]["brier"]
        prob[k]["brier_skill"] = 1.0 - prob[k]["brier"] / cb if cb > 0 else float("nan")
    return {"c": c, "dir": direction, "prevalence_tr": base_rate,
            "prevalence_te": float(ev_te.mean()), "n_event_te": int(ev_te.sum()),
            "policies": pol, "flip_per_1000": flips, "budget": budget, "prob": prob}


def monotone_block(r1, p_oof, p_te, g, Ks, lo, hi):
    """Spearman(ŷ, g_K) trên test và số bước giảm của g_K trên lưới ŷ.
    R1 cộng một hằng số theo bin, nên g_K có thể nhảy xuống ở mép bin."""
    grid = np.linspace(float(np.min(p_oof)), float(np.max(p_oof)), 2001)
    out = {}
    for K in Ks:
        kk = kkey(K)
        gg = r1.predict(grid, wfun=dl.step(K, K, lo, hi))
        dg = np.diff(gg)
        out[kk] = {"spearman": float(spearmanr(p_te, g[kk])[0]),
                   "grid_n_decrease": int(np.sum(dg < -1e-9)),
                   "grid_max_drop": float(max(0.0, -float(dg.min())))}
    return out


def pi_block(y_te, p_te, S_te, lo, hi, groups_te):
    lo_b, hi_b = pi_bounds(S_te)
    reg = {"Low tail": y_te < lo, "Middle": (y_te >= lo) & (y_te < hi), "High tail": y_te >= hi}
    edges = np.quantile(p_te, np.linspace(0, 1, 11))
    dec = np.searchsorted(edges[1:-1], p_te, side="right")
    out = {"level": gates.E9_PI_LEVEL, "overall": cover(y_te, lo_b, hi_b),
           "by_region": {k: cover(y_te, lo_b, hi_b, m) for k, m in reg.items()},
           "by_yhat_decile": [cover(y_te, lo_b, hi_b, dec == i) for i in range(10)],
           "by_group": {}}
    for name, v in groups_te.items():
        out["by_group"][name] = {str(int(u)): cover(y_te, lo_b, hi_b, v == u) for u in np.unique(v)}
    return out


def analyze_split(seed, d, F, center, thresholds, Ks, groups=None):
    """(a), (b), (c), (e) cho một lần chia, từ npz E1 của trung tâm `center`."""
    if center not in d["oof"]:
        raise KeyError(f"npz lần chia {seed} không có trung tâm '{center}'; có {sorted(d['oof'])}")
    y_tr, y_te = np.asarray(d["y_tr"], dtype=float), np.asarray(d["y_te"], dtype=float)
    p_oof, p_te = np.asarray(d["oof"][center], dtype=float), np.asarray(d["test"][center], dtype=float)
    lo, hi = dl.tail_cutoffs(y_tr)
    integer_y = bool(np.all(np.mod(y_tr, 1) == 0))
    r1 = dl.make_rule("R1").fit(p_oof, y_tr)
    S_te = r1.samples(p_te)
    g = {kkey(K): r1.predict(p_te, wfun=dl.step(K, K, lo, hi)) for K in Ks}
    g_tr = None if groups is None else np.asarray(groups)[np.asarray(d["idx_tr"])]
    S_cf = crossfit_r1_samples(p_oof, y_tr, seed, groups=g_tr)

    out = {"lo": lo, "hi": hi, "n_tr": int(len(y_tr)), "n_te": int(len(y_te)),
           "integer_y": integer_y, "r1": r1.info(), "thresholds": {}}
    for direction, c in thresholds:
        out["thresholds"][thr_key(direction, c)] = threshold_block(
            direction, c, Ks, y_tr, y_te, p_te, g, S_te, S_cf, integer_y)
    out["monotone"] = monotone_block(r1, p_oof, p_te, g, Ks, lo, hi)
    g_te = {name: preds_io.rows(d, "te", name, frame=F) for name in GROUPS_PI}
    g_te["prov_size_t3"] = province_size_tertile(F.prov_code)[np.asarray(d["idx_te"])]
    out["pi90"] = pi_block(y_te, p_te, S_te, lo, hi, g_te)
    g1 = g[kkey(1)] if kkey(1) in g else r1.predict(p_te)
    out["calib"] = {"R0": calib_stats(y_te, p_te), "R1_1": calib_stats(y_te, g1)}
    out["rmse"] = {"R0": float(np.sqrt(np.mean((y_te - p_te) ** 2))),
                   "R1_1": float(np.sqrt(np.mean((y_te - g1) ** 2)))}
    return out


# ---------------------------------------------------------------------------
# (d): gần đoán mò
# ---------------------------------------------------------------------------
def eval_contrasts(preds, y_tr, y_te, lo, hi, primary, bstar, r8name, te_mask=None,
                   Ks=gates.K_GRID):
    """cost_K theo quy tắc, C1..C3 và bảng phân rã (mục 6.5) trên một tập kiểm tra.

    preds: {trung tâm: (oof, test)} cùng tr/te. te_mask chọn tập con của test (dùng
    cho nhánh "khớp trên đầy đủ, chấm trên tập đã bỏ y <= 37"). Trung tâm thiếu thì ô
    tương ứng là None."""
    m = np.ones(len(y_te), dtype=bool) if te_mask is None else np.asarray(te_mask, dtype=bool)
    yt = np.asarray(y_te, dtype=float)[m]
    Kall = sorted(set(Ks) | {gates.PRIMARY_K})

    def r1_fit(c):
        return dl.make_rule("R1").fit(preds[c][0], y_tr) if c in preds else None

    p_oof, p_te = preds[primary][0], np.asarray(preds[primary][1])[m]
    r1 = r1_fit(primary)
    r1b = r1_fit(bstar)
    r1r = r1_fit("rs_tuned")
    cost, dec = {}, {}
    for K in Kall:
        wf = dl.step(K, K, lo, hi)

        def ck(pred):
            return sp.cost_k(yt, pred, K, K, lo, hi)
        row = {"R0": ck(p_te), "R1": ck(r1.predict(p_te, wfun=wf)), "R1_1": ck(r1.predict(p_te))}
        for code in DECOMP_RULES:
            row[code] = ck(dl.fit_apply(code, p_oof, y_tr, p_te, wf)[0])
        row["R1@bstar"] = ck(r1b.predict(np.asarray(preds[bstar][1])[m], wfun=wf)) if r1b else None
        row["R1@rs_tuned"] = ck(r1r.predict(np.asarray(preds["rs_tuned"][1])[m], wfun=wf)) if r1r else None
        row["R8*"] = (ck(np.asarray(preds[r8name][1])[m])
                      if (r8name in preds and K == gates.PRIMARY_K) else None)
        cost[kkey(K)] = row
        if K in Ks:
            dec[kkey(K)] = {
                "i_center": (row["R1@bstar"] - row["R1"]) if row["R1@bstar"] is not None else None,
                "ii_a_calib_K1": row["R0"] - row["R1_1"],
                "ii_b_tilt": row["R1_1"] - row["R1"],
                "iii": {x: (row[x] - row["R1"]) if row[x] is not None else None
                        for x in DECOMP_RULES + ["R8*"]}}
    c3 = cost[kkey(gates.PRIMARY_K)]
    C = {"C1": (c3["R1"] - c3["R8*"]) if c3["R8*"] is not None else None,
         "C2": c3["R1"] - c3["R5"],
         "C3": ((c3["R1@rs_tuned"] - c3["R1@bstar"])
                if (c3["R1@rs_tuned"] is not None and c3["R1@bstar"] is not None) else None)}
    return {"n_te": int(m.sum()), "cost": cost, "C": C, "decomp": dec}


def near_guess_split(seed, d, F, fset, a, rs_cfg, r8_cfg, n_jobs, groups=None):
    """(d) cho một lần chia. Dự đoán khớp lại lưu ở preds_dir/split<seed>_nearguess.npz
    (cùng dấu thì dùng lại, không khớp lại)."""
    y = F.y
    tr, te = np.asarray(d["idx_tr"]), np.asarray(d["idx_te"])
    plan = splits.fold_plan(tr, seed, groups)
    if "fold_of" in d and not np.array_equal(splits.fold_of(plan, len(tr)), np.asarray(d["fold_of"])):
        raise ValueError(f"lần chia {seed}: fold_plan không khớp fold_of của npz E1 "
                         "(khác nhóm --grouped, hay E1 chia fold khác?)")
    lo, hi = dl.tail_cutoffs(y[tr])                      # lựa chọn 8: giữ cho cả hai nhánh
    keep_tr, keep_te = y[tr] > gates.NEAR_GUESS_MAX, y[te] > gates.NEAR_GUESS_MAX
    primary, bstar = a["center"], a["bstar"]
    r8name = gates.r8_star(primary)
    specs = {}
    for c in dict.fromkeys([primary, bstar, "rs_tuned"]):
        cfg = rs_cfg if c.startswith("rs_tuned") else None
        if c.startswith("rs_tuned") and cfg is None:
            continue
        specs[c] = center_members(c, cfg, a["smoke"]) + (None,)
    if r8_cfg is not None:
        specs[r8name] = center_members(r8name, r8_cfg, a["smoke"]) + (
            dl.step(gates.PRIMARY_K, gates.PRIMARY_K, lo, hi),)
    max_trees = SMOKE_TREES if a["smoke"] else gates.MAX_TREES
    es_rounds = SMOKE_ES if a["smoke"] else gates.EARLY_STOP_ROUNDS

    fp = preds_io.fingerprint({"script": EXP, "part": "near_guess", "seed": int(seed),
                               "fset": fset, "specs": {k: v[:2] for k, v in specs.items()},
                               "in_sha256": a["in_sha256"][str(seed)], "smoke": a["smoke"],
                               "max_trees": max_trees, "es_rounds": es_rounds,
                               "near_guess_max": gates.NEAR_GUESS_MAX, "lo": lo, "hi": hi})
    path = preds_io.split_path(a["preds_dir"], seed, tag="nearguess")
    cached = preds_io.load_or_none(path, fp, on_mismatch=a["on_mismatch"])
    variants = {"full": (tr, te, plan), "excl": (tr[keep_tr], te[keep_te], subset_plan(plan, keep_tr))}
    preds, fitinfo = {}, {}
    for v, (t_r, t_e, pl) in variants.items():
        preds[v], fitinfo[v] = {}, {}
        for c, (members, es, wf) in specs.items():
            if cached is not None:
                preds[v][c] = (cached[f"oof_{v}"][c], cached[f"test_{v}"][c])
                continue
            oof, test, info = fit_center(F, fset, t_r, t_e, pl, members, es, n_jobs, wfun=wf,
                                         max_trees=max_trees, es_rounds=es_rounds)
            preds[v][c], fitinfo[v][c] = (oof, test), info
    if cached is None:
        arrays = {"idx_tr": tr, "idx_te": te, "keep_tr": keep_tr, "keep_te": keep_te,
                  "meta": {"seed": int(seed), "fingerprint": fp, "fit": fitinfo,
                           "provenance": provenance.stamp()}}
        for v in variants:
            arrays[f"oof_{v}"] = {c: p[0] for c, p in preds[v].items()}
            arrays[f"test_{v}"] = {c: p[1] for c, p in preds[v].items()}
        preds_io.save_split(path, **arrays)
    else:
        fitinfo = (cached.get("meta") or {}).get("fit", {})

    y_tr_f, y_te_f = y[tr], y[te]
    out = {"lo": lo, "hi": hi, "n_removed_tr": int((~keep_tr).sum()), "n_removed_te": int((~keep_te).sum()),
           "frac_removed": float((~keep_tr).sum() + (~keep_te).sum()) / (len(tr) + len(te)),
           "centers": list(specs), "r8_name": r8name if r8name in specs else None, "fit": fitinfo,
           "variants": {
               "full": eval_contrasts(preds["full"], y_tr_f, y_te_f, lo, hi, primary, bstar, r8name),
               "full_on_excl_test": eval_contrasts(preds["full"], y_tr_f, y_te_f, lo, hi, primary, bstar,
                                                   r8name, te_mask=keep_te),
               "excl": eval_contrasts(preds["excl"], y_tr_f[keep_tr], y_te_f[keep_te], lo, hi, primary,
                                      bstar, r8name)}}
    npz_preds = {c: (d["oof"][c], d["test"][c]) for c in dict.fromkeys([primary, bstar, "rs_tuned"])
                 if c in d["oof"]}
    if primary in npz_preds:
        out["npz_full"] = eval_contrasts(npz_preds, y_tr_f, y_te_f, lo, hi, primary, bstar, r8name)["C"]
    return out


# ---------------------------------------------------------------------------
# Một lần chia (chạy trong tiến trình joblib)
# ---------------------------------------------------------------------------
def run_split(seed, inp, a):
    t0 = time.time()
    F = load_frame_cached(a["data"])
    groups = record_groups(a["data"]) if a["grouped"] else None
    d = preds_io.load_split(inp["npz"])
    err = preds_io.validate_split(d)
    if err:
        raise ValueError(f"{inp['npz']}: {err}")
    if preds_io.file_sha256(inp["npz"]) != inp["npz_sha256"]:
        raise ValueError(f"{inp['npz']} đổi trong lúc chạy")
    res = analyze_split(seed, d, F, a["center"], a["thresholds"], a["Ks"], groups)
    if a["near_guess"]:
        fset = a["feature_set"] or (d.get("meta") or {}).get("feature_set")
        if fset not in features.FEATURE_SETS:
            raise ValueError(f"không rõ tập đặc trưng của npz lần chia {seed} (--feature-set)")
        res["near_guess"] = near_guess_split(seed, d, F, fset, a, inp["rs_cfg"], inp["r8_cfg"],
                                             a["n_jobs"], groups)
    res["inputs"] = {k: inp[k] for k in ("npz_sha256", "rs_cfg", "rs_src", "r8_cfg", "r8_src")}
    res["seconds"] = round(time.time() - t0, 1)
    return seed, res


# ---------------------------------------------------------------------------
# Tổng hợp qua lần chia, thống kê, cổng
# ---------------------------------------------------------------------------
def _get(obj, *path):
    for k in path:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def _vals(per, seeds, *path):
    out = []
    for s in seeds:
        v = _get(per[s], *path)
        out.append(float("nan") if v is None else float(v))
    return np.array(out)


def summarize(per, thresholds, Ks):
    seeds = sorted(per, key=int)
    light = [{k: v for k, v in per[s].items() if k not in ("inputs",)} for s in seeds]
    S = {"n_splits": len(seeds), "seeds": [int(s) for s in seeds], "mean": aggregate(light)}

    # (a) L(P2) - L(P1): Nadeau-Bengio, TOST ±5, Holm trên lưới c × K
    cells, pv = [], []
    tests = {}
    for direction, c in thresholds:
        tk = thr_key(direction, c)
        tests[tk] = {}
        for K in Ks:
            kk = kkey(K)
            L1 = _vals(per, seeds, "thresholds", tk, "policies", "P1", kk, "L")
            L2 = _vals(per, seeds, "thresholds", tk, "policies", "P2", kk, "L")
            L0 = _vals(per, seeds, "thresholds", tk, "policies", "P0", kk, "L")
            cell = {"P2_minus_P1": sp.nb_ttest(L2 - L1),
                    "tost_P2_P1": sp.tost_nb(L2 - L1, margin=gates.E9_FLAG_EFFECT),
                    "P1_minus_P0": sp.nb_ttest(L1 - L0), "P2_minus_P0": sp.nb_ttest(L2 - L0)}
            tests[tk][kk] = cell
            cells.append((tk, kk))
            pv.append(cell["P2_minus_P1"]["p"])
    for (tk, kk), ph in zip(cells, sp.holm(pv)):
        tests[tk][kk]["P2_minus_P1"]["p_holm"] = ph
    S["L_tests"] = tests

    G = {"P2_recommended": {}, "P2_P1_equivalent": {}}
    for direction, c in thresholds:
        tk = thr_key(direction, c)
        wins, eq = [], {}
        for K in Ks:
            if K <= 1:
                continue
            cell = tests[tk][kkey(K)]
            r = cell["P2_minus_P1"]
            ph = r.get("p_holm", float("nan"))
            if np.isfinite(r["mean"]) and r["mean"] <= -gates.E9_FLAG_EFFECT and np.isfinite(ph) and ph < gates.ALPHA:
                wins.append(kkey(K))
            eq[kkey(K)] = bool(cell["tost_P2_P1"]["passed"])
        G["P2_recommended"][tk] = {"passed": len(wins) >= gates.E9_MIN_K_WINS, "K_wins": wins,
                                   "need": gates.E9_MIN_K_WINS}
        G["P2_P1_equivalent"][tk] = {"per_K": eq, "all_K_gt1": bool(eq) and all(eq.values())}

    rho = np.array([[_get(per[s], "monotone", kkey(K), "spearman") for K in Ks] for s in seeds], dtype=float)
    G["spearman"] = {"min": float(np.nanmin(rho)) if rho.size else float("nan"),
                     "min_by_K": {kkey(K): float(np.nanmin(rho[:, i])) for i, K in enumerate(Ks)},
                     "threshold": gates.E9_SPEARMAN_MIN}
    G["spearman"]["passed"] = bool(G["spearman"]["min"] >= gates.E9_SPEARMAN_MIN)

    dec_cov = []
    for i in range(10):
        v = [(_get(per[s], "pi90", "by_yhat_decile") or [{}] * 10)[i].get("cover") for s in seeds]
        dec_cov.append(float(np.nanmean([np.nan if x is None else x for x in v])))
    tail_cov = {k: float(np.nanmean(_vals(per, seeds, "pi90", "by_region", k, "cover")))
                for k in ("Low tail", "High tail")}
    G["pi90_individual_use"] = {
        "decile_cover_mean": dec_cov, "tail_cover_mean": tail_cov,
        "decile_min_needed": gates.E9_PI_COVER_DECILE_MIN, "tail_min_needed": gates.E9_PI_COVER_TAIL_MIN,
        "passed": bool(min(dec_cov) >= gates.E9_PI_COVER_DECILE_MIN
                       and min(tail_cov.values()) >= gates.E9_PI_COVER_TAIL_MIN)}

    if any("near_guess" in per[s] for s in seeds):
        ng = {}
        for v in ("full", "full_on_excl_test", "excl"):
            ng[v] = {C: sp.nb_ttest(_vals(per, seeds, "near_guess", "variants", v, "C", C))
                     for C in gates.PRIMARY_CONTRASTS}
        ng["npz_full"] = {C: sp.nb_ttest(_vals(per, seeds, "near_guess", "npz_full", C))
                          for C in gates.PRIMARY_CONTRASTS}
        ng["excl_primary_holm"] = sp.primary_holm({C: ng["excl"][C]["p"] for C in gates.PRIMARY_CONTRASTS})
        flips = {}
        for C in gates.PRIMARY_CONTRASTS:
            mf, me = ng["full"][C]["mean"], ng["excl"][C]["mean"]
            flips[C] = (None if not (np.isfinite(mf) and np.isfinite(me))
                        else bool(np.sign(mf) != np.sign(me)))
        S["near_guess_tests"] = ng
        G["near_guess_sign_change"] = flips
    S["gates"] = G
    return S


def print_summary(S, thresholds, Ks):
    print("\n[a] L_K trên 1.000 (trung bình qua lần chia), P2 - P1 với p Holm:")
    for direction, c in thresholds:
        tk = thr_key(direction, c)
        for K in Ks:
            kk = kkey(K)
            m = S["mean"]["thresholds"][tk]["policies"]
            t = S["L_tests"][tk][kk]["P2_minus_P1"]
            print(f"  {tk:>6} K={kk:>2}: P0 {m['P0'][kk]['L']['mean']:7.1f}  P1 {m['P1'][kk]['L']['mean']:7.1f}  "
                  f"P2 {m['P2'][kk]['L']['mean']:7.1f}  P2-P1 {t['mean']:+6.1f} (p_holm {t.get('p_holm', float('nan')):.3g})")
    g = S["gates"]
    print(f"[b] Spearman(ŷ, g_K) nhỏ nhất = {g['spearman']['min']:.4f} (qua: {g['spearman']['passed']})")
    pi = g["pi90_individual_use"]
    print(f"[c] PI90: độ phủ thập phân vị nhỏ nhất {min(pi['decile_cover_mean']):.3f}, đuôi "
          + ", ".join(f"{k} {v:.3f}" for k, v in pi["tail_cover_mean"].items()) + f" (qua: {pi['passed']})")
    if "near_guess_tests" in S:
        ng = S["near_guess_tests"]
        print("[d] Gần đoán mò (trung bình C qua lần chia): "
              + "; ".join(f"{C}: đầy đủ {ng['full'][C]['mean']:+.3f}, bỏ y<=37 {ng['excl'][C]['mean']:+.3f}"
                          for C in gates.PRIMARY_CONTRASTS)
              + f"; đổi dấu: {g['near_guess_sign_change']}")


# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="E9: ngưỡng cố định, khoảng dự đoán, gần đoán mò")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--preds-dir", default=PREDS_DIR,
                    help="nơi lưu dự đoán khớp lại của (d); ngoài src/ và results_cost/")
    ap.add_argument("--in-preds", default=IN_PREDS, help="thư mục npz của E1")
    ap.add_argument("--in-tag", default=None, help="tag npz E1: split<seed>_<tag>.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=gates.SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1 lần chia, bag 2, trần 50 cây")
    ap.add_argument("--workers", type=int, default=4, help="số tiến trình joblib (theo lần chia)")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="luồng XGBoost; mặc định cpu_count // workers (kiểm thử ở Mac: 2)")
    ap.add_argument("--center", default="bag10", help="trung tâm chính (cổng G1 của E1)")
    ap.add_argument("--bstar", default="bag10", help="bag B* (cổng B* của E1)")
    ap.add_argument("--feature-set", default=None, choices=features.FEATURE_SETS,
                    help="mặc định lấy từ meta npz E1")
    ap.add_argument("--cutoffs", nargs="*", default=[],
                    help="ngưỡng THÊM: 85 hoặc ge:85 (y >= 85), lt:55 (y < 55)")
    ap.add_argument("--cutoffs-source", default="", help="URL và ngày truy cập của ngưỡng thêm")
    ap.add_argument("--near-guess", action="store_true", help="chạy (d): khớp lại khi bỏ y <= 37")
    ap.add_argument("--wtrain-json", default=WTRAIN_JSON, help="JSON E2b (cấu hình R8 ở K = 3)")
    ap.add_argument("--grouped", action="store_true",
                    help="chia theo nhóm bản ghi trùng (khi E0b tìm thấy dòng trùng)")
    ap.add_argument("--on-mismatch", choices=["raise", "recompute"], default="raise")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    guard_real_data_on_mac(a.data)
    provenance.print_versions()
    seeds = a.seeds[:1] if a.smoke else a.seeds
    thresholds = all_thresholds(a.cutoffs)
    Ks = list(gates.E9_KS)
    n_jobs = a.n_jobs or splits.xgb_threads(a.workers)

    inputs = {}
    for s in seeds:
        p = preds_io.split_path(a.in_preds, s, a.in_tag)
        if not os.path.exists(p):
            sys.exit(f"thiếu npz E1: {p}")
        d = preds_io.load_split(p)
        rs_cfg, rs_src = rs_tuned_config(d)
        r8_cfg, r8_src = r8_config(a.wtrain_json, s)
        if r8_cfg is None and a.smoke and rs_cfg is not None:
            # Chỉ để nhánh C1 chạy qua trong kiểm thử khói; không bao giờ ở lượt thật.
            r8_cfg, r8_src = rs_cfg, f"SMOKE: thay bằng cấu hình rs_tuned ({r8_src})"
        inputs[str(s)] = {"npz": p, "npz_sha256": preds_io.file_sha256(p),
                          "rs_cfg": rs_cfg if isinstance(rs_src, int) else None,
                          "rs_src": rs_src, "r8_cfg": r8_cfg, "r8_src": r8_src}

    config = {"script": EXP, "center": a.center, "bstar": a.bstar, "feature_set": a.feature_set,
              "in_tag": a.in_tag, "thresholds": thresholds, "Ks": Ks, "smoke": a.smoke,
              "near_guess": a.near_guess, "grouped": a.grouped,
              "data_sha256": preds_io.file_sha256(a.data), "n_bins": gates.N_BINS,
              "n_samples": gates.N_SAMPLES, "rule_folds": gates.RULE_CROSSFIT_FOLDS,
              "near_guess_max": gates.NEAR_GUESS_MAX, "pi_level": gates.E9_PI_LEVEL}
    fp = preds_io.fingerprint(config)
    res = preds_io.load_partial(a.out + ".partial", fp, {"per_split": {}}, on_mismatch=a.on_mismatch)
    for s, r in res["per_split"].items():
        if s in inputs and r.get("inputs", {}).get("npz_sha256") != inputs[s]["npz_sha256"]:
            raise preds_io.FingerprintMismatch(
                f"{a.out}.partial: lần chia {s} tính trên npz E1 khác bản hiện tại; xoá .partial để tính lại")
    pending = [s for s in seeds if str(s) not in res["per_split"]]
    print(f"E9: trung tâm {a.center}, bag B* {a.bstar}, ngưỡng {[thr_key(*t) for t in thresholds]}, "
          f"K {Ks}, (d) {'bật' if a.near_guess else 'tắt'}; {len(pending)}/{len(seeds)} lần chia cần tính, "
          f"{a.workers} tiến trình x {n_jobs} luồng", flush=True)

    shared = {"data": a.data, "center": a.center, "bstar": a.bstar, "feature_set": a.feature_set,
              "thresholds": thresholds, "Ks": Ks, "smoke": a.smoke, "near_guess": a.near_guess,
              "grouped": a.grouped, "preds_dir": a.preds_dir, "n_jobs": n_jobs,
              "on_mismatch": a.on_mismatch, "in_sha256": {s: v["npz_sha256"] for s, v in inputs.items()}}
    jobs = (delayed(run_split)(s, inputs[str(s)], shared) for s in pending)
    for s, r in Parallel(n_jobs=min(a.workers, max(1, len(pending))), return_as="generator_unordered")(jobs):
        res["per_split"][str(s)] = r
        preds_io.dump_json_atomic(sanitize(res), a.out + ".partial")
        th = r["thresholds"]
        print(f"  lần chia {s}: " + "; ".join(
            f"{tk} L3 P0/P1/P2 = {v['policies']['P0']['3']['L']:.1f}/{v['policies']['P1']['3']['L']:.1f}/"
            f"{v['policies']['P2']['3']['L']:.1f}" for tk, v in th.items() if "3" in v["policies"]["P1"])
            + f" ({r['seconds']:.0f}s)", flush=True)

    per = {s: res["per_split"][str(s)] for s in map(str, seeds)}
    S = summarize(per, thresholds, Ks)
    print_summary(S, thresholds, Ks)
    meta = res.get("meta", {})
    meta.update({
        "experiment": "E9", "config": config, "seeds": seeds,
        "cutoffs_source": a.cutoffs_source or None,
        "inputs": {s: {k: v for k, v in inp.items() if k != "npz"} | {"npz": os.path.basename(inp["npz"])}
                   for s, inp in inputs.items()},
        "choices": [
            "P2: p̂ OOF bằng R1 khớp chéo 5-fold trong OOF; isotonic khớp trên toàn OOF, áp cho p̂ test",
            "p̂ thô dùng hiệu chỉnh liên tục c - 0,5 khi y nguyên; P0, P1 so với c",
            "ngưỡng mặc định 60 (y < 60) và 100 (y >= 100); --cutoffs thêm ngưỡng",
            "Holm cho L(P2) - L(P1) trên cả lưới c × K, kể cả K = 1",
            "PI90 = điểm mẫu τ = 0,05 và 0,95 của R1; thập phân vị ŷ trên ŷ test",
            "tam phân vị cỡ tỉnh theo tỉnh (mỗi tỉnh một đơn vị), cỡ = số bản ghi toàn bộ",
            "cổng Spearman dùng min qua lần chia và K; cổng độ phủ dùng trung bình qua lần chia",
            "(d): lo/hi của tập huấn luyện đầy đủ; kế hoạch fold E1 bỏ dòng; khớp lại cả hai nhánh "
            "bằng cùng mã; R8* chỉ ở K = 3; số cây dò = trung vị best_iteration + 1 của 5 fold",
        ],
        "protocol": "hậu kỳ trên npz E1 (trung tâm --center); (d) khớp lại theo giao thức E1/E2b",
        "provenance": provenance.stamp()})
    out = {"meta": meta, "summary": S, "per_split": res["per_split"]}
    preds_io.dump_json_atomic(sanitize(out), a.out)
    print(f"\nĐã ghi {a.out}")
    return out


if __name__ == "__main__":
    main()
