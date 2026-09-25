# -*- coding: utf-8 -*-
"""E2b: huấn luyện có trọng số, chuẩn hoá và dò lại (khung bài 24/9, mục 6.6).

Câu hỏi: tầng hậu kỳ (R1 trên trung tâm khớp bằng squared loss) có còn hơn huấn
luyện có trọng số (R8) khi đối thủ được cơ hội công bằng không? final_compare chỉ so
với wtrain CHƯA dò lại và CHƯA chuẩn hoá (mục 0.3): trọng số thô K làm tổng hessian
lớn gấp nhiều lần so với min_child_weight, reg_lambda đã dò cho w ≡ 1, và số cây cố
định. Ở đây R8 được cùng ngân sách dò với trung tâm của E1, trọng số chuẩn hoá, và
dừng sớm theo đúng chi phí (Nhận xét 4).

Quy trình, cho mỗi lần chia và mỗi K ∈ {2; 3; 5; 8} (họ bậc thang đối xứng):
  1. Lần chia, fold trong và tập dừng sớm là của E1. idx_tr, idx_te đọc từ npz của
     E1 và phải trùng splits.outer_split(n, seed); kế hoạch fold dựng lại bằng
     splits.fold_plan(idx_tr, seed) và phải trùng fold_of trong npz. Lệch là báo lỗi:
     R8 và R1 phải thấy đúng cùng dòng ở mọi bước, nếu không C1 so hai thứ khác nhau.
  2. Cấu hình: bảng meta.trace_params của npz E1 (60 cấu hình, hoặc 40 nếu E0 hạ ngân
     sách), đúng mẫu mà rs_tuned đã dò ("hai bên luôn cùng ngân sách"). Bảng được đối
     chiếu với splits.sample_configs(len, seed); lệch chỉ ghi cờ configs_match_sampler
     vì npz của E1 mới là nguồn sự thật. --n-cfg lấy TIỀN TỐ của bảng: với cùng
     default_rng(seed), tiền tố n cấu hình chính là sample_configs(n, seed).
  3. Trọng số w_K(y) = K ở đuôi, 1 ở giữa, đuôi theo khối lượng của y huấn luyện
     (decision_layer.tail_cutoffs, trùng 60/100 trên khoá này). Không dùng
     tail_weights.weights vì hàm đó cứng 60/100; hai cách trùng nhau trên HSA nhưng
     chỉ cách theo khối lượng mang sang được E11. lo, hi phải trùng meta của npz E1.
  4. Mỗi cấu hình, mỗi fold: khớp trên phần `fit` của fold với sample_weight =
     w_K(y)/mean(w_K(y_fit)). Chọn đơn giản: chuẩn hoá theo đúng các hàng mô hình
     thấy khi khớp (fold huấn luyện trừ tập dừng sớm), vì thứ cần giữ là tổng
     hessian bằng số hàng khớp; chuẩn hoá theo cả fold huấn luyện (gồm tập dừng
     sớm) chỉ lệch một hệ số rất gần 1. Tập dừng sớm nhận trọng số chia cùng hằng
     số (thang không đổi RMSE có trọng số). Dừng sớm 50 vòng, tối đa 3.000 cây,
     eval_metric rmse với sample_weight_eval_set: RMSE có trọng số của XGBoost là
     sqrt(Σ w e² / Σ w), đúng cost_K trên tập dừng sớm. OOF chỉ trên fold giữ lại.
     random_state = 0 như vết cấu hình của E1.
  5. Chọn cấu hình có cost_K OOF nhỏ nhất (trọng số thô, cost không đổi theo thang).
     Khớp lại trên toàn tập huấn luyện với số cây = int(trung vị(best_iteration + 1))
     (best_iteration đánh số từ 0, cùng quy ước trace_n_trees của E1), trọng số
     chuẩn hoá trung bình 1 trên toàn tập huấn luyện. R8 = random_state 0; R8_bag5 =
     trung bình random_state 0..4 (R8 là thành viên 0, không khớp lại lần hai). Tên
     "R8_bag5" giữ nguyên cả khi --smoke dùng bag 2, để mã hậu kỳ và gates.r8_star
     đọc một tên; số thành viên thật ghi ở n_bag.
  6. C1 = cost_K(R1 trên trung tâm c) - cost_K(R8*(c)), R8* = gates.r8_star(c), tính
     cho MỌI trung tâm có trong npz E1 vì trung tâm chính chỉ biết sau cổng G1. R1
     khớp bằng decision_layer trên (ŷ OOF, y) của E1 với cùng w_K rồi áp cho ŷ test,
     như E2, nên cùng số với decomp_rules. --primary-center (khi đã có G1) đánh dấu
     trung tâm của phép so chính. Holm cho C1 ở K = 3 là TẠM: họ chính {C1; C2; C3}
     cố định m = 3 và C2, C3 chưa có nên được coi là p = 1 (stats_paired.primary_holm);
     cổng cuối phải tính lại khi có đủ ba p (p thô lưu ở summary). C1 ở K ∈ {2; 5; 8}
     là thứ cấp, Holm trong bảng của từng trung tâm. Kèm bootstrap cụm theo trường
     trong từng tập kiểm tra và số lần chia có CI loại 0.
  7. Thứ cấp (ghi rõ trong bài): họ prior (λ ∈ tail_prior.LDS_LAMBDAS) và φ (K ∈
     {2; 3; 5; 8}) chỉ chuẩn hoá, KHÔNG dò lại, KHÔNG dừng sớm lại: siêu tham số và
     số cây của trung tâm chính. Mặc định là cấu hình rs_tuned của E1 (meta
     rs_tuned_index, số cây trace_n_trees); --secondary-from bag dùng XGBoost mặc
     định với subsample = colsample_bytree = 0,8, cho trường hợp G1 chọn bag B*. Mỗi
     họ được chấm bằng chi phí của chính nó và so với R1 cùng trọng số; so giữa các
     họ không phải so hiệu năng (F14, Nhận xét 6).

Song song: joblib (loky) trên từng cặp (cấu hình, fold), 300 việc mỗi (lần chia, K);
cấu hình learning_rate nhỏ (nhiều cây nhất) xếp trước để lượt cuối không bị một việc
dài kéo đuôi. Ma trận thiết kế của 5 fold và của tập huấn luyện đầy đủ được dựng MỘT
lần cho mỗi lần chia bằng features.Frame.design (cột trường tính chỉ từ hàng huấn
luyện của từng mô hình, như E1), ghi thành .npy float32 trong một thư mục tạm dưới
--preds-dir và mở bằng memmap trong tiến trình con: truyền mảng 60 MB qua pickle cho
300 việc sẽ tốn hơn chính phép khớp. float32 không đổi số: XGBoost vốn ép đầu vào
về float32 khi dựng DMatrix. XGBoost n_jobs = max(1, cpu // workers) (splits.xgb_threads).

Chạy tiếp: đơn vị là (lần chia, K) và (lần chia, "secondary"). Mỗi đơn vị ghi npz
(preds_dir/split<seed>_K<K>.npz, split<seed>_secondary.npz) TRƯỚC, rồi mục JSON trong
<out>.partial; meta của npz mang sẵn mục JSON, nên job chết giữa hai lần ghi không
phải khớp lại. Mọi mục ghi sha256 của npz E1 đã đọc; chạy tiếp trên npz E1 khác thì
báo lỗi thay vì trộn. Dấu lượt chạy (preds_io.fingerprint) gồm mọi tham số đổi số
của một đơn vị và code_sha256, không gồm danh sách seed, K hay --workers.

Chưa hỗ trợ chia theo nhóm (khoá băm bản ghi, nếu E0b tìm thấy dòng trùng): khi đó
idx_tr của E1 khác outer_split không nhóm và script dừng với thông báo rõ.

Chạy (server): PYTHONPATH=src python -W ignore src/wtrain_tuned.py --workers 8
Kiểm thử khói (Mac, chỉ dữ liệu giả):
  PYTHONPATH=src python3 src/wtrain_tuned.py --smoke --data tests/fixtures/fake_hsa.csv \
      --e1-preds-dir tests/fixtures/preds/decomp --preds-dir <tạm> --out <tạm>/wt.json \
      --workers 2 --threads 1
"""
import argparse
import os
import platform
import shutil
import sys
import tempfile
import time

import numpy as np
from joblib import Parallel, delayed
from xgboost import XGBRegressor

import decision_layer as dl
import features
import preds_io
import provenance
import splits
import stats_paired as sp
from ga_xgb import CV_FOLDS
from gates import (ALPHA, BAG_SUBSAMPLE, CLUSTER_BOOT_B, EARLY_STOP_ROUNDS, K_GRID, MAX_TREES,
                   PRIMARY_K, R8_BAG, SEEDS, SESOI, r8_star)
from preprocess import DATA_PATH
from tail_prior import LDS_LAMBDAS, Relevance

PROTOCOL_VERSION = 1               # tăng khi đổi cách tính mà tham số dòng lệnh không thấy
R8_NAME, R8_BAG_NAME = "R8", "R8_bag5"
REPORT_KS = [1] + list(K_GRID)     # cost_K của mỗi dự đoán ở mọi K (mô tả chéo K)
SMOKE = {"n_cfg": 2, "n_bag": 2, "max_trees": 50, "es_rounds": 10, "boot_B": 200,
         "lambdas": [0.5], "phi_Ks": [PRIMARY_K]}


# ---------------------------------------------------------------------------
# Việc chạy trong tiến trình con (chỉ nhận đường dẫn .npy, không nhận ma trận)
# ---------------------------------------------------------------------------
def _load(path):
    # memmap: các tiến trình con đọc chung trang bộ nhớ đệm của hệ điều hành
    return np.load(path, mmap_mode="r")


def _parallel(workers, jobs):
    """Cùng cỡ pool cho mọi lời gọi: đổi n_jobs giữa các lời gọi làm loky co/giãn pool
    (giết và sinh lại tiến trình con) ở mỗi K. Việc ít hơn pool thì tiến trình thừa
    ngồi không; luồng XGBoost đã tính theo min(workers, số việc)."""
    return Parallel(n_jobs=int(workers), batch_size=1)(jobs)


def _xgb(threads, **kw):
    return XGBRegressor(tree_method="hist", verbosity=0, n_jobs=int(threads), **kw)


def fit_fold(j, fold, cfg, paths, y_fit, w_fit, y_es, w_es, max_trees, es_rounds, threads):
    """Một mô hình fold có trọng số, dừng sớm theo RMSE có trọng số trên tập dừng sớm.
    Trả (j, fold, best_iteration, ŷ trên fold giữ lại, chạm trần?, giây)."""
    t0 = time.time()
    m = _xgb(threads, random_state=0, n_estimators=int(max_trees),
             early_stopping_rounds=int(es_rounds), eval_metric="rmse", **cfg)
    m.fit(_load(paths["fit"]), y_fit, sample_weight=w_fit,
          eval_set=[(_load(paths["es"]), y_es)], sample_weight_eval_set=[w_es], verbose=False)
    p_va = m.predict(_load(paths["va"]))          # predict dùng best_iteration
    capped = m.get_booster().num_boosted_rounds() >= int(max_trees)
    return j, fold, int(m.best_iteration), np.asarray(p_va, dtype=float), bool(capped), time.time() - t0


def fit_full(cfg, n_trees, random_state, path_tr, path_te, y_tr, w_tr, threads):
    """Khớp lại trên toàn tập huấn luyện, số cây cố định; trả (ŷ test, giây).
    n_trees = None: giữ n_estimators mặc định của XGBoost (nhánh bag của họ thứ cấp)."""
    t0 = time.time()
    kw = dict(cfg)
    if n_trees is not None:
        kw["n_estimators"] = int(n_trees)
    m = _xgb(threads, random_state=int(random_state), **kw)
    m.fit(_load(path_tr), y_tr, sample_weight=w_tr)
    return np.asarray(m.predict(_load(path_te)), dtype=float), time.time() - t0


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def ms(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def kish_ratio(w):
    """Cỡ mẫu hiệu dụng Kish chia n: (Σw)² / (n·Σw²) (Nhận xét 4)."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (len(w) * np.sum(w * w)))


def unit_path(preds_dir, seed, unit):
    return preds_io.split_path(preds_dir, seed, f"K{unit}" if unit != "secondary" else unit)


def c1_verdict(mean, p_holm, tost_passed):
    """Cổng E2b (mục 6.6), âm = tầng hậu kỳ tốt hơn."""
    if np.isfinite(mean) and np.isfinite(p_holm) and p_holm < ALPHA:
        if mean <= -SESOI:
            return "post_hoc_better"
        if mean >= SESOI:
            return "wtrain_better"
    if tost_passed:
        return "equivalent"
    return "inconclusive"


def _guard_real_data(path):
    """Không chạy trên dữ liệu thật ở Mac (AGENTS.md, ràng buộc 1). Server là Linux."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real = os.path.realpath(os.path.join(repo, "data", "data_final.csv"))
    if platform.system() == "Darwin" and os.path.realpath(path) == real:
        sys.exit("Không chạy E2b trên data/data_final.csv ở máy Mac: thí nghiệm chỉ chạy ở "
                 "server; kiểm thử dùng tests/fixtures/fake_hsa.csv.")


# ---------------------------------------------------------------------------
# Ngữ cảnh một lần chia: đọc npz E1, đối chiếu, dựng ma trận
# ---------------------------------------------------------------------------
class SplitContext:
    def __init__(self, F, d, seed, args, e1_path, e1_sha):
        self.seed, self.e1_path, self.e1_sha = int(seed), e1_path, e1_sha
        tr, te = np.asarray(d["idx_tr"]), np.asarray(d["idx_te"])
        tr0, te0 = splits.outer_split(F.n, seed)
        if not (np.array_equal(tr, tr0) and np.array_equal(te, te0)):
            raise ValueError(f"{e1_path}: idx_tr/idx_te khác splits.outer_split(n={F.n}, seed={seed}). "
                             "E1 chia theo nhóm (E0b) hoặc dựng từ dữ liệu khác; wtrain_tuned chưa "
                             "hỗ trợ chia theo nhóm.")
        if not np.array_equal(F.y[tr], np.asarray(d["y_tr"], dtype=float)):
            raise ValueError(f"{e1_path}: y_tr khác y của --data: npz E1 dựng từ file dữ liệu khác")
        self.tr, self.te = tr, te
        self.y_tr, self.y_te = F.y[tr], F.y[te]
        self.plan = splits.fold_plan(tr, seed)
        self.fold_of = splits.fold_of(self.plan, len(tr))
        if "fold_of" in d and not np.array_equal(self.fold_of, np.asarray(d["fold_of"])):
            raise ValueError(f"{e1_path}: fold_of khác splits.fold_plan: E1 dùng fold khác")
        meta = d.get("meta") or {}
        table = meta.get("trace_params")
        if not table:
            raise ValueError(f"{e1_path}: không có meta.trace_params (bảng cấu hình của E1); "
                             "cần npz giai đoạn 2 của E1 (--e1-tag)")
        n_cfg = len(table) if args.n_cfg is None else int(args.n_cfg)
        if n_cfg > len(table):
            raise ValueError(f"--n-cfg {n_cfg} lớn hơn bảng cấu hình của E1 ({len(table)})")
        self.table_len = len(table)
        # Bảng đầy đủ giữ riêng: cấu hình rs_tuned của họ thứ cấp có thể nằm ngoài
        # tiền tố --n-cfg (ví dụ --smoke lấy 2 cấu hình đầu).
        self.configs_full = [dict(c) for c in table]
        self.configs = self.configs_full[:n_cfg]
        self.configs_match = self.configs == splits.sample_configs(len(table), seed)[:n_cfg]
        if not self.configs_match:
            print(f"  [cảnh báo] split {seed}: bảng cấu hình của E1 khác splits.sample_configs; "
                  "dùng bảng của E1", flush=True)
        self.fset = meta.get("feature_set") or args.feature_set
        if self.fset is None:
            raise ValueError(f"{e1_path}: meta không có feature_set; truyền --feature-set")
        if args.feature_set and args.feature_set != self.fset:
            raise ValueError(f"--feature-set {args.feature_set} khác tập của E1 ({self.fset})")
        self.lo, self.hi = dl.tail_cutoffs(self.y_tr)
        if "lo" in meta and "hi" in meta and (float(meta["lo"]), float(meta["hi"])) != (self.lo, self.hi):
            raise ValueError(f"{e1_path}: ngưỡng đuôi E1 {meta['lo']}/{meta['hi']} khác "
                             f"tail_cutoffs {self.lo}/{self.hi}")
        # Trung tâm của E1 cho C1 (R1 khớp trên OOF của nó, áp cho ŷ test)
        oof, test = d.get("oof", {}), d.get("test", {})
        self.centers = {c: (np.asarray(oof[c], dtype=float), np.asarray(test[c], dtype=float))
                        for c in oof if c in test}
        self.school_code, self.prov_code = np.asarray(d["school_code"]), np.asarray(d["prov_code"])
        self.school_te = self.school_code[te]
        # Cấu hình của trung tâm rs_tuned (cho họ thứ cấp)
        self.rs_index = meta.get("rs_tuned_index")
        if self.rs_index is None and "trace_oof_rmse" in d:
            self.rs_index = int(np.argmin(np.asarray(d["trace_oof_rmse"])))
        self.rs_n_trees = None
        if self.rs_index is not None:
            if "trace_n_trees" in d:
                self.rs_n_trees = int(np.asarray(d["trace_n_trees"])[self.rs_index])
            elif "trace_best_iter" in d:
                self.rs_n_trees = int(np.median(np.asarray(d["trace_best_iter"])[self.rs_index] + 1))
        self.tmp, self.paths, self.n_features = None, None, None


def write_designs(F, ctx, parent):
    """Ghi ma trận thiết kế float32 của 5 fold (fit, es, va) và của (train, test) vào
    một thư mục tạm; trả {"folds": [{fit, es, va}], "tr", "te"}.

    Cột trường của fold tính từ hàng huấn luyện của fold (fit ∪ es), như E1 và
    tests/make_fake_preds.py; của mô hình khớp lại tính từ toàn tập huấn luyện."""
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"_mm_split{ctx.seed}_", dir=parent)
    tr, te = ctx.tr, ctx.te

    def save(name, X):
        p = os.path.join(tmp, name + ".npy")
        np.save(p, np.ascontiguousarray(X, dtype=np.float32))
        return p

    paths = {"folds": []}
    for f in ctx.plan:
        _, Xfit, Xes, Xva = F.design(ctx.fset, tr[f["train"]], tr[f["fit"]], tr[f["es"]], tr[f["va"]])
        k = f["fold"]
        paths["folds"].append({"fit": save(f"f{k}_fit", Xfit), "es": save(f"f{k}_es", Xes),
                               "va": save(f"f{k}_va", Xva)})
    Xtr, Xte = F.design(ctx.fset, tr, te)
    paths["tr"], paths["te"] = save("tr", Xtr), save("te", Xte)
    ctx.tmp, ctx.paths, ctx.n_features = tmp, paths, int(Xtr.shape[1])
    return paths


def cleanup(ctx):
    if ctx is not None and ctx.tmp and os.path.isdir(ctx.tmp):
        shutil.rmtree(ctx.tmp, ignore_errors=True)
        ctx.tmp = None


# ---------------------------------------------------------------------------
# Chấm điểm
# ---------------------------------------------------------------------------
def step_metrics(y, pred, lo, hi, K):
    return {"region_rmse": sp.region_rmse(y, pred, lo, hi),
            "cost_K": sp.cost_k(y, pred, K, K, lo, hi),
            "cost_by_K": {str(k): sp.cost_k(y, pred, k, k, lo, hi) for k in REPORT_KS}}


def center_contrasts(ctx, wfun, preds, boot_B, K=None):
    """R0, R1 của từng trung tâm E1 và C1 = cost(R1) - cost(R8*) cùng trọng số wfun.

    preds: {"R8": ŷ, "R8_bag5": ŷ}. K cho họ bậc thang (để bootstrap dùng đúng
    trọng số bậc thang); None thì bootstrap nhận trọng số cho sẵn wfun(y_te)."""
    y_tr, y_te = ctx.y_tr, ctx.y_te
    w_te = dl.weights_of(wfun, y_te)
    out = {}
    for c, (p_oof, p_te) in ctx.centers.items():
        p1 = dl.make_rule("R1").fit(p_oof, y_tr, wfun).predict(p_te)
        star = r8_star(c)
        p8 = preds[star]
        kw = ({"K_low": K, "K_high": K} if K is not None else
              {"K_low": 1, "K_high": 1, "weights": w_te})
        boot = sp.cluster_boot_diff(y_te, p1, p8, ctx.school_te, lo=ctx.lo, hi=ctx.hi,
                                    B=boot_B, seed=ctx.seed, **kw)
        r0, r1, r8 = (sp.cost_w(y_te, p, w_te) for p in (p_te, p1, p8))
        out[c] = {"R0": r0, "R1": r1, "r8_star": star, "R8*": r8, "C1": r1 - r8,
                  "R1_region_rmse": sp.region_rmse(y_te, p1, ctx.lo, ctx.hi),
                  "boot": {k: boot[k] for k in ("ci_lo", "ci_hi", "se", "excludes_zero", "B", "n_clusters")}}
    return out


# ---------------------------------------------------------------------------
# Một đơn vị (lần chia, K)
# ---------------------------------------------------------------------------
def run_K(ctx, K, args, fp):
    y_tr, y_te, plan = ctx.y_tr, ctx.y_te, ctx.plan
    n_cfg, n_tr, n_folds = len(ctx.configs), len(y_tr), len(plan)
    wfun = dl.step(K, K, ctx.lo, ctx.hi)
    w_all = wfun(y_tr)

    # (1) Dò: mỗi cấu hình x mỗi fold, learning_rate nhỏ trước
    order = sorted(range(n_cfg), key=lambda j: ctx.configs[j]["learning_rate"])
    threads = args.threads or splits.xgb_threads(min(args.workers, n_cfg * n_folds))
    jobs = []
    for j in order:
        for f in plan:
            fit, es = f["fit"], f["es"]
            c = float(w_all[fit].mean())
            jobs.append(delayed(fit_fold)(j, f["fold"], ctx.configs[j], ctx.paths["folds"][f["fold"]],
                                          y_tr[fit], w_all[fit] / c, y_tr[es], w_all[es] / c,
                                          args.max_trees, args.es_rounds, threads))
    t0 = time.time()
    out = _parallel(args.workers, jobs)
    tune_s = time.time() - t0
    oof = np.full((n_cfg, n_tr), np.nan)
    best = np.zeros((n_cfg, n_folds), dtype=np.int32)
    capped = np.zeros((n_cfg, n_folds), dtype=bool)
    cpu_s = 0.0
    for j, k, bi, p_va, cap, secs in out:
        oof[j, plan[k]["va"]] = p_va
        best[j, k], capped[j, k] = bi, cap
        cpu_s += secs
    assert np.all(np.isfinite(oof)), "OOF còn NaN: có vị trí không thuộc fold giữ lại nào"
    oof_cost = np.array([sp.cost_w(y_tr, oof[j], w_all) for j in range(n_cfg)])
    oof_rmse = np.sqrt(np.mean((y_tr[None, :] - oof) ** 2, axis=1))
    n_trees = np.median(best + 1, axis=1).astype(np.int32)     # số cây, không phải chỉ số
    jb = int(np.argmin(oof_cost))

    # (2) Khớp lại cấu hình được chọn, trọng số chuẩn hoá trên toàn tập huấn luyện
    w_tr = w_all / w_all.mean()
    rthreads = args.threads or splits.xgb_threads(min(args.workers, args.n_bag))
    t1 = time.time()
    fits = _parallel(args.workers, [
        delayed(fit_full)(ctx.configs[jb], int(n_trees[jb]), b, ctx.paths["tr"], ctx.paths["te"],
                          y_tr, w_tr, rthreads) for b in range(args.n_bag)])
    fit_s = time.time() - t1
    preds = {R8_NAME: fits[0][0], R8_BAG_NAME: np.mean([p for p, _ in fits], axis=0)}

    entry = {
        "K": K, "family": "step", "lo": ctx.lo, "hi": ctx.hi, "feature_set": ctx.fset,
        "e1_npz_sha256": ctx.e1_sha, "n_cfg": n_cfg, "e1_table_len": ctx.table_len,
        "configs_match_sampler": bool(ctx.configs_match),
        "best_index": jb, "best_params": ctx.configs[jb], "best_iter": best[jb].tolist(),
        "n_trees": int(n_trees[jb]), "oof_cost": float(oof_cost[jb]), "oof_rmse": float(oof_rmse[jb]),
        "trace_oof_cost": oof_cost.tolist(), "trace_oof_rmse": oof_rmse.tolist(),
        "trace_n_trees": n_trees.tolist(),
        "capped_fold_fits": int(capped.sum()), "n_fold_fits": int(capped.size),
        "weights": {"train_mean_raw": float(w_all.mean()), "kish_ratio": kish_ratio(w_all)},
        "test": {v: step_metrics(y_te, p, ctx.lo, ctx.hi, K) for v, p in preds.items()},
        "cost_K": {v: sp.cost_k(y_te, p, K, K, ctx.lo, ctx.hi) for v, p in preds.items()},
        "centers": center_contrasts(ctx, wfun, preds, args.boot_B, K=K),
        "tune_s": tune_s, "tune_cpu_s": cpu_s, "fit_s": fit_s,
        "fit_cpu_s": float(sum(s for _, s in fits)), "n_bag": int(args.n_bag),
        "workers": int(args.workers), "xgb_threads": int(threads),
    }
    meta = {"seed": ctx.seed, "K": K, "family": "step", "feature_set": ctx.fset, "lo": ctx.lo,
            "hi": ctx.hi, "trace_params": ctx.configs, "best_index": jb, "n_bag": int(args.n_bag),
            "max_trees": args.max_trees, "es_rounds": args.es_rounds,
            "e1_npz": os.path.abspath(ctx.e1_path), "e1_npz_sha256": ctx.e1_sha,
            "fingerprint": fp, "entry": entry, "provenance": provenance.stamp()}
    arrays = dict(idx_tr=ctx.tr, idx_te=ctx.te, y_tr=y_tr, y_te=y_te, fold_of=ctx.fold_of,
                  school_code=ctx.school_code, prov_code=ctx.prov_code,
                  oof={R8_NAME: oof[jb]}, test=preds, trace_oof_cost=oof_cost,
                  trace_oof_rmse=oof_rmse, trace_best_iter=best, trace_n_trees=n_trees, meta=meta)
    return entry, arrays


# ---------------------------------------------------------------------------
# Họ thứ cấp: prior và φ, chỉ chuẩn hoá, không dò lại
# ---------------------------------------------------------------------------
def secondary_params(ctx, args):
    """(cfg, n_trees, mô tả) của trung tâm chính cho họ thứ cấp."""
    if args.secondary_from == "bag":
        return ({"subsample": BAG_SUBSAMPLE, "colsample_bytree": BAG_SUBSAMPLE}, None,
                "XGBoost mặc định, subsample = colsample_bytree = 0,8 (trung tâm bag)")
    if ctx.rs_index is None or ctx.rs_n_trees is None:
        raise ValueError(f"{ctx.e1_path}: không tìm được rs_tuned (rs_tuned_index / trace_n_trees) "
                         "cho họ thứ cấp; dùng --secondary-from bag hoặc --no-secondary")
    return (dict(ctx.configs_full[ctx.rs_index]), ctx.rs_n_trees,
            f"cấu hình rs_tuned của E1 (chỉ số {ctx.rs_index}), số cây trace_n_trees của E1")


def run_secondary(ctx, args, fp):
    y_tr, y_te = ctx.y_tr, ctx.y_te
    dens, rel = dl.density(y_tr), Relevance(y_tr)
    fams = {"prior": [(lam, dl.prior(lam, dens)) for lam in args.lambdas],
            "phi": [(K, dl.phi(K, rel)) for K in args.phi_Ks]}
    cfg, n_trees, source = secondary_params(ctx, args)
    items = [(fam, prm, wf) for fam, lst in fams.items() for prm, wf in lst]
    n_jobs = len(items) * args.n_bag
    threads = args.threads or splits.xgb_threads(min(args.workers, n_jobs))
    jobs = []
    for fam, prm, wf in items:
        w = wf(y_tr)
        for b in range(args.n_bag):
            jobs.append(delayed(fit_full)(cfg, n_trees, b, ctx.paths["tr"], ctx.paths["te"], y_tr,
                                          w / w.mean(), threads))
    t0 = time.time()
    fits = _parallel(args.workers, jobs)
    fit_s = time.time() - t0
    entry = {"e1_npz_sha256": ctx.e1_sha, "params_source": source, "params": cfg,
             "n_trees": n_trees, "n_bag": int(args.n_bag), "fit_s": fit_s,
             "fit_cpu_s": float(sum(s for _, s in fits)), "families": {}}
    test = {}
    for i, (fam, prm, wf) in enumerate(items):
        ps = [fits[i * args.n_bag + b][0] for b in range(args.n_bag)]
        preds = {R8_NAME: ps[0], R8_BAG_NAME: np.mean(ps, axis=0)}
        w_te = wf(y_te)
        entry["families"].setdefault(fam, {})[str(prm)] = {
            "test": {v: {"region_rmse": sp.region_rmse(y_te, p, ctx.lo, ctx.hi),
                         "cost": sp.cost_w(y_te, p, w_te)} for v, p in preds.items()},
            "weights": {"train_mean_raw": float(wf(y_tr).mean()), "kish_ratio": kish_ratio(wf(y_tr))},
            "centers": center_contrasts(ctx, wf, preds, args.boot_B, K=None)}
        test[f"{fam}_{prm}"] = preds[R8_NAME]
        test[f"{fam}_{prm}_bag"] = preds[R8_BAG_NAME]
    meta = {"seed": ctx.seed, "unit": "secondary", "feature_set": ctx.fset, "lo": ctx.lo, "hi": ctx.hi,
            "params": cfg, "n_trees": n_trees, "params_source": source,
            "e1_npz": os.path.abspath(ctx.e1_path), "e1_npz_sha256": ctx.e1_sha,
            "fingerprint": fp, "entry": entry, "provenance": provenance.stamp()}
    arrays = dict(idx_tr=ctx.tr, idx_te=ctx.te, y_tr=y_tr, y_te=y_te, fold_of=ctx.fold_of,
                  school_code=ctx.school_code, prov_code=ctx.prov_code, test=test, meta=meta)
    return entry, arrays


# ---------------------------------------------------------------------------
# Một lần chia: bỏ qua đơn vị đã xong, khôi phục từ npz, tính phần còn lại
# ---------------------------------------------------------------------------
def _num(s):
    v = float(s)
    return int(v) if v.is_integer() else v


def run_seed(F, seed, args, fp, res):
    e1_path = preds_io.split_path(args.e1_preds_dir, seed, args.e1_tag)
    if not os.path.exists(e1_path):
        raise FileNotFoundError(f"{e1_path}: chưa có npz của E1 cho lần chia {seed} (E2b cần "
                                "lần chia, fold và bảng cấu hình của E1)")
    e1_sha = preds_io.file_sha256(e1_path)
    entry = res["per_split"].setdefault(str(seed), {})
    units = [str(K) for K in args.Ks] + (["secondary"] if args.secondary else [])
    for u in units:
        if u in entry and entry[u].get("e1_npz_sha256") != e1_sha:
            raise preds_io.FingerprintMismatch(
                f"split {seed}, đơn vị {u}: mục trong {args.out}.partial tính trên npz E1 khác "
                f"({str(entry[u].get('e1_npz_sha256'))[:12]} so với {e1_sha[:12]}); E1 đã chạy lại? "
                "Xoá/đổi tên .partial và preds của E2b để tính lại.")
    pending = []
    for u in units:
        if u in entry:
            continue
        p = unit_path(args.preds_dir, seed, u)
        d_u = preds_io.load_or_none(p, fp)
        if d_u is not None:
            m = d_u["meta"]
            if m.get("e1_npz_sha256") != e1_sha:
                raise preds_io.FingerprintMismatch(f"{p}: tính trên npz E1 khác bản hiện tại")
            entry[u] = m["entry"]
            preds_io.dump_json_atomic(res, args.out + ".partial")
            print(f"  split {seed} {u}: khôi phục từ {p}", flush=True)
            continue
        pending.append(u)
    if not pending:
        print(f"  split {seed}: đủ {units} -> bỏ qua", flush=True)
        return
    ctx = SplitContext(F, preds_io.load_split(e1_path), seed, args, e1_path, e1_sha)
    try:
        t0 = time.time()
        write_designs(F, ctx, args.preds_dir)
        print(f"  split {seed}: {ctx.fset}, {ctx.n_features} cột, n_tr={len(ctx.tr)}, n_te={len(ctx.te)}, "
              f"đuôi {ctx.lo:g}/{ctx.hi:g}, {len(ctx.configs)}/{ctx.table_len} cấu hình, trung tâm E1 "
              f"{sorted(ctx.centers)}; ma trận {time.time() - t0:.0f}s", flush=True)
        for u in pending:
            if u == "secondary":
                e, arrays = run_secondary(ctx, args, fp)
                required = ("idx_tr", "idx_te", "y_tr", "y_te", "test")
            else:
                e, arrays = run_K(ctx, _num(u), args, fp)
                required = preds_io.E1_REQUIRED
            p = unit_path(args.preds_dir, seed, u)
            preds_io.save_split(p, **arrays)          # npz trước, JSON sau (xem docstring)
            err = preds_io.validate_split(preds_io.load_split(p), required=required)
            assert not err, f"{p}: {err}"
            entry[u] = e
            preds_io.dump_json_atomic(res, args.out + ".partial")
            _print_unit(seed, u, e)
    finally:
        cleanup(ctx)


def _print_unit(seed, u, e):
    if u == "secondary":
        parts = []
        for fam, d in e["families"].items():
            for prm, v in d.items():
                parts.append(f"{fam}{prm}={v['test'][R8_NAME]['cost']:.3f}")
        print(f"  split {seed} thứ cấp ({e['params_source']}): " + " ".join(parts)
              + f"; khớp {e['fit_s']:.0f}s", flush=True)
        return
    cfg = e["best_params"]
    c1 = " ".join(f"C1[{c}]={v['C1']:+.3f}" for c, v in e["centers"].items())
    print(f"  split {seed} K={u}: cfg {e['best_index']} (lr={cfg['learning_rate']:.3f}, "
          f"depth={cfg['max_depth']}), {e['n_trees']} cây, OOF cost {e['oof_cost']:.3f}; test "
          f"R8={e['cost_K'][R8_NAME]:.3f} {R8_BAG_NAME}={e['cost_K'][R8_BAG_NAME]:.3f}; {c1}; "
          f"dò {e['tune_s']:.0f}s, khớp lại {e['fit_s']:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Tổng hợp
# ---------------------------------------------------------------------------
REGIONS = ["Low tail", "Middle", "High tail", "Tails", "All"]


def _units(per, u):
    return {s: v[u] for s, v in per.items() if u in v}


def _contrast_table(rows_by_key, roles):
    """rows_by_key: {khoá (K hoặc tham số): [(seed, mục trung tâm)]}. Trả bảng C1 có
    nb, TOST, bootstrap; Holm trong bảng trên các khoá vai 'secondary'."""
    tab = {}
    for key, rows in rows_by_key.items():
        if not rows:
            continue
        st = sp.paired([x["C1"] for _, x in rows])
        tab[key] = {"role": roles.get(key, "secondary"), "r8_star": rows[0][1]["r8_star"],
                    "R0": ms([x["R0"] for _, x in rows]), "R1": ms([x["R1"] for _, x in rows]),
                    "R8*": ms([x["R8*"] for _, x in rows]), "nb": st["nb"], "tost": st["tost"],
                    "boot_excludes_zero": int(sum(bool(x["boot"]["excludes_zero"]) for _, x in rows)),
                    "n_splits": len(rows), "diffs": {s: x["C1"] for s, x in rows}}
    sec = [k for k in tab if tab[k]["role"] == "secondary"]
    for k, a in zip(sec, sp.holm([tab[k]["nb"]["p"] for k in sec])):
        tab[k]["p_holm_table"] = a
    return tab


def summarize(per, Ks, primary_center=None):
    out = {"by_K": {}, "C1": {}, "gate": {}, "secondary": {}, "compute": {}}
    for K in Ks:
        u = _units(per, str(K))
        if not u:
            continue
        rows = list(u.values())
        bk = {v: {"cost_K": ms([r["test"][v]["cost_K"] for r in rows]),
                  "region_rmse": {g: ms([r["test"][v]["region_rmse"][g] for r in rows]) for g in REGIONS}}
              for v in (R8_NAME, R8_BAG_NAME)}
        bk["bag_minus_single"] = sp.nb_ttest([r["cost_K"][R8_BAG_NAME] - r["cost_K"][R8_NAME] for r in rows])
        bk["oof_cost"] = ms([r["oof_cost"] for r in rows])
        bk["n_trees"] = ms([r["n_trees"] for r in rows])
        bk["best_index"] = {s: r["best_index"] for s, r in u.items()}
        bk["capped_share"] = (sum(r["capped_fold_fits"] for r in rows) / max(1, sum(r["n_fold_fits"] for r in rows)))
        bk["kish_ratio"] = ms([r["weights"]["kish_ratio"] for r in rows])
        bk["tune_s"] = ms([r["tune_s"] for r in rows])
        bk["fit_s"] = ms([r["fit_s"] for r in rows])
        bk["n_splits"] = len(rows)
        out["by_K"][str(K)] = bk

    centers = sorted({c for s in per.values() for k, e in s.items() if k != "secondary"
                      for c in e.get("centers", {})})
    roles = {str(K): ("primary" if K == PRIMARY_K else "secondary") for K in Ks}
    for c in centers:
        by = {str(K): [(s, r["centers"][c]) for s, r in _units(per, str(K)).items() if c in r["centers"]]
              for K in Ks}
        tab = _contrast_table(by, roles)
        pk = str(PRIMARY_K)
        if pk in tab:
            # Họ chính m = 3 cố định; C2, C3 chưa có coi là p = 1 -> Holm TẠM (bảo thủ)
            ph = sp.primary_holm({"C1": tab[pk]["nb"]["p"]})["C1"]
            tab[pk]["p_holm_provisional"] = ph["p_holm"]
            tab[pk]["verdict_provisional"] = c1_verdict(tab[pk]["nb"]["mean"], ph["p_holm"],
                                                        tab[pk]["tost"]["passed"])
        out["C1"][c] = tab

    note = ("Holm tạm: họ {C1; C2; C3} m = 3, C2 và C3 coi là p = 1 cho tới khi E1, E2 xong; "
            "cổng cuối tính lại bằng stats_paired.primary_holm với đủ ba p thô")
    g = out["C1"].get(primary_center, {}).get(str(PRIMARY_K)) if primary_center else None
    if g:
        out["gate"] = {"primary_center": primary_center, "r8_star": g["r8_star"], "K": PRIMARY_K,
                       "mean": g["nb"]["mean"], "ci": [g["nb"]["ci_lo"], g["nb"]["ci_hi"]],
                       "p": g["nb"]["p"], "p_holm_provisional": g["p_holm_provisional"],
                       "tost_passed": g["tost"]["passed"], "tost_p": g["tost"]["p"],
                       "boot_excludes_zero": g["boot_excludes_zero"], "n_splits": g["n_splits"],
                       "verdict_provisional": g["verdict_provisional"], "note": note}
    else:
        out["gate"] = {"primary_center": primary_center, "note": "chưa có trung tâm chính (cổng G1) "
                       "hoặc trung tâm đó không có trong npz E1; xem C1 theo từng trung tâm. " + note}

    sec = _units(per, "secondary")
    fams = {}
    for s, e in sec.items():
        for fam, d in e["families"].items():
            for prm, v in d.items():
                fams.setdefault(fam, {}).setdefault(prm, []).append((s, v))
    for fam, d in fams.items():
        fs = {"label": "thứ cấp: chỉ chuẩn hoá, không dò lại; chấm bằng chi phí của chính họ",
              "params": {}, "C1": {}}
        for prm, rows in d.items():
            fs["params"][prm] = {v: {"cost": ms([x["test"][v]["cost"] for _, x in rows]),
                                     "All": ms([x["test"][v]["region_rmse"]["All"] for _, x in rows])}
                                 for v in (R8_NAME, R8_BAG_NAME)}
        cs = sorted({c for rows in d.values() for _, x in rows for c in x["centers"]})
        for c in cs:
            by = {prm: [(s, x["centers"][c]) for s, x in rows if c in x["centers"]] for prm, rows in d.items()}
            fs["C1"][c] = _contrast_table(by, {})
        out["secondary"][fam] = fs

    k_units = [e for s in per.values() for k, e in s.items() if k != "secondary"]
    wall = sum(e["tune_s"] + e["fit_s"] for e in k_units) + sum(e["fit_s"] for e in sec.values())
    cpu = (sum(e["tune_cpu_s"] + e["fit_cpu_s"] for e in k_units)
           + sum(e["fit_cpu_s"] for e in sec.values()))
    out["compute"] = {"wall_h": wall / 3600, "cpu_h": cpu / 3600, "n_units_K": len(k_units),
                      "wall_s_per_split_K": ms([e["tune_s"] + e["fit_s"] for e in k_units]),
                      "note": "R8 cần n_cfg × CV_FOLDS lần khớp fold + n_bag lần khớp lại cho MỖI K; "
                              "tầng hậu kỳ dùng một trung tâm cho mọi K (thời gian trung tâm ở E1)"}
    return out


def print_summary(S):
    f = lambda d: f"{d['mean']:.3f}±{d['sd']:.3f}"      # noqa: E731
    print("\n[1] R8 theo K (test, mean±sd qua lần chia)")
    print("  K   cost_K R8      cost_K R8_bag5  OOF cost       cây     chạm trần  dò (s)   khớp lại (s)")
    for K, b in S["by_K"].items():
        print(f"  {K:3s} {f(b[R8_NAME]['cost_K']):14s} {f(b[R8_BAG_NAME]['cost_K']):15s} {f(b['oof_cost']):14s} "
              f"{b['n_trees']['mean']:7.0f} {b['capped_share']:9.1%}  {b['tune_s']['mean']:7.0f}  {b['fit_s']['mean']:7.0f}")
    print("\n[2] C1 = cost_K(R1, trung tâm) - cost_K(R8*) (âm = tầng hậu kỳ tốt hơn; Nadeau-Bengio)")
    for c, tab in S["C1"].items():
        for K, t in tab.items():
            nb = t["nb"]
            hol = (f"Holm tạm {t['p_holm_provisional']:.3g} -> {t['verdict_provisional']}"
                   if "p_holm_provisional" in t else f"Holm bảng {t.get('p_holm_table', float('nan')):.3g}")
            print(f"  {c:14s} vs {t['r8_star']:8s} K={K:3s} {nb['mean']:+.3f} [{nb['ci_lo']:+.3f}; {nb['ci_hi']:+.3f}] "
                  f"p={nb['p']:.3g} {hol}; TOST {'qua' if t['tost']['passed'] else 'không qua'}; "
                  f"boot loại 0 ở {t['boot_excludes_zero']}/{t['n_splits']}")
    g = S["gate"]
    print(f"\n[3] Cổng C1: {g.get('verdict_provisional', '-')} (trung tâm chính: {g.get('primary_center')})")
    for fam, fs in S["secondary"].items():
        print(f"\n[4] Thứ cấp, họ {fam} (chi phí của chính họ, R8 một mô hình / bag):")
        for prm, v in fs["params"].items():
            print(f"  {prm:6s} {f(v[R8_NAME]['cost'])} / {f(v[R8_BAG_NAME]['cost'])}")
    c = S["compute"]
    print(f"\n[5] Thời gian: {c['wall_h']:.2f} giờ đồng hồ, {c['cpu_h']:.2f} giờ CPU "
          f"({c['n_units_K']} đơn vị (lần chia, K))")


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="E2b: huấn luyện có trọng số, chuẩn hoá, dò lại (mục 6.6)")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default="results_cost/wtrain_tuned.json")
    ap.add_argument("--preds-dir", default=os.path.join("preds", "wtrain"),
                    help="npz dự đoán R8 (ngoài src/ và results_cost/; trên server ~/ga4xgb/preds/wtrain)")
    ap.add_argument("--e1-preds-dir", default=os.path.join("preds", "decomp"),
                    help="npz của E1 (lần chia, fold, bảng cấu hình, OOF/test của các trung tâm)")
    ap.add_argument("--e1-tag", default=None, help="tag của npz E1: split<seed>_<tag>.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--Ks", type=_num, nargs="+", default=K_GRID)
    ap.add_argument("--n-cfg", type=int, default=None,
                    help="số cấu hình (tiền tố bảng E1); mặc định cả bảng, đúng ngân sách của E1")
    ap.add_argument("--max-trees", type=int, default=MAX_TREES)
    ap.add_argument("--es-rounds", type=int, default=EARLY_STOP_ROUNDS)
    ap.add_argument("--n-bag", type=int, default=R8_BAG)
    ap.add_argument("--boot-B", type=int, default=CLUSTER_BOOT_B)
    ap.add_argument("--feature-set", default=None, choices=features.FEATURE_SETS,
                    help="mặc định lấy từ meta của npz E1; truyền vào thì phải khớp")
    ap.add_argument("--primary-center", default=None,
                    help="trung tâm chính sau cổng G1 (ví dụ rs_tuned_bag5), để đánh dấu C1 chính")
    ap.add_argument("--secondary-from", default="rs_tuned", choices=["rs_tuned", "bag"],
                    help="siêu tham số cho họ prior/φ: rs_tuned của E1, hoặc bag mặc định 0,8")
    ap.add_argument("--no-secondary", dest="secondary", action="store_false")
    ap.add_argument("--workers", type=int, default=1, help="số tiến trình joblib")
    ap.add_argument("--threads", type=int, default=None,
                    help="luồng XGBoost mỗi mô hình; mặc định max(1, cpu // workers)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 lần chia, 2 cấu hình, bag 2, tối đa 50 cây: chỉ để kiểm mã")
    args = ap.parse_args(argv)
    args.lambdas, args.phi_Ks = list(LDS_LAMBDAS), list(K_GRID)
    if args.smoke:
        args.seeds = args.seeds[:1]
        args.n_cfg = SMOKE["n_cfg"] if args.n_cfg is None else args.n_cfg
        args.n_bag = min(args.n_bag, SMOKE["n_bag"])
        args.max_trees = min(args.max_trees, SMOKE["max_trees"])
        args.es_rounds = min(args.es_rounds, SMOKE["es_rounds"])
        args.boot_B = min(args.boot_B, SMOKE["boot_B"])
        args.lambdas, args.phi_Ks = list(SMOKE["lambdas"]), list(SMOKE["phi_Ks"])
    if args.workers < 1 or args.n_bag < 1:
        ap.error("--workers và --n-bag phải >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    _guard_real_data(args.data)
    provenance.print_versions()
    # Mọi thứ làm đổi số của MỘT đơn vị; không có seed, K, workers, đường dẫn.
    config = {"script": "wtrain_tuned", "protocol_version": PROTOCOL_VERSION,
              "data_sha256": preds_io.file_sha256(args.data), "e1_tag": args.e1_tag,
              "feature_set": args.feature_set, "n_cfg": args.n_cfg, "max_trees": args.max_trees,
              "es_rounds": args.es_rounds, "n_bag": args.n_bag, "boot_B": args.boot_B,
              "lambdas": args.lambdas, "phi_Ks": args.phi_Ks, "secondary_from": args.secondary_from,
              "smoke": bool(args.smoke), "cv_folds": CV_FOLDS}
    fp = preds_io.fingerprint(config)
    res = preds_io.load_partial(args.out + ".partial", fp, {"per_split": {}})
    F = features.load_frame(args.data)
    print(f"{args.data}: n={F.n}; lần chia {args.seeds}; K {args.Ks}; workers {args.workers}, "
          f"luồng XGBoost {args.threads or splits.xgb_threads(args.workers)}; dấu {fp[:12]}", flush=True)
    t0 = time.time()
    for s in args.seeds:
        run_seed(F, s, args, fp, res)
    per = {str(s): res["per_split"][str(s)] for s in args.seeds if str(s) in res["per_split"]}
    res["summary"] = summarize(per, args.Ks, args.primary_center)
    res["meta"].update({
        "experiment": "E2b", "protocol": (
            "Lần chia, fold, tập dừng sớm và bảng cấu hình của E1; mỗi K (bậc thang đối xứng, đuôi "
            "theo khối lượng) dò lại mọi cấu hình với trọng số chuẩn hoá trung bình 1 trên hàng khớp "
            "của fold, dừng sớm theo RMSE có trọng số (sample_weight_eval_set), chọn theo cost_K OOF, "
            "khớp lại với trung vị best_iteration + 1 cây; R8 = random_state 0, R8_bag5 = trung bình "
            "random_state 0..n_bag-1. C1 = cost_K(R1 trên trung tâm E1) - cost_K(gates.r8_star)."),
        "seeds": args.seeds, "summary_seeds": list(per), "Ks": args.Ks, "primary_K": PRIMARY_K,
        "n_cfg": args.n_cfg, "max_trees": args.max_trees, "es_rounds": args.es_rounds,
        "n_bag": args.n_bag, "boot_B": args.boot_B, "cv_folds": CV_FOLDS,
        "e1_preds_dir": args.e1_preds_dir, "e1_tag": args.e1_tag, "preds_dir": args.preds_dir,
        "primary_center": args.primary_center, "secondary_from": args.secondary_from,
        "secondary_lambdas": args.lambdas, "secondary_phi_Ks": args.phi_Ks,
        "workers": args.workers, "threads": args.threads, "smoke": bool(args.smoke),
        "run_wall_s": time.time() - t0, "config": config, "provenance": provenance.stamp()})
    print_summary(res["summary"])
    preds_io.dump_json_atomic(res, args.out)
    print(f"\nĐã ghi {args.out}", flush=True)
    return res


if __name__ == "__main__":
    main()
