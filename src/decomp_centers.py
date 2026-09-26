# -*- coding: utf-8 -*-
"""E1 (khung bài 24/9, mục 6.4): tập đặc trưng, trung tâm và vết 60 cấu hình.

Mục đích: chọn tập đặc trưng chính theo tính dùng được lúc tư vấn, đo lợi ích trung
tâm từ mốc bag, và dựng vết cấu hình để E4 hỏi "HPO nên tối ưu gì" mà không phải
khớp lại mô hình. Mọi thước đo theo RMSE: cost_K = sqrt(Σ w e² / Σ w), họ bậc thang
đối xứng, đuôi theo khối lượng của y huấn luyện (decision_layer.tail_cutoffs, trên
khoá này trùng 60/100 của src/bins.py).

Giai đoạn 1 (mỗi lần chia, mỗi tập đặc trưng F_full, F_dt, F_dt-cn, F_dt-cn-bc):
  default  XGBRegressor(tree_method="hist", random_state=0), còn lại mặc định.
  bag10    10 mô hình random_state 0..9, subsample = colsample_bytree = 0,8.
  Riêng F_dt: sub1 (một mô hình 0,8, random_state 0) và bag40 (random_state 0..39),
  trung bình tiền tố cho B ∈ {1; 2; 5; 10; 20; 40}.
  Mỗi trung tâm: OOF 5-fold (mô hình fold khớp trên cả fold huấn luyện, không dừng
  sớm) và dự đoán test của mô hình khớp trên toàn tập huấn luyện; áp R0, R1, R1₁ ở
  K ∈ {1; 2; 3; 5; 8}.
Cổng tập đặc trưng (sau giai đoạn 1): tập chính là F_dt-cn nếu TOST(RMSE bag10
  F_dt-cn - F_dt, ±0,10) qua; là F_dt-cn-bc nếu TOST tiếp tục qua khi so F_dt-cn-bc
  với F_dt-cn; không thì giữ tập lớn hơn. B* = B nhỏ nhất có trung bình theo lần chia
  |cost_3(R1, bag B) - cost_3(R1, bag 40)| ≤ 0,02 (tính trên F_dt, nơi có bag40).
  --feature-set bỏ qua cổng.
Giai đoạn 2 (trên tập chính): rs_tuned từ 60 cấu hình splits.sample_configs(60, seed)
  (không gian ga_xgb.BASE_GENES bỏ n_estimators). Mỗi cấu hình: 5 mô hình fold, mỗi
  mô hình dừng sớm 50 vòng trên tập dừng sớm của fold (splits.fold_plan), tối đa
  3.000 cây; RMSE OOF trên fold giữ lại. Chọn RMSE OOF nhỏ nhất (hoà thì chỉ số nhỏ
  nhất). Khớp lại CẢ 60 cấu hình trên toàn tập huấn luyện với số cây
  int(trung vị(best_iteration + 1)) cho oracle của E4. rs_tuned_bag5: cấu hình được
  chọn với random_state 0..4, lấy trung bình.
  OOF mà R1 (và mọi quy tắc hậu kỳ) khớp trên: xem lựa chọn 12.

Các lựa chọn khi khung bài không nói rõ (chọn cách đơn giản, ghi lại ở đây):
 1. Bố cục npz trong --preds-dir (mặc định preds/decomp, ngoài src/ và results_cost/):
    split<seed>_p1_<tập>.npz  giai đoạn 1, một file cho mỗi tập đặc trưng (E10 (d)
                              đọc bag10 trên F_dt, F_dt-cn, F_dt-cn-bc ở đây);
    split<seed>.npz           file GỘP cho tập chính: mọi trung tâm giai đoạn 1 của
                              tập đó + rs_tuned, rs_tuned_bag5 + vết 60 cấu hình.
                              Lược đồ E1 của preds_io, như tests/make_fake_preds.py;
                              E2, E3, E4, E5, E9 đọc file này.
 2. Trên F_dt, bag10 là tiền tố 10 của bag40 và sub1 trùng bag1 (cùng mô hình
    random_state 0): mỗi mô hình chỉ khớp một lần. npz lưu cả sub1 và bag1 để mã hậu
    kỳ lặp được qua "bag B"; meta.identical_centers ghi cặp trùng.
 3. Nếu tập chính không phải F_dt, giai đoạn 2 khớp thêm sub1 và bag40 trên tập chính
    (C3 cần bag B* trên cùng tập, E4 cần {default, sub1, bag 1..40} để tính Kendall τ).
    bag10 của tập đó giữ bản giai đoạn 1 (cổng tập đặc trưng đã dùng nó); tiền tố 10
    của bag40 mới chỉ có thể lệch nó ở mức làm tròn số thực nếu số luồng khác.
 4. B* tính trên F_dt như khung bài rồi dùng cho tập chính. Chỉ khi F_dt chưa có
    giai đoạn 1 (--feature-set --phase 2) thì tính trên tập chính, ghi "fallback".
 5. rs_tuned_bag5: thành viên 0 chính là rs_tuned; thành viên b = 1..4 có mô hình fold
    riêng (cùng tập dừng sớm) và số cây khớp lại riêng theo cùng quy tắc trung vị.
    Vết cấu hình dùng random_state 0.
 6. Cổng G1 ở đây là TẠM: C1 (E2b) và C2 (E2) chưa có nên stats_paired.primary_holm
    coi chúng là p = 1. Holm tạm này không bao giờ nhỏ hơn Holm cuối cùng, nên G1 tạm
    là bảo thủ; tính lại khi đủ ba phép so. Nhánh rs_tuned_bag5 cần thắng cả rs_tuned
    và bag B* (≤ -0,10); Holm trên đúng hai phép so đó (m = 2). Bảng III thứ cấp:
    Holm trên 5 phép so của khung bài, đo bằng cost_3(R1).
 7. fit_s, oof_s, tune_s là giây tính toán của từng lần fit (không gồm predict) ở
    `threads` luồng, cộng lại; không phải giờ đồng hồ. tune_s của rs_tuned là tổng
    thời gian 60 × 5 lần fit dừng sớm (phần khớp lại 60 cấu hình phục vụ E4, không
    tính vào chi phí dò). oof_s của rs_tuned và rs_tuned_bag5 = fit dừng sớm + khớp
    lại fold (lựa chọn 12); riêng phần khớp lại fold ghi ở oof_refit_s.
 8. Bootstrap cụm theo trường (2.000 lần) cho C3 và Bảng III tính trong summary từ
    npz; không có npz (ví dụ JSON đã kéo về Mac) thì bỏ qua và ghi chú.
 9. JSON ghi NaN thành null để là JSON chuẩn (stats_paired.primary_holm nhận None).
10. Không chạy trên data/data_final.csv khi máy là macOS (AGENTS.md: không chạy thí
    nghiệm trên Mac). Trên server (Linux) chạy bình thường.
11. --smoke: một lần chia, 2 cấu hình, bag 2 (bag40 thành bag2, rs_tuned_bag2), tối đa
    50 cây, dừng sớm 10 vòng, bootstrap 200 lần. Tên trung tâm theo cỡ thật.
12. OOF của rs_tuned và rs_tuned_bag5 (sửa theo phản biện E1/E2, #5). OOF dừng sớm
    đến từ mô hình fold khớp trên `fit` (90% fold huấn luyện), trong khi default/bag
    khớp trên fit ∪ es và ŷ test của mọi trung tâm đến từ mô hình khớp lại trên 100%
    tập huấn luyện. Nếu R1 khớp trên OOF dừng sớm thì phần dư theo bin của rs_tuned
    rộng hơn thật (mô hình yếu hơn), và cùng OOF đó vừa chọn cấu hình vừa khớp R1
    (winner's curse): C3 bị lệch theo cả hai hướng không biết trước. Vì vậy:
      - OOF dừng sớm (`oof_es`, `test_foldavg_es` trong npz; trace_oof cho 60 cấu hình)
        CHỈ dùng để chọn cấu hình và lấy best_iteration.
      - Với cấu hình được chọn (và từng thành viên b của rs_tuned_bag5), mỗi fold khớp
        lại trên fit ∪ es với best_iteration + 1 cây CỦA FOLD ĐÓ, cùng random_state,
        dự đoán fold giữ lại: đó là `oof[rs_tuned]` mà R1 và mọi quy tắc hậu kỳ khớp
        trên, và `test_foldavg[rs_tuned]` là trung bình dự đoán test của chính các mô
        hình này. Cùng quy trình với lần khớp lại ngoài (số cây chọn bằng dừng sớm rồi
        khớp trên toàn bộ phần huấn luyện), và cùng lượng dữ liệu với OOF của bag.
      - Không rò rỉ: tập dừng sớm là con của fold huấn luyện, fold giữ lại không bao
        giờ vào khớp hay dừng sớm. Chọn cấu hình vẫn nhìn fold giữ lại (qua OOF dừng
        sớm) nên phần winner's curse chỉ giảm chứ không mất; E5 có thể so hai OOF.
      Lược đồ tương thích ngược: khoá `oof`, `test_foldavg` giữ nghĩa "OOF để khớp quy
      tắc" cho E2, E2b, E3, E4, E5, E9; khoá mới `oof_es`, `test_foldavg_es` chỉ có
      cho hai trung tâm rs. trace_oof[rs_tuned_index] vì thế KHÁC oof["rs_tuned"].
13. Cổng E0b (#1): đọc --data-audit (mặc định results_cost/data_audit.json). Nếu
    gates.group_split_required hoặc gates.stop_E1_until_IDT là true thì dừng: script
    này chưa có đường chia theo nhóm. Không có file thì chỉ cảnh báo (kiểm thử khói);
    file của dữ liệu khác (data_sha256 lệch) thì dừng, trừ --smoke. sha256 file và giá
    trị cổng ghi vào meta.data_audit (không vào dấu: chạy lại E0b cùng kết quả không
    được làm mất giờ khớp của E1).
14. Đủ lần chia (#3): C3 và G1 chỉ "cuối" khi số lần chia có số bằng len(gates.SEEDS)
    (--smoke: số lần chia khói). Thiếu thì C3 được coi là vắng trong Holm chính (p = 1,
    bảo thủ), G1 ghi splits_complete = false kèm danh sách phép so thiếu; E2 từ chối
    đọc vai trò tự động từ G1 như vậy.
15. Ghi chú thiết kế (không phải lỗi mã, ghi để người đọc biết):
    - Nhánh rs_tuned_bag5 của G1 dùng Holm riêng m = 2 (lựa chọn 6); khung bài chỉ nói
      "cùng điều kiện", chưa chốt họ.
    - G1 tạm chọn trung tâm chính, rồi C1 (E2b) và C2 (E2) tính trên trung tâm đó. Khi
      đủ họ, Holm cuối có thể lật G1; không mã nào tự kiểm lại, phải chạy lại E1
      summary (và E2, E2b nếu trung tâm đổi).
    - B* và cổng tập đặc trưng chọn theo thước đo trên tập kiểm tra, rồi cùng tập kiểm
      tra dùng cho suy luận C3 (khung bài cho phép; lệch nhỏ vì B* chọn theo độ gần
      bag40, không theo tốt nhất).
    - Dấu gồm code_sha256 của MỌI module src/ đang import (kể cả phần tóm tắt): sửa
      một dòng in ấn cũng làm mọi npz lệch dấu. Đừng dùng --on-mismatch recompute với
      một phần --seeds rồi để E2 đọc cả thư mục: E2 giờ từ chối npz lệch dấu giữa các
      lần chia.

Chạy tiếp sau khi bị ngắt: npz theo (lần chia, tập) và JSON dở dang
<out>.phase1.partial, <out>.phase2.partial mang dấu preds_io.fingerprint (cấu hình +
code_sha256); cùng dấu thì bỏ qua phần đã xong, khác dấu thì báo lỗi (không trộn hai
bản mã). JSON cuối <out> dựng lại từ hai file dở dang sau mỗi lượt chạy.

Chạy (server): PYTHONPATH=src .venv/bin/python -W ignore src/decomp_centers.py --workers 8
Khói (Mac, dữ liệu giả): xem run_decomp_centers.sh và tests/make_fake_data.py.
"""
import argparse
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
from joblib import Parallel, delayed
from xgboost import XGBRegressor

import decision_layer as dl
import features
import preds_io
import provenance
import splits
import stats_paired as sp
from gates import (ALPHA, B_STAR_TOL, BAG_SIZES, BAG_SUBSAMPLE, CLUSTER_BOOT_B, E1_FEATURE_TOST_MARGIN,
                   E1_KS, E1_R2_LOSS_REPORT, EARLY_STOP_ROUNDS, ES_FRAC, MAX_TREES, N_BINS, N_SAMPLES,
                   PRIMARY_K, R8_BAG, SEEDS, SESOI, TAIL_MASS_HIGH, TAIL_MASS_LOW, TUNE_BUDGET)
# CV_FOLDS và không gian gene: ga_xgb là nguồn sự thật duy nhất (AGENTS.md); import ga_xgb
# không có tác dụng phụ (chỉ hằng số và hàm), splits.py đã kiểm và dùng cùng cách.
from ga_xgb import CV_FOLDS
from preprocess import DATA_PATH

EXPERIMENT = "E1"
DEFAULT_OUT = "results_cost/decomp_centers.json"
DEFAULT_PREDS = os.path.join("preds", "decomp")
EXT_FSET = "F_dt"                  # tập có sub1 và bag40 ở giai đoạn 1
TIMING_REF = "F_full"              # chỉ để đo mất mát do ràng buộc thời điểm
GATE_BASE, GATE_CHAIN = "F_dt", ["F_dt-cn", "F_dt-cn-bc"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DATA = os.path.realpath(os.path.join(REPO, DATA_PATH))
DEFAULT_AUDIT = "results_cost/data_audit.json"
# Nguồn OOF mà R1 khớp trên cho rs_tuned, rs_tuned_bag5 (lựa chọn 12). Ghi vào dấu
# giai đoạn 2 và meta để npz trước và sau lần sửa không bao giờ bị trộn.
OOF_SOURCE = "fold_refit_fit_es_best_iter_plus_1"
RS_ES_KEYS = ("oof_es", "test_foldavg_es")


@dataclass(frozen=True)
class Budget:
    n_bag: int              # "bag10"
    bag_max: int            # "bag40"
    bag_sizes: tuple        # tiền tố của bag_max
    default_trees: int      # n_estimators của default và bag (XGBoost mặc định 100)
    n_configs: int
    max_trees: int
    es_rounds: int
    r8_bag: int             # rs_tuned_bag5
    boot_B: int

    @property
    def bag_main(self):
        return f"bag{self.n_bag}"

    @property
    def rs_bag(self):
        return f"rs_tuned_bag{self.r8_bag}"


def make_budget(smoke, n_configs):
    if smoke:
        return Budget(n_bag=2, bag_max=2, bag_sizes=(1, 2), default_trees=50,
                      n_configs=max(1, min(int(n_configs), 2)), max_trees=50, es_rounds=10,
                      r8_bag=2, boot_B=200)
    return Budget(n_bag=10, bag_max=max(BAG_SIZES), bag_sizes=tuple(BAG_SIZES), default_trees=100,
                  n_configs=int(n_configs), max_trees=MAX_TREES, es_rounds=EARLY_STOP_ROUNDS,
                  r8_bag=R8_BAG, boot_B=CLUSTER_BOOT_B)


def kkey(K):
    return str(int(K)) if float(K).is_integer() else str(K)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


# ---------------------------------------------------------------------------
# Một lần fit (chạy trong tiến trình joblib)
# ---------------------------------------------------------------------------
def _fit_task(X_fit, y_fit, X_apply, params, threads, es=None):
    """Khớp một XGBoost và dự đoán cho từng ma trận trong X_apply.

    es = (X_es, y_es, max_trees, rounds): dừng sớm trên tập dừng sớm, predict dùng
    best_iteration (XGBoost sklearn tự làm khi có dừng sớm). Trả (best_iteration hoặc
    None, [dự đoán], giây fit). best_iteration đánh số từ 0."""
    kw = dict(tree_method="hist", n_jobs=int(threads), verbosity=0)
    t0 = time.perf_counter()
    if es is None:
        m = XGBRegressor(**kw, **params).fit(X_fit, y_fit)
        best = None
    else:
        X_es, y_es, max_trees, rounds = es
        m = XGBRegressor(**kw, n_estimators=int(max_trees), early_stopping_rounds=int(rounds),
                         eval_metric="rmse", **params)
        m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        best = int(m.best_iteration)
    secs = time.perf_counter() - t0
    return best, [m.predict(X) for X in X_apply], secs


def run_tasks(par, specs):
    """specs: [(khoá, độ nặng ước lượng, đối số của _fit_task)]. Việc nặng gửi trước để
    đuôi của lượt Parallel không phải chờ một cấu hình lr 0,01 sâu 12 chạy một mình."""
    specs = sorted(specs, key=lambda s: -s[1])
    out = par(delayed(_fit_task)(*args) for _, _, args in specs)
    return {k: r for (k, _, _), r in zip(specs, out)}


def es_weight(cfg, n_trees=None):
    """Độ nặng tương đối để xếp việc: số cây (∝ 1/lr khi dừng sớm) × số lá tối đa."""
    trees = float(n_trees) if n_trees is not None else 1.0 / float(cfg["learning_rate"])
    return trees * 2.0 ** min(int(cfg["max_depth"]), 10)


# ---------------------------------------------------------------------------
# Thước đo của một trung tâm
# ---------------------------------------------------------------------------
def center_metrics(y_tr, y_te, oof, test, lo, hi, ks=E1_KS):
    """RMSE theo vùng, R², độ co, độ dốc hiệu chỉnh và cost_K của R0, R1, R1₁.

    R1 khớp MỘT lần trên (ŷ OOF, y) rồi dự đoán với trọng số của từng K (phân vị phần
    dư không phụ thuộc K). Ở K = 1 trọng số bằng 1 nên R1 trùng R1₁."""
    oof, test = np.asarray(oof, float), np.asarray(test, float)
    y_tr, y_te = np.asarray(y_tr, float), np.asarray(y_te, float)
    r1 = dl.ResidualBinBayes().fit(oof, y_tr)
    r11 = dl.ResidualBinBayes(uniform=True).fit(oof, y_tr).predict(test)
    cost = {"R0": {}, "R1": {}, "R1_1": {}}
    for K in ks:
        k = kkey(K)
        cost["R0"][k] = sp.cost_k(y_te, test, K, K, lo, hi)
        cost["R1_1"][k] = sp.cost_k(y_te, r11, K, K, lo, hi)
        cost["R1"][k] = sp.cost_k(y_te, r1.predict(test, wfun=dl.step(K, K, lo, hi)), K, K, lo, hi)
    reg = sp.region_rmse(y_te, test, lo, hi)
    sst = float(np.sum((y_te - y_te.mean()) ** 2))
    return {"region_rmse": reg, "all_rmse": reg["All"],
            "r2": 1.0 - float(np.sum((y_te - test) ** 2)) / sst,
            "sd_ratio": float(test.std(ddof=1) / y_te.std(ddof=1)),
            # hệ số của y theo ŷ: > 1 là ŷ co quá, < 1 là ŷ giãn quá (calibration_table.calib_stats)
            "calib_slope": float(np.polyfit(test, y_te, 1)[0]),
            "cost_K": cost, "oof_rmse": rmse(y_tr, oof)}


def eval_centers(d, centers):
    """{trung tâm: thước đo + thông tin khớp} từ một npz (mới khớp hay đọc lại)."""
    m = d["meta"]
    out = {}
    for c in centers:
        out[c] = center_metrics(d["y_tr"], d["y_te"], d["oof"][c], d["test"][c], m["lo"], m["hi"]) \
            | m["info"][c]
    return out


def split_info(d):
    lo, hi = d["meta"]["lo"], d["meta"]["hi"]
    ml, mh = dl.tail_masses(d["y_tr"], lo, hi)
    return {"lo": lo, "hi": hi, "n_tr": int(len(d["idx_tr"])), "n_te": int(len(d["idx_te"])),
            "mass_low_tr": ml, "mass_high_tr": mh}


# ---------------------------------------------------------------------------
# Mô hình không dừng sớm (default, bag): OOF, test, trung bình test của mô hình fold
# ---------------------------------------------------------------------------
def bag_params(b, bud):
    return {"random_state": int(b), "subsample": BAG_SUBSAMPLE, "colsample_bytree": BAG_SUBSAMPLE,
            "n_estimators": bud.default_trees}


def fit_plain_members(F, fset, tr, te, plan, members, par, threads, folds=None):
    """Mỗi thành viên: 5 mô hình fold khớp trên fold huấn luyện (fit ∪ es) cho OOF và
    trung bình test, một mô hình khớp trên toàn tập huấn luyện cho test.

    X_te của từng fold dựng CÙNG lời gọi F.design với hàng huấn luyện của fold nên tần
    suất trường chỉ đến từ fold đó (như X_va; preds_io, test_foldavg)."""
    y = F.y
    if folds is None:
        folds = [F.design(fset, tr[f["train"]], tr[f["va"]], te) for f in plan]
    Xtr, Xte = F.design(fset, tr, te)
    y_fold = [y[tr[f["train"]]] for f in plan]
    y_tr = y[tr]
    specs = []
    for name, params in members.items():
        for f, (Xa, Xva, Xte_f), yf in zip(plan, folds, y_fold):
            specs.append(((name, f["fold"]), 1.0, (Xa, yf, [Xva, Xte_f], params, threads)))
        specs.append(((name, "full"), 1.25, (Xtr, y_tr, [Xte], params, threads)))
    res = run_tasks(par, specs)
    out = {}
    for name in members:
        oof, fa, oof_s = np.zeros(len(tr)), np.zeros(len(te)), 0.0
        for f in plan:
            _, (p_va, p_te), secs = res[(name, f["fold"])]
            oof[f["va"]] = p_va
            fa += np.asarray(p_te, float) / len(plan)
            oof_s += secs
        _, (p_test,), fit_s = res[(name, "full")]
        out[name] = {"oof": oof, "test": np.asarray(p_test, float), "foldavg": fa,
                     "oof_s": oof_s, "fit_s": fit_s}
    return out


def bag_prefix_centers(members, sizes, bud, with_sub1):
    """Trung bình tiền tố của các thành viên m0, m1, ...: bag B dùng m0..m(B-1)."""
    out = {}
    for B in sizes:
        mem = [members[f"m{b}"] for b in range(B)]
        out[f"bag{B}"] = {
            "oof": np.mean([m["oof"] for m in mem], axis=0),
            "test": np.mean([m["test"] for m in mem], axis=0),
            "foldavg": np.mean([m["foldavg"] for m in mem], axis=0),
            "info": {"fit_s": float(sum(m["fit_s"] for m in mem)),
                     "oof_s": float(sum(m["oof_s"] for m in mem)), "tune_s": 0.0,
                     "best_params": bag_params(0, bud) | {"random_state": list(range(B))},
                     "best_iter": None, "n_trees": bud.default_trees}}
    if with_sub1:
        m0 = members["m0"]
        out["sub1"] = {"oof": m0["oof"], "test": m0["test"], "foldavg": m0["foldavg"],
                       "info": {"fit_s": m0["fit_s"], "oof_s": m0["oof_s"], "tune_s": 0.0,
                                "best_params": bag_params(0, bud), "best_iter": None,
                                "n_trees": bud.default_trees}}
    return out


def split_arrays(F, seed):
    tr, te = splits.outer_split(F.n, seed)
    plan = splits.fold_plan(tr, seed)
    return tr, te, plan


# ---------------------------------------------------------------------------
# Giai đoạn 1
# ---------------------------------------------------------------------------
def phase1_centers(fset, bud):
    if fset != EXT_FSET:
        return ["default", bud.bag_main]
    return ["default", "sub1"] + [f"bag{B}" for B in bud.bag_sizes]


def fit_phase1(F, seed, fset, bud, par, threads, fp, run_meta=None):
    """run_meta: {smoke, data_sha256, ...} ghi thẳng vào meta npz. Hai giá trị này đã
    nằm trong dấu, nhưng dấu là băm một chiều: E2 cần đọc chúng ở dạng rõ để từ chối
    npz khói hay npz của file dữ liệu khác mà không phải tính lại dấu của E1."""
    t0 = time.time()
    tr, te, plan = split_arrays(F, seed)
    y_tr, y_te = F.y[tr], F.y[te]
    lo, hi = dl.tail_cutoffs(y_tr)
    n_members = bud.bag_max if fset == EXT_FSET else bud.n_bag
    members = {"default": {"random_state": 0, "n_estimators": bud.default_trees}}
    members |= {f"m{b}": bag_params(b, bud) for b in range(n_members)}
    mem = fit_plain_members(F, fset, tr, te, plan, members, par, threads)

    dflt = mem.pop("default")
    cen = {"default": {"oof": dflt["oof"], "test": dflt["test"], "foldavg": dflt["foldavg"],
                       "info": {"fit_s": dflt["fit_s"], "oof_s": dflt["oof_s"], "tune_s": 0.0,
                                "best_params": members["default"], "best_iter": None,
                                "n_trees": bud.default_trees}}}
    sizes = bud.bag_sizes if fset == EXT_FSET else (bud.n_bag,)
    cen |= bag_prefix_centers(mem, sizes, bud, with_sub1=(fset == EXT_FSET))
    names = phase1_centers(fset, bud)
    meta = {"experiment": EXPERIMENT, "phase": 1, "seed": int(seed), "feature_set": fset,
            "lo": lo, "hi": hi, "centers": names, "phase1_centers": names,
            "n_features": int(len(F.columns(fset))), "info": {c: cen[c]["info"] for c in names},
            "identical_centers": [["sub1", "bag1"]] if fset == EXT_FSET and 1 in bud.bag_sizes else [],
            "budget": asdict(bud), "threads": int(threads), "fingerprint": fp,
            "wall_s": time.time() - t0, "provenance": provenance.stamp()} | (run_meta or {})
    return dict(idx_tr=tr, idx_te=te, y_tr=y_tr, y_te=y_te, fold_of=splits.fold_of(plan, len(tr)),
                school_code=F.school_code, prov_code=F.prov_code,
                oof={c: cen[c]["oof"] for c in names}, test={c: cen[c]["test"] for c in names},
                test_foldavg={c: cen[c]["foldavg"] for c in names}, meta=meta)


# ---------------------------------------------------------------------------
# Giai đoạn 2
# ---------------------------------------------------------------------------
def fold_refit_specs(tag, plan, D, cfg, rs, best_iter, threads):
    """Khớp lại mô hình fold trên fit ∪ es với best_iteration + 1 cây của CHÍNH fold đó
    (lựa chọn 12): cùng quy trình với lần khớp lại ngoài, cùng lượng dữ liệu với OOF
    của default/bag. Tập dừng sớm là con của fold huấn luyện nên fold giữ lại vẫn chưa
    bao giờ được nhìn."""
    return [((tag, rs, f["fold"]), es_weight(cfg, int(best_iter[f["fold"]]) + 1),
             (Dk["a"], Dk["y_a"], [Dk["va"], Dk["te"]],
              cfg | {"random_state": int(rs), "n_estimators": int(best_iter[f["fold"]]) + 1}, threads))
            for f, Dk in zip(plan, D)]


def collect_fold_refit(res, tag, rs, plan, n_tr, n_te):
    """(OOF, trung bình dự đoán test của mô hình fold, giây fit) từ fold_refit_specs."""
    oof, fa, secs = np.full(n_tr, np.nan), np.zeros(n_te), 0.0
    for f in plan:
        _, (p_va, p_te), s = res[(tag, rs, f["fold"])]
        oof[f["va"]] = p_va
        fa += np.asarray(p_te, float) / len(plan)
        secs += s
    assert np.all(np.isfinite(oof)), "OOF khớp lại fold còn vị trí trống"
    return oof, fa, float(secs)


def fit_phase2(F, seed, fset, bud, par, threads, fp, p1d, run_meta=None):
    """Vết n_configs cấu hình, rs_tuned, rs_tuned_bag5 (và sub1, bag B nếu tập chính
    không phải F_dt), gộp với các trung tâm giai đoạn 1 của tập chính.

    OOF dừng sớm chỉ để chọn cấu hình; OOF lưu ở `oof` cho hai trung tâm rs là OOF của
    mô hình fold khớp lại trên fit ∪ es (lựa chọn 12). OOF dừng sớm lưu ở `oof_es`."""
    t0 = time.time()
    tr, te, plan = split_arrays(F, seed)
    assert np.array_equal(tr, p1d["idx_tr"]) and np.array_equal(te, p1d["idx_te"]), \
        "npz giai đoạn 1 không cùng lần chia"
    y, y_tr, y_te = F.y, F.y[tr], F.y[te]
    n_tr, n_te, nf = len(tr), len(te), len(plan)
    configs = splits.sample_configs(bud.n_configs, seed)
    # sub1 chỉ có ở giai đoạn 1 của F_dt: thiếu nó nghĩa là tập chính chưa có sub1 và
    # bag 1..40 (lựa chọn 3 trong docstring).
    need_bag = "sub1" not in p1d["test"]

    # Ma trận của fold: hàng huấn luyện (fit ∪ es) quyết định tần suất trường, như
    # tests/make_fake_preds.py. Ma trận "train" (fit ∪ es) luôn giữ: khớp lại fold của
    # cấu hình được chọn (lựa chọn 12) và bag không dừng sớm đều khớp trên nó.
    D = []
    for f in plan:
        Xa, Xfit, Xes, Xva, Xte_f = F.design(fset, tr[f["train"]], tr[f["fit"]], tr[f["es"]],
                                             tr[f["va"]], te)
        D.append({"a": Xa, "fit": Xfit, "es": Xes, "va": Xva, "te": Xte_f, "y_a": y[tr[f["train"]]],
                  "y_fit": y[tr[f["fit"]]], "y_es": y[tr[f["es"]]]})
    Xtr, Xte = F.design(fset, tr, te)
    es_arg = lambda Dk: (Dk["es"], Dk["y_es"], bud.max_trees, bud.es_rounds)  # noqa: E731

    # Lượt A: 60 x 5 mô hình fold dừng sớm (+ thành viên bag nếu cần)
    specs = []
    for j, cfg in enumerate(configs):
        for f, Dk in zip(plan, D):
            specs.append((("es", j, f["fold"]), es_weight(cfg),
                          (Dk["fit"], Dk["y_fit"], [Dk["va"], Dk["te"]], cfg | {"random_state": 0},
                           threads, es_arg(Dk))))
    res = run_tasks(par, specs)
    t_oof = np.zeros((len(configs), n_tr))
    t_te_f = np.zeros((len(configs), n_te))
    best_iter = np.zeros((len(configs), nf), dtype=np.int32)
    es_s = np.zeros((len(configs), nf))
    for j in range(len(configs)):
        for f in plan:
            bi, (p_va, p_te), secs = res[("es", j, f["fold"])]
            best_iter[j, f["fold"]] = bi
            t_oof[j, f["va"]] = p_va
            t_te_f[j] += np.asarray(p_te, float) / nf
            es_s[j, f["fold"]] = secs
    # best_iteration đánh số từ 0 nên số cây là best_iteration + 1 (preds_io, trace_n_trees)
    n_trees = np.median(best_iter + 1, axis=1).astype(np.int32)
    t_rmse = np.sqrt(np.mean((y_tr[None, :] - t_oof) ** 2, axis=1))
    jb = int(np.argmin(t_rmse))
    cfg_b = configs[jb]

    # Lượt B: khớp lại cả 60 cấu hình + khớp lại fold của cấu hình được chọn (lựa chọn
    # 12) + mô hình fold dừng sớm của thành viên bag 1..r8_bag-1
    specs = [(("refit", j), es_weight(cfg, n_trees[j]),
              (Xtr, y_tr, [Xte], cfg | {"random_state": 0, "n_estimators": int(n_trees[j])}, threads))
             for j, cfg in enumerate(configs)]
    specs += fold_refit_specs("fold_refit", plan, D, cfg_b, 0, best_iter[jb], threads)
    for b in range(1, bud.r8_bag):
        for f, Dk in zip(plan, D):
            specs.append((("bag_es", b, f["fold"]), es_weight(cfg_b),
                          (Dk["fit"], Dk["y_fit"], [Dk["va"], Dk["te"]], cfg_b | {"random_state": b},
                           threads, es_arg(Dk))))
    res = run_tasks(par, specs)
    t_test = np.zeros((len(configs), n_te))
    fit_s = np.zeros(len(configs))
    for j in range(len(configs)):
        _, (p,), fit_s[j] = res[("refit", j)]
        t_test[j] = p
    oof0, fa0, rs0 = collect_fold_refit(res, "fold_refit", 0, plan, n_tr, n_te)
    mem = [{"oof": oof0, "foldavg": fa0, "oof_es": t_oof[jb], "foldavg_es": t_te_f[jb],
            "test": t_test[jb], "best_iter": best_iter[jb], "n_trees": int(n_trees[jb]),
            "oof_s": float(es_s[jb].sum()) + rs0, "oof_refit_s": rs0, "fit_s": float(fit_s[jb])}]
    for b in range(1, bud.r8_bag):
        m = {"oof_es": np.zeros(n_tr), "foldavg_es": np.zeros(n_te), "best_iter": np.zeros(nf, np.int32),
             "oof_s": 0.0}
        for f in plan:
            bi, (p_va, p_te), secs = res[("bag_es", b, f["fold"])]
            m["best_iter"][f["fold"]] = bi
            m["oof_es"][f["va"]] = p_va
            m["foldavg_es"] += np.asarray(p_te, float) / nf
            m["oof_s"] += secs
        m["n_trees"] = int(np.median(m["best_iter"] + 1))
        mem.append(m)

    # Lượt C: khớp lại thành viên bag 1..r8_bag-1 với số cây của chính nó, và khớp lại
    # fold của chúng trên fit ∪ es với best_iteration + 1 của từng fold (lựa chọn 12)
    specs = [(("bag_refit", b), es_weight(cfg_b, mem[b]["n_trees"]),
              (Xtr, y_tr, [Xte], cfg_b | {"random_state": b, "n_estimators": mem[b]["n_trees"]}, threads))
             for b in range(1, bud.r8_bag)]
    for b in range(1, bud.r8_bag):
        specs += fold_refit_specs("fold_refit", plan, D, cfg_b, b, mem[b]["best_iter"], threads)
    res = run_tasks(par, specs) if specs else {}
    for b in range(1, bud.r8_bag):
        _, (p,), mem[b]["fit_s"] = res[("bag_refit", b)]
        mem[b]["test"] = np.asarray(p, float)
        mem[b]["oof"], mem[b]["foldavg"], rsb = collect_fold_refit(res, "fold_refit", b, plan, n_tr, n_te)
        mem[b]["oof_s"] += rsb
        mem[b]["oof_refit_s"] = rsb

    tune_s = float(es_s.sum())
    avg = lambda key: np.mean([m[key] for m in mem], axis=0)  # noqa: E731
    new = {"rs_tuned": {"oof": oof0.copy(), "test": t_test[jb].copy(), "foldavg": fa0.copy(),
                        "oof_es": t_oof[jb].copy(), "foldavg_es": t_te_f[jb].copy(),
                        "info": {"fit_s": float(fit_s[jb]), "oof_s": mem[0]["oof_s"], "oof_refit_s": rs0,
                                 "tune_s": tune_s, "best_params": cfg_b, "best_iter": best_iter[jb].tolist(),
                                 "n_trees": int(n_trees[jb]), "config_index": jb, "oof_source": OOF_SOURCE,
                                 "oof_es_rmse": rmse(y_tr, t_oof[jb])}},
           bud.rs_bag: {"oof": avg("oof"), "test": avg("test"), "foldavg": avg("foldavg"),
                        "oof_es": avg("oof_es"), "foldavg_es": avg("foldavg_es"),
                        "info": {"fit_s": float(sum(m["fit_s"] for m in mem)),
                                 "oof_s": float(sum(m["oof_s"] for m in mem)),
                                 "oof_refit_s": float(sum(m["oof_refit_s"] for m in mem)), "tune_s": tune_s,
                                 "best_params": cfg_b | {"random_state": list(range(bud.r8_bag))},
                                 "best_iter": [np.asarray(m["best_iter"]).tolist() for m in mem],
                                 "n_trees": [int(m["n_trees"]) for m in mem], "config_index": jb,
                                 "oof_source": OOF_SOURCE, "oof_es_rmse": rmse(y_tr, avg("oof_es"))}}}
    rs_names = ["rs_tuned", bud.rs_bag]
    if need_bag:
        folds = [(Dk["a"], Dk["va"], Dk["te"]) for Dk in D]
        bmem = fit_plain_members(F, fset, tr, te, plan,
                                 {f"m{b}": bag_params(b, bud) for b in range(bud.bag_max)},
                                 par, threads, folds=folds)
        sizes = [B for B in bud.bag_sizes if B != bud.n_bag]      # bag10 giữ bản giai đoạn 1
        new |= bag_prefix_centers(bmem, sizes, bud, with_sub1=True)

    p1m = p1d["meta"]
    order = ["default", "sub1"] + [f"bag{B}" for B in bud.bag_sizes] + ["rs_tuned", bud.rs_bag]
    names = [c for c in order if c in p1d["test"] or c in new]
    get = lambda part, c, key: (new[c][key] if c in new else p1d[part][c])  # noqa: E731
    info = {c: (new[c]["info"] if c in new else p1m["info"][c]) for c in names}
    meta = {"experiment": EXPERIMENT, "phase": 2, "seed": int(seed), "feature_set": fset,
            "lo": p1m["lo"], "hi": p1m["hi"], "centers": names,
            "phase1_centers": [c for c in names if c not in new],
            "phase2_centers": [c for c in names if c in new], "info": info,
            "identical_centers": [["sub1", "bag1"]] if 1 in bud.bag_sizes else [],
            "trace_params": configs, "rs_tuned_index": jb, "n_features": int(Xtr.shape[1]),
            "max_trees": bud.max_trees, "es_rounds": bud.es_rounds,
            "trace_timing": {"es_s": es_s.tolist(), "fit_s": fit_s.tolist()},
            "budget": asdict(bud), "threads": int(threads), "fingerprint": fp,
            "phase1_fingerprint": p1m.get("fingerprint"), "wall_s": time.time() - t0,
            # oof[rs] là OOF khớp lại fold; oof_es[rs] là OOF dừng sớm đã dùng để chọn cấu
            # hình, cũng là trace_oof[rs_tuned_index] (lựa chọn 12)
            "oof_source": {c: (OOF_SOURCE if c in rs_names else "fold_fit_es_no_early_stop") for c in names},
            "es_oof_keys": {k: rs_names for k in RS_ES_KEYS},
            "provenance": provenance.stamp()} | (run_meta or {})
    return dict(idx_tr=tr, idx_te=te, y_tr=y_tr, y_te=y_te, fold_of=p1d["fold_of"],
                school_code=F.school_code, prov_code=F.prov_code,
                oof={c: get("oof", c, "oof") for c in names},
                test={c: get("test", c, "test") for c in names},
                test_foldavg={c: get("test_foldavg", c, "foldavg") for c in names},
                oof_es={c: new[c]["oof_es"] for c in rs_names},
                test_foldavg_es={c: new[c]["foldavg_es"] for c in rs_names},
                trace_oof=t_oof, trace_test=t_test, trace_test_foldavg=t_te_f,
                trace_best_iter=best_iter, trace_n_trees=n_trees, trace_oof_rmse=t_rmse, meta=meta)


def validate_e1(d):
    """preds_io.validate_split cộng phần riêng của E1 mà preds_io không biết: oof_es,
    test_foldavg_es (lựa chọn 12) phải đủ dài, hữu hạn và chỉ cho trung tâm có trong test."""
    err = preds_io.validate_split(d)
    n = {"oof_es": len(d["idx_tr"]), "test_foldavg_es": len(d["idx_te"])}
    for part in RS_ES_KEYS:
        for c, a in d.get(part, {}).items():
            if c not in d["test"]:
                err.append(f"{part}[{c}]: trung tâm không có trong test")
            if len(a) != n[part] or not np.all(np.isfinite(a)):
                err.append(f"{part}[{c}] dài {len(a)} (cần {n[part]}) hoặc có NaN/inf")
    return err


def trace_json(d):
    m = d["meta"]
    bi = np.asarray(d["trace_best_iter"])
    capped = (bi + 1 >= m["max_trees"]).any(axis=1)
    cfgs = [{"j": j, "params": p, "oof_rmse": float(d["trace_oof_rmse"][j]),
             "test_rmse": rmse(d["y_te"], d["trace_test"][j]), "n_trees": int(d["trace_n_trees"][j]),
             "best_iter": bi[j].tolist(), "capped": bool(capped[j]),
             "es_s": float(np.sum(m["trace_timing"]["es_s"][j])), "fit_s": m["trace_timing"]["fit_s"][j]}
            for j, p in enumerate(m["trace_params"])]
    return {"feature_set": m["feature_set"], "rs_tuned_index": m["rs_tuned_index"],
            "n_configs": len(cfgs), "n_capped": int(capped.sum()), "max_trees": m["max_trees"],
            "es_rounds": m["es_rounds"], "tune_s": float(np.sum(m["trace_timing"]["es_s"])),
            "refit_all_s": float(np.sum(m["trace_timing"]["fit_s"])), "configs": cfgs}


# ---------------------------------------------------------------------------
# Cổng và tổng hợp
# ---------------------------------------------------------------------------
def vals(per, seeds, fset, center, *path):
    out = []
    for s in seeds:
        v = per.get(str(s), {}).get(fset, {}).get(center)
        for p in path:
            v = None if v is None else v.get(p)
        out.append(float("nan") if v is None else float(v))
    return np.array(out)


def r1k(K=PRIMARY_K):
    return ("cost_K", "R1", kkey(K))


def feature_gate(per, seeds, bud):
    """Tập chính theo tính dùng được: bỏ thuộc tính nhạy cảm chỉ khi mất không quá
    SESOI RMSE (TOST ±0,10 trên RMSE test của bag10, Nadeau-Bengio theo lần chia)."""
    primary, steps, prev = GATE_BASE, [], GATE_BASE
    for cand in GATE_CHAIN:
        d = vals(per, seeds, cand, bud.bag_main, "all_rmse") - vals(per, seeds, prev, bud.bag_main, "all_rmse")
        t = sp.paired(d, margin=E1_FEATURE_TOST_MARGIN)
        steps.append({"compare": f"{cand} - {prev}", "metric": f"all_rmse {bud.bag_main}",
                      "diffs": d.tolist(), **t})
        if not t["tost"]["passed"]:
            break
        primary = prev = cand
    return {"primary": primary, "margin": E1_FEATURE_TOST_MARGIN, "steps": steps}


def b_star(per, seeds, bud, K=PRIMARY_K, fset=EXT_FSET):
    ref = vals(per, seeds, fset, f"bag{bud.bag_max}", *r1k(K))
    rows, bst = {}, None
    for B in bud.bag_sizes:
        d = np.abs(vals(per, seeds, fset, f"bag{B}", *r1k(K)) - ref)
        d = d[np.isfinite(d)]
        m = float(d.mean()) if len(d) else float("nan")
        rows[str(B)] = {"mean_abs_diff": m, "n": int(len(d))}
        if bst is None and np.isfinite(m) and m <= B_STAR_TOL:
            bst = B
    return {"B_star": bst, "tol": B_STAR_TOL, "K": K, "feature_set": fset, "by_B": rows}


def _describe(per, seeds, fsets, ks=E1_KS):
    out = {}
    for fs in fsets:
        cs = sorted({c for s in seeds for c in per.get(str(s), {}).get(fs, {})})
        for c in cs:
            row = {}
            for key in ("all_rmse", "r2", "sd_ratio", "calib_slope", "oof_rmse", "oof_es_rmse", "fit_s",
                        "tune_s"):
                row[key] = _ms(vals(per, seeds, fs, c, key))
            for reg in ("Low tail", "Middle", "High tail", "Tails"):
                row[f"rmse_{reg}"] = _ms(vals(per, seeds, fs, c, "region_rmse", reg))
            row["cost_K"] = {r: {kkey(K): _ms(vals(per, seeds, fs, c, "cost_K", r, kkey(K))) for K in ks}
                             for r in ("R0", "R1", "R1_1")}
            out.setdefault(fs, {})[c] = row
    return out


def _ms(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": int(len(v))}


class _Boot:
    """Bootstrap cụm theo trường cho chênh cost_K(R1) giữa hai (tập, trung tâm), từng lần chia.

    Đọc npz trong preds_dir; dự đoán R1 ở K tính lại (rẻ) từ OOF và test đã lưu."""

    def __init__(self, preds_dir, primary, B):
        self.preds_dir, self.primary, self.B = preds_dir, primary, B
        self._npz, self._r1 = {}, {}

    def npz(self, seed, fset):
        key = (seed, fset)
        if key not in self._npz:
            p = preds_io.split_path(self.preds_dir, seed)
            if fset != self.primary or not os.path.exists(p):
                p = preds_io.split_path(self.preds_dir, seed, f"p1_{fset}")
            self._npz[key] = preds_io.load_split(p) if os.path.exists(p) else None
        return self._npz[key]

    def r1(self, seed, fset, center, K):
        key = (seed, fset, center, K)
        if key not in self._r1:
            d = self.npz(seed, fset)
            if d is None or center not in d["test"]:
                self._r1[key] = None
            else:
                lo, hi = d["meta"]["lo"], d["meta"]["hi"]
                r = dl.ResidualBinBayes().fit(d["oof"][center], d["y_tr"])
                self._r1[key] = r.predict(d["test"][center], wfun=dl.step(K, K, lo, hi))
        return self._r1[key]

    def contrast(self, seeds, a, b, K=PRIMARY_K):
        rows = []
        for s in seeds:
            pa, pb = self.r1(s, *a, K), self.r1(s, *b, K)
            if pa is None or pb is None:
                continue
            d = self.npz(s, a[0])
            lo, hi = d["meta"]["lo"], d["meta"]["hi"]
            cb = sp.cluster_boot_diff(d["y_te"], pa, pb, preds_io.rows(d, "te", "school_code"),
                                      K, K, lo, hi, B=self.B, seed=int(s))
            rows.append({"seed": int(s)} | cb)
        return {"B": self.B, "n_splits": len(rows), "n_excl_zero": int(sum(r["excludes_zero"] for r in rows)),
                "per_split": rows}


def _cond(nb_mean, p):
    return bool(np.isfinite(nb_mean) and np.isfinite(p) and nb_mean <= -SESOI and p < ALPHA)


def summarize(per, trace, seeds, bud, primary, gate, preds_dir, boot=True, n_ref=len(SEEDS)):
    """n_ref: số lần chia để C3, G1 là "cuối" (len(gates.SEEDS); --smoke: số lần chia khói)."""
    S = {"centers": _describe(per, seeds, features.FEATURE_SETS), "n_ref": int(n_ref)}
    S["feature_gate"] = gate
    bs = b_star(per, seeds, bud)
    if bs["B_star"] is None and primary != EXT_FSET:
        # Chạy --feature-set --phase 2 mà chưa có giai đoạn 1 của F_dt: giai đoạn 2 đã
        # khớp bag 1..40 trên tập chính, nên tính B* ở đó thay vì bỏ trống C3.
        alt = b_star(per, seeds, bud, fset=primary)
        if alt["B_star"] is not None:
            bs = alt | {"fallback": f"không có bag{bud.bag_max} trên {EXT_FSET}; B* tính trên {primary}"}
    bs["complete"] = all(r["n"] == n_ref for r in bs["by_B"].values())
    S["b_star"] = bs

    # Ràng buộc thời điểm: F_dt so với F_full (chỉ để đo mất mát, mục 5.1)
    tc = {}
    for c in ("default", bud.bag_main):
        d_rmse = vals(per, seeds, EXT_FSET, c, "all_rmse") - vals(per, seeds, TIMING_REF, c, "all_rmse")
        d_r2 = vals(per, seeds, TIMING_REF, c, "r2") - vals(per, seeds, EXT_FSET, c, "r2")
        tc[c] = {"d_rmse_Fdt_minus_Ffull": sp.nb_ttest(d_rmse), "r2_loss_Ffull_minus_Fdt": _ms(d_r2)}
    loss = tc[bud.bag_main]["r2_loss_Ffull_minus_Fdt"]["mean"]
    tc["report_both"] = bool(np.isfinite(loss) and loss >= E1_R2_LOSS_REPORT)
    tc["r2_loss_threshold"] = E1_R2_LOSS_REPORT
    S["timing_constraint"] = tc

    bt = _Boot(preds_dir, primary, bud.boot_B) if boot else None
    K = PRIMARY_K
    have_p2 = any(per.get(str(s), {}).get(primary, {}).get("rs_tuned") for s in seeds)
    # Bảng III thứ cấp (cost_3 của R1, Holm trong bảng, m = 5)
    t3_def = [(f"{bud.bag_main} - default", (primary, bud.bag_main), (primary, "default")),
              ("sub1 - default", (primary, "sub1"), (primary, "default")),
              (f"{bud.bag_main} - sub1", (primary, bud.bag_main), (primary, "sub1")),
              (f"{bud.rs_bag} - rs_tuned", (primary, bud.rs_bag), (primary, "rs_tuned")),
              (f"F_dt - F_full ({bud.bag_main})", (EXT_FSET, bud.bag_main), (TIMING_REF, bud.bag_main))]
    t3 = {}
    for name, a, b in t3_def:
        d = vals(per, seeds, *a, *r1k(K)) - vals(per, seeds, *b, *r1k(K))
        t3[name] = {"a": list(a), "b": list(b), "diffs": d.tolist(), **sp.paired(d)}
        if bt is not None:
            t3[name]["cluster_boot"] = bt.contrast(seeds, a, b, K)
    names = list(t3)
    for name, ph in zip(names, sp.holm([t3[n]["nb"]["p"] for n in names])):
        t3[name]["p_holm"] = ph
    S["table_III"] = {"metric": f"cost_{K}(R1)", "family": "Holm trong bảng (thứ cấp)", "contrasts": t3}

    if not have_p2:
        S["C3"] = S["G1"] = {"status": "chưa có giai đoạn 2"}
        return S
    Bs = bs["B_star"]
    bstar = f"bag{Bs}" if Bs is not None else None
    if bstar is None or not any(per.get(str(s), {}).get(primary, {}).get(bstar) for s in seeds):
        S["C3"] = S["G1"] = {"status": f"chưa có bag B* ({bstar}) trên tập chính {primary}"}
        return S
    d3 = vals(per, seeds, primary, "rs_tuned", *r1k(K)) - vals(per, seeds, primary, bstar, *r1k(K))
    c3 = {"desc": f"cost_{K}(R1, rs_tuned) - cost_{K}(R1, {bstar}) trên {primary}", "B_star": Bs,
          "diffs": d3.tolist(), **sp.paired(d3)}
    # Lựa chọn 14: C3 thiếu lần chia (nb bỏ NaN nên n tự co) không được vào Holm chính
    # như một phép so đủ; coi là vắng (p = 1) và gắn nhãn tạm.
    c3 |= {"n_ref": int(n_ref), "final": bool(c3["nb"]["n"] == n_ref)}
    c3["status"] = "final" if c3["final"] else f"provisional: {c3['nb']['n']}/{n_ref} lần chia"
    c3["primary_holm_provisional"] = sp.primary_holm({"C3": c3["nb"]["p"]} if c3["final"] else {})["C3"]
    if bt is not None:
        c3["cluster_boot"] = bt.contrast(seeds, (primary, "rs_tuned"), (primary, bstar), K)
    S["C3"] = c3

    # Cổng G1 (tạm, xem docstring mục 6)
    g5 = {}
    for name, other in ((f"{bud.rs_bag} - rs_tuned", "rs_tuned"), (f"{bud.rs_bag} - {bstar}", bstar)):
        d = vals(per, seeds, primary, bud.rs_bag, *r1k(K)) - vals(per, seeds, primary, other, *r1k(K))
        g5[name] = {"diffs": d.tolist(), **sp.paired(d)}
    for name, ph in zip(list(g5), sp.holm([g5[n]["nb"]["p"] for n in g5])):
        g5[name]["p_holm_m2"] = ph
    # Lựa chọn 14: mọi phép so mà G1 dựa vào phải đủ n_ref lần chia (cổng tập đặc trưng,
    # B*, C3, hai phép so của rs_tuned_bag5); thiếu thì G1 vẫn tính nhưng không "cuối".
    incomplete = [f"feature_gate {st['compare']}" for st in gate.get("steps", []) if st["nb"]["n"] != n_ref]
    incomplete += [f"B* bag{B}" for B, r in bs["by_B"].items() if r["n"] != n_ref]
    incomplete += ([] if c3["final"] else ["C3"])
    incomplete += [n for n, g in g5.items() if g["nb"]["n"] != n_ref]
    bag5_wins = (all(_cond(g["nb"]["mean"], g["p_holm_m2"]) for g in g5.values())
                 and all(g["nb"]["n"] == n_ref for g in g5.values()))
    c3_wins = _cond(c3["nb"]["mean"], c3["primary_holm_provisional"]["p_holm"])
    center = bud.rs_bag if bag5_wins else ("rs_tuned" if c3_wins else bstar)
    wording = None
    if center == bstar:
        wording = ("dò siêu tham số sau bagging thêm dưới 0,10" if c3["tost"]["passed"]
                   else "nhỏ, chưa chứng minh được dưới ngưỡng")
    S["G1"] = {"provisional": True, "primary_center": center, "feature_set": primary,
               "bag_Bstar": bstar, "rs_tuned_wins_C3": c3_wins, "rs_bag_wins_both": bag5_wins,
               "rs_bag_contrasts": g5, "wording_if_bag": wording,
               "splits_complete": not incomplete, "n_ref": int(n_ref), "incomplete": incomplete,
               "note": "C1, C2 chưa có: primary_holm coi p = 1 nên Holm tạm ≥ Holm cuối; tính lại khi đủ họ"}
    S["trace"] = {"n_capped": _ms([t["n_capped"] for t in trace.values()]),
                  "rs_tuned_index": {s: t["rs_tuned_index"] for s, t in trace.items()},
                  "tune_s": _ms([t["tune_s"] for t in trace.values()]),
                  "refit_all_s": _ms([t["refit_all_s"] for t in trace.values()])}
    return S


def print_summary(S, bud, primary):
    f = lambda d: f"{d['mean']:.3f}±{d['sd']:.3f}" if np.isfinite(d["sd"]) else f"{d['mean']:.3f}"  # noqa: E731
    k = kkey(PRIMARY_K)
    print(f"\n[1] Trung tâm (RMSE test; cost_{k} của R0/R1/R1_1; mean±sd qua lần chia)")
    for fs, cs in S["centers"].items():
        for c, r in cs.items():
            ck = r["cost_K"]
            print(f"  {fs:11s} {c:16s} RMSE {f(r['all_rmse'])}  R² {r['r2']['mean']:.4f}  "
                  f"sd_ratio {r['sd_ratio']['mean']:.3f}  slope {r['calib_slope']['mean']:.3f}  "
                  f"cost_{k} {ck['R0'][k]['mean']:.3f}/{ck['R1'][k]['mean']:.3f}/{ck['R1_1'][k]['mean']:.3f}")
    g = S["feature_gate"]
    print(f"\n[2] Cổng tập đặc trưng -> {primary}" + ("" if g.get("steps") else " (--feature-set, bỏ qua cổng)"))
    for st in g.get("steps", []):
        print(f"  {st['compare']}: {st['nb']['mean']:+.3f} (p {st['nb']['p']:.3g}, n {st['nb']['n']}); "
              f"TOST ±{g['margin']} p {st['tost']['p']:.3g} -> {'qua' if st['tost']['passed'] else 'không qua'}")
    b = S["b_star"]
    print(f"\n[3] B* = {b['B_star']} (|cost_3(R1, bag B) - bag {bud.bag_max}| ≤ {b['tol']}): "
          + ", ".join(f"B={B}: {r['mean_abs_diff']:.3f}" for B, r in b["by_B"].items()))
    tc = S["timing_constraint"]
    print(f"\n[4] F_dt - F_full ({bud.bag_main}): ΔRMSE {tc[bud.bag_main]['d_rmse_Fdt_minus_Ffull']['mean']:+.3f}, "
          f"mất R² {tc[bud.bag_main]['r2_loss_Ffull_minus_Fdt']['mean']:+.4f}; báo cả hai: {tc['report_both']}")
    print(f"\n[5] Bảng III thứ cấp (cost_{k}(R1), Holm trong bảng):")
    for n, c in S["table_III"]["contrasts"].items():
        ph = c.get("p_holm")
        print(f"  {n:28s} {c['nb']['mean']:+.3f} (p {c['nb']['p']:.3g}, Holm "
              f"{ph if ph is None else f'{ph:.3g}'}, thắng {c['nb']['wins']}/{c['nb']['n']})")
    if "nb" in S.get("C3", {}):
        c3, g1 = S["C3"], S["G1"]
        print(f"\n[6] C3 = {c3['desc']}: {c3['nb']['mean']:+.3f} (p {c3['nb']['p']:.3g}, "
              f"Holm tạm {c3['primary_holm_provisional']['p_holm']:.3g}); TOST p {c3['tost']['p']:.3g}; "
              f"{c3['status']}" + ("" if c3["final"] else " -> coi là vắng trong Holm chính"))
        print(f"    G1 (tạm): trung tâm chính = {g1['primary_center']} trên {g1['feature_set']}"
              + (f"; '{g1['wording_if_bag']}'" if g1["wording_if_bag"] else ""))
        if not g1["splits_complete"]:
            print(f"    G1 CHƯA ĐỦ {g1['n_ref']} lần chia ở: {', '.join(g1['incomplete'])} "
                  "(E2 sẽ không đọc vai trò tự động từ G1 này)")
    else:
        print(f"\n[6] C3, G1: {S.get('C3', {}).get('status')}")


def _nan_to_none(o):
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    return o


# ---------------------------------------------------------------------------
def check_data_audit(path, data_sha, smoke):
    """Cổng E0b (lựa chọn 13). Trả dict ghi vào meta.data_audit, hoặc dừng.

    Vì sao dừng mà không tự chia theo nhóm: script này chưa có đường chia theo nhóm
    (E0b trên dữ liệu thật: 0 nhóm trùng). Nếu một lần chạy E0b sau này ra khác, chạy
    E1 bằng chia theo dòng sẽ để bản ghi trùng nằm hai phía test/huấn luyện, fold trong
    và tập dừng sớm: cost test lạc quan, OOF (và R1) rò rỉ. Dừng rõ ràng tốt hơn ra số sai."""
    if not os.path.exists(path):
        print(f"[cảnh báo] không có {path} (E0b): KHÔNG kiểm được cổng chia theo nhóm. Chỉ chấp "
              "nhận cho kiểm thử khói; lượt chạy khẳng định phải có data_audit.json.", flush=True)
        return {"path": path, "status": "missing", "sha256": None, "group_split_required": None,
                "stop_E1_until_IDT": None}
    audit = preds_io.load_json(path)
    g = audit.get("gates") or {}
    a_sha = (audit.get("meta") or {}).get("data_sha256")
    info = {"path": os.path.abspath(path), "status": "checked", "sha256": preds_io.file_sha256(path),
            "audit_data_sha256": a_sha, "audit_smoke": (audit.get("meta") or {}).get("smoke"),
            "group_split_required": g.get("group_split_required"),
            "stop_E1_until_IDT": g.get("stop_E1_until_IDT"),
            "exact_dup_groups": g.get("exact_dup_groups")}
    if a_sha is not None and a_sha != data_sha:
        msg = (f"{path} là kiểm toán của file dữ liệu khác (data_sha256 {str(a_sha)[:12]} khác "
               f"{data_sha[:12]} của --data)")
        if not smoke:
            sys.exit(msg + ": chạy lại E0b trên đúng file, hoặc trỏ --data-audit tới file đúng.")
        print(f"[cảnh báo] {msg}; --smoke nên chỉ ghi lại", flush=True)
        info["status"] = "other_data"
        return info
    if "group_split_required" not in g:
        sys.exit(f"{path} không có gates.group_split_required: không kiểm được cổng E0b.")
    if g.get("group_split_required") or g.get("stop_E1_until_IDT"):
        sys.exit(f"Cổng E0b ({path}): group_split_required = {g.get('group_split_required')}, "
                 f"stop_E1_until_IDT = {g.get('stop_E1_until_IDT')} ({g.get('exact_dup_groups')} nhóm "
                 "trùng chính xác). decomp_centers.py chưa có đường chia theo nhóm "
                 "(splits.outer_split/fold_plan với groups=data_audit.record_groups): dừng E1, "
                 "không chạy chia theo dòng trên dữ liệu có bản ghi trùng.")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--preds-dir", default=DEFAULT_PREDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all")
    ap.add_argument("--feature-set", choices=features.FEATURE_SETS, default=None,
                    help="tập chính cho giai đoạn 2, bỏ qua cổng tập đặc trưng")
    ap.add_argument("--n-configs", type=int, default=TUNE_BUDGET)
    ap.add_argument("--workers", type=int, default=4, help="số tiến trình joblib")
    ap.add_argument("--threads", type=int, default=None,
                    help="n_jobs của mỗi XGBoost (mặc định max(1, cpu_count // workers))")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-boot", action="store_true", help="bỏ bootstrap cụm trong summary")
    ap.add_argument("--on-mismatch", choices=["raise", "recompute"], default="raise")
    ap.add_argument("--data-audit", default=DEFAULT_AUDIT,
                    help="JSON của E0b (data_audit.py); cổng chia theo nhóm đọc ở đây")
    args = ap.parse_args(argv)

    if platform.system() == "Darwin" and os.path.realpath(args.data) == REAL_DATA:
        sys.exit("Không chạy E1 trên data/data_final.csv ở máy Mac (AGENTS.md): chạy ở server, "
                 "hoặc dùng --data tests/fixtures/fake_hsa.csv cho kiểm thử khói.")
    provenance.print_versions()
    t_start = time.time()
    seeds = args.seeds[:1] if args.smoke else list(args.seeds)
    # Lựa chọn 14: số lần chia để C3, G1 là "cuối"
    n_ref = len(seeds) if args.smoke else len(SEEDS)
    bud = make_budget(args.smoke, args.n_configs)
    threads = args.threads or splits.xgb_threads(args.workers)
    data_sha = preds_io.file_sha256(args.data)
    audit = check_data_audit(args.data_audit, data_sha, args.smoke)
    # smoke và data_sha256 ở dạng rõ trong meta npz để E2 đối chiếu (fit_phase1 docstring)
    run_meta = {"smoke": bool(args.smoke), "data_sha256": data_sha}
    base_cfg = {"script": "decomp_centers", "data_sha256": data_sha, "smoke": bool(args.smoke),
                "cv_folds": CV_FOLDS, "es_frac": ES_FRAC, "ks": E1_KS, "n_bins": N_BINS,
                "n_samples": N_SAMPLES, "tail_mass": [TAIL_MASS_LOW, TAIL_MASS_HIGH]}
    fp1 = preds_io.fingerprint(base_cfg | {"phase": 1, "n_bag": bud.n_bag, "bag_max": bud.bag_max,
                                            "bag_sizes": list(bud.bag_sizes),
                                            "default_trees": bud.default_trees})
    p1_path, p2_path = args.out + ".phase1.partial", args.out + ".phase2.partial"
    res1 = preds_io.load_partial(p1_path, fp1, {"per_split": {}, "splits_info": {}},
                                 on_mismatch=args.on_mismatch)

    F = features.load_frame(args.data)
    print(f"{args.data}: n={F.n}; seeds {seeds}; workers {args.workers} x {threads} luồng; "
          f"ngân sách {asdict(bud)}", flush=True)
    npz_sha = {}

    def unit_par():
        # Một ngữ cảnh Parallel cho MỖI đơn vị (lần chia, tập đặc trưng), không một ngữ
        # cảnh cho cả lượt: joblib ghi mỗi ma trận design gửi sang tiến trình con (~50 MB
        # trên dữ liệu thật) vào thư mục tạm và chỉ dọn khi ngữ cảnh đóng. Lượt 25/9 mở
        # một ngữ cảnh cho cả 10 lần chia × 4 tập, làm đầy /dev/shm (7,9 GB) sau 9 phút.
        # Pool tiến trình loky vẫn được giữ giữa các ngữ cảnh nên không tốn khởi động lại.
        return Parallel(n_jobs=args.workers, backend="loky", batch_size=1)

    def ensure_phase1(seed, fset):
        path = preds_io.split_path(args.preds_dir, seed, f"p1_{fset}")
        d = preds_io.load_or_none(path, fp1, on_mismatch=args.on_mismatch)
        fresh = d is None
        if fresh:
            t0 = time.time()
            with unit_par() as par:
                arrays = fit_phase1(F, seed, fset, bud, par, threads, fp1, run_meta)
            preds_io.save_split(path, **arrays)
            d = preds_io.load_split(path)
            err = validate_e1(d)
            assert not err, err
        entry = res1["per_split"].setdefault(str(seed), {})
        if fresh or fset not in entry:
            entry[fset] = eval_centers(d, d["meta"]["centers"])
            res1["splits_info"][str(seed)] = split_info(d)
            preds_io.dump_json_atomic(res1, p1_path)
            if fresh:
                e = entry[fset]
                print(f"  giai đoạn 1, split {seed}, {fset}: RMSE "
                      + " ".join(f"{c}={e[c]['all_rmse']:.3f}" for c in e)
                      + f" ({time.time() - t0:.0f}s)", flush=True)
        npz_sha.setdefault(str(seed), {})[f"p1_{fset}"] = preds_io.file_sha256(path)
        return d

    if args.phase in ("1", "all"):
        for s in seeds:
            for fs in features.FEATURE_SETS:
                ensure_phase1(s, fs)

    # Cổng tập đặc trưng (hoặc --feature-set)
    if args.feature_set:
        primary, gate = args.feature_set, {"primary": args.feature_set, "override": True, "steps": []}
    else:
        missing = [(s, fs) for s in seeds for fs in [GATE_BASE] + GATE_CHAIN
                   if fs not in res1["per_split"].get(str(s), {})]
        if missing and args.phase == "2":
            sys.exit(f"Cổng tập đặc trưng cần giai đoạn 1 cho {missing[:5]}...: chạy --phase 1 "
                     "trước, hoặc cho --feature-set")
        gate = feature_gate(res1["per_split"], seeds, bud) | {"override": False}
        primary = gate["primary"]

    fp2 = res2 = None
    if args.phase in ("2", "all"):
        fp2 = preds_io.fingerprint(base_cfg | {"phase": 2, "feature_set": primary, "phase1": fp1,
                                                "n_configs": bud.n_configs, "max_trees": bud.max_trees,
                                                "es_rounds": bud.es_rounds, "r8_bag": bud.r8_bag,
                                                "bag_max": bud.bag_max, "bag_sizes": list(bud.bag_sizes),
                                                "oof_source": OOF_SOURCE})
        res2 = preds_io.load_partial(p2_path, fp2, {"per_split": {}}, on_mismatch=args.on_mismatch)
        print(f"\nGiai đoạn 2 trên {primary} ({'--feature-set' if args.feature_set else 'cổng'}); "
              f"{bud.n_configs} cấu hình, tối đa {bud.max_trees} cây, dừng sớm {bud.es_rounds} vòng",
              flush=True)
        for s in seeds:
            p1d = ensure_phase1(s, primary)
            path = preds_io.split_path(args.preds_dir, s)
            d = preds_io.load_or_none(path, fp2, on_mismatch=args.on_mismatch)
            fresh = d is None
            if fresh:
                t0 = time.time()
                with unit_par() as par:
                    arrays = fit_phase2(F, s, primary, bud, par, threads, fp2, p1d, run_meta)
                preds_io.save_split(path, **arrays)
                d = preds_io.load_split(path)
                err = validate_e1(d)
                assert not err, err
            if fresh or str(s) not in res2["per_split"]:
                res2["per_split"][str(s)] = {"feature_set": primary,
                                             "centers": eval_centers(d, d["meta"]["phase2_centers"]),
                                             "trace": trace_json(d)}
                preds_io.dump_json_atomic(res2, p2_path)
                if fresh:
                    e, tj = res2["per_split"][str(s)]["centers"], res2["per_split"][str(s)]["trace"]
                    print(f"  giai đoạn 2, split {s}: RMSE "
                          + " ".join(f"{c}={e[c]['all_rmse']:.3f}" for c in e)
                          + f"; cấu hình {tj['rs_tuned_index']}, {tj['n_capped']} chạm trần "
                          f"({time.time() - t0:.0f}s)", flush=True)
            npz_sha.setdefault(str(s), {})["merged"] = preds_io.file_sha256(path)
    elif os.path.exists(p2_path):
        # Chỉ chạy giai đoạn 1: giữ kết quả giai đoạn 2 cũ nếu cùng tập chính và cùng mã
        fp2 = preds_io.fingerprint(base_cfg | {"phase": 2, "feature_set": primary, "phase1": fp1,
                                                "n_configs": bud.n_configs, "max_trees": bud.max_trees,
                                                "es_rounds": bud.es_rounds, "r8_bag": bud.r8_bag,
                                                "bag_max": bud.bag_max, "bag_sizes": list(bud.bag_sizes),
                                                "oof_source": OOF_SOURCE})
        old = preds_io.load_json(p2_path)
        res2 = old if (old.get("meta") or {}).get("fingerprint") == fp2 else None

    # JSON cuối: gộp hai giai đoạn, per_split[seed][tập][trung tâm]
    per = {s: {fs: dict(c) for fs, c in e.items()} for s, e in res1["per_split"].items()}
    trace = {}
    for s, e in ((res2 or {}).get("per_split") or {}).items():
        per.setdefault(s, {}).setdefault(e["feature_set"], {}).update(e["centers"])
        trace[s] = e["trace"]
    S = summarize(per, trace, seeds, bud, primary, gate, args.preds_dir, boot=not args.no_boot,
                  n_ref=n_ref)
    print_summary(S, bud, primary)
    meta = {"experiment": EXPERIMENT, "script": "decomp_centers",
            "protocol": "khung bài 24/9 mục 6.4; lần chia splits.outer_split (80/20), fold splits.fold_plan "
                        f"({CV_FOLDS}-fold, tập dừng sớm {ES_FRAC:.0%} mỗi fold huấn luyện); đuôi theo khối "
                        f"lượng {TAIL_MASS_LOW}/{TAIL_MASS_HIGH} của y huấn luyện; mọi thước đo RMSE",
            "seeds": seeds, "feature_sets": features.FEATURE_SETS, "primary_feature_set": primary,
            "feature_set_source": "override" if args.feature_set else "gate", "phase_arg": args.phase,
            "phase2_done_splits": sorted(trace), "budget": asdict(bud), "n_configs": bud.n_configs,
            "ks": E1_KS, "primary_k": PRIMARY_K, "n_bins": N_BINS, "n_samples": N_SAMPLES,
            "cv_folds": CV_FOLDS, "es_frac": ES_FRAC, "workers": args.workers, "threads": threads,
            "timing_note": "fit_s, oof_s, tune_s: tổng giây fit của từng mô hình ở `threads` luồng "
                           "(không gồm predict), không phải giờ đồng hồ",
            "npz_layout": {"phase1": "split<seed>_p1_<tập>.npz", "merged": "split<seed>.npz (tập chính)"},
            "preds_dir": os.path.abspath(args.preds_dir), "npz_sha256": npz_sha,
            "fingerprints": {"phase1": fp1, "phase2": fp2}, "smoke": bool(args.smoke),
            "data_sha256": data_sha, "data_audit": audit, "n_ref": n_ref, "oof_source": OOF_SOURCE,
            "wall_s": time.time() - t_start,
            "provenance": provenance.stamp()}
    out = _nan_to_none({"meta": meta, "summary": S, "splits_info": res1["splits_info"],
                        "per_split": per, "trace": trace})
    preds_io.dump_json_atomic(out, args.out)
    print(f"\nĐã ghi {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
