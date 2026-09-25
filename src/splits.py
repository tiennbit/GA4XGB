# -*- coding: utf-8 -*-
"""Kế hoạch chia dữ liệu và ngẫu nhiên hoá dùng CHUNG cho mọi thí nghiệm E1 đến E11.

Vì sao gom về một chỗ (khung bài 24/9, mục 6): thăm dò cũ mỗi script tự gọi
train_test_split/KFold với seed riêng, nên hai "5 lần chia" của hai script không
chắc là cùng lần chia, và OOF của cấu hình được chọn vừa dùng để dừng sớm vừa dùng
làm phân phối phần dư. Ở đây:

- Lần chia ngoài: train_test_split(arange(n), test_size, random_state=seed), hoặc
  GroupShuffleSplit theo khoá băm bản ghi nếu E0b tìm thấy dòng trùng.
- Fold trong: KFold(CV_FOLDS, shuffle=True, random_state=seed) (GroupKFold nếu có
  nhóm). CV_FOLDS import từ ga_xgb.py, nguồn sự thật duy nhất (AGENTS.md).
- Tập dừng sớm: 10% dòng của MỖI fold huấn luyện trong, rng = default_rng(1000*seed
  + fold). Tập này là con của fold huấn luyện, nên không bao giờ giao với fold giữ
  lại của cùng fold: OOF chỉ lấy trên fold giữ lại, không bao giờ trên tập dừng sớm.

Quy ước chỉ số (chọn cách đơn giản, khớp mã cũ `p_oof[ova]`, `X[tr[otr]]`):
  outer_split trả CHỈ SỐ DÒNG gốc (0..n-1). inner_folds, es_split và fold_plan trả
  VỊ TRÍ trong idx_tr (0..len(idx_tr)-1), để mảng OOF dài len(idx_tr) đánh chỉ số
  thẳng bằng chúng; muốn ra chỉ số dòng gốc thì lấy idx_tr[vị trí].
  outer_split KHÔNG sắp lại idx_tr: thứ tự do train_test_split trả về quyết định
  fold nào nhận dòng nào, và giữ nguyên nó là cách duy nhất để lần chia tái lập
  đúng lời gọi viết trong khung bài.

Nhóm (groups): outer_split, inner_folds, fold_plan nhận mảng dài n, thẳng hàng với
dòng gốc. Riêng es_split nhận nhãn nhóm CỦA CHÍNH các vị trí itr (dài len(itr)),
vì hàm này chỉ thấy vị trí, không thấy idx_tr; fold_plan tự chuyển. Khi có nhóm,
tập dừng sớm bốc cả nhóm (một khoá bản ghi không bao giờ nằm cả ở phần khớp lẫn
phần dừng sớm), cùng rng default_rng(1000*seed + fold). Không có nhóm thì kết quả
giữ nguyên từng bit như trước.
"""
import os

import numpy as np
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     train_test_split)

from gates import ES_FRAC, TEST_SIZE
# ga_xgb chỉ định nghĩa hằng số và hàm khi import (đã kiểm: không đọc dữ liệu,
# không chạy GA), nên import thẳng thay vì chép giá trị.
from ga_xgb import BASE_GENES, CV_FOLDS

# Khung bài chốt OOF 5-fold cho mọi trung tâm, quy tắc và nhánh có trọng số.
# Ngày 24/9, CV_FOLDS = 5 mới có trong bản làm việc của ga_xgb.py (HEAD còn 3, chờ
# commit ở E0 bước 5). Chặn ngay lúc import, kể cả trên server, để một checkout
# sạch của commit cũ hay một lần quay lại bản cũ không âm thầm chạy 3 fold. Đây là phép kiểm, không phải nguồn thứ hai
# của giá trị: đổi số fold thì sửa ga_xgb.py và dòng này cùng lúc, có lý do.
if CV_FOLDS != 5:
    raise ImportError(f"ga_xgb.CV_FOLDS = {CV_FOLDS}, giao thức E1..E11 cần 5 "
                      "(ga_xgb.py ở commit cũ? xem AGENTS.md, E0 bước 5)")


def outer_split(n, seed, groups=None, test_size=TEST_SIZE):
    """(idx_tr, idx_te): chỉ số dòng gốc của tập huấn luyện và kiểm tra."""
    idx = np.arange(n)
    if groups is None:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
        return np.asarray(tr), np.asarray(te)
    groups = np.asarray(groups)
    assert len(groups) == n, "groups phải dài n, thẳng hàng với dòng gốc"
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(idx, groups=groups))
    return tr, te


def inner_folds(idx_tr, seed, groups=None, n_folds=CV_FOLDS):
    """[(tr_pos, va_pos)] cho từng fold trong; VỊ TRÍ trong idx_tr.

    Có nhóm thì dùng GroupKFold(shuffle=True) để hai dòng cùng khoá không nằm ở hai
    phía của một fold (sklearn >= 1.6 mới có shuffle cho GroupKFold)."""
    idx_tr = np.asarray(idx_tr)
    pos = np.arange(len(idx_tr))
    if groups is None:
        kf = KFold(n_folds, shuffle=True, random_state=seed)
        return [(a, b) for a, b in kf.split(pos)]
    groups = np.asarray(groups)
    assert len(groups) > idx_tr.max(), "groups phải dài n (thẳng hàng dòng gốc), không phải dài idx_tr"
    g = groups[idx_tr]
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(a, b) for a, b in gkf.split(pos, groups=g)]


def es_split(itr, seed, fold, frac=ES_FRAC, groups=None):
    """(fit_pos, es_pos): tách tập dừng sớm khỏi fold huấn luyện trong `itr`.

    rng riêng cho từng (seed, fold) để đổi số fold hay thứ tự chạy không làm đổi
    tập dừng sớm của fold khác. Trả về con của itr, giữ thứ tự của itr.

    `groups` (tuỳ chọn) là nhãn nhóm của từng phần tử itr (dài len(itr)). Khi có,
    bốc NGUYÊN NHÓM theo một hoán vị ngẫu nhiên của các nhóm cho tới khi đủ
    round(frac·len(itr)) dòng (tiền tố ngắn nhất đạt mục tiêu), nên hai dòng trùng
    khoá không nằm hai phía: dừng sớm trên bản gần như sao chép của dòng khớp sẽ lạc
    quan, làm phồng best_iteration và số cây khi khớp lại."""
    itr = np.asarray(itr)
    rng = np.random.default_rng(1000 * int(seed) + int(fold))
    n_es = int(round(frac * len(itr)))
    pick = np.zeros(len(itr), dtype=bool)
    if groups is None:
        pick[rng.choice(len(itr), size=n_es, replace=False)] = True
        return itr[~pick], itr[pick]
    groups = np.asarray(groups)
    assert len(groups) == len(itr), "groups của es_split là nhãn của từng phần tử itr (dài len(itr))"
    uniq, inv = np.unique(groups, return_inverse=True)
    size = np.bincount(inv, minlength=len(uniq))
    order = rng.permutation(len(uniq))
    k = int(np.searchsorted(np.cumsum(size[order]), n_es, side="left")) + 1 if n_es > 0 else 0
    chosen = np.zeros(len(uniq), dtype=bool)
    chosen[order[:k]] = True
    pick = chosen[inv]
    return itr[~pick], itr[pick]


def fold_plan(idx_tr, seed, groups=None, n_folds=CV_FOLDS, es_frac=ES_FRAC):
    """Kế hoạch đầy đủ cho một lần chia: [{fold, fit, es, va}] (vị trí trong idx_tr).

    Mô hình có dừng sớm khớp trên `fit`, dừng trên `es`, dự đoán OOF trên `va`.
    Mô hình không dừng sớm (default, bag) khớp trên fit ∪ es."""
    plan = []
    idx_tr = np.asarray(idx_tr)
    g_tr = None if groups is None else np.asarray(groups)[idx_tr]
    for k, (a, b) in enumerate(inner_folds(idx_tr, seed, groups, n_folds)):
        fit, es = es_split(a, seed, k, es_frac, groups=None if g_tr is None else g_tr[a])
        plan.append({"fold": k, "fit": fit, "es": es, "va": b, "train": a})
    return plan


def fold_of(plan, n_tr):
    """Mảng dài n_tr: fold mà mỗi vị trí nằm ở phía giữ lại (để lưu vào npz)."""
    out = np.full(n_tr, -1, dtype=np.int8)
    for f in plan:
        out[f["va"]] = f["fold"]
    assert (out >= 0).all(), "có vị trí không thuộc fold giữ lại nào"
    return out


# ---------------------------------------------------------------------------
# Mẫu cấu hình cho rs_tuned (E1) và R8 (E2b): cùng một bảng cho cả hai phía
# ---------------------------------------------------------------------------
def sample_configs(n, seed):
    """n cấu hình lấy mẫu MỘT lần cho mỗi lần chia, rng = default_rng(seed).

    Không gian lấy từ ga_xgb.BASE_GENES, bỏ n_estimators (thay bằng dừng sớm 50
    vòng, tối đa 3.000 cây, vì 29/40 lượt GA cũ chạm trần 800). Gene log-thang lấy
    log-uniform, gene nguyên lấy nguyên đều. E1 và E2b gọi cùng hàm với cùng seed
    nên hai phía dò trên đúng một mẫu cấu hình ("ngân sách như nhau")."""
    rng = np.random.default_rng(int(seed))
    out = []
    for _ in range(int(n)):
        cfg = {}
        for name, lo, hi, typ, log in BASE_GENES:
            if name == "n_estimators":
                continue
            if typ is int:
                cfg[name] = int(rng.integers(int(lo), int(hi) + 1))
            elif log:
                cfg[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                cfg[name] = float(rng.uniform(lo, hi))
        out.append(cfg)
    return out


def xgb_threads(workers):
    """n_jobs cho mỗi XGBoost khi chạy `workers` tiến trình joblib song song."""
    return max(1, (os.cpu_count() or 1) // max(1, int(workers)))
