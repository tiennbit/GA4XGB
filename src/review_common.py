# -*- coding: utf-8 -*-
"""Tiện ích dùng chung cho các phân tích bổ sung sau vòng phản biện (Tầng 1).

Mọi script Tầng 1 nạp dữ liệu và cấu hình qua đây để KHÔNG có hai nơi định nghĩa
split hay đường dẫn cấu hình — đúng lý do bins.py tồn tại, áp cho một tầng khác.

⚠️ NGUỒN CẤU HÌNH: chỉ đọc từ results_mseed/ (server, 5 seed). Thư mục results/
là lượt chạy Mac single-run cũ; trộn hai nguồn đã từng làm sai số liệu trong bài
và đã đánh lừa cả reviewer. Không script nào ở đây được đọc results/.
"""
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from preprocess import load_and_preprocess
from ga_xgb import split_params, sample_weights

# Cho phép ghi đè khi smoke-test (REVIEW_SEEDS=42) — chạy thật thì để mặc định.
SEEDS = [int(s) for s in os.environ.get("REVIEW_SEEDS", "42 1 2 3 4").split()]
MSEED_DIR = os.environ.get("REVIEW_DIR", "results_mseed")

# Fold seed RIÊNG cho việc sinh out-of-fold prediction. Cố ý KHÁC random_state=42
# mà GA đã tối ưu trên đó: nếu dùng lại đúng phân hoạch ấy thì `s` được chọn trên
# chính phân hoạch đã bị tối ưu hoá hàng trăm lần -> ước lượng chệch (optimization
# bias, không phải test leakage). Dùng fold khác làm lập luận "không rò rỉ" chặt hơn.
OOF_FOLD_SEED = 7

_DATA = None


def get_data():
    """Trả về (X_tr, X_te, y_tr, y_te) — split CỐ ĐỊNH, giống hệt ga_xgb.py."""
    global _DATA
    if _DATA is None:
        X, y = load_and_preprocess()
        _DATA = train_test_split(X, y, test_size=0.2, random_state=42)
    return _DATA


def cfg_path(name, seed):
    """Đường dẫn *_best.json trong results_mseed cho (tên cấu hình, seed)."""
    suffix = "" if seed == 42 else f"_seed{seed}"
    return os.path.join(MSEED_DIR, f"{name}{suffix}_best.json")


def load_params(name, seed):
    """Trả về (xgb_params, beta). beta là None nếu cấu hình không có gene thứ 8."""
    with open(cfg_path(name, seed)) as f:
        return split_params(json.load(f)["best_params"])


def fit_predict(xgb_params, beta, X_fit, y_fit, X_pred, n_jobs=2):
    m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=n_jobs,
                     verbosity=0, **xgb_params)
    sw = sample_weights(y_fit, beta) if beta is not None else None
    m.fit(X_fit, y_fit, sample_weight=sw)
    return m.predict(X_pred)


def ms(xs):
    """mean ± sd (sd mẫu, ddof=1) — định dạng dùng thống nhất ở Bảng 4/5."""
    a = np.asarray([v for v in xs if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "n": int(a.size)}
