# -*- coding: utf-8 -*-
"""Tập đặc trưng F_full, F_dt, F_dt-cn, F_dt-cn-bc (khung bài 24/9, mục 5.1).

Vì sao không sửa preprocess.load_and_preprocess: các script thăm dò cũ (cost_aware,
tail_prior, center_ablation, final_compare) vẫn gọi nó, và số trong JSON cũ phải
sinh lại được đúng. Thay vào đó, load_frame gọi load_and_preprocess (giữ nguyên
mã hoá ordinal, one-hot, đổi tên cột) rồi BỎ hai cột trường của nó:

- truong_freq: tính trên TOÀN BỘ dữ liệu (kể cả tập kiểm tra) và khoá theo tên
  trần (988 tên, trong khi có 1.125 trường thật vì tên trùng giữa tỉnh).
- truong_chuyen: vô hại về rò rỉ (hàm của từng dòng) nhưng tính lại cùng chỗ.

Hai cột này được tính lại bằng school_basic_features: tần suất trường CHỈ trên hàng
huấn luyện, khoá (tỉnh, tên trường); cờ chuyên giữ phép khớp chuỗi 'huyên' của
preprocess.py cho tới khi có danh sách chính thức (giới hạn ghi trong bài). Hàng
áp dụng có trường chưa thấy trong huấn luyện nhận tần suất 0: đó là đúng điều mô
hình biết lúc tư vấn.

Tập đặc trưng:
  F_full     : mọi cột của preprocess (trừ hai cột trường cũ) + 2 cột trường tính lại.
               Chỉ để đo mất mát do ràng buộc thời điểm.
  F_dt       : F_full bỏ 24 cột lớp 12 học kỳ II và cả năm (tổng kết, học lực,
               hạnh kiểm và 9 môn, mỗi loại hai cột). HSA 2024 thi từ tháng 3 đến
               tháng 6; tư vấn trước kỳ thi chỉ có lớp 10, 11 và 12 học kỳ I.
               Giữ 12.Môn ngoại ngữ (biết từ đầu năm).
  F_dt-cn    : F_dt bỏ giới tính (gioiTinh) và tháng sinh (thangSinh).
  F_dt-cn-bc : F_dt-cn bỏ khuVuc_* và Tỉnh_* one-hot.

Frame không giữ tên trường, tên tỉnh hay ngày sinh: chỉ mã số nguyên (sắp theo
tên để tái lập được) cho bootstrap cụm và kiểm toán, và các cột nhóm nhỏ cho E10.
Các cột nhóm E10 KHÔNG được chép vào npz dự đoán (preds_io, để preds/ không thành
vi dữ liệu nhân khẩu); E10 lấy chúng ở đây, qua preds_io.rows(d, part, tên, frame=F).
"""
import warnings

import numpy as np
import pandas as pd

from preprocess import DATA_PATH, load_and_preprocess

FEATURE_SETS = ["F_full", "F_dt", "F_dt-cn", "F_dt-cn-bc"]
SCHOOL_BASIC_COLS = ["truong_freq", "truong_chuyen"]
SUBJECTS = ["Toán", "Văn", "Vật lí", "Hóa học", "Sinh học", "Lịch sử", "Địa lí",
            "GDCD", "Ngoại ngữ"]
# 12 loại cột x 2 kỳ (HK II, CN) = 24 cột biết SAU thời điểm tư vấn
G12_LATE = [f"12.{k} {s}" for k in ["Điểm tổng kết", "Học lực", "Hạnh kiểm"] + SUBJECTS
            for s in ["HK II", "CN"]]
DEMO_COLS = ["gioiTinh", "thangSinh"]
GEO_PREFIXES = ("khuVuc_", "Tỉnh_")
GROUP_COLS = ["gender", "region", "chuyen", "school_size_q", "birth_quarter"]
SCHOOL_SIZE_Q = 5
_KEY_SEP = "||"


def _is_chuyen(names):
    # Giữ đúng phép khớp của preprocess.py (bắt cả "Chuyên" và "chuyên").
    return pd.Series(names).astype(str).str.contains("huyên", na=False).to_numpy().astype(np.int8)


def school_basic_features(prov, school, idx_tr, idx_apply, chuyen=None):
    """Mảng (len(idx_apply), 2): [tần suất trường, cờ chuyên] cho hàng idx_apply.

    Tần suất = số hàng HUẤN LUYỆN của trường / len(idx_tr), khoá (tỉnh, trường).
    Dùng tỉ lệ chứ không dùng số đếm để mô hình fold (khớp trên ~64% n) và mô hình
    khớp lại (80% n) thấy cùng thang. `prov`, `school` là mảng dài n (tên hoặc mã);
    `chuyen` (dài n) bắt buộc khi `school` là mã, vì cờ chuyên cần tên trường.
    """
    prov, school = np.asarray(prov), np.asarray(school)
    idx_tr, idx_apply = np.asarray(idx_tr), np.asarray(idx_apply)
    key = pd.Series(prov.astype(str)).str.cat(pd.Series(school.astype(str)), sep=_KEY_SEP)
    freq = key.iloc[idx_tr].value_counts() / len(idx_tr)
    f = key.iloc[idx_apply].map(freq).fillna(0.0).to_numpy(float)
    if chuyen is None:
        if not (school.dtype.kind in "OUS"):
            raise ValueError("school là mã số: cần truyền chuyen (cờ chuyên tính từ tên)")
        ch = _is_chuyen(school[idx_apply])
    else:
        ch = np.asarray(chuyen)[idx_apply]
    return np.column_stack([f, ch.astype(float)])


def _codes(values):
    """Mã nguyên sắp theo giá trị (tái lập được giữa các máy), không giữ nhãn."""
    codes, _ = pd.factorize(pd.Series(values).astype(str), sort=True)
    return codes.astype(np.int32)


class Frame:
    """Dữ liệu đã tiền xử lý cho các script thí nghiệm mới.

    Thuộc tính: X (DataFrame, cột gốc chưa có 2 cột trường), y (float), n,
    school_code, prov_code (int32, dài n), chuyen (int8), groups (dict tên -> mảng
    int8 dài n, khoá GROUP_COLS), n_schools, n_provs.
    """

    def __init__(self, X, y, school_code, prov_code, chuyen, groups):
        self.X, self.y = X, y
        self.n = len(y)
        self.school_code, self.prov_code, self.chuyen = school_code, prov_code, chuyen
        self.groups = groups
        self.n_schools = int(school_code.max()) + 1
        self.n_provs = int(prov_code.max()) + 1
        self._Xv = X.to_numpy(float)
        self._col = {c: i for i, c in enumerate(X.columns)}

    def base_columns(self, fset):
        """Cột gốc (chưa gồm 2 cột trường) của một tập đặc trưng."""
        if fset not in FEATURE_SETS:
            raise ValueError(f"tập đặc trưng lạ: {fset}; có {FEATURE_SETS}")
        cols = list(self.X.columns)
        if fset == "F_full":
            return cols
        cols = [c for c in cols if c not in set(G12_LATE)]
        if fset == "F_dt":
            return cols
        cols = [c for c in cols if c not in DEMO_COLS]
        if fset == "F_dt-cn":
            return cols
        return [c for c in cols if not c.startswith(GEO_PREFIXES)]

    def columns(self, fset):
        """Tên cột của ma trận design (cột gốc + 2 cột trường ở cuối)."""
        return self.base_columns(fset) + SCHOOL_BASIC_COLS

    def design(self, fset, idx_fit, *idx_apply):
        """(X_fit, X_apply_1, ...): mảng float, cột trường tính CHỈ từ idx_fit.

        idx_* là chỉ số dòng gốc. Gọi một lần cho mỗi mô hình: fold trong thì
        idx_fit là hàng huấn luyện của fold (kể cả tập dừng sớm), khớp lại thì
        idx_fit là toàn bộ tập huấn luyện ngoài."""
        cols = [self._col[c] for c in self.base_columns(fset)]
        idx_fit = np.asarray(idx_fit)
        # school_code đã là khoá (tỉnh, trường) nên bincount cho đúng tần suất của
        # school_basic_features mà không phải dựng lại khoá chuỗi cho 57k dòng mỗi lần.
        freq = np.bincount(self.school_code[idx_fit], minlength=self.n_schools) / len(idx_fit)
        out = []
        for idx in (idx_fit,) + idx_apply:
            idx = np.asarray(idx)
            sb = np.column_stack([freq[self.school_code[idx]], self.chuyen[idx].astype(float)])
            out.append(np.hstack([self._Xv[np.ix_(idx, cols)], sb]))
        return tuple(out)


def load_frame(path=DATA_PATH):
    """Đọc CSV, tái dùng preprocess.load_and_preprocess, trả Frame."""
    with warnings.catch_warnings():
        # preprocess chèn cột từng cái một nên pandas cảnh báo phân mảnh; vô hại,
        # và sửa preprocess thì đổi hành vi của các script cũ.
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        X, y = load_and_preprocess(path)
    X = X.drop(columns=SCHOOL_BASIC_COLS).copy()
    missing = [c for c in G12_LATE if c not in X.columns]
    assert not missing, f"thiếu cột lớp 12 cần bỏ ở F_dt: {missing}"
    raw = pd.read_csv(path, usecols=["Trường", "Tỉnh", "Giới tính", "Ngày sinh", "khuVuc"])
    assert len(raw) == len(X), "preprocess đã đổi số dòng; Frame giả định giữ nguyên"
    prov = raw["Tỉnh"].astype(str)
    school = raw["Trường"].astype(str)
    # Khoá trường gồm cả tỉnh: tên như "THPT Nguyễn Du" trùng giữa nhiều tỉnh.
    school_code = _codes(prov.str.cat(school, sep=_KEY_SEP))
    prov_code = _codes(prov)
    chuyen = _is_chuyen(school)

    month = pd.to_datetime(raw["Ngày sinh"], format="%d/%m/%Y", errors="coerce").dt.month
    birth_q = ((month - 1) // 3 + 1).fillna(0).astype(np.int8).to_numpy()   # 0 = không rõ
    # Ngũ phân vị cỡ trường theo số em (mỗi em mang cỡ trường của mình), cắt bằng qcut
    # trên cỡ nên một trường không bị xẻ sang hai nhóm. Đây là biến nhóm mô tả cho
    # kiểm toán E10, không phải đặc trưng mô hình, nên được tính trên mọi hàng.
    size = np.bincount(school_code)[school_code]
    size_q = pd.qcut(size, SCHOOL_SIZE_Q, labels=False, duplicates="drop").astype(np.int8)
    groups = {"gender": (raw["Giới tính"] == "Nam").astype(np.int8).to_numpy(),
              "region": _codes(raw["khuVuc"]).astype(np.int8),
              "chuyen": chuyen,
              "school_size_q": np.asarray(size_q),
              "birth_quarter": birth_q}
    return Frame(X, y.to_numpy(float), school_code, prov_code, chuyen, groups)
