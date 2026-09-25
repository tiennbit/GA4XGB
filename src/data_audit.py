# -*- coding: utf-8 -*-
"""E0b: kiểm dữ liệu trước mọi lần chia (khung bài 24/9, mục 6.3 và 5.1).

Vì sao cần: header không có mã thí sinh, đợt thi, ngày thi hay mã đề, trong khi HSA
2024 thi nhiều đợt. Chưa biết một dòng là một thí sinh hay một lượt thi. Nếu hai lượt
của cùng một em rơi vào hai phía của lần chia, tập kiểm tra chứa bản gần như sao chép
của tập huấn luyện và mọi số đều lạc quan. E0b đếm trước khi E1 chạy, và cổng của nó
quyết định có phải chia theo nhóm (khoá băm bản ghi) hay không.

ĐẦU RA CHỈ CÓ SỐ ĐẾM VÀ PHÂN VỊ. Không dòng dữ liệu, không tên trường, không tên tỉnh,
không ngày sinh, không giá trị băm. File này đọc data/data_final.csv tại chỗ trên
server, không chép và không ghi file dữ liệu mới. Trước khi ghi JSON, _assert_no_pii
quét mọi chuỗi của kết quả (khoá lẫn giá trị) và dừng nếu gặp tên trường, tên tỉnh
hay chuỗi ngày sinh của dữ liệu: JSON và log được pulldir về Mac, nên đây là chốt
cuối. Nhãn danh mục (khuVuc, năm sinh, nhãn học lực lạ) có ít hơn MIN_CELL dòng bị
gộp vào "khác".

Quy trình (đánh số như mục 6.3):
 (1) Khoá chính xác = SHA-256 của chuỗi ghép (ngày sinh, giới tính, tỉnh, trường, mọi
     cột học bạ đã chuẩn hoá chuỗi). Đếm số nhóm có từ 2 dòng, số dòng trong các nhóm
     đó, phân bố |Δy| trong nhóm (max - min), và y dòng sau trừ y dòng trước theo thứ
     tự file.
 (2) Khoá gần = SHA-256 của (tỉnh, trường, mọi cột học bạ), không có ngày sinh và
     giới tính. Đếm như trên, thêm số nhóm gần gồm từ hai khoá chính xác trở lên (cùng
     học bạ, khác ngày sinh hoặc giới: lỗi nhập hoặc đăng ký lại).
 (3) Đếm y <= NEAR_GUESS_MAX (37) và phân bố theo khuVuc và tam phân vị cỡ tỉnh.
 (4) Luồng mẫu cho Bảng I: số dòng đầu vào, số dòng bị ảnh hưởng ở từng bước của
     preprocess.py, số cuối (từ features.load_frame, đúng đường các thí nghiệm dùng);
     số trường theo khoá (tỉnh, tên), số tên trường trần, số tỉnh.
 (5) Danh sách câu hỏi gửi IDT (mục 10, ý 5) kèm số đếm làm bằng chứng.
Thêm (rẻ, cùng mục đích): khoá toàn dòng (mọi cột trừ STT, kể cả y và khuVuc) để tách
bản sao nhập trùng khỏi thi lại; STT có đánh lại từ 1 không (dấu hiệu file ghép theo
đợt); histogram y theo từng điểm cho Hình 3; ngưỡng đuôi theo khối lượng trên toàn khoá
và trên tập huấn luyện của từng seed (kiểm "trùng 60/100" của mục 5.1).

Lựa chọn khi khung bài chưa rõ (ghi cả vào meta.choices của JSON):
- "129 cột học bạ" của mục 6.3 không khớp header: header có 111 cột học bạ (37 mỗi
  lớp: 3 điểm tổng kết, 3 học lực, 3 hạnh kiểm, 27 điểm môn, 1 môn ngoại ngữ). Khoá
  dùng MỌI cột có tên bắt đầu bằng "10.", "11.", "12." (TRANSCRIPT_PREFIXES), nên
  đúng ý "toàn bộ học bạ" bất kể con số.
- Chuẩn hoá chuỗi: Unicode NFC, gộp khoảng trắng, chữ thường; ô trống hay dạng NA
  thành ""; ô đọc được thành số được viết lại bằng repr(float) ("8", "8.0", "8,0"
  cùng thành "8.0"); ngày sinh đọc được theo %d/%m/%Y (như preprocess.py) viết lại
  thành YYYY-MM-DD. Chữ thường để hai lần nhập khác hoa thường vẫn trùng khoá.
- "Dòng trùng vượt 1%" của cổng: tỉ lệ số dòng NẰM TRONG nhóm trùng chính xác trên n
  (cách đếm lớn hơn, nên thận trọng hơn); số dòng thừa (dòng trừ nhóm) báo kèm.
- Tam phân vị cỡ tỉnh: như features.py làm với ngũ phân vị cỡ trường, mỗi dòng mang cỡ
  tỉnh của mình rồi pd.qcut theo dòng (duplicates="drop"), nên một tỉnh không bị xẻ.
- |Δy| trong nhóm là max - min (với nhóm 2 dòng đúng là |y1 - y2|).
- Ngưỡng DUP_STOP_FRAC = 0,01 khai ở đây vì gates.py là hạ tầng đã duyệt, không sửa;
  nên chuyển vào gates.py ở lần sửa hạ tầng tới.
- Không có .partial và chạy tiếp: cả script chạy vài chục giây. --preds-dir và
  --workers nhận cho đồng bộ giao diện nhưng không dùng (E0b không ghi file theo dòng,
  không khớp mô hình).

Thí nghiệm sau lấy nhóm cho lần chia bằng record_groups(path, "exact") (E5 bỏ nhóm
trùng gần bằng record_groups(path, "near")), chỉ trong bộ nhớ, trên server:

    groups = data_audit.record_groups(args.data) if audit["gates"]["group_split_required"] else None
    tr, te = splits.outer_split(F.n, seed, groups=groups)

Chạy (server): PYTHONPATH=src python3 src/data_audit.py --out results_cost/data_audit.json
Kiểm thử khói (Mac, CSV giả): PYTHONPATH=src python3 src/data_audit.py \
    --data tests/fixtures/fake_hsa.csv --out <scratch>/data_audit.json --smoke
"""
import argparse
import hashlib
import math
import os
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

import decision_layer as dl
import features
import preds_io
import provenance
import splits
from bins import HIGH_TAIL_MIN, LOW_TAIL_MAX
from gates import (NEAR_GUESS_MAX, SEEDS, SEM_ASSUMED, TAIL_MASS_HIGH, TAIL_MASS_LOW,
                   TUNE_BUDGET)
from preprocess import DATA_PATH, HANH_KIEM, HOC_LUC, TARGET

COL_STT, COL_DOB, COL_GENDER = "STT", "Ngày sinh", "Giới tính"
COL_SCHOOL, COL_PROV, COL_REGION = "Trường", "Tỉnh", "khuVuc"
TRANSCRIPT_PREFIXES = ("10.", "11.", "12.")
DOB_FORMAT = "%d/%m/%Y"                    # như preprocess.py
KEY_SEP = "\x1f"                           # ký tự phân cách đơn vị: không gặp trong dữ liệu
NA_LIKE = {"", "nan", "none", "null", "na", "n/a", "#n/a", "<na>", "nat"}
MIN_CELL = 10                              # nhãn danh mục ít hơn số dòng này gộp vào "khác"
PII_SUBSTR_MIN = 8                         # tên cấm dài từ chừng này ký tự thì kiểm cả chứa-trong
DUP_STOP_FRAC = 0.01                       # cổng E0b: dòng trùng > 1% thì dừng E1 chờ IDT
QS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
DY_BINS = [(0, 0), (0, 2), (2, 5), (5, 10), (10, 20), (20, math.inf)]   # (lo, hi]; (0,0] = đúng 0
RELIABILITIES = (0.85, 0.90, 0.95)
GPA_COL = "11.Điểm tổng kết CN"            # học bạ có trước kỳ thi, so nhóm gần đoán mò


# ---------------------------------------------------------------------------
# Chặn chạy trên dữ liệu thật ở Mac
# ---------------------------------------------------------------------------
def guard_real_data(path):
    """Dừng nếu đang ở Mac mà --data là file thật (AGENTS.md, ràng buộc 1: thí nghiệm chỉ
    chạy ở server; Mac chỉ chạy kiểm thử khói trên CSV giả). Server là Linux nên không
    bị chặn. Không có biến môi trường nào để vượt: cần chạy thật thì chạy ở server."""
    real = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         DATA_PATH))
    if sys.platform == "darwin" and (os.path.realpath(path) == real
                                     or os.path.basename(path) == os.path.basename(DATA_PATH)):
        sys.exit(f"Không chạy trên dữ liệu thật ở máy này ({path}); dùng --data "
                 "tests/fixtures/fake_hsa.csv, chạy thật ở server (AGENTS.md, ràng buộc 1).")


# ---------------------------------------------------------------------------
# Chuẩn hoá chuỗi và khoá băm
# ---------------------------------------------------------------------------
def _norm_one(v):
    t = " ".join(unicodedata.normalize("NFC", str(v)).split()).casefold()
    if t in NA_LIKE:
        return ""
    try:
        x = float(t.replace(",", "."))
    except ValueError:
        return t
    return repr(x) if math.isfinite(x) else t


def norm_column(s):
    """Chuẩn hoá một cột chuỗi. Ánh xạ qua giá trị duy nhất: cột điểm chỉ có vài trăm
    giá trị khác nhau, nên nhanh hơn nhiều so với gọi hàm cho từng ô."""
    s = s.astype(str)
    return s.map({u: _norm_one(u) for u in pd.unique(s)})


def norm_dob(s):
    """Ngày sinh đọc được theo %d/%m/%Y thành YYYY-MM-DD; không đọc được thì chuẩn hoá chuỗi."""
    d = pd.to_datetime(s.astype(str).str.strip(), format=DOB_FORMAT, errors="coerce")
    iso = d.dt.strftime("%Y-%m-%d")
    return iso.where(d.notna(), norm_column(s))


def transcript_columns(columns):
    return [c for c in columns if str(c).startswith(TRANSCRIPT_PREFIXES)]


def read_raw(path):
    """CSV dạng chuỗi thô: ô trống là "" (không NaN), để khoá không phụ thuộc cách pandas
    đoán kiểu. Chỉ dùng trong bộ nhớ."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _hash_rows(parts):
    joined = parts[0].str.cat(parts[1:], sep=KEY_SEP) if len(parts) > 1 else parts[0]
    return np.array([hashlib.sha256(k.encode("utf-8")).hexdigest() for k in joined], dtype=object)


def record_keys(raw, kind="exact"):
    """Mảng sha256 hex dài n, thẳng hàng với dòng CSV. KHÔNG ghi ra file hay JSON.

    exact: (ngày sinh, giới tính, tỉnh, trường, học bạ). near: (tỉnh, trường, học bạ).
    full: mọi cột trừ STT, theo thứ tự header (kể cả y và khuVuc)."""
    tcols = transcript_columns(raw.columns)
    if kind == "exact":
        parts = [norm_dob(raw[COL_DOB])] + [norm_column(raw[c])
                                             for c in [COL_GENDER, COL_PROV, COL_SCHOOL] + tcols]
    elif kind == "near":
        parts = [norm_column(raw[c]) for c in [COL_PROV, COL_SCHOOL] + tcols]
    elif kind == "full":
        parts = [norm_dob(raw[c]) if c == COL_DOB else norm_column(raw[c])
                 for c in raw.columns if c != COL_STT]
    else:
        raise ValueError(f"kind lạ: {kind}")
    return _hash_rows(parts)


def record_groups(path=DATA_PATH, kind="exact"):
    """Mã nhóm int32 dài n (thứ tự dòng CSV, trùng thứ tự features.load_frame) theo khoá
    băm bản ghi, cho splits.outer_split(..., groups=...). Chỉ trong bộ nhớ."""
    codes, _ = pd.factorize(pd.Series(record_keys(read_raw(path), kind)), sort=True)
    return codes.astype(np.int32)


# ---------------------------------------------------------------------------
# Thống kê nhỏ
# ---------------------------------------------------------------------------
def _qd(x, qs=QS):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    return {f"p{round(q * 100, 2):g}": float(np.quantile(x, q)) for q in qs}


def _desc(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "quantiles": _qd(x)}


def _pooled_counts(values, min_cell=MIN_CELL, other="khác"):
    """{nhãn: số dòng}, nhãn có ít hơn min_cell dòng gộp vào `other`."""
    vc = pd.Series(values).astype(str).value_counts()
    out = {str(k): int(v) for k, v in vc.items() if v >= min_cell}
    rest = int(vc[vc < min_cell].sum())
    if rest:
        out[f"{other} (<{min_cell} dòng mỗi nhãn, {int((vc < min_cell).sum())} nhãn)"] = rest
    return out


def _pool_labels(values, min_cell=MIN_CELL, other="khác"):
    s = pd.Series(values).astype(str)
    vc = s.value_counts()
    keep = set(vc[vc >= min_cell].index)
    return s.where(s.isin(keep), other).to_numpy()


def dup_stats(keys, y):
    """Số nhóm có từ 2 dòng, số dòng trong đó, cỡ nhóm, |Δy| và y sau trừ y trước."""
    n = len(keys)
    codes, _ = pd.factorize(pd.Series(keys), sort=False)
    size = np.bincount(codes)
    in_dup = size[codes] >= 2
    n_groups = int((size >= 2).sum())
    n_rows = int(in_dup.sum())
    out = {"n_distinct": int(len(size)), "n_groups": n_groups, "n_rows": n_rows,
           "n_excess_rows": n_rows - n_groups,
           "row_frac": n_rows / n if n else 0.0,
           "excess_frac": (n_rows - n_groups) / n if n else 0.0,
           "max_group_size": int(size.max()) if len(size) else 0,
           "group_size": {"2": int((size == 2).sum()), "3": int((size == 3).sum()),
                          "4": int((size == 4).sum()), ">=5": int((size >= 5).sum())}}
    if n_groups == 0:
        return out
    d = pd.DataFrame({"g": codes[in_dup], "y": np.asarray(y, dtype=float)[in_dup]})
    agg = d.groupby("g", sort=False)["y"].agg(["min", "max", "first", "last"])
    rng = (agg["max"] - agg["min"]).to_numpy()
    signed = (agg["last"] - agg["first"]).to_numpy()
    hist = {}
    for lo, hi in DY_BINS:
        m = (rng == 0) if hi == 0 else ((rng > lo) & (rng <= hi))
        hist["0" if hi == 0 else (f">{lo:g}" if math.isinf(hi) else f"({lo:g},{hi:g}]")] = int(m.sum())
    out["abs_dy_range"] = {"desc": _desc(rng), "n_zero": int((rng == 0).sum()), "hist": hist}
    out["dy_last_minus_first"] = {"desc": _desc(signed), "n_pos": int((signed > 0).sum()),
                                  "n_neg": int((signed < 0).sum()), "n_zero": int((signed == 0).sum())}
    return out


def near_not_exact(near, exact):
    """Nhóm gần chứa từ hai khoá chính xác: cùng trường và học bạ, khác ngày sinh/giới."""
    df = pd.DataFrame({"near": pd.factorize(pd.Series(near))[0],
                       "exact": pd.factorize(pd.Series(exact))[0]})
    k = df.groupby("near")["exact"].nunique()
    many = k[k >= 2].index
    return {"n_groups": int(len(many)), "n_rows": int(df["near"].isin(many).sum())}


def stt_structure(stt):
    """STT có là 1..n liên tục không, và có đánh lại (giảm) không: file ghép theo đợt
    thường đánh lại STT từ 1 ở mỗi phần. Chỉ đếm; cỡ các đoạn là số đếm."""
    v = pd.to_numeric(pd.Series(stt), errors="coerce").to_numpy(float)
    ok = np.isfinite(v)
    out = {"n_non_numeric": int((~ok).sum())}
    if ok.sum() < 2:
        return out
    vv = v[ok]
    resets = np.flatnonzero(np.diff(vv) < 0)
    seg = np.diff(np.r_[0, resets + 1, len(vv)])
    out.update({"n_unique": int(len(np.unique(vv))), "min": float(vv.min()), "max": float(vv.max()),
                "is_1_to_n": bool(np.array_equal(vv, np.arange(1, len(vv) + 1))),
                "n_resets": int(len(resets)), "segment_sizes": [int(s) for s in seg[:50]],
                "n_segments": int(len(seg))})
    return out


# ---------------------------------------------------------------------------
# Các phần của báo cáo
# ---------------------------------------------------------------------------
def header_info(columns):
    cols = [str(c) for c in columns]
    tcols = transcript_columns(cols)
    other = [c for c in cols if c not in tcols]
    low = [c.casefold() for c in other]

    def has(*tokens):
        return any(t in c for c in low for t in tokens)

    return {"n_cols": len(cols), "n_transcript_cols": len(tcols),
            "transcript_cols_per_grade": {p: sum(c.startswith(p) for c in tcols) for p in TRANSCRIPT_PREFIXES},
            "non_transcript_cols": other,
            # Trường khung bài nói header KHÔNG có (mục 5.1): kiểm lại bằng tên cột.
            "has_candidate_id": has("sbd", "số báo danh", "mã thí sinh", "mã hs", "cccd", "cmnd"),
            "has_round": has("đợt", "lượt", "ca thi"),
            "has_exam_date": has("ngày thi"),
            "has_test_form": has("mã đề", "đề thi"),
            "has_section_scores": has("định lượng", "định tính", "khoa học", "phần ", "thành phần")}


def target_info(y_raw):
    y = pd.to_numeric(y_raw, errors="coerce").to_numpy(float)
    fin = np.isfinite(y)
    yf = y[fin]
    out = {"n": int(len(y)), "n_missing_or_non_numeric": int((~fin).sum()),
           "n_non_integer": int(np.sum(yf != np.round(yf))),
           "n_out_of_0_150": int(np.sum((yf < 0) | (yf > 150))),
           "desc": _desc(yf),
           "min": float(yf.min()) if len(yf) else None, "max": float(yf.max()) if len(yf) else None}
    if len(yf):
        lo, hi = dl.tail_cutoffs(yf)
        ml, mh = dl.tail_masses(yf, lo, hi)
        sd = float(yf.std(ddof=1))
        out["regions_60_100"] = {"n_low": int(np.sum(yf < LOW_TAIL_MAX)),
                                 "n_high": int(np.sum(yf >= HIGH_TAIL_MIN)),
                                 "mass_low": float(np.mean(yf < LOW_TAIL_MAX)),
                                 "mass_high": float(np.mean(yf >= HIGH_TAIL_MIN))}
        out["mass_cutoffs_all"] = {"target_mass_low": TAIL_MASS_LOW, "target_mass_high": TAIL_MASS_HIGH,
                                   "lo": lo, "hi": hi, "mass_low": ml, "mass_high": mh,
                                   "equals_60_100": bool(lo == LOW_TAIL_MAX and hi == HIGH_TAIL_MIN)}
        # SEM = SD·sqrt(1 - độ tin cậy): độ tin cậy thật chưa có (câu hỏi IDT), nên báo
        # theo vài giá trị giả định; gates.SEM_ASSUMED = 4,3 ứng với 0,90.
        out["sem_assumed"] = {"sem_gates": SEM_ASSUMED,
                              "by_reliability": {f"{r:.2f}": sd * math.sqrt(1 - r) for r in RELIABILITIES}}
        # Histogram theo từng điểm (floor) cho Hình 3: số đếm, không có dòng.
        vc = pd.Series(np.floor(yf).astype(int)).value_counts().sort_index()
        out["hist_by_score"] = {str(int(k)): int(v) for k, v in vc.items()}
    return out


def near_guess_info(raw, y, prov_code):
    """(3) y <= NEAR_GUESS_MAX: tổng, theo khuVuc, theo tam phân vị cỡ tỉnh, và học bạ
    lớp 11 của nhóm này so với toàn bộ (học bạ trung bình mà điểm gần đoán mò gợi ý
    làm bài không nghiêm túc, điều học bạ không dự đoán được)."""
    y = np.asarray(y, dtype=float)
    ng = np.isfinite(y) & (y <= NEAR_GUESS_MAX)
    out = {"threshold": NEAR_GUESS_MAX, "n": int(ng.sum()), "frac": float(ng.mean()) if len(y) else 0.0}

    reg = norm_column(raw[COL_REGION]).str.upper().replace("", "(trống)")
    reg = _pool_labels(reg.to_numpy())
    by_reg = {}
    for lab in sorted(set(reg)):
        m = reg == lab
        by_reg[lab] = {"n": int(m.sum()), "n_near_guess": int((ng & m).sum()),
                       "rate": float((ng & m).sum() / m.sum())}
    out["by_region"] = by_reg

    size = np.bincount(prov_code)[prov_code]
    # Mọi tỉnh cùng cỡ thì qcut trả NaN: gộp thành một nhóm thay vì lỗi.
    terc = np.nan_to_num(np.asarray(pd.qcut(size, 3, labels=False, duplicates="drop"), dtype=float)).astype(int)
    by_t = {}
    for t in sorted(set(terc.tolist())):
        m = terc == t
        by_t[f"T{int(t) + 1}"] = {"n_provinces": int(len(np.unique(prov_code[m]))), "n": int(m.sum()),
                                  "prov_size_min": int(size[m].min()), "prov_size_max": int(size[m].max()),
                                  "n_near_guess": int((ng & m).sum()),
                                  "rate": float((ng & m).sum() / m.sum())}
    out["by_province_size_tercile"] = by_t

    if GPA_COL in raw.columns:
        g = pd.to_numeric(raw[GPA_COL].str.replace(",", ".", regex=False), errors="coerce").to_numpy(float)
        out["gpa_11_cn"] = {"near_guess": _desc(g[ng]), "all": _desc(g)}
    return out


def sample_flow(path, raw):
    """(4) Luồng mẫu theo đúng các bước của preprocess.py. preprocess KHÔNG bỏ dòng nào:
    ngày sinh lỗi thành tháng 0, giới tính khác "Nam" thành 0, còn NaN (kể cả nhãn học
    lực/hạnh kiểm lạ, trường trống) thì assert của nó dừng cả lượt. Vì vậy mỗi bước báo
    số dòng bị ảnh hưởng và cách xử lý, và số cuối lấy từ features.load_frame."""
    pp = pd.read_csv(path)                  # đúng cách preprocess đọc (NA mặc định của pandas)
    n0 = len(pp)
    steps = [{"step": "đọc CSV", "n_rows": n0}]

    def add(name, mask, action):
        steps.append({"step": name, "n_affected": int(np.asarray(mask).sum()), "action": action})

    y = pd.to_numeric(pp[TARGET], errors="coerce")
    add("y trống hoặc không phải số", y.isna(), "chặn: y NaN, mô hình không khớp được")
    add("y ngoài [0; 150]", (y < 0) | (y > 150), "giữ (không có bước lọc)")
    dob_na = pp[COL_DOB].isna()
    dob = pd.to_datetime(pp[COL_DOB], format=DOB_FORMAT, errors="coerce")
    add("ngày sinh trống", dob_na, "giữ, thangSinh = 0")
    add("ngày sinh sai định dạng %d/%m/%Y", dob.isna() & ~dob_na, "giữ, thangSinh = 0")
    g = pp[COL_GENDER]
    add("giới tính không phải 'Nam' hay 'Nữ'", ~g.isin(["Nam", "Nữ"]), "giữ, gioiTinh = 0 (như Nữ)")

    unmapped = {}
    for label, mapping, key in [("Học lực", HOC_LUC, "hoc_luc"), ("Hạnh kiểm", HANH_KIEM, "hanh_kiem")]:
        cols = [c for c in pp.columns if label in c]
        bad = pd.DataFrame({c: pp[c].notna() & ~pp[c].isin(list(mapping)) for c in cols})
        miss = pd.DataFrame({c: pp[c].isna() for c in cols})
        add(f"nhãn {label.lower()} lạ", bad.any(axis=1), "chặn: map ra NaN, assert của preprocess dừng")
        add(f"nhãn {label.lower()} trống", miss.any(axis=1), "chặn: NaN, assert của preprocess dừng")
        vals = pd.concat([pp.loc[bad[c], c] for c in cols]) if len(cols) else pd.Series([], dtype=str)
        unmapped[key] = _pooled_counts(vals) if len(vals) else {}

    tcols = transcript_columns(pp.columns)
    score_cols = [c for c in tcols if not any(t in c for t in ("Học lực", "Hạnh kiểm", "Môn ngoại ngữ"))]
    num = pp[score_cols].apply(pd.to_numeric, errors="coerce")
    add("ô điểm học bạ trống", pp[score_cols].isna().any(axis=1), "chặn: NaN, assert của preprocess dừng")
    add("ô điểm học bạ không phải số", (num.isna() & pp[score_cols].notna()).any(axis=1),
        "chặn: cột kiểu object, XGBoost không nhận")
    add("điểm học bạ ngoài [0; 10]", ((num < 0) | (num > 10)).any(axis=1), "giữ (không có bước lọc)")
    # pandas 3: astype(str) GIỮ NaN (không thành "nan"), nên ở features.load_frame một
    # trường hay tỉnh trống cho mã -1 và np.bincount báo lỗi. Đếm riêng để biết trước.
    add("trường trống", pp[COL_SCHOOL].isna(),
        "chặn: preprocess (truong_freq NaN) và features.load_frame (mã -1)")
    add("tỉnh trống", pp[COL_PROV].isna(), "preprocess giữ (one-hot toàn 0); features.load_frame chặn (mã -1)")
    cat_cols = [COL_REGION] + [c for c in tcols if "Môn ngoại ngữ" in c]
    add("khuVuc hoặc môn ngoại ngữ trống", pp[cat_cols].isna().any(axis=1), "giữ, one-hot toàn 0")

    final = {"ok": False, "n_rows": None, "error": None, "n_features": {}}
    F = None
    try:
        F = features.load_frame(path)
        final.update(ok=True, n_rows=int(F.n),
                     n_features={fs: len(F.columns(fs)) for fs in features.FEATURE_SETS})
    except Exception as exc:        # báo loại lỗi, không để lượt kiểm dừng giữa chừng
        final["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    steps.append({"step": "sau preprocess (features.load_frame)", "n_rows": final["n_rows"],
                  "ok": final["ok"], "error": final["error"]})
    removed = None if final["n_rows"] is None else n0 - final["n_rows"]

    years = dob.dt.year.dropna().astype(int).astype(str)
    schools = {
        "n_school_key_prov_name": int(raw.groupby([COL_PROV, COL_SCHOOL]).ngroups),
        "n_school_key_normalized": int(pd.Series(norm_column(raw[COL_PROV]).str.cat(
            norm_column(raw[COL_SCHOOL]), sep=KEY_SEP)).nunique()),
        "n_school_names_bare": int(raw[COL_SCHOOL].nunique()),
        "n_provinces": int(raw[COL_PROV].nunique()),
        "n_region_labels": int(raw[COL_REGION].nunique()),
    }
    size = raw.groupby([COL_PROV, COL_SCHOOL]).size().to_numpy()
    schools["school_size"] = {"desc": _desc(size), "n_size_1": int((size == 1).sum()),
                              "n_size_lt5": int((size < 5).sum()), "n_size_lt10": int((size < 10).sum()),
                              "n_size_lt30": int((size < 30).sum())}
    ch = features._is_chuyen(raw[COL_SCHOOL].to_numpy()).astype(bool)
    chuyen_keys = raw.loc[ch, [COL_PROV, COL_SCHOOL]].drop_duplicates()
    schools["chuyen"] = {"n_schools": int(len(chuyen_keys)), "n_rows": int(ch.sum()),
                         "rule": "khớp chuỗi 'huyên' như preprocess.py; cần danh sách chính thức"}
    return {"steps": steps, "n_input": n0, "n_final": final["n_rows"], "n_removed": removed,
            "n_features": final["n_features"], "unmapped_labels": unmapped,
            "birth_year": _pooled_counts(years), "n_birth_month_unknown": int(dob.isna().sum()),
            "schools": schools}, F


def cutoffs_by_seed(F, seeds, groups):
    """Ngưỡng đuôi theo khối lượng trên y huấn luyện của từng lần chia (mục 5.1: phải
    trùng 60/100 trên khoá này, chỉ lệch do điểm trùng)."""
    out = {}
    for s in seeds:
        tr, te = splits.outer_split(F.n, s, groups=groups)
        lo, hi = dl.tail_cutoffs(F.y[tr])
        ml, mh = dl.tail_masses(F.y[tr], lo, hi)
        tl, th = dl.tail_masses(F.y[te], lo, hi)
        out[str(s)] = {"lo": lo, "hi": hi, "mass_low_tr": ml, "mass_high_tr": mh,
                       "mass_low_te": tl, "mass_high_te": th, "n_tr": int(len(tr)), "n_te": int(len(te)),
                       "equals_60_100": bool(lo == LOW_TAIL_MAX and hi == HIGH_TAIL_MIN)}
    return out


def idt_questions(res):
    """(5) Câu hỏi gửi IDT (khung bài mục 10, ý 5) kèm số đếm làm bằng chứng."""
    d, h, t, ng = res["duplicates"], res["header"], res["target"], res["near_guess"]
    ex, nr, full = d["exact"], d["near"], d["full_row"]
    return [
        {"id": 1, "question": "Một dòng là một thí sinh hay một lượt thi?",
         "evidence": {"exact_dup_groups": ex["n_groups"], "exact_dup_rows": ex["n_rows"],
                      "exact_dup_row_frac": ex["row_frac"], "near_dup_groups": nr["n_groups"],
                      "near_dup_rows": nr["n_rows"], "full_row_dup_groups": full["n_groups"],
                      "near_groups_with_several_exact_keys": d["near_not_exact"]["n_groups"],
                      "stt_resets": d["stt"].get("n_resets"), "has_candidate_id": h["has_candidate_id"]}},
        {"id": 2, "question": "Khi thí sinh thi nhiều đợt, y là lượt đầu, lượt cao nhất hay lượt cuối?",
         "evidence": {"exact_groups_same_y": ex.get("abs_dy_range", {}).get("n_zero"),
                      "exact_abs_dy_range": ex.get("abs_dy_range", {}).get("desc"),
                      "exact_dy_last_minus_first": ex.get("dy_last_minus_first", {}).get("desc")}},
        {"id": 3, "question": "Có thể cung cấp số lượt, đợt thi, mã đề cho từng bản ghi không?",
         "evidence": {"has_round": h["has_round"], "has_test_form": h["has_test_form"],
                      "has_exam_date": h["has_exam_date"], "has_section_scores": h["has_section_scores"]}},
        {"id": 4, "question": "Hệ số tin cậy (KR-20 hoặc alpha theo mã đề) và SEM của HSA 2024?",
         "evidence": {"y_sd": t["desc"].get("sd"), "sem_by_assumed_reliability": t.get("sem_assumed")}},
        {"id": 5, "question": "Số câu và loại câu (trắc nghiệm mấy lựa chọn, tự luận) của đề 2024?",
         "evidence": {"y_min": t["min"], "near_guess_threshold": ng["threshold"], "near_guess_n": ng["n"],
                      "near_guess_frac": ng["frac"]}},
        {"id": 6, "question": "Danh sách ngưỡng sàn xét tuyển theo điểm HSA, kèm nguồn?",
         "evidence": {"regions_60_100": t.get("regions_60_100"),
                      "mass_cutoffs_all": t.get("mass_cutoffs_all")}},
        {"id": 7, "question": "Văn bản chính thức về cấu trúc đề và lịch thi 2024, kèm URL?",
         "evidence": {"has_exam_date": h["has_exam_date"]}},
        {"id": 8, "question": "Có thể cấp khoá 2025 cùng schema không (cần văn bản cho phép mới)?",
         "evidence": {"n_cols": h["n_cols"]}},
        {"id": 9, "question": "Nhà nghiên cứu độc lập xin truy cập dữ liệu theo đường nào?", "evidence": {}},
        {"id": 10, "question": "Có được công bố bộ sinh dữ liệu giả cùng schema không?",
         "evidence": {"n_cols": h["n_cols"], "n_transcript_cols": h["n_transcript_cols"]}},
        {"id": 11, "question": "Điểm thô giữa các mã đề đã được quy đổi (equating) chưa? (bổ sung từ mục 5.1)",
         "evidence": {"y_n_non_integer": t["n_non_integer"], "has_test_form": h["has_test_form"]}},
    ]


def gate_info(dups):
    ex, nr = dups["exact"], dups["near"]
    return {"group_key": "exact", "dup_stop_frac": DUP_STOP_FRAC,
            "exact_dup_groups": ex["n_groups"], "exact_dup_rows": ex["n_rows"],
            "exact_dup_row_frac": ex["row_frac"], "exact_dup_excess_frac": ex["excess_frac"],
            # Có dòng trùng chính xác thì MỌI thí nghiệm chia theo nhóm (khoá băm).
            "group_split_required": ex["n_groups"] > 0,
            # Dòng trùng vượt 1% thì dừng E1 cho tới khi IDT trả lời quy tắc chọn lượt.
            "stop_E1_until_IDT": ex["row_frac"] > DUP_STOP_FRAC,
            # Nếu IDT xác nhận y là lượt cao nhất: E5 báo độ nhạy khi bỏ nhóm trùng gần.
            "near_dup_groups": nr["n_groups"], "near_dup_rows": nr["n_rows"],
            "near_dup_sensitivity_E5": nr["n_groups"] > 0,
            "note": "Không có trả lời của IDT về đơn vị và lượt thì bài không được nộp (mục 6.3)."}


# ---------------------------------------------------------------------------
# Chốt chống lộ dữ liệu
# ---------------------------------------------------------------------------
def forbidden_strings(raw):
    """Tên trường, tên tỉnh, ngày sinh (thô và chuẩn hoá) có trong dữ liệu."""
    bad = set()
    for c in (COL_SCHOOL, COL_PROV):
        for v in pd.unique(raw[c].astype(str)):
            v = v.strip()
            if v:
                bad.add(v)
                bad.add(_norm_one(v))
    for v in pd.unique(raw[COL_DOB].astype(str)):
        v = v.strip()
        if v:
            bad.add(v)
    return {b for b in bad if b and b not in NA_LIKE}


def _strings(obj, path="$"):
    """(đường dẫn JSON, chuỗi) cho mọi khoá và giá trị chuỗi."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.<khoá>", str(k)
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def _assert_no_pii(obj, forbidden):
    """Dừng nếu một chuỗi của kết quả trùng (hoặc chứa, với tên dài) tên trường, tỉnh hay
    ngày sinh của dữ liệu. Chuỗi thuần số không tính (điểm, đếm). Thông báo lỗi chỉ nêu
    ĐƯỜNG DẪN trong JSON, không nêu giá trị (log cũng được pulldir về Mac)."""
    long_names = [f.casefold() for f in forbidden if len(f) >= PII_SUBSTR_MIN]
    for path, s in _strings(obj):
        hit = any(c in forbidden and not _is_number(c) for c in (s, s.strip(), _norm_one(s)))
        low = s.casefold()
        if hit or any(f in low for f in long_names):
            raise RuntimeError(f"kết quả chứa một giá trị định danh của dữ liệu tại {path}; "
                               "không ghi JSON")


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Chạy
# ---------------------------------------------------------------------------
def audit(path, seeds):
    raw = read_raw(path)
    y_raw = raw[TARGET]
    y = pd.to_numeric(y_raw, errors="coerce").to_numpy(float)
    exact = record_keys(raw, "exact")
    near = record_keys(raw, "near")
    full = record_keys(raw, "full")
    dups = {"exact": dup_stats(exact, y), "near": dup_stats(near, y), "full_row": dup_stats(full, y),
            "near_not_exact": near_not_exact(near, exact), "stt": stt_structure(raw[COL_STT]),
            "key_fields": {"exact": ["Ngày sinh", "Giới tính", "Tỉnh", "Trường", "học bạ 10./11./12."],
                           "near": ["Tỉnh", "Trường", "học bạ 10./11./12."],
                           "full_row": "mọi cột trừ STT (kể cả y, khuVuc)"}}
    prov_code, _ = pd.factorize(raw[COL_PROV].astype(str), sort=True)
    flow, F = sample_flow(path, raw)
    res = {"header": header_info(raw.columns), "target": target_info(y_raw), "duplicates": dups,
           "near_guess": near_guess_info(raw, y, prov_code.astype(np.int64)), "sample_flow": flow}
    res["gates"] = gate_info(dups)
    if F is not None:
        groups = None
        if res["gates"]["group_split_required"]:
            groups = pd.factorize(pd.Series(exact), sort=True)[0].astype(np.int32)
        res["cutoffs_by_seed"] = {"group_split": groups is not None,
                                  "by_seed": cutoffs_by_seed(F, seeds, groups)}
    res["idt_questions"] = idt_questions(res)
    return res, forbidden_strings(raw)


def print_summary(res):
    d, g, ng, fl = res["duplicates"], res["gates"], res["near_guess"], res["sample_flow"]
    t = res["target"]
    print(f"Header: {res['header']['n_cols']} cột, {res['header']['n_transcript_cols']} cột học bạ", flush=True)
    print(f"y: n={t['n']} mean={t['desc'].get('mean', float('nan')):.2f} sd={t['desc'].get('sd') or float('nan'):.2f} "
          f"min={t['min']} max={t['max']} không nguyên={t['n_non_integer']}")
    mc = t.get("mass_cutoffs_all") or {}
    print(f"Ngưỡng đuôi theo khối lượng (toàn khoá): {mc.get('lo')}/{mc.get('hi')} "
          f"trùng 60/100: {mc.get('equals_60_100')}")
    for k in ("full_row", "exact", "near"):
        x = d[k]
        print(f"Trùng {k:>8}: {x['n_groups']} nhóm, {x['n_rows']} dòng ({x['row_frac']:.3%}), "
              f"cỡ lớn nhất {x['max_group_size']}")
    print(f"Nhóm gần gồm >= 2 khoá chính xác: {d['near_not_exact']['n_groups']}; "
          f"STT đánh lại: {d['stt'].get('n_resets')}")
    print(f"Gần đoán mò (y <= {ng['threshold']}): {ng['n']} ({ng['frac']:.3%})")
    print(f"Luồng mẫu: vào {fl['n_input']}, cuối {fl['n_final']}, bỏ {fl['n_removed']}; "
          f"{fl['schools']['n_school_key_prov_name']} trường (tỉnh, tên), "
          f"{fl['schools']['n_school_names_bare']} tên trần, {fl['schools']['n_provinces']} tỉnh")
    print(f"Cổng: chia theo nhóm = {g['group_split_required']}; dừng E1 chờ IDT = {g['stop_E1_until_IDT']}; "
          f"độ nhạy trùng gần ở E5 = {g['near_dup_sensitivity_E5']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="E0b: kiểm dữ liệu (chỉ số đếm và phân vị)")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default="results_cost/data_audit.json")
    ap.add_argument("--preds-dir", default=os.path.join("preds", "data_audit"),
                    help="không dùng: E0b không ghi file theo dòng (nhận cho đồng bộ giao diện)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--smoke", action="store_true", help="chỉ seed đầu cho phần ngưỡng theo lần chia")
    ap.add_argument("--workers", type=int, default=1, help="không dùng (không khớp mô hình)")
    args = ap.parse_args(argv)
    guard_real_data(args.data)
    provenance.print_versions()
    seeds = args.seeds[:1] if args.smoke else args.seeds
    t0 = time.time()
    res, forbidden = audit(args.data, seeds)
    res["meta"] = {
        "experiment": "E0b", "spec": "notes/khung-bai-bao-2026-09-24.md mục 6.3",
        "data_sha256": preds_io.file_sha256(args.data), "seeds": seeds, "smoke": bool(args.smoke),
        "min_cell": MIN_CELL, "elapsed_s": round(time.time() - t0, 2),
        "tune_budget_default": TUNE_BUDGET,
        "choices": [
            "Khoá dùng mọi cột bắt đầu bằng 10./11./12. (111 cột theo header), không phải '129'.",
            "Chuẩn hoá: NFC, gộp khoảng trắng, chữ thường, NA thành '', số theo repr(float), "
            "ngày sinh %d/%m/%Y thành YYYY-MM-DD.",
            "Cổng 1%: số dòng nằm trong nhóm trùng chính xác chia n (thận trọng hơn số dòng thừa).",
            "|Δy| trong nhóm = max - min; y sau trừ y trước theo thứ tự file.",
            "Tam phân vị cỡ tỉnh: qcut theo dòng trên cỡ tỉnh của từng dòng (như ngũ phân vị cỡ trường).",
            f"Nhãn danh mục < {MIN_CELL} dòng gộp vào 'khác'.",
        ],
        "provenance": provenance.stamp(),
    }
    _assert_no_pii(res, forbidden)
    preds_io.dump_json_atomic(res, args.out)
    print_summary(res)
    print(f"Đã ghi {args.out} ({res['meta']['elapsed_s']} s)", flush=True)
    return res


if __name__ == "__main__":
    main()
