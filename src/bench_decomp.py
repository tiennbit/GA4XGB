# -*- coding: utf-8 -*-
"""E11 (khung bài 24/9, mục 6.13): lặp lại E1, E2, E2b, E4 trên dữ liệu công khai.

Câu hỏi: thứ tự đo được trên HSA (C1, C2, C3 và các chính sách E4) có lặp lại trên bộ
dữ liệu khác không, hay chỉ là chuyện "một bộ dữ liệu riêng, một khoá"? Phần kết luận
về ML dựa trên bốn bộ bảng OpenML-CTR23; câu về giáo dục chỉ dựa trên Saber.

Mã thí nghiệm KHÔNG viết lại: mỗi bộ dữ liệu được bọc thành một đối tượng có đúng giao
diện features.Frame (n, y, school_code, prov_code, columns(fset), design(fset, idx_fit,
*idx_apply)), rồi gọi thẳng các hàm đã dùng cho HSA:
  E1   decomp_centers.fit_phase1, fit_phase2 (default, bag10, rs_tuned; cùng không gian
       cấu hình splits.sample_configs, cùng dừng sớm 50 vòng / tối đa 3.000 cây, cùng
       OOF khớp lại fold) và decomp_centers.center_metrics (R0, R1, R1₁ ở K = 1..8).
  E2   decision_layer (R2, R5 khớp trên ŷ OOF, áp cho ŷ test; R1 như E1).
  E2b  wtrain_tuned.SplitContext, write_designs, run_K ở K = 3: R8 dò lại trên CÙNG bảng
       cấu hình với rs_tuned, trọng số chuẩn hoá, dừng sớm theo cost_K; R8_bag5.
  E4   select_policy.run_split và summarize ở K = 3: chính sách (a), (b), (c), (c+), (o).
Thống kê: stats_paired (Nadeau-Bengio, Holm, bootstrap cụm), lần chia gates.SEEDS
(10 lần 80/20, seed 100..109), đuôi theo khối lượng 9,25%/5,72% của y huấn luyện
(decision_layer.tail_cutoffs). Không hàm nào của các module trên bị sửa.

Phép so trên mỗi bộ (âm = vế trái tốt hơn, cùng quy ước gates.CONTRASTS), K = 3:
  C1  cost_3(R1, bag10) - cost_3(R8_bag5)      trung tâm chính tương tự HSA là bag B*,
                                               nên R8* = gates.r8_star("bag10") = R8_bag5
  C2  cost_3(R1, bag10) - cost_3(R5, bag10)
  C3  cost_3(R1, rs_tuned) - cost_3(R1, bag10)
  E4  (b)-(a), (c+)-(a), (c)-(a), (o)-(a) qua select_policy.summarize
Giữa các bộ chỉ ĐẾM số bộ cùng dấu với HSA (đọc từ results_cost/decomp_rules.json,
decomp_centers.json, select_policy.json), không gộp p. Cổng (mục 6.13): "thứ tự lặp
lại" nếu ít nhất 4/5 bộ chính cùng dấu với HSA ở C2 và C3, và ít nhất 4/5 ở C1.

Các lựa chọn khi khung bài chưa nói rõ (ghi cả vào meta.choices):
 1. Bộ dữ liệu. Năm bộ chính của khung bài: california_housing, diamonds, kings_county,
    cps88wages (OpenML-CTR23) và saber. student_performance_por (CTR23, n = 649) có sẵn
    trong data_public nên chạy kèm như bộ PHỤ: báo cáo đủ nhưng không vào cổng 4/5
    (tập kiểm tra 130 dòng, đuôi cao chỉ ~7 dòng mỗi lần chia).
 2. Đặc trưng. ARFF: thuộc tính số giữ nguyên; thuộc tính danh nghĩa thành mã nguyên
    theo thứ tự khai báo trong ARFF (với diamonds đó là thứ tự chất lượng của cut,
    color, clarity; với zipcode là thứ tự số), thiếu là NaN để XGBoost tự xử lý.
    Không one-hot: cây tách được mọi nhóm mức bằng nhiều lần cắt, và giữ số cột nhỏ.
    Bỏ cột: kings_county bỏ date_day (ngày trong tháng không mang thông tin về giá;
    date_year, date_month giữ dưới dạng SỐ năm, tháng); OpenML đã bỏ id và tách date
    sẵn. Các bộ CTR23 khác không có cột định danh hay cột đo sau mục tiêu.
    student_performance_por giữ G1, G2 (điểm hai kỳ TRƯỚC G3, như học bạ trước HSA).
    Saber: danh sách cột ĐƯỢC DÙNG viết tường minh (SABER_NUMERIC, SABER_CATEG);
    mọi cột khác bị bỏ: COD_S11 (mã), Cod_SPro, UNIVERSITY, ACADEMIC_PROGRAM, mọi *_PRO,
    PERCENTILE, 2ND_DECILE, QUARTILE (đo tại hoặc sau kết quả), SEL_IHE (chỉ số kinh
    tế xã hội của trường ĐẠI HỌC, chỉ có sau khi nhập học). Danh nghĩa của Saber thành
    mã nguyên theo nhãn sắp xếp; giá trị "0" (mã thiếu của file gốc) thành NaN.
 3. Cụm. Saber: SCHOOL_NAME (trường THPT), chỉ dùng cho bootstrap cụm, không làm đặc
    trưng và không chia theo nhóm (HSA cũng chia theo dòng). Bộ OpenML không có cụm tự
    nhiên: mỗi dòng là một cụm, nên "bootstrap cụm" thành bootstrap dòng i.i.d. Không
    có cột tần suất trường như HSA (school_basic_features) vì OpenML không có trường.
 4. Trung tâm. decomp_centers.Budget với n_bag = 10, bag_sizes = (10,) và bag_max = 1:
    giai đoạn 2 của decomp_centers khớp thêm "sub1" (một mô hình subsample 0,8) khi
    tập không phải F_dt; bag_max = 1 giữ việc đó ở một thành viên thay vì khớp lại 10.
    r8_bag = 1: khung bài E11 chỉ cần rs_tuned, nên không dựng rs_tuned_bag5 (tiết kiệm
    ~30% thời gian giai đoạn 2); decomp_centers vẫn ghi một trung tâm "rs_tuned_bag1"
    trùng rs_tuned, bị loại khỏi mọi bảng ở đây. bag10 đóng vai bag B* của HSA (B* =
    20 trên HSA; ở đây không dựng bag40 để tìm B*, khung bài E11 ghi bag10).
 5. Ngân sách dò 30 cấu hình (--n-configs) cho cả rs_tuned và R8: tiền tố của
    splits.sample_configs(…, seed) nên cùng mẫu cho hai bên ("hai bên cùng ngân sách").
 6. Quy tắc E4 là R1 (như lượt HSA, select_policy.json meta.rule), K = 3.
 7. SESOI = 0,10 là điểm HSA, không mang sang thang khác (giá nhà, đô la...). JSON vẫn
    ghi TOST ±0,10 của stats_paired cho đủ lược đồ, nhưng chỉ dấu, p Nadeau-Bengio và
    hai cỡ chuẩn hoá (chia SD của y; phần trăm cost_3(R1, bag10)) được dùng để đọc.
 8. Chạy tiếp: mỗi (bộ, lần chia) có npz riêng ở --preds-dir/<bộ>/ (giai đoạn 1, gộp
    E1, R8 ở K = 3) mang dấu preds_io.fingerprint; JSON dở dang <out>.partial giữ kết
    quả theo (bộ, lần chia), và mỗi bộ ghi data_sha256 cùng dấu riêng: chạy tiếp với
    file dữ liệu hay mã khác thì báo lỗi thay vì trộn. Dữ liệu công khai nên npz có thể
    nằm ở đâu cũng được; mặc định preds/bench (ngoài src/ bị mirror --delete).
 9. Nguồn gốc: meta.provenance (provenance.stamp), sha256 của từng file dữ liệu đối
    chiếu với data_public/SHA256SUMS (lệch thì dừng, trừ --smoke), sha256 của ba JSON
    HSA đã đọc dấu.
10. --smoke: một lần chia, 3 cấu hình, bag10 với 50 cây mỗi mô hình, tối đa 50 cây, dừng sớm 10 vòng, R8_bag
    2 thành viên, bootstrap 200 lần. Chỉ để kiểm mã chạy hết và ghi JSON hợp lệ.

Chạy (server): PYTHONPATH=src .venv/bin/python -W ignore src/bench_decomp.py --workers 8
Khói (Mac, dữ liệu công khai, không có PII):
  PYTHONPATH=src python3 -W ignore src/bench_decomp.py --smoke --workers 4 --threads 1 \\
      --out <scratch>/bench_decomp.json --preds-dir <scratch>/preds
"""
import argparse
import io
import os
import re
import sys
import time
from argparse import Namespace
from dataclasses import asdict

import numpy as np
import pandas as pd
from joblib import Parallel

import decision_layer as dl
import decomp_centers as dc
import preds_io
import provenance
import select_policy as spol
import splits
import stats_paired as sp
import wtrain_tuned as wt
from ga_xgb import CV_FOLDS
from gates import (ALPHA, CLUSTER_BOOT_B, E1_KS, EARLY_STOP_ROUNDS, ES_FRAC, MAX_TREES, N_BINS,
                   N_SAMPLES, PRIMARY_K, R8_BAG, RULE_CROSSFIT_FOLDS, SEEDS, TAIL_MASS_HIGH,
                   TAIL_MASS_LOW, r8_star)

EXPERIMENT = "E11"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO, "data_public")
DEFAULT_OUT = "results_bench/bench_decomp.json"
DEFAULT_PREDS = os.path.join("preds", "bench")
FSET = "bench"                     # tên tập đặc trưng ghi vào meta npz (một tập mỗi bộ)
N_CONFIGS = 30                     # khung bài 6.13: 30 cấu hình cho rs_tuned và cho R8
E4_RULE = "R1"
PRIMARY = "bag10"                  # vai bag B* của HSA
CENTERS = ["default", "sub1", "bag10", "rs_tuned"]   # trung tâm báo cáo (bỏ rs_tuned_bag1)
RULES = ["R0", "R1_1", "R1", "R2", "R5"]
CONTRASTS = ["C1", "C2", "C3"]
E4_CONTRASTS = ["b-a", "c+-a", "c-a", "o-a"]
SECONDARY = ["R2-R1", "ii", "ii_a", "ii_b", "bag10-default", "bag10-sub1"]
GATE_MIN = 4                       # ít nhất 4/5 bộ chính cùng dấu với HSA

# ---------------------------------------------------------------------------
# Bộ dữ liệu
# ---------------------------------------------------------------------------
SABER_NUMERIC = ["MAT_S11", "CR_S11", "CC_S11", "BIO_S11", "ENG_S11", "SEL"]
SABER_CATEG = ["GENDER", "EDU_FATHER", "EDU_MOTHER", "OCC_FATHER", "OCC_MOTHER", "STRATUM", "SISBEN",
               "PEOPLE_HOUSE", "INTERNET", "TV", "COMPUTER", "WASHING_MCH", "MIC_OVEN", "CAR", "DVD",
               "FRESH", "PHONE", "MOBILE", "REVENUE", "JOB", "SCHOOL_NAT", "SCHOOL_TYPE"]
# Cột Saber bị bỏ có chủ ý (lựa chọn 2). Kiểm lúc đọc: cột nào không thuộc ba nhóm
# (dùng, bỏ, mục tiêu/cụm) là file đã đổi lược đồ -> dừng, đừng lặng lẽ bỏ qua.
SABER_DROP = ["COD_S11", "Cod_SPro", "UNIVERSITY", "ACADEMIC_PROGRAM", "QR_PRO", "CR_PRO", "CC_PRO",
              "ENG_PRO", "WC_PRO", "FEP_PRO", "PERCENTILE", "2ND_DECILE", "QUARTILE", "SEL_IHE"]

DATASETS = {
    "california_housing": {"file": "california_housing.arff", "target": "medianHouseValue",
                           "drop": [], "as_number": [], "core": True, "domain": "housing",
                           "note": "medianHouseValue bị chặn trên ở 500001 (khoảng 4,7% dòng): đuôi cao bị kiểm duyệt"},
    "diamonds": {"file": "diamonds.arff", "target": "price", "drop": [], "as_number": [],
                 "core": True, "domain": "retail"},
    "kings_county": {"file": "kings_county.arff", "target": "price", "drop": ["date_day"],
                     "as_number": ["date_year", "date_month"], "core": True, "domain": "housing",
                     "note": "OpenML đã bỏ id và tách date; giữ năm, tháng dạng số, bỏ ngày trong tháng"},
    "cps88wages": {"file": "cps88wages.arff", "target": "wage", "drop": [], "as_number": [],
                   "core": True, "domain": "labour"},
    "saber": {"file": "saber.csv.gz", "target": "G_SC", "cluster": "SCHOOL_NAME", "core": True,
              "domain": "education",
              "note": "Saber 11 -> Saber Pro (Delahoz-Dominguez et al. 2020, CC BY 4.0); cụm = SCHOOL_NAME"},
    "student_performance_por": {"file": "student_performance_por.arff", "target": "G3", "drop": [],
                                "as_number": [], "core": False, "domain": "education",
                                "note": "bộ phụ, n = 649; giữ G1, G2 (điểm kỳ trước G3)"},
}
CORE = [k for k, v in DATASETS.items() if v["core"]]


# ---------------------------------------------------------------------------
# Đọc ARFF (chỉ stdlib + pandas; không thêm gói liac-arff)
# ---------------------------------------------------------------------------
_ATTR = re.compile(r"""@attribute\s+(?:'([^']*)'|"([^"]*)"|(\S+))\s+(.+)$""", re.IGNORECASE)


def _unquote(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def read_arff(path):
    """(DataFrame, {cột: [mức] hoặc None}). Hỗ trợ đúng phần ARFF mà CTR23 dùng:
    thuộc tính numeric/real/integer và danh nghĩa {…}, dữ liệu dạng dày, '?' là thiếu.
    Kiểu khác (string, date, relational) hay dữ liệu thưa thì báo lỗi, không đoán."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    attrs, start = [], None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@relation"):
            continue
        if low.startswith("@attribute"):
            m = _ATTR.match(s)
            if not m:
                raise ValueError(f"{path}:{i + 1}: không đọc được dòng thuộc tính: {s[:80]}")
            name = next(g for g in m.groups()[:3] if g is not None)
            typ = m.group(4).strip()
            if typ.startswith("{"):
                if not typ.endswith("}"):
                    raise ValueError(f"{path}:{i + 1}: danh sách mức không đóng: {s[:80]}")
                attrs.append((name, [_unquote(v) for v in typ[1:-1].split(",")]))
            elif typ.lower() in ("numeric", "real", "integer"):
                attrs.append((name, None))
            else:
                raise ValueError(f"{path}: thuộc tính {name} kiểu {typ!r} chưa hỗ trợ")
            continue
        if low.startswith("@data"):
            start = i + 1
            break
        raise ValueError(f"{path}:{i + 1}: dòng lạ trong phần đầu: {s[:80]}")
    if start is None or not attrs:
        raise ValueError(f"{path}: không có @attribute hoặc @data")
    body = [ln for ln in lines[start:] if ln.strip() and not ln.lstrip().startswith("%")]
    if body and body[0].lstrip().startswith("{"):
        raise ValueError(f"{path}: ARFF thưa chưa hỗ trợ")
    names = [a for a, _ in attrs]
    raw = pd.read_csv(io.StringIO("\n".join(body)), header=None, names=names, dtype=str,
                      quotechar='"', skipinitialspace=True, na_values=["?"], keep_default_na=False)
    if raw.shape[1] != len(names):
        raise ValueError(f"{path}: {raw.shape[1]} cột dữ liệu, khai báo {len(names)}")
    out, levels = {}, {}
    for name, lv in attrs:
        col = raw[name].map(lambda v: v if pd.isna(v) else _unquote(v))
        if lv is None:
            out[name] = pd.to_numeric(col, errors="raise").astype(float)
            levels[name] = None
        else:
            bad = sorted(set(col.dropna()) - set(lv))
            if bad:
                raise ValueError(f"{path}: {name} có giá trị ngoài danh sách mức: {bad[:5]}")
            out[name] = pd.Categorical(col, categories=lv)
            levels[name] = lv
    return pd.DataFrame(out), levels


# ---------------------------------------------------------------------------
# Frame công khai: cùng giao diện features.Frame mà decomp_centers, wtrain_tuned dùng
# ---------------------------------------------------------------------------
class PublicFrame:
    """Ma trận số cố định (không có cột tính theo hàng huấn luyện như tần suất trường
    của HSA), nên design chỉ cắt hàng. school_code là mã cụm cho bootstrap."""

    def __init__(self, name, X, y, cols, cluster_code, info):
        self.name, self.y = name, np.asarray(y, dtype=float)
        self.n = len(self.y)
        self._X = np.ascontiguousarray(X, dtype=float)
        self._cols = list(cols)
        self.school_code = np.asarray(cluster_code, dtype=np.int32)
        self.prov_code = np.zeros(self.n, dtype=np.int32)
        self.info = info
        assert self._X.shape == (self.n, len(self._cols))
        assert np.all(np.isfinite(self.y)), f"{name}: y có NaN/inf"

    def columns(self, fset=FSET):
        return list(self._cols)

    def design(self, fset, idx_fit, *idx_apply):
        return tuple(self._X[np.asarray(idx)] for idx in (idx_fit,) + idx_apply)


def _codes(series):
    """Mã nguyên theo nhãn sắp xếp; NaN giữ NaN."""
    s = pd.Series(series)
    codes, _ = pd.factorize(s, sort=True)
    out = codes.astype(float)
    out[codes < 0] = np.nan
    return out


def load_arff_frame(name, spec, data_dir):
    df, levels = read_arff(os.path.join(data_dir, spec["file"]))
    tgt = spec["target"]
    if tgt not in df or levels[tgt] is not None:
        raise ValueError(f"{name}: mục tiêu {tgt} không có hoặc không phải số")
    y = df[tgt].to_numpy(float)
    keep = ~np.isnan(y)
    feats = [c for c in df.columns if c != tgt and c not in spec["drop"]]
    cols, mats, encoding = [], [], {}
    for c in feats:
        if levels[c] is None:
            v, enc = df[c].to_numpy(float), "numeric"
        elif c in spec["as_number"]:
            v, enc = pd.to_numeric(df[c].astype(object), errors="raise").to_numpy(float), "label_as_number"
        else:
            codes = df[c].cat.codes.to_numpy()
            v = codes.astype(float)
            v[codes < 0] = np.nan
            enc = "arff_level_order"
        cols.append(c)
        mats.append(v)
        encoding[c] = enc
    X = np.column_stack(mats)[keep]
    info = {"n_raw": int(len(df)), "n_dropped_missing_y": int((~keep).sum()), "target": tgt,
            "dropped": list(spec["drop"]), "encoding": encoding, "cluster": None}
    return PublicFrame(name, X, y[keep], cols, np.arange(int(keep.sum())), info)


def load_saber(name, spec, data_dir):
    path = os.path.join(data_dir, spec["file"])
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    known = set(SABER_NUMERIC) | set(SABER_CATEG) | set(SABER_DROP) | {spec["target"], spec["cluster"]}
    unknown = sorted(set(df.columns) - known)
    missing = sorted(known - set(df.columns))
    if unknown or missing:
        raise ValueError(f"{path}: lược đồ khác dự kiến (cột lạ {unknown}, thiếu {missing}); "
                         "chạy lại src/convert_saber.py hoặc cập nhật danh sách cột có lý do")
    y = pd.to_numeric(df[spec["target"]], errors="raise").to_numpy(float)
    mats, encoding = [], {}
    for c in SABER_NUMERIC:
        mats.append(pd.to_numeric(df[c], errors="raise").to_numpy(float))
        encoding[c] = "numeric"
    for c in SABER_CATEG:
        s = df[c].str.strip()
        s = s.where(~s.isin(["0", ""]))          # "0" là mã thiếu của file gốc
        mats.append(_codes(s))
        encoding[c] = "sorted_label_code"
    cl = df[spec["cluster"]].fillna("").astype(str)
    cl_code, _ = pd.factorize(cl, sort=True)
    info = {"n_raw": int(len(df)), "n_dropped_missing_y": 0, "target": spec["target"],
            "dropped": SABER_DROP, "encoding": encoding, "cluster": spec["cluster"],
            "n_clusters": int(cl_code.max()) + 1}
    return PublicFrame(name, np.column_stack(mats), y, SABER_NUMERIC + SABER_CATEG, cl_code, info)


def load_dataset(name, data_dir):
    spec = DATASETS[name]
    F = load_saber(name, spec, data_dir) if name == "saber" else load_arff_frame(name, spec, data_dir)
    F.info |= {"n": F.n, "n_features": len(F.columns()), "file": spec["file"], "core": spec["core"],
               "domain": spec["domain"], "note": spec.get("note"),
               "y_mean": float(F.y.mean()), "y_sd": float(F.y.std(ddof=1)),
               "y_min": float(F.y.min()), "y_max": float(F.y.max()),
               "n_unique_y": int(len(np.unique(F.y)))}
    return F


def read_sums(data_dir):
    p = os.path.join(data_dir, "SHA256SUMS")
    out = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                parts = ln.split()
                if len(parts) == 2:
                    out[os.path.basename(parts[1])] = parts[0]
    return out


def check_sha(name, data_dir, sums, smoke):
    f = DATASETS[name]["file"]
    got = preds_io.file_sha256(os.path.join(data_dir, f))
    want = sums.get(f)
    ok = want is not None and want == got
    if not ok:
        msg = (f"{f}: sha256 {got[:12]} " + ("không có trong SHA256SUMS" if want is None
                                             else f"khác SHA256SUMS ({want[:12]})"))
        if not smoke:
            sys.exit(msg + ": dữ liệu khác bản đã tải ngày 24/9; kiểm lại trước khi chạy.")
        print(f"[cảnh báo] {msg} (--smoke nên chỉ ghi lại)", flush=True)
    return {"file": f, "sha256": got, "matches_SHA256SUMS": ok}


# ---------------------------------------------------------------------------
# Một lần chia của một bộ
# ---------------------------------------------------------------------------
def make_budget(smoke, n_configs):
    """Lựa chọn 4: bag_max = 1 (chỉ sub1 ở giai đoạn 2), r8_bag = 1 (không rs_tuned_bag5)."""
    if smoke:
        # bag10 giữ đúng tên (50 cây mỗi mô hình nên vẫn rẻ): mọi khoá JSON như lượt thật
        return dc.Budget(n_bag=10, bag_max=1, bag_sizes=(10,), default_trees=50, n_configs=int(n_configs),
                         max_trees=50, es_rounds=10, r8_bag=1, boot_B=200)
    return dc.Budget(n_bag=10, bag_max=1, bag_sizes=(10,), default_trees=100, n_configs=int(n_configs),
                     max_trees=MAX_TREES, es_rounds=EARLY_STOP_ROUNDS, r8_bag=1, boot_B=CLUSTER_BOOT_B)


def kkey(K):
    return dc.kkey(K)


def rule_block(p_oof, y_tr, p_te, y_te, lo, hi, ks=E1_KS):
    """cost_K của R0, R1₁, R1, R2, R5 ở mọi K của E1; dự đoán ở K = 3 cho phép so.

    R1 khớp một lần (phân vị phần dư không phụ thuộc K) như decomp_centers.center_metrics;
    R2, R5 khớp lại ở từng K vì w vào chính phép khớp (như decomp_rules.eval_center)."""
    r1 = dl.make_rule("R1").fit(p_oof, y_tr)
    r11 = dl.make_rule("R1_1").fit(p_oof, y_tr).predict(p_te)
    cost, preds3, info3 = {r: {} for r in RULES}, {}, {}
    for K in ks:
        wf = dl.step(K, K, lo, hi)
        pr = {"R0": np.asarray(p_te, float), "R1_1": r11, "R1": r1.predict(p_te, wfun=wf)}
        for code in ("R2", "R5"):
            rule = dl.make_rule(code).fit(p_oof, y_tr, wf)
            pr[code] = rule.predict(p_te)
            if K == PRIMARY_K and code == "R5":
                info3["R5"] = rule.info()
        for r in RULES:
            cost[r][kkey(K)] = sp.cost_k(y_te, pr[r], K, K, lo, hi)
        if K == PRIMARY_K:
            preds3 = pr
    region3 = {r: sp.region_rmse(y_te, preds3[r], lo, hi) for r in RULES}
    return {"cost_K": cost, "region_rmse_K3": region3, "info_K3": info3}, preds3


def boot_pair(y, p1, p2, clusters, lo, hi, B, seed):
    b = sp.cluster_boot_diff(y, p1, p2, clusters, PRIMARY_K, PRIMARY_K, lo, hi, B=B, seed=int(seed))
    return {k: b[k] for k in ("est", "ci_lo", "ci_hi", "se", "excludes_zero", "B", "n_clusters")}


def run_split(F, name, seed, bud, args, fps, ds_dir, threads):
    """Mọi số của một (bộ, lần chia). Mỗi bước nặng ghi npz trước (chạy tiếp được)."""
    t0 = time.time()
    run_meta = {"smoke": bool(args.smoke), "data_sha256": fps["data_sha256"], "dataset": name}
    timing = {}

    def par():
        # Một ngữ cảnh Parallel cho mỗi bước, như decomp_centers.unit_par: thư mục tạm
        # của joblib được dọn khi ngữ cảnh đóng.
        return Parallel(n_jobs=args.workers, backend="loky", batch_size=1)

    # E1 giai đoạn 1: default, bag10
    p1_path = preds_io.split_path(ds_dir, seed, "p1")
    p1d = preds_io.load_or_none(p1_path, fps["p1"], on_mismatch=args.on_mismatch)
    if p1d is None:
        t1 = time.time()
        with par() as pp:
            arrays = dc.fit_phase1(F, seed, FSET, bud, pp, threads, fps["p1"], run_meta)
        preds_io.save_split(p1_path, **arrays)
        p1d = preds_io.load_split(p1_path)
        timing["phase1_s"] = time.time() - t1
    err = dc.validate_e1(p1d)
    assert not err, err

    # E1 giai đoạn 2: vết cấu hình, rs_tuned, sub1 (npz gộp theo lược đồ E1)
    e1_path = preds_io.split_path(ds_dir, seed)
    d = preds_io.load_or_none(e1_path, fps["p2"], on_mismatch=args.on_mismatch)
    if d is None:
        t1 = time.time()
        with par() as pp:
            arrays = dc.fit_phase2(F, seed, FSET, bud, pp, threads, fps["p2"], p1d, run_meta)
        preds_io.save_split(e1_path, **arrays)
        d = preds_io.load_split(e1_path)
        timing["phase2_s"] = time.time() - t1
    err = dc.validate_e1(d)
    assert not err, err
    e1_sha = preds_io.file_sha256(e1_path)
    meta = d["meta"]
    lo, hi = float(meta["lo"]), float(meta["hi"])
    y_tr, y_te = np.asarray(d["y_tr"], float), np.asarray(d["y_te"], float)
    cl_te = preds_io.rows(d, "te", "school_code")

    # E1 thước đo trung tâm và E2 quy tắc
    centers = [c for c in CENTERS if c in d["test"]]
    cen = dc.eval_centers(d, centers)
    rules, p3 = {}, {}
    for c in centers:
        rules[c], p3[c] = rule_block(d["oof"][c], y_tr, d["test"][c], y_te, lo, hi)

    # E2b: R8 dò lại ở K = 3 trên cùng bảng cấu hình, R8_bag5
    wargs = Namespace(n_cfg=None, smoke=bool(args.smoke), feature_set=None, max_trees=bud.max_trees,
                      es_rounds=bud.es_rounds, threads=args.threads, workers=args.workers,
                      n_bag=(2 if args.smoke else R8_BAG), boot_B=bud.boot_B)
    fp_r8 = preds_io.fingerprint(fps["r8_cfg"] | {"e1_npz_sha256": e1_sha})
    r8_path = preds_io.split_path(ds_dir, seed, f"K{PRIMARY_K}")
    d8 = preds_io.load_or_none(r8_path, fp_r8, on_mismatch=args.on_mismatch)
    if d8 is None:
        t1 = time.time()
        wt.sweep_memmaps(ds_dir)
        ctx = wt.SplitContext(F, d, seed, wargs, e1_path, e1_sha)
        try:
            wt.write_designs(F, ctx, ds_dir)
            entry, arrays = wt.run_K(ctx, PRIMARY_K, wargs, fp_r8)
        finally:
            wt.cleanup(ctx)
        preds_io.save_split(r8_path, **arrays)
        d8 = preds_io.load_split(r8_path)
        timing["r8_s"] = time.time() - t1
    r8e = d8["meta"]["entry"]
    if r8e.get("e1_npz_sha256") != e1_sha:
        raise preds_io.FingerprintMismatch(f"{r8_path}: R8 tính trên npz E1 khác bản hiện tại")

    # E4 ở K = 3, quy tắc R1 (select_policy đọc npz gộp của E1)
    t1 = time.time()
    _, e4_out, e4_det = spol.run_split(seed, ds_dir, None, E4_RULE, [PRIMARY_K], bud.boot_B, None,
                                       RULE_CROSSFIT_FOLDS, None, fps["e4"])
    timing["e4_s"] = time.time() - t1
    k3 = spol.kkey(PRIMARY_K)
    e4_slim = {"by_K": {k: {"boot": v["boot"], "rank": v["rank"]} for k, v in e4_det["by_K"].items()},
               "checks": e4_det["checks"]}

    # Phép so ở K = 3 (giá trị theo lần chia; Nadeau-Bengio tính ở phần tóm tắt)
    c1_center = r8e["centers"][PRIMARY]
    star = r8_star(PRIMARY)
    assert c1_center["r8_star"] == star, (c1_center["r8_star"], star)
    k = kkey(PRIMARY_K)
    cost3 = {c: rules[c]["cost_K"] for c in centers}
    diffs = {
        "C1": float(c1_center["C1"]),
        "C2": cost3[PRIMARY]["R1"][k] - cost3[PRIMARY]["R5"][k],
        "C3": cost3["rs_tuned"]["R1"][k] - cost3[PRIMARY]["R1"][k],
        "R2-R1": cost3[PRIMARY]["R2"][k] - cost3[PRIMARY]["R1"][k],
        "ii": cost3[PRIMARY]["R0"][k] - cost3[PRIMARY]["R1"][k],
        "ii_a": cost3[PRIMARY]["R0"][k] - cost3[PRIMARY]["R1_1"][k],
        "ii_b": cost3[PRIMARY]["R1_1"][k] - cost3[PRIMARY]["R1"][k],
        "bag10-default": cost3[PRIMARY]["R1"][k] - cost3["default"]["R1"][k],
    }
    if "sub1" in cost3:
        diffs["bag10-sub1"] = cost3[PRIMARY]["R1"][k] - cost3["sub1"]["R1"][k]
    # R1 của wtrain_tuned (C1) và của rule_block phải là cùng một số
    # (sai số tương đối: thang y khác nhau tới 10^5 lần giữa các bộ)
    assert abs(c1_center["R1"] - cost3[PRIMARY]["R1"][k]) <= 1e-9 * max(1.0, abs(c1_center["R1"])), \
        (c1_center["R1"], cost3[PRIMARY]["R1"][k])
    boot = {"C1": {kk: c1_center["boot"][kk] for kk in ("ci_lo", "ci_hi", "se", "excludes_zero", "B",
                                                         "n_clusters")} | {"est": diffs["C1"]},
            "C2": boot_pair(y_te, p3[PRIMARY]["R1"], p3[PRIMARY]["R5"], cl_te, lo, hi, bud.boot_B, seed),
            "C3": boot_pair(y_te, p3["rs_tuned"]["R1"], p3[PRIMARY]["R1"], cl_te, lo, hi, bud.boot_B, seed)}

    ml, mh = dl.tail_masses(y_tr, lo, hi)
    out = {
        "split_info": {"lo": lo, "hi": hi, "n_tr": int(len(y_tr)), "n_te": int(len(y_te)),
                       "mass_low_tr": ml, "mass_high_tr": mh,
                       "mass_low_te": dl.tail_masses(y_te, lo, hi)[0],
                       "mass_high_te": dl.tail_masses(y_te, lo, hi)[1],
                       "n_clusters_te": int(len(np.unique(cl_te))), "sd_y_te": float(np.std(y_te, ddof=1))},
        "centers": cen, "rules": rules, "diffs": diffs, "boot": boot,
        "r8": {kk: r8e[kk] for kk in ("best_index", "best_params", "n_trees", "oof_cost", "oof_rmse",
                                       "cost_K", "test", "weights", "capped_fold_fits", "n_fold_fits",
                                       "bag_n_trees", "tune_s", "fit_s", "n_bag")},
        "e4": e4_out[k3], "e4_detail": e4_slim,
        "npz": {"p1": preds_io.file_sha256(p1_path), "e1": e1_sha, "r8": preds_io.file_sha256(r8_path)},
        "timing": timing | {"split_wall_s": time.time() - t0},
        "trace": {"rs_tuned_index": int(meta["rs_tuned_index"]),
                  "n_capped": int((np.asarray(d["trace_best_iter"]) + 1 >= bud.max_trees).any(axis=1).sum()),
                  "tune_s": float(np.sum(meta["trace_timing"]["es_s"]))},
    }
    print(f"  {name} split {seed}: lo/hi {lo:g}/{hi:g}; RMSE bag10 {cen[PRIMARY]['all_rmse']:.4g}, "
          f"R² {cen[PRIMARY]['r2']:.3f}; C1 {diffs['C1']:+.4g} C2 {diffs['C2']:+.4g} C3 {diffs['C3']:+.4g}; "
          f"E4 b-a {e4_out[k3]['b']['test_cost'] - e4_out[k3]['a']['test_cost']:+.4g} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Tóm tắt một bộ và đếm dấu giữa các bộ
# ---------------------------------------------------------------------------
def _ms(v):
    v = np.asarray([np.nan if x is None else x for x in v], dtype=float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": int(len(v))}


def summarize_dataset(entry, n_ref):
    per = entry["per_split"]
    seeds = sorted(per, key=int)
    sd_y = entry["info"]["y_sd"]
    base = _ms([per[s]["rules"][PRIMARY]["cost_K"]["R1"][kkey(PRIMARY_K)] for s in seeds])["mean"]
    S = {"seeds": [int(s) for s in seeds], "n_splits": len(seeds), "n_ref": int(n_ref),
         "complete": len(seeds) == n_ref, "sd_y": sd_y, "cost3_R1_bag10": base}
    # Mô tả trung tâm và quy tắc
    S["centers"] = {}
    for c in CENTERS:
        if not all(c in per[s]["centers"] for s in seeds):
            continue
        S["centers"][c] = {m: _ms([per[s]["centers"][c][m] for s in seeds])
                           for m in ("all_rmse", "r2", "sd_ratio", "calib_slope", "oof_rmse")}
        S["centers"][c]["cost_K"] = {r: {kkey(K): _ms([per[s]["rules"][c]["cost_K"][r][kkey(K)] for s in seeds])
                                         for K in E1_KS} for r in RULES}
    S["tails"] = {"lo": _ms([per[s]["split_info"]["lo"] for s in seeds]),
                  "hi": _ms([per[s]["split_info"]["hi"] for s in seeds]),
                  "mass_low_tr": _ms([per[s]["split_info"]["mass_low_tr"] for s in seeds]),
                  "mass_high_tr": _ms([per[s]["split_info"]["mass_high_tr"] for s in seeds])}
    S["r5_edge_K3"] = int(sum(bool(per[s]["rules"][PRIMARY]["info_K3"]["R5"]["edge_low"])
                              + bool(per[s]["rules"][PRIMARY]["info_K3"]["R5"]["edge_high"]) for s in seeds))

    def cell(name):
        d = [float("nan") if per[s]["diffs"].get(name) is None else float(per[s]["diffs"][name])
             for s in seeds]
        r = {"diffs": d, **sp.paired(d)}
        m = r["nb"]["mean"]
        r["std_by_sd_y"] = m / sd_y if np.isfinite(m) else float("nan")
        r["pct_of_cost3_R1_bag10"] = 100.0 * m / base if np.isfinite(m) and base else float("nan")
        r["n_pos"] = int(sum(x > 0 for x in d if np.isfinite(x)))
        return r

    con = {c: cell(c) for c in CONTRASTS}
    for c in ("C1", "C2", "C3"):
        bs = [per[s]["boot"][c] for s in seeds]
        con[c]["boot_splits_excluding_zero"] = int(sum(bool(b["excludes_zero"]) for b in bs))
        con[c]["boot_n"] = len(bs)
    ph = sp.primary_holm({c: con[c]["nb"]["p"] for c in CONTRASTS})
    for c in CONTRASTS:
        con[c]["p_holm_within_dataset"] = ph[c]["p_holm"]
    S["contrasts"] = con
    sec = {c: cell(c) for c in SECONDARY if all(c in per[s]["diffs"] for s in seeds)}
    adj = sp.holm([sec[c]["nb"]["p"] for c in sec])
    for c, p in zip(sec, adj):
        sec[c]["p_holm"] = p
    S["secondary"] = sec
    # E4: select_policy.summarize trên đúng các lần chia này
    k3 = spol.kkey(PRIMARY_K)
    e4_per = {s: {k3: per[s]["e4"]} for s in seeds}
    e4_det = {s: per[s]["e4_detail"] for s in seeds}
    e4_sum, rank = spol.summarize(e4_per, e4_det, [PRIMARY_K])
    for name, r in e4_sum["contrasts"][k3].items():
        m = r["nb"]["mean"]
        r["std_by_sd_y"] = m / sd_y if np.isfinite(m) else float("nan")
        r["pct_of_cost3_R1_bag10"] = 100.0 * m / base if np.isfinite(m) and base else float("nan")
    S["e4"] = {"policies": e4_sum["policies"][k3], "contrasts": e4_sum["contrasts"][k3],
               "holm_family_size": e4_sum["holm_family_size"], "rank_corr": rank[k3]}
    S["r8"] = {"oof_cost": _ms([per[s]["r8"]["oof_cost"] for s in seeds]),
               "cost3_R8": _ms([per[s]["r8"]["cost_K"]["R8"] for s in seeds]),
               "cost3_R8_bag5": _ms([per[s]["r8"]["cost_K"]["R8_bag5"] for s in seeds]),
               "kish_ratio": _ms([per[s]["r8"]["weights"]["kish_ratio"] for s in seeds]),
               "capped_fold_fits": int(sum(per[s]["r8"]["capped_fold_fits"] for s in seeds))}
    S["timing_s"] = {"split_wall": _ms([per[s]["timing"]["split_wall_s"] for s in seeds])}
    return S


def _hsa_value(obj, name):
    """Trung bình chênh của một phép so trong JSON HSA; C2 có thể theo K ({'3': ...})."""
    if not isinstance(obj, dict):
        return None
    k = kkey(PRIMARY_K)
    if "nb" not in obj and k in obj:
        obj = obj[k]
    nb = obj.get("nb") if isinstance(obj, dict) else None
    m = (nb or {}).get("mean", obj.get("mean") if isinstance(obj, dict) else None)
    return None if m is None else float(m)


def read_hsa(paths):
    """Dấu tham chiếu của HSA: C1, C2 từ decomp_rules.json (E2 tính lại sau E2b), C3 từ
    decomp_rules.json hoặc decomp_centers.json, E4 từ select_policy.json ở K = 3."""
    out, src = {}, {}
    rules = preds_io.load_json(paths["rules"])
    centers = preds_io.load_json(paths["centers"])
    policy = preds_io.load_json(paths["policy"])
    for key, obj in (("rules", rules), ("centers", centers), ("policy", policy)):
        src[key] = {"path": paths[key], "sha256": preds_io.file_sha256(paths[key]) if obj is not None else None,
                    "present": obj is not None}
    con = ((rules or {}).get("summary") or {}).get("contrasts") or {}
    for c in CONTRASTS:
        out[c] = _hsa_value(con.get(c), c)
    if out.get("C3") is None:
        out["C3"] = _hsa_value(((centers or {}).get("summary") or {}).get("C3"), "C3")
    pc = (((policy or {}).get("summary") or {}).get("contrasts") or {}).get(spol.kkey(PRIMARY_K)) or {}
    for c in E4_CONTRASTS:
        out[f"E4 {c}"] = _hsa_value(pc.get(c), c)
    roles = ((rules or {}).get("summary") or {}).get("roles")
    return {"means": out, "signs": {k: (None if v is None else int(np.sign(v))) for k, v in out.items()},
            "roles": roles, "sources": src,
            "note": "HSA: trung tâm chính bag B* (= bag20), R8* = R8_bag5, C3 so rs_tuned với bag B*"}


def sign_counts(datasets, hsa):
    """Số bộ cùng dấu với HSA cho từng phép so (chỉ đếm, không gộp p)."""
    rows = {}
    keys = CONTRASTS + [f"E4 {c}" for c in E4_CONTRASTS]
    for key in keys:
        ref = hsa["signs"].get(key)
        per = {}
        for name, e in datasets.items():
            S = e.get("summary") or {}
            r = S.get("contrasts", {}).get(key) if not key.startswith("E4 ") else \
                S.get("e4", {}).get("contrasts", {}).get(key[3:])
            m = None if r is None else r["nb"]["mean"]
            p = None if r is None else r["nb"]["p"]
            sgn = None if m is None or not np.isfinite(m) else int(np.sign(m))
            per[name] = {"mean": m, "p": p, "sign": sgn, "core": DATASETS[name]["core"],
                         "complete": bool(S.get("complete")),
                         "same_sign": None if ref is None or sgn is None else bool(sgn == ref and ref != 0),
                         "same_sign_p05": None if ref is None or sgn is None or p is None or not np.isfinite(p)
                         else bool(sgn == ref and ref != 0 and p < ALPHA)}

        def count(filt):
            sel = [v for v in per.values() if filt(v)]
            return {"n": len(sel), "same_sign": int(sum(bool(v["same_sign"]) for v in sel)),
                    "same_sign_p05": int(sum(bool(v["same_sign_p05"]) for v in sel))}
        rows[key] = {"hsa_mean": hsa["means"].get(key), "hsa_sign": ref, "per_dataset": per,
                     "core_complete": count(lambda v: v["core"] and v["complete"]),
                     "all_run": count(lambda v: True)}
    return rows


def gate_e11(counts, n_core=len(CORE)):
    """Cổng mục 6.13 trên năm bộ chính đủ lần chia."""
    cc = {k: counts[k]["core_complete"] for k in CONTRASTS}
    complete = all(v["n"] == n_core for v in cc.values())
    ok = all(cc[k]["same_sign"] >= GATE_MIN for k in ("C1", "C2", "C3"))
    not_rep = {k: sorted(n for n, v in counts[k]["per_dataset"].items()
                         if v["core"] and v["complete"] and not v["same_sign"]) for k in CONTRASTS}
    return {"complete": complete, "n_core": n_core, "threshold": f"{GATE_MIN}/{n_core}",
            "counts": {k: v["same_sign"] for k, v in cc.items()},
            "replicates": bool(complete and ok),
            "verdict": ("chưa đủ năm bộ chính với đủ lần chia" if not complete else
                        ("thứ tự lặp lại" if ok else "phụ lục: bảng điều kiện biên")),
            "not_replicating": not_rep,
            "note": "Chỉ Saber được dùng cho câu về giáo dục; bốn bộ OpenML cho câu về ML"}


# ---------------------------------------------------------------------------
def _nan_to_none(o):
    return dc._nan_to_none(preds_io.to_jsonable(o))


def print_dataset(name, S):
    k = kkey(PRIMARY_K)
    c = S["contrasts"]
    f = lambda r: f"{r['nb']['mean']:+.4g} (p {r['nb']['p']:.3g}, {r['pct_of_cost3_R1_bag10']:+.2f}%)" \
        if r["nb"]["p"] is not None and np.isfinite(r["nb"]["p"]) else f"{r['nb']['mean']:+.4g}"  # noqa: E731
    r2 = S["centers"][PRIMARY]["r2"]["mean"]
    print(f"\n[{name}] {S['n_splits']}/{S['n_ref']} lần chia; R² bag10 {r2:.3f}; cost_{k}(R1, bag10) "
          f"{S['cost3_R1_bag10']:.4g}")
    for cn in CONTRASTS:
        print(f"  {cn}: {f(c[cn])}")
    for cn, r in S["e4"]["contrasts"].items():
        print(f"  E4 {cn}: {f(r)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--preds-dir", default=DEFAULT_PREDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-configs", type=int, default=None,
                    help=f"cấu hình cho rs_tuned và R8 (mặc định {N_CONFIGS}; --smoke: 3)")
    ap.add_argument("--workers", type=int, default=4, help="số tiến trình joblib")
    ap.add_argument("--threads", type=int, default=None,
                    help="n_jobs của mỗi XGBoost (mặc định max(1, cpu_count // workers))")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--on-mismatch", choices=["raise", "recompute"], default="raise")
    ap.add_argument("--hsa-rules", default="results_cost/decomp_rules.json")
    ap.add_argument("--hsa-centers", default="results_cost/decomp_centers.json")
    ap.add_argument("--hsa-policy", default="results_cost/select_policy.json")
    args = ap.parse_args(argv)

    provenance.print_versions()
    t_start = time.time()
    seeds = args.seeds[:1] if args.smoke else list(args.seeds)
    n_ref = len(seeds) if args.smoke else len(SEEDS)
    n_cfg = args.n_configs if args.n_configs is not None else (3 if args.smoke else N_CONFIGS)
    bud = make_budget(args.smoke, n_cfg)
    threads = args.threads or splits.xgb_threads(args.workers)
    sums = read_sums(args.data_dir)
    base_cfg = {"script": "bench_decomp", "experiment": EXPERIMENT, "smoke": bool(args.smoke),
                "budget": asdict(bud), "cv_folds": CV_FOLDS, "es_frac": ES_FRAC, "ks": E1_KS,
                "n_bins": N_BINS, "n_samples": N_SAMPLES, "tail_mass": [TAIL_MASS_LOW, TAIL_MASS_HIGH],
                "primary_k": PRIMARY_K, "e4_rule": E4_RULE, "oof_source": dc.OOF_SOURCE}
    fp_run = preds_io.fingerprint(base_cfg)
    partial = args.out + ".partial"
    res = preds_io.load_partial(partial, fp_run, {"datasets": {}}, on_mismatch=args.on_mismatch)
    print(f"E11: bộ {args.datasets}; seeds {seeds}; workers {args.workers} x {threads} luồng; "
          f"ngân sách {asdict(bud)}", flush=True)

    for name in args.datasets:
        t_ds = time.time()
        sha = check_sha(name, args.data_dir, sums, args.smoke)
        F = load_dataset(name, args.data_dir)
        ds_cfg = base_cfg | {"dataset": name, "data_sha256": sha["sha256"], "encoding": F.info["encoding"],
                             "dropped": F.info["dropped"], "cluster": F.info["cluster"]}
        fps = {"data_sha256": sha["sha256"],
               "p1": preds_io.fingerprint(ds_cfg | {"unit": "phase1"}),
               "p2": preds_io.fingerprint(ds_cfg | {"unit": "phase2"}),
               "e4": preds_io.fingerprint(ds_cfg | {"unit": "e4"}),
               "r8_cfg": ds_cfg | {"unit": "r8", "K": PRIMARY_K, "n_bag": 2 if args.smoke else R8_BAG}}
        fp_ds = preds_io.fingerprint(ds_cfg)
        entry = res["datasets"].get(name)
        if entry is not None and entry.get("fingerprint") != fp_ds:
            msg = (f"{partial}: bộ {name} tính bằng mã/cấu hình/dữ liệu khác (dấu "
                   f"{str(entry.get('fingerprint'))[:12]} khác {fp_ds[:12]})")
            if args.on_mismatch != "recompute":
                raise preds_io.FingerprintMismatch(msg + "; xoá .partial hoặc --on-mismatch recompute")
            print(f"[bench] {msg} -> tính lại", flush=True)
            entry = None
        if entry is None:
            entry = {"fingerprint": fp_ds, "per_split": {}}
        entry |= {"info": F.info, "data": sha}
        res["datasets"][name] = entry
        ds_dir = os.path.join(args.preds_dir, name)
        os.makedirs(ds_dir, exist_ok=True)
        print(f"\n== {name}: n={F.n}, {len(F.columns())} đặc trưng, y {F.info['y_min']:g}..{F.info['y_max']:g} "
              f"(sd {F.info['y_sd']:.4g})" + (f"; cụm {F.info['cluster']} ({F.info['n_clusters']})"
                                              if F.info["cluster"] else ""), flush=True)
        for s in seeds:
            if str(s) in entry["per_split"]:
                continue
            entry["per_split"][str(s)] = run_split(F, name, s, bud, args, fps, ds_dir, threads)
            preds_io.dump_json_atomic(_nan_to_none(res), partial)
        entry["summary"] = summarize_dataset(entry, n_ref)
        entry["wall_s"] = time.time() - t_ds
        preds_io.dump_json_atomic(_nan_to_none(res), partial)
        print_dataset(name, entry["summary"])

    hsa = read_hsa({"rules": args.hsa_rules, "centers": args.hsa_centers, "policy": args.hsa_policy})
    done = {n: e for n, e in res["datasets"].items() if "summary" in e}
    counts = sign_counts(done, hsa)
    gate = gate_e11(counts)
    print("\nĐếm dấu so với HSA (bộ chính đủ lần chia / mọi bộ đã chạy):")
    for k, r in counts.items():
        print(f"  {k:9s} HSA {r['hsa_sign']}: {r['core_complete']['same_sign']}/{r['core_complete']['n']} "
              f"chính, {r['all_run']['same_sign']}/{r['all_run']['n']} tất cả")
    print(f"Cổng E11: {gate['verdict']} ({gate['counts']})")
    # Lựa chọn 1..10 của docstring, ghi nguyên văn để JSON tự giải thích
    block = __doc__.split("(ghi cả vào meta.choices):")[1].split("Chạy (server)")[0]
    choices = [" ".join(item.split()) for item in re.split(r"\n\s*(?=\d+\. )", block.strip())]
    meta = {"experiment": EXPERIMENT, "script": "bench_decomp", "fingerprint": fp_run,
            "protocol": "khung bài 24/9 mục 6.13; lần chia splits.outer_split (80/20), fold splits.fold_plan "
                        f"({CV_FOLDS}-fold, dừng sớm {ES_FRAC:.0%}); đuôi theo khối lượng {TAIL_MASS_LOW}/"
                        f"{TAIL_MASS_HIGH} của y huấn luyện; mọi thước đo theo RMSE; K = {PRIMARY_K}",
            "seeds": seeds, "n_ref": n_ref, "datasets_requested": args.datasets, "core": CORE,
            "budget": asdict(bud), "n_configs": n_cfg, "r8_n_bag": 2 if args.smoke else R8_BAG,
            "primary_center": PRIMARY, "r8_star": r8_star(PRIMARY), "e4_rule": E4_RULE,
            "centers": CENTERS, "rules": RULES, "smoke": bool(args.smoke), "workers": args.workers,
            "threads": threads, "data_dir": os.path.abspath(args.data_dir),
            "preds_dir": os.path.abspath(args.preds_dir), "choices": choices,
            "sign_convention": "âm = vế trái tốt hơn (gates.CONTRASTS)",
            "wall_s": time.time() - t_start, "provenance": provenance.stamp()}
    out = {"meta": meta, "hsa_reference": hsa, "sign_counts": counts, "gate": gate,
           "datasets": res["datasets"]}
    preds_io.dump_json_atomic(_nan_to_none(out), args.out)
    print(f"\nĐã ghi {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
