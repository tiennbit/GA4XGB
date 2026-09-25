# -*- coding: utf-8 -*-
"""Sinh CSV GIẢ cùng schema với data/data_final.csv để kiểm thử trên Mac.

Vì sao: data/data_final.csv có ngày sinh, trường, tỉnh của 57.174 thí sinh và chỉ
được đồng bộ lên server thí nghiệm (AGENTS.md, ràng buộc 1). Trên Mac không script
nào được chạy trên file thật; mọi kiểm thử khói dùng file giả này.

Header nhúng sẵn ở dưới (lấy bằng `head -1`, chỉ tên cột, không có dữ liệu). Nếu
file thật có mặt, script đọc DUY NHẤT dòng đầu của nó để kiểm header còn khớp; nó
không đọc thêm dòng nào.

Cấu trúc dữ liệu giả, đủ giống thật để các nhánh mã đều chạy:
- 20 tỉnh ("Thành phố ...", "Tỉnh ..." như nhãn thật), 60 trường, mỗi tỉnh một
  trường chuyên; tên trường thường trùng giữa các tỉnh để khoá (tỉnh, trường) có ý nghĩa.
- Năng lực tiềm ẩn a; điểm học bạ = hàm của a + độ "nới điểm" của trường + nhiễu;
  CN = (HK I + 2·HK II)/3 như quy chế; học lực suy từ điểm tổng kết; hạnh kiểm
  phần lớn Tốt. Chỉ dùng nhãn mà preprocess.py ánh xạ được (Yếu/Trung bình/Khá/
  Giỏi, Yếu/Trung bình/Khá/Tốt), nếu không preprocess ra NaN và dừng.
- Ngày sinh dạng %d/%m/%Y, vài dòng thi lại (sinh 2005) và vài dòng ngày lỗi hoặc
  trống (preprocess cho tháng 0).
- y = làm tròn (hàm của a + hiệu ứng trường, tỉnh, giới + nhiễu), cắt [29; 129],
  trung bình ~77, SD ~13,6; khoảng 0,5% bản ghi "đoán mò" (Binomial(112; 0,25))
  để vùng y <= 37 có mặt.

Chạy: python3 tests/make_fake_data.py [--n 3000] [--seed 0] [--out tests/fixtures/fake_hsa.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "tests", "fixtures", "fake_hsa.csv")
REAL = os.path.join(ROOT, "data", "data_final.csv")

HEADER = "STT,Ngày sinh,Giới tính,Trường,Tỉnh,khuVuc,10.Điểm tổng kết HK I,10.Điểm tổng kết HK II,10.Điểm tổng kết CN,10.Học lực HK I,10.Học lực HK II,10.Học lực CN,10.Hạnh kiểm HK I,10.Hạnh kiểm HK II,10.Hạnh kiểm CN,10.Toán HK I,10.Toán HK II,10.Toán CN,10.Văn HK I,10.Văn HK II,10.Văn CN,10.Vật lí HK I,10.Vật lí HK II,10.Vật lí CN,10.Hóa học HK I,10.Hóa học HK II,10.Hóa học CN,10.Sinh học HK I,10.Sinh học HK II,10.Sinh học CN,10.Lịch sử HK I,10.Lịch sử HK II,10.Lịch sử CN,10.Địa lí HK I,10.Địa lí HK II,10.Địa lí CN,10.GDCD HK I,10.GDCD HK II,10.GDCD CN,10.Ngoại ngữ HK I,10.Ngoại ngữ HK II,10.Ngoại ngữ CN,10.Môn ngoại ngữ,11.Điểm tổng kết HK I,11.Điểm tổng kết HK II,11.Điểm tổng kết CN,11.Học lực HK I,11.Học lực HK II,11.Học lực CN,11.Hạnh kiểm HK I,11.Hạnh kiểm HK II,11.Hạnh kiểm CN,11.Toán HK I,11.Toán HK II,11.Toán CN,11.Văn HK I,11.Văn HK II,11.Văn CN,11.Vật lí HK I,11.Vật lí HK II,11.Vật lí CN,11.Hóa học HK I,11.Hóa học HK II,11.Hóa học CN,11.Sinh học HK I,11.Sinh học HK II,11.Sinh học CN,11.Lịch sử HK I,11.Lịch sử HK II,11.Lịch sử CN,11.Địa lí HK I,11.Địa lí HK II,11.Địa lí CN,11.GDCD HK I,11.GDCD HK II,11.GDCD CN,11.Ngoại ngữ HK I,11.Ngoại ngữ HK II,11.Ngoại ngữ CN,11.Môn ngoại ngữ,12.Điểm tổng kết HK I,12.Điểm tổng kết HK II,12.Điểm tổng kết CN,12.Học lực HK I,12.Học lực HK II,12.Học lực CN,12.Hạnh kiểm HK I,12.Hạnh kiểm HK II,12.Hạnh kiểm CN,12.Toán HK I,12.Toán HK II,12.Toán CN,12.Văn HK I,12.Văn HK II,12.Văn CN,12.Vật lí HK I,12.Vật lí HK II,12.Vật lí CN,12.Hóa học HK I,12.Hóa học HK II,12.Hóa học CN,12.Sinh học HK I,12.Sinh học HK II,12.Sinh học CN,12.Lịch sử HK I,12.Lịch sử HK II,12.Lịch sử CN,12.Địa lí HK I,12.Địa lí HK II,12.Địa lí CN,12.GDCD HK I,12.GDCD HK II,12.GDCD CN,12.Ngoại ngữ HK I,12.Ngoại ngữ HK II,12.Ngoại ngữ CN,12.Môn ngoại ngữ,Điểm HSA"  # noqa: E501
COLUMNS = HEADER.split(",")

SUBJECTS = ["Toán", "Văn", "Vật lí", "Hóa học", "Sinh học", "Lịch sử", "Địa lí", "GDCD", "Ngoại ngữ"]
# Độ nặng của năng lực chung lên từng môn: Toán, Lí, Hoá gắn mạnh với HSA hơn GDCD.
LOAD = np.array([1.00, 0.70, 0.90, 0.90, 0.75, 0.55, 0.55, 0.35, 0.80])
PROVINCES = ["Thành phố Hà Nội", "Tỉnh Nam Định", "Tỉnh Thái Bình", "Tỉnh Nghệ An", "Tỉnh Thanh Hóa",
             "Tỉnh Hà Tĩnh", "Tỉnh Bắc Ninh", "Tỉnh Hải Dương", "Thành phố Hải Phòng", "Tỉnh Hưng Yên",
             "Tỉnh Vĩnh Phúc", "Tỉnh Phú Thọ", "Tỉnh Ninh Bình", "Tỉnh Hà Nam", "Tỉnh Bắc Giang",
             "Tỉnh Thái Nguyên", "Tỉnh Quảng Ninh", "Tỉnh Lào Cai", "Tỉnh Yên Bái", "Tỉnh Hòa Bình"]
SCHOOL_POOL = ["THPT Nguyễn Du", "THPT Trần Phú", "THPT Lê Quý Đôn", "THPT Nguyễn Trãi",
               "THPT Phan Bội Châu", "THPT Lê Lợi", "THPT Quang Trung", "THPT Lý Thường Kiệt",
               "THPT Chu Văn An", "THPT Ngô Quyền", "THPT Trần Hưng Đạo", "THPT Hoàng Văn Thụ"]
LANGS = ["Tiếng Anh", "Tiếng Pháp", "Tiếng Trung", "Tiếng Nga", "Tiếng Nhật"]
LANG_P = [0.92, 0.03, 0.02, 0.01, 0.02]
REGIONS = ["KV1", "KV2", "KV2-NT", "KV3"]


def check_header():
    """So header nhúng với dòng đầu file thật nếu có. Chỉ đọc MỘT dòng."""
    if not os.path.exists(REAL):
        return None
    with open(REAL, encoding="utf-8") as fh:
        first = fh.readline().rstrip("\r\n").lstrip("﻿")
    return first == HEADER


def standing(g):
    """Học lực từ điểm tổng kết (ngưỡng quen thuộc 8,0 / 6,5 / 5,0)."""
    return np.select([g >= 8.0, g >= 6.5, g >= 5.0], ["Giỏi", "Khá", "Trung bình"], "Yếu")


def make(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    # --- trường và tỉnh ---
    schools = []                                      # (tỉnh, tên, chuyên?, KV)
    for i, p in enumerate(PROVINCES):
        short = p.replace("Thành phố ", "").replace("Tỉnh ", "")
        kv = REGIONS[i % 4]
        schools.append((p, f"THPT Chuyên {short}", True, "KV2" if kv == "KV1" else kv))
        for name in rng.choice(SCHOOL_POOL, size=2, replace=False):
            schools.append((p, str(name), False, kv))
    S = len(schools)
    prov_eff = dict(zip(PROVINCES, rng.normal(0, 1.5, len(PROVINCES))))
    school_eff = rng.normal(0, 2.5, S) + np.array([6.0 if s[2] else 0.0 for s in schools])
    lenient = rng.normal(0, 0.3, S)                   # trường nới điểm: học bạ cao, HSA không cao
    size_w = rng.lognormal(0, 0.8, S) * np.array([0.3 if s[2] else 1.0 for s in schools])
    sch = rng.choice(S, size=n, p=size_w / size_w.sum())
    is_ch = np.array([schools[s][2] for s in sch])

    # --- học sinh ---
    a = rng.normal(0, 1, n) + 0.6 * is_ch
    male = rng.random(n) < 0.47
    lang = rng.choice(LANGS, size=n, p=LANG_P)
    df = {c: None for c in COLUMNS}
    df["STT"] = np.arange(1, n + 1)
    year = np.where(rng.random(n) < 0.02, 2005, 2006)
    month, day = rng.integers(1, 13, n), rng.integers(1, 29, n)
    dob = np.array([f"{d:02d}/{m:02d}/{y}" for d, m, y in zip(day, month, year)], dtype=object)
    bad = rng.choice(n, size=max(2, n // 1000), replace=False)
    dob[bad[: len(bad) // 2]] = ""                     # trống -> NaN -> tháng 0
    dob[bad[len(bad) // 2:]] = "31/02/2006"            # ngày không tồn tại -> NaT -> tháng 0
    df["Ngày sinh"] = dob
    df["Giới tính"] = np.where(male, "Nam", "Nữ")
    df["Trường"] = [schools[s][1] for s in sch]
    df["Tỉnh"] = [schools[s][0] for s in sch]
    kv = np.array([schools[s][3] for s in sch], dtype=object)
    flip = rng.random(n) < 0.1
    kv[flip] = rng.choice(REGIONS, size=flip.sum())
    df["khuVuc"] = kv

    for gi, grade in enumerate([10, 11, 12]):
        subj = {}
        for si, sub in enumerate(SUBJECTS):
            base = 6.9 + 0.12 * gi + 0.95 * LOAD[si] * a + lenient[sch] + rng.normal(0, 0.35, n)
            h1 = np.clip(np.round(base + rng.normal(0, 0.45, n), 1), 0, 10)
            h2 = np.clip(np.round(base + 0.1 + rng.normal(0, 0.45, n), 1), 0, 10)
            cn = np.round((h1 + 2 * h2) / 3, 1)
            subj[sub] = (h1, h2, cn)
            for tag, v in zip(["HK I", "HK II", "CN"], subj[sub]):
                df[f"{grade}.{sub} {tag}"] = v
        for k, tag in enumerate(["HK I", "HK II", "CN"]):
            tk = np.round(np.mean([subj[s][k] for s in SUBJECTS], axis=0), 1)
            df[f"{grade}.Điểm tổng kết {tag}"] = tk
            df[f"{grade}.Học lực {tag}"] = standing(tk)
            df[f"{grade}.Hạnh kiểm {tag}"] = rng.choice(["Tốt", "Khá", "Trung bình", "Yếu"], size=n,
                                                       p=[0.90, 0.08, 0.015, 0.005])
        df[f"{grade}.Môn ngoại ngữ"] = lang

    ystar = (75.6 + 9.4 * a + school_eff[sch] + np.array([prov_eff[schools[s][0]] for s in sch])
             + 1.2 * male + rng.gamma(4.0, 1.6, n) - 6.4 + rng.normal(0, 7.7, n))
    y = np.round(ystar)
    guess = rng.random(n) < 0.005
    y[guess] = rng.binomial(112, 0.25, guess.sum())
    df["Điểm HSA"] = np.clip(y, 29, 129).astype(int)
    out = pd.DataFrame({c: df[c] for c in COLUMNS})
    assert list(out.columns) == COLUMNS
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    ok = check_header()
    if ok is False:
        sys.exit("Header nhúng KHÔNG khớp dòng đầu data/data_final.csv: cập nhật HEADER (head -1).")
    df = make(args.n, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, args.out)
    y = df["Điểm HSA"].to_numpy(float)
    print(f"{args.out}: {len(df)} dòng, {df.shape[1]} cột, header "
          f"{'khớp file thật' if ok else 'chưa kiểm (không có file thật)'}; "
          f"y mean={y.mean():.1f} sd={y.std(ddof=1):.1f} min={y.min():.0f} max={y.max():.0f}; "
          f"y<60 {np.mean(y < 60):.1%}, y>=100 {np.mean(y >= 100):.1%}, y<=37 {int((y <= 37).sum())}; "
          f"{df['Tỉnh'].nunique()} tỉnh, {df.groupby(['Tỉnh', 'Trường']).ngroups} trường")
    return args.out


if __name__ == "__main__":
    main()
