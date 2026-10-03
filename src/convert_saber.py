# -*- coding: utf-8 -*-
"""Chuyển bộ Saber 11 -> Saber Pro từ .xlsx sang data_public/saber.csv.gz (chạy MỘT lần ở Mac).

Vì sao có file này (E11, khung bài 24/9 mục 6.13): server không có openpyxl và
requirements.lock.txt không được thêm gói, nên server không đọc được .xlsx. Mac đọc
được (pandas.read_excel + openpyxl có sẵn trong môi trường cá nhân), nên chuyển một
lần ở đây rồi đẩy CSV lên cùng thư mục data_public/ (dữ liệu công khai CC BY 4.0,
Delahoz-Dominguez et al. 2020, Mendeley Data 83tcx8psxv v1; không có PII của HSA).

Chuyển ĐÚNG ĐỊNH DẠNG, không chọn cột: giữ mọi cột của sheet, chỉ bỏ cột rỗng hoàn
toàn ('Unnamed: 9', tiêu đề trống trong file gốc). Việc bỏ cột rò rỉ (*_PRO,
PERCENTILE, ...) làm ở bench_decomp.load_saber bằng danh sách cột ĐƯỢC DÙNG, để mọi
quyết định về đặc trưng nằm một chỗ và file CSV kiểm lại được với file gốc.

Tái lập từng byte: gzip ghi mtime vào header, nên chạy lại hai lần sẽ cho hai sha256
khác nhau dù nội dung như nhau. Đặt mtime = 0 để sha256 ghi trong SHA256SUMS kiểm
lại được bất cứ lúc nào.

Chạy (Mac): python3 src/convert_saber.py
"""
import argparse
import hashlib
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "data_public", "saber_academic_performance.xlsx")
DST = os.path.join(REPO, "data_public", "saber.csv.gz")
SUMS = os.path.join(REPO, "data_public", "SHA256SUMS")
SHEET = "SABER11_SABERPRO"
N_ROWS, N_COLS = 12411, 45          # số đo khi tải ngày 24/9 (45 cột gồm cột rỗng)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def update_sums(path, name, digest):
    """Thay (hoặc thêm) dòng của `name` trong SHA256SUMS, giữ nguyên các dòng khác."""
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    lines = [ln for ln in lines if ln.split()[-1] != name] + [f"{digest}  {name}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--sums", default=SUMS)
    args = ap.parse_args(argv)

    df = pd.read_excel(args.src, sheet_name=SHEET)
    if df.shape != (N_ROWS, N_COLS):
        sys.exit(f"{args.src}: dạng {df.shape}, cần {(N_ROWS, N_COLS)} (file khác bản đã tải?)")
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty != ["Unnamed: 9"]:
        sys.exit(f"cột rỗng hoàn toàn là {empty}, cần đúng ['Unnamed: 9']: kiểm lại file gốc")
    df = df.drop(columns=empty)
    df.to_csv(args.dst, index=False, encoding="utf-8",
              compression={"method": "gzip", "mtime": 0})
    # Đọc lại để chắc CSV giữ đúng nội dung (cùng dạng, cùng mục tiêu)
    back = pd.read_csv(args.dst)
    assert back.shape == df.shape, (back.shape, df.shape)
    assert (back["G_SC"].to_numpy() == df["G_SC"].to_numpy()).all()
    digest = sha256(args.dst)
    update_sums(args.sums, os.path.basename(args.dst), digest)
    print(f"{args.dst}: {df.shape[0]} dòng x {df.shape[1]} cột; sha256 {digest} -> {args.sums}")


if __name__ == "__main__":
    main()
