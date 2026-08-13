# -*- coding: utf-8 -*-
"""Tiền xử lý dữ liệu học bạ THPT -> ma trận đặc trưng cho bài toán dự đoán Điểm HSA.

Quy ước mã hoá:
- Học lực  : Yếu=0, Trung bình=1, Khá=2, Giỏi=3   (thang ordinal)
- Hạnh kiểm: Yếu=0, Trung bình=1, Khá=2, Tốt=3    (thang ordinal)
- Giới tính: Nam=1, Nữ=0
- khuVuc, Tỉnh, Môn ngoại ngữ: one-hot
- Trường   : frequency encoding + cờ trường chuyên (tránh 988 cột one-hot)
- Ngày sinh: tách tháng sinh; STT bỏ
"""
import pandas as pd
import numpy as np

DATA_PATH = "data/data_final.csv"
TARGET = "Điểm HSA"

HOC_LUC = {"Yếu": 0, "Trung bình": 1, "Khá": 2, "Giỏi": 3}
HANH_KIEM = {"Yếu": 0, "Trung bình": 1, "Khá": 2, "Tốt": 3}


def load_and_preprocess(path: str = DATA_PATH):
    df = pd.read_csv(path)
    y = df[TARGET].astype(float)
    df = df.drop(columns=[TARGET, "STT"])

    # Ngày sinh -> tháng sinh (cùng cohort nên năm ít thông tin)
    ns = pd.to_datetime(df["Ngày sinh"], format="%d/%m/%Y", errors="coerce")
    df["thangSinh"] = ns.dt.month.fillna(0).astype(int)
    df = df.drop(columns=["Ngày sinh"])

    df["gioiTinh"] = (df["Giới tính"] == "Nam").astype(int)
    df = df.drop(columns=["Giới tính"])

    # Học lực / Hạnh kiểm -> ordinal
    for c in df.columns:
        if "Học lực" in c:
            df[c] = df[c].map(HOC_LUC).astype(float)
        elif "Hạnh kiểm" in c:
            df[c] = df[c].map(HANH_KIEM).astype(float)

    # Trường: frequency encoding + cờ trường chuyên
    freq = df["Trường"].value_counts(normalize=True)
    df["truong_freq"] = df["Trường"].map(freq).astype(float)
    df["truong_chuyen"] = df["Trường"].str.contains("huyên", na=False).astype(int)
    df = df.drop(columns=["Trường"])

    # One-hot cho các categorical ít giá trị
    df = pd.get_dummies(
        df,
        columns=["khuVuc", "Tỉnh", "10.Môn ngoại ngữ", "11.Môn ngoại ngữ", "12.Môn ngoại ngữ"],
        dtype=int,
    )

    # Tên cột an toàn cho XGBoost (bỏ ký tự đặc biệt)
    df.columns = (
        df.columns.str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", "_lt_", regex=False)
    )
    assert df.isna().sum().sum() == 0, "Còn giá trị thiếu sau tiền xử lý"
    return df, y


if __name__ == "__main__":
    X, y = load_and_preprocess()
    print("X:", X.shape, "| y:", y.shape)
    print("dtypes:", X.dtypes.value_counts().to_dict())
