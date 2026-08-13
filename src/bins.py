# -*- coding: utf-8 -*-
"""Định nghĩa bin điểm dùng CHUNG cho fitness, bảng và hình.

⚠️ LÝ DO TỒN TẠI: fitness trong ga_xgb.py dùng np.digitize -> khoảng [lo, hi)
(đóng trái, mở phải). Nếu phần báo cáo dùng pd.cut (mặc định khoảng (lo, hi],
mở trái, đóng phải) thì học sinh có điểm ĐÚNG BẰNG mép bin sẽ rơi sang bin khác
-> bảng trong bài không khớp với bin mà thuật toán thật sự tối ưu.

Lỗi này đã xảy ra: đuôi thấp đếm 1,257 (pd.cut) vs 1,105 (digitize) — lệch 152
học sinh có điểm đúng bằng 60.

MỌI phân tích theo bin PHẢI import từ đây.
"""
import numpy as np

# Mép bin — khớp TAIL_BINS trong ga_xgb.py
EDGES = [0, 50, 60, 70, 80, 90, 100, 110, 150]
# Nhãn hiển thị: chú ý ký hiệu khoảng [lo, hi) để người đọc biết quy ước
LABELS = ["<50", "[50,60)", "[60,70)", "[70,80)", "[80,90)", "[90,100)",
          "[100,110)", ">=110"]

# Vùng gộp cho phân tích đuôi (dùng CÙNG quy ước đóng trái/mở phải)
LOW_TAIL_MAX = 60      # đuôi thấp: y < 60
HIGH_TAIL_MIN = 100    # đuôi cao: y >= 100


def bin_index(y):
    """Chỉ số bin cho từng mẫu — GIỐNG HỆT np.digitize trong fitness."""
    return np.digitize(np.asarray(y, dtype=float), EDGES[1:-1])


def bin_label(y):
    """Nhãn bin dạng chuỗi cho từng mẫu."""
    idx = bin_index(y)
    return np.asarray(LABELS, dtype=object)[idx]


def regions(y):
    """Mặt nạ 3 vùng + toàn bộ. Trả về dict tên -> mảng bool."""
    y = np.asarray(y, dtype=float)
    return {
        "Low tail": y < LOW_TAIL_MAX,
        "Middle": (y >= LOW_TAIL_MAX) & (y < HIGH_TAIL_MIN),
        "High tail": y >= HIGH_TAIL_MIN,
        "All": np.ones(len(y), dtype=bool),
    }


def bin_counts(y):
    """Số mẫu mỗi bin, theo thứ tự LABELS."""
    idx = bin_index(y)
    return np.array([(idx == i).sum() for i in range(len(LABELS))])


if __name__ == "__main__":
    # Kiểm tra tính nhất quán trên chính dữ liệu thật
    import sys
    sys.path.insert(0, "src")
    from preprocess import load_and_preprocess
    import pandas as pd
    _, y = load_and_preprocess()

    print("Số mẫu mỗi bin (digitize — quy ước của fitness):")
    for lab, c in zip(LABELS, bin_counts(y)):
        print(f"  {lab:>10}: {c:>6} ({100*c/len(y):>5.2f}%)")
    print(f"  {'TỔNG':>10}: {bin_counts(y).sum():>6}")

    r = regions(y)
    print("\nVùng gộp:")
    for k, m in r.items():
        print(f"  {k:>10}: {m.sum():>6} ({100*m.sum()/len(y):>5.2f}%)")

    # Chứng minh pd.cut cho kết quả KHÁC -> lý do file này tồn tại
    cut = pd.cut(y, bins=EDGES).value_counts().sort_index()
    print(f"\nSo sánh đuôi thấp:  digitize (y<60) = {(np.asarray(y)<60).sum()}"
          f"  |  pd.cut (0,50]+(50,60] = {cut.iloc[0] + cut.iloc[1]}"
          f"  |  lệch = {cut.iloc[0]+cut.iloc[1] - (np.asarray(y)<60).sum()}"
          f" học sinh có điểm đúng bằng 60")
