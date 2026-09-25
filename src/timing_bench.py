# -*- coding: utf-8 -*-
"""Đo thời gian huấn luyện mô hình CUỐI và thời gian suy luận, cho Bảng 6.

VÌ SAO PHẢI CÓ: cột "Final training (s)" của Bảng 6 trước đây lấy từ trường
`fit_seconds` ghi trong các file *_best.json — nhưng những con số đó đến từ lượt
chạy trên máy Mac (nay ở results_legacy_mac/), trong khi bài khẳng định mọi kết
quả đều sinh trên MỘT máy Xeon. Đó đúng là kiểu trộn hai máy mà chính bài cảnh
báo ở phần hạn chế.

Thêm nữa, `fit_seconds` trong các file đó không được đo dưới cùng một mức phân
bổ luồng: baseline chạy một mình, các lượt GA chạy 8 job đồng thời. Không so
sánh ngang hàng được.

Script này đo LẠI toàn bộ dưới CÙNG một cấu hình luồng, nêu rõ trong kết quả,
nên các dòng của Bảng 6 so sánh được với nhau — đó là điều duy nhất bảng đó cần.

Chạy: PYTHONPATH=src python3 src/timing_bench.py --n-jobs 4 --repeat 3
Kết quả: results_mseed/timing_bench.json
"""
import argparse
import json
import time

import numpy as np
from xgboost import XGBRegressor

import review_common as rc

# (nhãn trong Bảng 6, tên cấu hình trong results_mseed, có nhiều seed không)
ROWS = [
    ("Default XGBoost",      None,             False),
    ("Grid search",          "grid_search",    False),
    ("Random search",        "random_search",  True),
    ("GA-RMSE",              "ga_rmse",        True),
    ("GA4XGB (a=0.75)",      "ga_tail_a0.75",  True),
    ("GA4XGB (a=1)",         "ga_tail_a1",     True),
    ("GA4XGB (a=1, +LW)",    "ga_tail_a1_lw",  True),
]


def time_one(xgb_params, beta, X_tr, y_tr, X_te, n_jobs, repeat):
    """Trả về (giây huấn luyện, giây suy luận) — lấy TRUNG VỊ của `repeat` lần.

    Trung vị chứ không phải trung bình: máy đang chạy việc khác, một lần đo bị
    nhiễu nặng sẽ kéo lệch trung bình nhưng không kéo lệch trung vị."""
    fits, infs = [], []
    for _ in range(repeat):
        m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=n_jobs,
                         verbosity=0, **xgb_params)
        sw = rc.sample_weights(y_tr, beta) if beta is not None else None
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr, sample_weight=sw)
        fits.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        m.predict(X_te)
        infs.append(time.perf_counter() - t0)
    return float(np.median(fits)), float(np.median(infs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    X_tr, X_te, y_tr, y_te = rc.get_data()
    out = {"n_jobs": args.n_jobs, "repeat": args.repeat,
           "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
           "note": ("Trung vị của %d lần đo, cùng n_jobs=%d cho MỌI dòng nên các "
                    "dòng so sánh được với nhau." % (args.repeat, args.n_jobs)),
           "rows": {}}

    for label, name, multi in ROWS:
        fits, infs = [], []
        seeds = rc.SEEDS if multi else [42]
        for s in seeds:
            if name is None:
                xp, beta = {}, None          # XGBoost mặc định
            else:
                try:
                    xp, beta = rc.load_params(name, s)
                except FileNotFoundError:
                    continue
            f, inf = time_one(xp, beta, X_tr, y_tr, X_te, args.n_jobs, args.repeat)
            fits.append(f)
            infs.append(inf)
        if not fits:
            print(f"[timing] {label}: KHÔNG có cấu hình, bỏ qua", flush=True)
            continue
        out["rows"][label] = {"fit_s": rc.ms(fits), "infer_s": rc.ms(infs),
                              "n_config": len(fits)}
        print(f"[timing] {label:<20} fit {np.mean(fits):6.2f}s  "
              f"infer {1000*np.mean(infs):6.1f}ms  (n={len(fits)})", flush=True)

    with open(f"{rc.MSEED_DIR}/timing_bench.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"\n{'method':<22}{'fit (s)':>16}{'inference (ms)':>18}")
    for label, d in out["rows"].items():
        f, i = d["fit_s"], d["infer_s"]
        fs = f"{f['mean']:.1f}" + (f" ± {f['sd']:.1f}" if f["sd"] else "")
        is_ = f"{1000*i['mean']:.0f}" + (f" ± {1000*i['sd']:.0f}" if i["sd"] else "")
        print(f"{label:<22}{fs:>16}{is_:>18}")


if __name__ == "__main__":
    main()
