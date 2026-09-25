# -*- coding: utf-8 -*-
"""MAE theo vùng cho chiến dịch multi-seed — luận điểm trung tâm của bài nằm ở đây.

Các file *_best.json chỉ lưu chỉ số TỔNG HỢP (rmse/mae/r2). Bài báo lại khẳng
định ở hai ĐUÔI, nên phải fit lại từng cấu hình rồi chấm theo vùng. Mỗi cấu hình
một lần fit (~6 giây) nên cả chiến dịch chỉ vài phút.

PHẢI chạy trên cùng máy đã sinh ra các cấu hình đó: kết quả khác nhau giữa các
kiến trúc CPU (đã đo: cùng seed, cùng code, lệch tới 0,025 điểm RMSE).

  python src/mseed_regions.py [--dir results_mseed]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

import bins as B
from preprocess import load_and_preprocess
from ga_xgb import split_params, sample_weights

REGIONS = ["Low tail", "Middle", "High tail", "All"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_mseed")
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()

    X, y = load_and_preprocess()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    y_np = y_te.values
    masks = B.regions(y_np)

    runs = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*_best.json"))):
        base = os.path.basename(f)[: -len("_best.json")]
        m = re.search(r"_seed(\d+)$", base)
        seed = int(m.group(1)) if m else 42
        cfg = base[: m.start()] if m else base
        d = json.load(open(f, encoding="utf-8"))
        if "best_params" not in d:
            continue
        p, beta = split_params(d["best_params"])
        mdl = XGBRegressor(tree_method="hist", random_state=42,
                           n_jobs=a.n_jobs, verbosity=0, **p)
        sw = sample_weights(y_tr, beta) if beta is not None else None
        mdl.fit(X_tr, y_tr, sample_weight=sw)
        pr = mdl.predict(X_te)
        runs.setdefault(cfg, {})[seed] = {
            r: float(mean_absolute_error(y_np[mk], pr[mk])) for r, mk in masks.items()}
        print(f"  {cfg:<22} seed {seed:<3} "
              + "  ".join(f"{r} {runs[cfg][seed][r]:.3f}" for r in REGIONS))

    # Dòng "Fixed: cấu hình RMSE + beta = 1" của Bảng 5. Nó KHÔNG phải một lượt
    # tìm kiếm riêng: lấy đúng siêu tham số mà GA-RMSE chọn ở TỪNG seed rồi áp
    # trọng số nghịch mật độ beta = 1 mà không tối ưu lại — tách phần công của
    # việc tìm kiếm chung khỏi phần công của loss weighting đặt tay [20]. Vì mỗi
    # seed cho một cấu hình khác nhau nên dòng này cũng có SD như các dòng khác.
    if "ga_rmse" in runs:
        fixed = {}
        for seed in runs["ga_rmse"]:
            sfx = "" if seed == 42 else f"_seed{seed}"
            fp = os.path.join(a.dir, f"ga_rmse{sfx}_best.json")
            if not os.path.exists(fp):
                continue
            pr_, _ = split_params(json.load(open(fp, encoding="utf-8"))["best_params"])
            mdl = XGBRegressor(tree_method="hist", random_state=42,
                               n_jobs=a.n_jobs, verbosity=0, **pr_)
            mdl.fit(X_tr, y_tr, sample_weight=sample_weights(y_tr, 1.0))
            pv = mdl.predict(X_te)
            fixed[seed] = {r: float(mean_absolute_error(y_np[mk], pv[mk]))
                           for r, mk in masks.items()}
            print(f"  {'fixed_beta1':<22} seed {seed:<3} "
                  + "  ".join(f"{r} {fixed[seed][r]:.3f}" for r in REGIONS))
        if fixed:
            runs["fixed_beta1"] = fixed

    out = {"n_by_region": {r: int(mk.sum()) for r, mk in masks.items()}, "runs": runs}
    with open(os.path.join(a.dir, "regions.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- tổng hợp mean +- SD, so với GA-RMSE bằng Welch t-test ----
    try:
        from scipy import stats
    except ImportError:
        stats = None

    def ms(xs):
        n = len(xs); mu = sum(xs) / n
        sd = (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else float("nan")
        return mu, sd

    base = "ga_rmse"
    print("\n" + "=" * 92)
    print("MAE THEO VÙNG — mean ± SD qua các seed")
    print("=" * 92)
    hdr = f"{'cấu hình':<22}{'n':>3}"
    for r in REGIONS:
        hdr += f"{r:>18}"
    print(hdr)
    order = sorted(runs, key=lambda c: (("tail" in c), c))
    for cfg in order:
        seeds = runs[cfg]
        line = f"{cfg:<22}{len(seeds):>3}"
        for r in REGIONS:
            mu, sd = ms([v[r] for v in seeds.values()])
            line += f"{mu:>11.3f}±{sd:<6.3f}"
        print(line)

    if base in runs and len(runs[base]) >= 2:
        print("\n" + "=" * 92)
        print(f"So với {base} — Welch t-test theo từng vùng "
              f"(dấu âm = TỐT HƠN {base})")
        print("=" * 92)
        print(f"{'cấu hình':<22}" + "".join(f"{r:>22}" for r in REGIONS[:3]))
        for cfg in order:
            if cfg == base:
                continue
            if len(runs[cfg]) < 2:
                continue
            line = f"{cfg:<22}"
            for r in REGIONS[:3]:
                xs = [v[r] for v in runs[cfg].values()]
                bs = [v[r] for v in runs[base].values()]
                diff = sum(xs) / len(xs) - sum(bs) / len(bs)
                pv = stats.ttest_ind(xs, bs, equal_var=False)[1] if stats else float("nan")
                mark = "*" if pv == pv and pv < 0.05 else " "
                line += f"{diff:>+13.3f} p={pv:<5.3f}{mark}"
            print(line)
        print("\n* = p < 0,05. n = 3–5 nên lực kiểm định thấp: không có * nghĩa là")
        print("CHƯA ĐỦ BẰNG CHỨNG, không phải đã chứng minh hai bên như nhau.")
    print(f"\n-> {a.dir}/regions.json")


if __name__ == "__main__":
    main()
