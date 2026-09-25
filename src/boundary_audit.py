# -*- coding: utf-8 -*-
"""Tầng 1 — mục 1.6: tỷ lệ chạm biên của từng gene, trên toàn bộ lượt chạy.

VÌ SAO PHẢI CÓ: bài chỉ thừa nhận beta chạm biên (Limitation 8). Nhưng
n_estimators chạm trần 800 ở phần lớn lượt chạy, nghĩa là MỌI cấu hình được báo
cáo ở Bảng 4/5/6 đều là nghiệm bị RÀNG BUỘC. Hai hệ quả bài chưa nêu:

  - Xu hướng "objective nhấn đuôi chọn cây nông hơn" bị lẫn với trần n_estimators:
    khi số cây bị chặn, max_depth phải gánh toàn bộ capacity.
  - Nghiệm nằm trên một MẶT của hộp thì mặt đó có diện tích lớn, nên random search
    chạm nó dễ ngang GA — điều này góp phần giải thích kết quả null của Section IV-G.

Bảng boundary-hit là thực hành chuẩn trong empirical evolutionary computation và
hiện hoàn toàn vắng mặt trong bài.

Script chỉ đọc file, không huấn luyện gì.

Chạy: PYTHONPATH=src python3 src/boundary_audit.py
Kết quả: results_mseed/boundary_audit.json
"""
import glob
import json
import os

import numpy as np

import review_common as rc
from ga_xgb import BASE_GENES, WEIGHT_GENE

TOL = 1e-6
NEAR = 0.02      # "gần biên" = trong 2% biên độ của gene


def main():
    genes = {g[0]: (g[1], g[2]) for g in list(BASE_GENES) + [WEIGHT_GENE]}
    files = sorted(glob.glob(os.path.join(rc.MSEED_DIR, "*_best.json")))
    rows, evals = [], []
    for path in files:
        base = os.path.basename(path)
        if not base.startswith("ga_"):          # bỏ grid/random/baseline
            continue
        with open(path) as f:
            d = json.load(f)
        p = d.get("best_params", {})
        rows.append({"run": base.replace("_best.json", ""), "params": p,
                     "total_evals": d.get("total_evals")})
        if d.get("total_evals"):
            evals.append(d["total_evals"])

    audit = {}
    for name, (lo, hi) in genes.items():
        vals = [r["params"][name] for r in rows if name in r["params"]]
        if not vals:
            continue
        v = np.asarray(vals, float)
        span = hi - lo
        audit[name] = {
            "range": [lo, hi], "n_runs": len(v),
            "at_upper": int((v >= hi - TOL).sum()),
            "at_lower": int((v <= lo + TOL).sum()),
            "near_upper": int((v >= hi - NEAR * span).sum()),
            "near_lower": int((v <= lo + NEAR * span).sum()),
            "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()),
            "pct_at_boundary": float(((v >= hi - TOL) | (v <= lo + TOL)).mean())}

    if not rows:
        raise SystemExit(f"Không thấy file ga_*_best.json nào trong {rc.MSEED_DIR}/ "
                         "— chạy script này từ thư mục gốc của repo.")
    ev = np.asarray(evals, float)
    budget = {"ceiling_claimed": 632, "n_runs": len(ev),
              "min": int(ev.min()), "max": int(ev.max()),
              "mean": float(ev.mean()), "sd": float(ev.std(ddof=1)),
              "n_over_ceiling": int((ev > 632).sum()),
              "pct_over_ceiling": float((ev > 632).mean())}

    with open(f"{rc.MSEED_DIR}/boundary_audit.json", "w") as f:
        json.dump({"genes": audit, "budget": budget, "runs": rows},
                  f, indent=2, ensure_ascii=False)

    print(f"{'gene':<20}{'range':>16}{'at upper':>10}{'at lower':>10}"
          f"{'near up':>10}{'% biên':>9}")
    for name, d in audit.items():
        print(f"{name:<20}{str(d['range']):>16}{d['at_upper']:>6}/{d['n_runs']:<3}"
              f"{d['at_lower']:>6}/{d['n_runs']:<3}{d['near_upper']:>6}/{d['n_runs']:<3}"
              f"{100*d['pct_at_boundary']:>8.1f}%")
    print(f"\nNgân sách: {budget['n_runs']} lượt, {budget['min']}–{budget['max']} "
          f"(mean {budget['mean']:.0f} ± {budget['sd']:.0f}), "
          f"vượt trần 632: {budget['n_over_ceiling']}/{budget['n_runs']} "
          f"({100*budget['pct_over_ceiling']:.0f}%)")


if __name__ == "__main__":
    main()
