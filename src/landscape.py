# -*- coding: utf-8 -*-
"""Tầng 1 — mục 1.10: phân tích fitness landscape để CHẨN ĐOÁN kết quả null.

VÌ SAO PHẢI CÓ: Section IV-G đo được rằng GA không hơn random search, nhưng chỉ
báo cáo điều đó như một limitation. Có ba giả thuyết cạnh tranh mà bài không tách:

  (a) landscape phẳng / basin tối ưu rất lớn  -> mọi sampler đều trúng
  (b) GA bị cấu hình quá yếu (pop 24 cho 7-8 chiều là ~3 cá thể/chiều)
  (c) search space đặt sai, nghiệm nằm trên mặt biên (xem boundary_audit.py)

Nếu (a) đúng thì kết quả null CHUYỂN TỪ ĐIỂM YẾU THÀNH PHÁT HIỆN: "landscape
hyperparameter của GBDT trên bài toán này là plateau, và đó là lý do phần lớn
'gains' mà literature GA-XGBoost báo cáo từ một lượt chạy đơn là ảo."

Nhưng nó cũng đặt ra một cảnh báo phải viết vào bài: nếu landscape phẳng thì
luận điểm "objective quan trọng hơn search" gần như tautology — trên plateau mọi
search đều tương đương nên tất nhiên chỉ objective mới tạo khác biệt. Bài phải
phát biểu có điều kiện thay vì phát biểu phổ quát.

Hai phần:
  A. MIỄN PHÍ — đọc random_search_log.jsonl đã có: phân bố fitness, tỷ lệ
     near-optimal, fitness-distance correlation (FDC) toàn cục và trong top quartile.
  B. TỐN MÁY — random walk trong không gian gene để đo autocorrelation
     (--walk-steps 0 để bỏ qua).

Chạy: PYTHONPATH=src python3 src/landscape.py [--walk-steps 200] [--n-jobs 2]
Kết quả: results_mseed/landscape.json
"""
import argparse
import json
import os

import numpy as np

import review_common as rc
from ga_xgb import BASE_GENES, decode, split_params, GA

NEAR_OPT_THRESHOLDS = [0.02, 0.05, 0.10, 0.20]


def normalize(params_list):
    """Đưa từng gene về [0,1] theo range của nó để khoảng cách có nghĩa."""
    out = []
    for p in params_list:
        out.append([(p[n] - lo) / (hi - lo) for n, lo, hi, _, _ in BASE_GENES])
    return np.asarray(out, float)


def fdc(fits, pts, best_pt):
    """Fitness-distance correlation: tương quan giữa fitness và khoảng cách tới optimum.

    Quy ước cho bài toán MINIMIZE: FDC cao (-> 1) nghĩa là càng gần optimum fitness
    càng thấp, tức landscape có hướng dẫn đường cho search. Ngưỡng thường dùng
    trong literature EC: > 0,75 mới coi là landscape thân thiện với GA."""
    d = np.linalg.norm(pts - best_pt, axis=1)
    if d.std() < 1e-12 or fits.std() < 1e-12:
        return None
    return float(np.corrcoef(fits, d)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-steps", type=int, default=200)
    ap.add_argument("--walk-sigma", type=float, default=0.10,
                    help="bước đi ngẫu nhiên, tính theo tỷ lệ biên độ mỗi gene")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    log = os.path.join(rc.MSEED_DIR, "random_search_log.jsonl")
    recs = [json.loads(l) for l in open(log)]
    params = [r["params"] for r in recs]
    fits = np.asarray([r["cv_rmse"] for r in recs], float)   # fitness của random search
    pts = normalize(params)
    best_i = int(np.argmin(fits))
    fmin = float(fits[best_i])

    near = {f"within_{t}": float((fits <= fmin + t).mean())
            for t in NEAR_OPT_THRESHOLDS}
    q1 = fits <= np.quantile(fits, 0.25)
    out = {
        "n_samples": len(fits),
        "fitness": {"min": fmin, "median": float(np.median(fits)),
                    "max": float(fits.max()), "sd": float(fits.std(ddof=1)),
                    "iqr": float(np.subtract(*np.percentile(fits, [75, 25])))},
        "near_optimal_fraction": near,
        "fdc_all": fdc(fits, pts, pts[best_i]),
        "fdc_top_quartile": fdc(fits[q1], pts[q1], pts[best_i]),
        "interpretation": ("FDC > 0,75 = landscape thân thiện với GA; 0,15–0,75 = "
                           "khó/không cung cấp thông tin. FDC tụt khi zoom vào vùng "
                           "tốt = plateau quanh optimum."),
    }

    if args.walk_steps > 0:
        ga = GA("rmse", args.seed, args.n_jobs)
        rng = np.random.default_rng(args.seed)
        ind = ga.random_individual()
        walk = []
        for step in range(args.walk_steps):
            walk.append(ga.scores_of(ind)["rmse"])
            ind = [float(np.clip(v + rng.normal(0, args.walk_sigma * (hi - lo)), lo, hi))
                   for v, (_, lo, hi, _, _) in zip(ind, BASE_GENES)]
            if (step + 1) % 20 == 0:
                print(f"[walk] {step+1}/{args.walk_steps} rmse={walk[-1]:.4f}", flush=True)
        w = np.asarray(walk, float)
        acf = {f"lag{k}": float(np.corrcoef(w[:-k], w[k:])[0, 1])
               for k in (1, 2, 5, 10) if len(w) > k + 1}
        out["random_walk"] = {"steps": len(w), "sigma_frac": args.walk_sigma,
                              "autocorrelation": acf, "series": walk,
                              "note": ("acf(1) gần 1 = landscape trơn; gần 0 = gồ ghề. "
                                       "Trơn + FDC thấp = plateau.")}

    with open(f"{rc.MSEED_DIR}/landscape.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nMẫu uniform: {out['n_samples']}, best CV-RMSE = {fmin:.4f}")
    for k, v in near.items():
        print(f"  {k}: {100*v:.1f}% cấu hình ngẫu nhiên")
    print(f"FDC toàn cục      = {out['fdc_all']:.3f}")
    print(f"FDC top quartile  = {out['fdc_top_quartile']:.3f}")
    if "random_walk" in out:
        print("Autocorrelation:", {k: round(v, 3)
                                   for k, v in out["random_walk"]["autocorrelation"].items()})


if __name__ == "__main__":
    main()
