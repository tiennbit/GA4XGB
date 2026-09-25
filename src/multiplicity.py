# -*- coding: utf-8 -*-
"""Tầng 1 — mục 1.4: hiệu chỉnh đa so sánh + kiểm định phi tham số + power.

VÌ SAO PHẢI CÓ: Bảng 5 chứa 9 cấu hình × 4 vùng = 36 phép kiểm Welch, và bài
không nêu một chữ nào về Bonferroni/Holm/FDR. Với n = 5 mỗi nhóm còn có hai vấn
đề nữa: (a) không kiểm được normality, và (b) p hai phía NHỎ NHẤT mà một phép
hoán vị chính xác có thể cho là 2/C(10,5) = 0,0079 — nên mọi p < 0,008 in trong
bài là hệ quả của giả định tham số, không phải của bằng chứng phi tham số.

Script này KHÔNG huấn luyện mô hình nào; nó chỉ đọc results_mseed/regions.json.

Chạy: PYTHONPATH=src python3 src/multiplicity.py
Kết quả: results_mseed/multiplicity.json
"""
import json
from itertools import combinations

import numpy as np
from scipy import stats

import review_common as rc

BASE = "ga_rmse"
REGIONS = ["Low tail", "Middle", "High tail", "All"]


def holm(pvals):
    """Holm–Bonferroni step-down, trả về p đã hiệu chỉnh theo thứ tự đầu vào."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def bh(pvals):
    """Benjamini–Hochberg FDR."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        running = min(running, n * pvals[i] / (rank + 1))
        adj[i] = min(running, 1.0)
    return adj


def exact_perm_p(a, b):
    """p hai phía của phép hoán vị chính xác trên hiệu trung bình (n nhỏ)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    n = len(a)
    cnt = tot = 0
    for idx in combinations(range(len(pool)), n):
        m = np.zeros(len(pool), dtype=bool)
        m[list(idx)] = True
        if abs(pool[m].mean() - pool[~m].mean()) >= obs - 1e-12:
            cnt += 1
        tot += 1
    return cnt / tot


def power_welch(d, n, alpha=0.05, n_sim=200_000, seed=0):
    """Power mô phỏng của Welch t-test hai phía cho effect size Cohen's d."""
    rng = np.random.default_rng(seed)
    a = rng.normal(d, 1.0, size=(n_sim, n))
    b = rng.normal(0.0, 1.0, size=(n_sim, n))
    t, p = stats.ttest_ind(a, b, axis=1, equal_var=False)
    return float((p < alpha).mean())


def main():
    with open(f"{rc.MSEED_DIR}/regions.json") as f:
        runs = json.load(f)["runs"]

    base = {r: [runs[BASE][str(s)][r] for s in rc.SEEDS] for r in REGIONS}
    tests = []
    for name, seeds in runs.items():
        if name == BASE or len(seeds) < 2:      # grid_search chỉ có 1 seed
            continue
        for r in REGIONS:
            other = [seeds[str(s)][r] for s in rc.SEEDS if str(s) in seeds]
            if len(other) < 2:
                continue
            t, p = stats.ttest_ind(other, base[r], equal_var=False)
            a, b = np.asarray(other), np.asarray(base[r])
            sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            tests.append({
                "method": name, "region": r,
                "delta": float(a.mean() - b.mean()),
                "sd_method": float(a.std(ddof=1)), "sd_base": float(b.std(ddof=1)),
                "cohens_d": float((a.mean() - b.mean()) / sp) if sp > 0 else None,
                "p_welch": float(p),
                "p_perm": exact_perm_p(other, base[r]),
                "p_mwu": float(stats.mannwhitneyu(other, base[r],
                                                  alternative="two-sided").pvalue)})

    praw = np.array([t["p_welch"] for t in tests])
    for t, ph, pb in zip(tests, holm(praw), bh(praw)):
        t["p_holm"] = float(ph)
        t["p_bh"] = float(pb)

    perm_floor = 2 / len(list(combinations(range(10), 5)))
    powers = {f"d={d}": power_welch(d, 5) for d in (0.5, 0.65, 0.9, 1.5, 2.0, 3.0)}
    n_for_80 = {}
    for d in (0.65, 0.9):
        for n in range(5, 201):
            if power_welch(d, n, n_sim=20_000, seed=1) >= 0.80:
                n_for_80[f"d={d}"] = n
                break

    out = {"family_size": len(tests),
           "perm_p_floor_n5": perm_floor,
           "power_at_n5": powers,
           "n_needed_for_power_80": n_for_80,
           "note": ("p_welch < %.4f là không thể tái tạo bằng kiểm định phi tham số "
                    "ở n=5; mọi p nhỏ hơn ngưỡng này phải được báo cáo kèm p_perm."
                    % perm_floor),
           "tests": sorted(tests, key=lambda t: t["p_welch"])}
    with open(f"{rc.MSEED_DIR}/multiplicity.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Family = {len(tests)} phép kiểm | sàn p hoán vị (n=5) = {perm_floor:.4f}")
    print("Power ở n=5:", {k: round(v, 3) for k, v in powers.items()})
    print("n cần cho power 80%:", n_for_80)
    surv = lambda k: sum(1 for t in tests if t[k] < 0.05)
    print(f"\nSống sót ở 0.05 — raw {surv('p_welch')}/{len(tests)} | "
          f"Holm {surv('p_holm')}/{len(tests)} | BH {surv('p_bh')}/{len(tests)}")
    print(f"\n{'method':<18}{'region':<11}{'delta':>9}{'p_welch':>10}"
          f"{'p_holm':>9}{'p_bh':>9}{'p_perm':>9}")
    for t in out["tests"]:
        flag = "" if t["p_holm"] < 0.05 else "   <- chết dưới Holm"
        print(f"{t['method']:<18}{t['region']:<11}{t['delta']:>9.3f}"
              f"{t['p_welch']:>10.4f}{t['p_holm']:>9.3f}{t['p_bh']:>9.3f}"
              f"{t['p_perm']:>9.4f}{flag}")


if __name__ == "__main__":
    main()
