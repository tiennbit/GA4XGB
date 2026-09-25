# -*- coding: utf-8 -*-
"""Đo thời gian fit XGBoost trên server. Hai chế độ, cùng mục đích lập kế hoạch chạy.

1. E0 bước (8) (mặc định; khung bài 24/9, mục 6.2 và 6.1): đo thời gian fit của 20
   cấu hình ngẫu nhiên trên F_dt với early stopping, đúng như một cấu hình của E1 giai
   đoạn 2 và E2b được khớp, rồi áp cổng ngân sách: thời gian fit trung bình > 5 s
   (gates.FIT_TIME_MAX_S) thì ngân sách dò của E1 và E2b cùng giảm từ 60 xuống 40.

       PYTHONPATH=src python3 src/bench_eval.py --workers 4 --out results_cost/bench_eval.json

2. Chế độ cũ (có --jobs hoặc --reps): so thông lượng của một lần đánh giá GA (5-fold,
   cấu hình GA-RMSE) ở các mức n_jobs. Giữ nguyên hành vi, không sinh số cho bài.

       python src/bench_eval.py --jobs 16 8 4 2 --reps 2

Chế độ E0, một cấu hình = đúng các bước của E1 giai đoạn 2 cho một cấu hình:
- Lần chia, fold trong, tập dừng sớm từ splits.py (seed đầu của --seeds, mặc định
  100); cấu hình = splits.sample_configs(n, seed), tức ĐÚNG n cấu hình đầu mà E1 sẽ
  dò ở lần chia đó (cùng rng), không phải một mẫu riêng cho phép đo.
- Mỗi fold: ma trận từ features.Frame.design (cột trường tính từ hàng huấn luyện của
  fold), XGBRegressor(tree_method="hist", random_state=0, n_estimators=MAX_TREES,
  early_stopping_rounds=EARLY_STOP_ROUNDS, eval_metric="rmse") khớp trên `fit`, dừng
  trên `es`, dự đoán `va` và tập kiểm tra.
- Khớp lại trên toàn tập huấn luyện với số cây = int(trung vị(best_iteration + 1)).
Thời gian đọc dữ liệu không tính; thời gian dựng ma trận và dự đoán ghi riêng.

Vì sao cổng dùng "giây máy mỗi fit" chứ không dùng giây đồng hồ của một fit: khung
bài giả định 3 s mỗi fit và tính E2b = 4 K x 60 cấu hình x 5 fold x 10 lần chia =
12.000 fit ≈ 10,5 giờ, tức 3 s là thời gian của CẢ MÁY cho một fit (thông lượng).
E1/E2b chạy `workers` tiến trình song song, mỗi tiến trình n_jobs = cpu // workers
luồng, nên một fit trong tiến trình mất fit_wall giây đồng hồ nhưng máy xong
`workers` fit trong khoảng đó. Vì vậy:
    fit_wall_mean_s = trung bình giây đồng hồ của một fit trong fold, đo khi `workers`
                      tiến trình cùng chạy (có tranh chấp bộ nhớ, bộ đệm như thật);
    fit_machine_s   = fit_wall_mean_s / workers  (giả định mọi tiến trình luôn bận,
                      đúng với E1/E2b vì chúng có hàng nghìn fit).
Cổng so fit_machine_s với 5 s. Với --workers 1 hai số trùng nhau. Muốn cổng phản ánh
đúng E1/E2b thì chạy phép đo với cùng --workers mà run_decomp_centers.sh và
run_wtrain_tuned.sh sẽ dùng (run_data_audit.sh mặc định 4 trên máy 16 nhân). Vài cấu
hình cuối chạy khi đã ít tiến trình bận nên hơi nhanh hơn; với 20 cấu hình và 4
tiến trình độ lệch này nhỏ, và nó làm cổng lỏng chứ không chặt, nên ghi rõ ở JSON.

Lựa chọn khi khung bài chưa rõ (ghi cả vào meta.choices):
- Chỉ dùng seed đầu của --seeds: khung bài nói 20 cấu hình, không nói bao nhiêu lần chia.
- Chia không theo nhóm (groups=None) kể cả khi E0b thấy dòng trùng: thời gian fit
  không phụ thuộc cách chia.
- Không đo huấn luyện có trọng số: sample_weight không đổi đáng kể thời gian của hist;
  dự báo giờ của E2b dùng cùng giây mỗi cấu hình.
- --preds-dir nhận cho đồng bộ giao diện nhưng không dùng (không lưu dự đoán).

Chạy tiếp: <out>.partial giữ từng cấu hình đã đo (preds_io.load_partial, dấu gồm cả
workers và n_jobs vì chúng đổi số đo). Tiến trình con (loky) đọc CSV một lần rồi giữ
Frame trong bộ nhớ của module bench_eval (import theo tên để cache sống qua các tác vụ).
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from preprocess import DATA_PATH, load_and_preprocess
from ga_xgb import CV_FOLDS

import features
import preds_io
import provenance
import splits
from data_audit import guard_real_data
from gates import (EARLY_STOP_ROUNDS, FIT_TIME_MAX_S, K_GRID, MAX_TREES, R8_BAG, SEEDS,
                   TUNE_BUDGET, TUNE_BUDGET_FALLBACK)

# Cấu hình đại diện: lấy đúng best_params của GA-RMSE 5-fold. Dùng cấu hình rẻ
# hơn sẽ cho ước tính lạc quan giả tạo, vì GA dành phần lớn thời gian ở vùng
# cây nhiều/sâu.
REP_PARAMS = dict(n_estimators=791, max_depth=8, learning_rate=0.0412273275528882,
                  subsample=0.921518743752697, colsample_bytree=0.5002142225284837,
                  min_child_weight=6, reg_lambda=8.47385093920334)

N_CONFIGS = 20
SMOKE = {"n_configs": 2, "max_trees": 50, "es_rounds": 10}
LEGACY_FLAGS = ("--jobs", "--reps")


# ---------------------------------------------------------------------------
# Chế độ cũ: thông lượng một lần đánh giá GA theo n_jobs (giữ nguyên)
# ---------------------------------------------------------------------------
def one_eval(X, y, n_jobs):
    """Một lần đánh giá = CV_FOLDS lần fit, đúng như GA làm."""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    t0 = time.time()
    for tr, va in kf.split(X):
        m = XGBRegressor(**REP_PARAMS, n_jobs=n_jobs, verbosity=0,
                         tree_method="hist", random_state=42)
        m.fit(X.iloc[tr], y.iloc[tr])
        mean_squared_error(y.iloc[va], m.predict(X.iloc[va]))
    return time.time() - t0


def legacy_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, nargs="+", default=[16, 8, 4, 2])
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args(argv)

    X, y = load_and_preprocess()
    print(f"X {X.shape} | CV_FOLDS = {CV_FOLDS}")
    print("Làm nóng cache...")
    one_eval(X, y, max(a.jobs))

    print(f"\n{'n_jobs':>6} {'giây/eval':>10} {'tăng tốc':>9} {'hiệu suất':>10} "
          f"{'song song':>10} {'thông lượng':>12}")
    print("-" * 64)
    base = None
    rows = []
    for nj in sorted(a.jobs, reverse=True):
        ts = [one_eval(X, y, nj) for _ in range(a.reps)]
        t = float(np.median(ts))
        if base is None:
            base = t * max(a.jobs)   # thời gian * luồng = "công" chuẩn hoá
        speed = base / max(a.jobs) / t
        eff = speed / (max(a.jobs) / nj) if nj else 0
        # Nếu chạy đầy máy 16 nhân bằng các job nj luồng thì được mấy job song song,
        # và tổng số eval mỗi giờ là bao nhiêu.
        par = 16 // nj
        thr = par * 3600.0 / t
        rows.append((nj, t, thr))
        print(f"{nj:>6} {t:>10.1f} {speed:>9.2f}x {eff:>9.0%} {par:>10} {thr:>12.1f}")

    best = max(rows, key=lambda r: r[2])
    print(f"\nThông lượng cao nhất: n_jobs = {best[0]} "
          f"({16 // best[0]} job song song, {best[2]:.0f} eval/giờ)")
    print(f"Một lượt GA ~632 eval => {632 / best[2] * (16 // best[0]):.1f} giờ/job "
          f"khi chạy {16 // best[0]} job cùng lúc.")


# ---------------------------------------------------------------------------
# E0 bước (8): thời gian fit có early stopping của cấu hình E1/E2b
# ---------------------------------------------------------------------------
_FRAMES = {}


def _frame(data):
    """Frame đọc một lần cho mỗi tiến trình (loky dùng lại tiến trình con giữa các tác vụ)."""
    key = os.path.realpath(data)
    if key not in _FRAMES:
        _FRAMES[key] = features.load_frame(data)
    return _FRAMES[key]


def bench_config(j, cfg, data, fset, seed, n_jobs, max_trees, es_rounds):
    """Đo một cấu hình: CV_FOLDS fit có dừng sớm + một lần khớp lại. Trả dict số đo."""
    F = _frame(data)
    t_cfg = time.perf_counter()
    tr, te = splits.outer_split(F.n, seed)
    plan = splits.fold_plan(tr, seed)
    y = F.y
    oof = np.zeros(len(tr))
    fit_s, pred_s, best, rounds = [], [], [], []
    design_s = 0.0
    for f in plan:
        t0 = time.perf_counter()
        rows = [tr[f[k]] for k in ("train", "fit", "es", "va")]
        # Như make_fake_preds/E1: cột trường tính từ hàng huấn luyện của fold (fit ∪ es).
        _, Xfit, Xes, Xva, Xte = F.design(fset, *rows, te)
        design_s += time.perf_counter() - t0
        m = XGBRegressor(tree_method="hist", n_jobs=n_jobs, verbosity=0, random_state=0,
                         n_estimators=max_trees, early_stopping_rounds=es_rounds,
                         eval_metric="rmse", **cfg)
        t0 = time.perf_counter()
        m.fit(Xfit, y[rows[1]], eval_set=[(Xes, y[rows[2]])], verbose=False)
        fit_s.append(time.perf_counter() - t0)
        best.append(int(m.best_iteration))                       # đánh số từ 0
        rounds.append(int(m.get_booster().num_boosted_rounds()))  # số vòng đã chạy
        t0 = time.perf_counter()
        oof[f["va"]] = m.predict(Xva)
        m.predict(Xte)
        pred_s.append(time.perf_counter() - t0)

    n_trees = int(np.median(np.asarray(best) + 1))
    t0 = time.perf_counter()
    Xtr, Xte = F.design(fset, tr, te)
    design_s += time.perf_counter() - t0
    m = XGBRegressor(tree_method="hist", n_jobs=n_jobs, verbosity=0, random_state=0,
                     n_estimators=n_trees, **cfg)
    t0 = time.perf_counter()
    m.fit(Xtr, y[tr])
    refit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.predict(Xte)
    refit_pred_s = time.perf_counter() - t0
    return {"index": int(j), "params": cfg, "fold_fit_s": fit_s, "fold_best_iter": best,
            "fold_rounds": rounds, "fold_hit_cap": [r >= max_trees for r in rounds],
            "fold_predict_s": pred_s, "design_s": design_s, "n_trees_refit": n_trees,
            "refit_s": refit_s, "refit_predict_s": refit_pred_s,
            # RMSE OOF chỉ để kiểm hợp lý của phép đo, không phải số của bài.
            "oof_rmse": float(np.sqrt(np.mean((y[tr] - oof) ** 2))),
            "config_wall_s": time.perf_counter() - t_cfg, "n_features": int(Xtr.shape[1])}


def _stats(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)),
            "min": float(x.min()), "max": float(x.max())}


def summarize(per_config, workers, max_trees):
    pcs = [per_config[k] for k in sorted(per_config, key=int)]
    fits = np.concatenate([p["fold_fit_s"] for p in pcs])
    rounds = np.concatenate([p["fold_rounds"] for p in pcs]).astype(float)
    trees = np.concatenate([np.asarray(p["fold_best_iter"]) + 1 for p in pcs]).astype(float)
    cfg_wall = np.array([p["config_wall_s"] for p in pcs])
    refit = np.array([p["refit_s"] for p in pcs])
    fit_wall = float(fits.mean())
    return {
        "n_configs": len(pcs), "n_fold_fits": int(len(fits)),
        "fit_wall_s": _stats(fits),
        "fit_wall_mean_s": fit_wall,
        "fit_machine_s": fit_wall / workers,
        "s_per_round": _stats(fits / np.maximum(rounds, 1)),
        "trees_best": _stats(trees),
        "frac_hit_cap": float(np.mean(rounds >= max_trees)),
        "refit_s": _stats(refit),
        "n_trees_refit": _stats([p["n_trees_refit"] for p in pcs]),
        "config_wall_s": _stats(cfg_wall),
        "config_machine_s": float(cfg_wall.mean()) / workers,
        "predict_s_per_fold": float(np.mean(np.concatenate([p["fold_predict_s"] for p in pcs]))),
        "design_s_per_config": float(np.mean([p["design_s"] for p in pcs])),
        "oof_rmse": _stats([p["oof_rmse"] for p in pcs]),
    }


def budget_gate(summary, smoke):
    x = summary["fit_machine_s"]
    exceeded = bool(x > FIT_TIME_MAX_S)
    return {"metric": "fit_machine_s", "value": x, "threshold_s": FIT_TIME_MAX_S,
            "exceeded": exceeded,
            "tune_budget": TUNE_BUDGET_FALLBACK if exceeded else TUNE_BUDGET,
            "applies_to": ["E1 giai đoạn 2 (rs_tuned)", "E2b (R8, mọi K)"],
            "valid": not smoke,
            "definition": "fit_wall_mean_s / workers: giây của cả máy cho một fit trong fold "
                          "có early stopping (đơn vị của giả định 3 s ở mục 6.1)"}


def projection(summary, workers):
    """Giờ dự kiến trên máy đã đo, cho hai mức ngân sách. Mỗi cấu hình = CV_FOLDS fit dừng
    sớm + 1 lần khớp lại (E1 khớp lại cả 60 cấu hình cho oracle của E4); cộng R8_BAG - 1
    lần khớp lại cho *_bag5. Huấn luyện có trọng số (E2b) coi như cùng giá."""
    per_cfg = summary["config_machine_s"]
    extra = (R8_BAG - 1) * summary["refit_s"]["mean"] / workers
    n_splits = len(SEEDS)
    out = {"assumptions": f"{n_splits} lần chia, {len(K_GRID)} K cho E2b, giây máy mỗi cấu hình "
                          f"{per_cfg:.2f}, thêm {R8_BAG - 1} lần khớp lại cho *_bag5; chưa gồm E1 giai "
                          "đoạn 1 (default, bag) và hậu kỳ"}
    for B in (TUNE_BUDGET, TUNE_BUDGET_FALLBACK):
        one = B * per_cfg + extra
        out[f"budget_{B}"] = {"E1_stage2_h": n_splits * one / 3600,
                              "E2b_h": len(K_GRID) * n_splits * one / 3600}
    return out


def e0_main(argv=None):
    ap = argparse.ArgumentParser(description="E0 bước (8): thời gian fit có early stopping trên F_dt")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default="results_cost/bench_eval.json")
    ap.add_argument("--preds-dir", default=os.path.join("preds", "bench_eval"),
                    help="không dùng: phép đo không lưu dự đoán (nhận cho đồng bộ giao diện)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                    help="chỉ dùng seed đầu (lần chia và mẫu cấu hình của E1 cho seed đó)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="số tiến trình joblib chạy cùng lúc")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="luồng XGBoost mỗi tiến trình; mặc định max(1, cpu // workers)")
    ap.add_argument("--feature-set", default="F_dt", choices=features.FEATURE_SETS)
    ap.add_argument("--n-configs", type=int, default=N_CONFIGS)
    ap.add_argument("--max-trees", type=int, default=MAX_TREES)
    ap.add_argument("--es-rounds", type=int, default=EARLY_STOP_ROUNDS)
    args = ap.parse_args(argv)
    guard_real_data(args.data)
    provenance.print_versions()
    if args.smoke:
        args.n_configs = min(args.n_configs, SMOKE["n_configs"])
        args.max_trees = min(args.max_trees, SMOKE["max_trees"])
        args.es_rounds = min(args.es_rounds, SMOKE["es_rounds"])
    workers = max(1, int(args.workers))
    n_jobs = int(args.n_jobs) if args.n_jobs else splits.xgb_threads(workers)
    seed = int(args.seeds[0])
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None

    # Import theo tên module (không phải __main__) để tiến trình con loky giữ cache Frame.
    import bench_eval as mod
    F = mod._frame(args.data)
    configs = splits.sample_configs(args.n_configs, seed)
    config = {"script": "bench_eval", "mode": "e0_fit_time", "data_sha256": preds_io.file_sha256(args.data),
              "feature_set": args.feature_set, "seed": seed, "n_configs": args.n_configs,
              "max_trees": args.max_trees, "es_rounds": args.es_rounds, "workers": workers,
              "n_jobs": n_jobs, "smoke": bool(args.smoke)}
    fp = preds_io.fingerprint(config)
    part = args.out + ".partial"
    res = preds_io.load_partial(part, fp, {"per_config": {}})
    todo = [j for j in range(args.n_configs) if str(j) not in res["per_config"]]
    print(f"{args.data}: n={F.n}, {len(F.columns(args.feature_set))} cột {args.feature_set}; seed {seed}; "
          f"{args.n_configs} cấu hình ({len(todo)} còn lại); workers={workers} x n_jobs={n_jobs} "
          f"(cpu={os.cpu_count()}, affinity={affinity}); trần {args.max_trees} cây, dừng sớm "
          f"{args.es_rounds}", flush=True)

    t0 = time.perf_counter()
    if todo:
        from joblib import Parallel, delayed
        jobs = (delayed(mod.bench_config)(j, configs[j], args.data, args.feature_set, seed, n_jobs,
                                          args.max_trees, args.es_rounds) for j in todo)
        for r in Parallel(n_jobs=workers, return_as="generator_unordered")(jobs):
            res["per_config"][str(r["index"])] = r
            preds_io.dump_json_atomic(res, part)
            print(f"  cfg {r['index']:>2}: fit {np.mean(r['fold_fit_s']):6.2f}s/fold, cây "
                  f"{int(np.median(np.asarray(r['fold_best_iter']) + 1)):>4}, chạm trần "
                  f"{sum(r['fold_hit_cap'])}/{len(r['fold_hit_cap'])}, khớp lại {r['refit_s']:.2f}s "
                  f"({len(res['per_config'])}/{args.n_configs})", flush=True)
    phase_wall = time.perf_counter() - t0

    s = summarize(res["per_config"], workers, args.max_trees)
    res["summary"] = s
    res["gate"] = budget_gate(s, args.smoke)
    res["projection_hours"] = projection(s, workers)
    res["meta"].update({
        "experiment": "E0 bước (8)", "spec": "notes/khung-bai-bao-2026-09-24.md mục 6.2, 6.1",
        "config": config, "cpu_count": os.cpu_count(), "affinity": affinity,
        "n_rows": int(F.n), "n_features": len(F.columns(args.feature_set)),
        "phase_wall_s_this_launch": phase_wall, "n_measured_this_launch": len(todo),
        "tail_effect": "vài cấu hình cuối chạy khi ít tiến trình bận nên nhanh hơn; làm cổng lỏng",
        "choices": [
            "Cổng dùng fit_machine_s = fit_wall_mean_s / workers (thông lượng, như giả định 3 s).",
            "Chỉ seed đầu của --seeds; cấu hình = n cấu hình đầu E1 sẽ dò ở seed đó.",
            "Chia không theo nhóm; không đo huấn luyện có trọng số (coi cùng giá).",
        ],
        "provenance": provenance.stamp(),
    })
    preds_io.dump_json_atomic(res, args.out)
    g = res["gate"]
    p = res["projection_hours"]
    print(f"fit trong fold: {s['fit_wall_mean_s']:.2f}s đồng hồ / {workers} tiến trình = "
          f"{s['fit_machine_s']:.2f}s máy (ngưỡng {FIT_TIME_MAX_S:g}); chạm trần {s['frac_hit_cap']:.0%}; "
          f"cây trung vị {s['trees_best']['median']:.0f}", flush=True)
    print(f"Cổng: ngân sách dò = {g['tune_budget']}{'' if g['valid'] else ' (smoke, không dùng)'}; "
          + "; ".join(f"B={b.split('_')[1]}: E1 gđ2 {v['E1_stage2_h']:.1f} h, E2b {v['E2b_h']:.1f} h"
                      for b, v in p.items() if b.startswith("budget_")), flush=True)
    print(f"Đã ghi {args.out}", flush=True)
    try:
        os.remove(part)
    except OSError:
        pass
    return res


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if any(a.split("=")[0] in LEGACY_FLAGS for a in argv):
        return legacy_main(argv)
    return e0_main(argv)


if __name__ == "__main__":
    main()
