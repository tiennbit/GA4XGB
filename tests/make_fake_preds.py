# -*- coding: utf-8 -*-
"""Sinh npz dự đoán GIẢ theo lược đồ E1 (preds_io) từ CSV giả, để các script hậu kỳ
(E2, E3, E4, E5, E9, E10) chạy thử được trước khi E1 có kết quả thật.

Đúng giao thức E1 ở mức cấu trúc, thu nhỏ ở mức ngân sách:
- Lần chia, fold trong, tập dừng sớm lấy từ src/splits.py (seed mặc định 100, 101).
- Ma trận đặc trưng từ features.Frame.design: cột trường tính CHỈ từ hàng huấn
  luyện của từng mô hình.
- default: XGBRegressor(tree_method="hist", random_state=0), không dừng sớm, khớp
  trên cả fold huấn luyện (fit ∪ es).
- bag10: 10 mô hình random_state 0..9, subsample = colsample_bytree = 0,8.
- Vết cấu hình: n_cfg cấu hình từ splits.sample_configs(n_cfg, seed); mỗi cấu hình
  5 mô hình fold dừng sớm trên tập dừng sớm của fold, OOF trên fold giữ lại; khớp
  lại trên toàn tập huấn luyện với số cây = int(trung vị(best_iteration + 1)),
  lưu ở trace_n_trees (best_iteration của XGBoost đánh số từ 0).
  rs_tuned = cấu hình có RMSE OOF nhỏ nhất.
- test_foldavg / trace_test_foldavg: mỗi mô hình fold (chính mô hình sinh OOF) dự
  đoán cả tập kiểm tra; X_te của fold dựng trong CÙNG lời gọi F.design với hàng
  huấn luyện của fold, nên tần suất trường chỉ đến từ fold đó. E5 dùng để so nguồn
  ŷ test {mô hình khớp lại; trung bình 5 mô hình fold} mà không chạy lại E1.
- Biến nhóm E10 không lưu vào npz (preds_io: lấy từ features.load_frame trên server).
- meta.fingerprint = preds_io.fingerprint(config): chạy lại bỏ qua lần chia đã có
  cùng dấu. Fixture này mặc định TÍNH LẠI khi dấu lệch (--on-mismatch recompute) vì
  mã hạ tầng đổi thường xuyên; script thí nghiệm thật giữ mặc định "raise".
Trần cây và số vòng dừng sớm nhỏ (300, 20) để chạy trong khoảng một phút; E1 thật
dùng 3.000 và 50 (gates.MAX_TREES, EARLY_STOP_ROUNDS).

Chạy: PYTHONPATH=src python3 tests/make_fake_preds.py [--seeds 100 101] [--n-cfg 6] [--n-jobs 2]
Kết quả: tests/fixtures/preds/decomp/split<seed>.npz (thư mục bị gitignore).
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from xgboost import XGBRegressor  # noqa: E402

import decision_layer as dl  # noqa: E402
import features  # noqa: E402
import preds_io  # noqa: E402
import provenance  # noqa: E402
import splits  # noqa: E402
from gates import BAG_SUBSAMPLE  # noqa: E402

DEFAULT_DATA = os.path.join(ROOT, "tests", "fixtures", "fake_hsa.csv")
DEFAULT_OUT = os.path.join(ROOT, "tests", "fixtures", "preds", "decomp")
N_BAG = 10


def _xgb(n_jobs, **kw):
    return XGBRegressor(tree_method="hist", n_jobs=n_jobs, verbosity=0, **kw)


def center_predict(kind, Xa, ya, Xs, n_jobs):
    """Dự đoán của trung tâm không dừng sớm cho từng ma trận trong Xs."""
    if kind == "default":
        m = _xgb(n_jobs, random_state=0).fit(Xa, ya)
        return [m.predict(X) for X in Xs]
    if kind == "bag10":
        ms = [_xgb(n_jobs, random_state=b, subsample=BAG_SUBSAMPLE, colsample_bytree=BAG_SUBSAMPLE)
              .fit(Xa, ya) for b in range(N_BAG)]
        return [np.mean([m.predict(X) for m in ms], axis=0) for X in Xs]
    raise ValueError(kind)


def run_split(F, fset, seed, n_cfg, n_jobs, max_trees, es_rounds, fp=None):
    t0 = time.time()
    tr, te = splits.outer_split(F.n, seed)
    y, y_tr, y_te = F.y, F.y[tr], F.y[te]
    plan = splits.fold_plan(tr, seed)
    configs = splits.sample_configs(n_cfg, seed)

    oof = {c: np.zeros(len(tr)) for c in ["default", "bag10"]}
    foldavg = {c: np.zeros(len(te)) for c in oof}
    t_oof = np.zeros((n_cfg, len(tr)))
    t_te_f = np.zeros((n_cfg, len(te)))
    best_iter = np.zeros((n_cfg, len(plan)), dtype=np.int32)
    for f in plan:
        rows = [tr[f[k]] for k in ("train", "fit", "es", "va")]
        # te trong CÙNG lời gọi: cột trường của X_te tính từ hàng huấn luyện của fold
        Xa, Xfit, Xes, Xva, Xte_f = F.design(fset, *rows, te)
        for c in oof:
            p_va, p_te = center_predict(c, Xa, y[rows[0]], [Xva, Xte_f], n_jobs)
            oof[c][f["va"]] = p_va
            foldavg[c] += p_te / len(plan)
        for j, cfg in enumerate(configs):
            m = _xgb(n_jobs, random_state=0, n_estimators=max_trees, early_stopping_rounds=es_rounds,
                     eval_metric="rmse", **cfg)
            m.fit(Xfit, y[rows[1]], eval_set=[(Xes, y[rows[2]])], verbose=False)
            best_iter[j, f["fold"]] = m.best_iteration          # đánh số từ 0
            t_oof[j, f["va"]] = m.predict(Xva)          # predict dùng best_iteration
            t_te_f[j] += m.predict(Xte_f) / len(plan)

    Xtr, Xte = F.design(fset, tr, te)
    test = {c: center_predict(c, Xtr, y_tr, [Xte], n_jobs)[0] for c in ["default", "bag10"]}
    t_test = np.zeros((n_cfg, len(te)))
    n_trees = np.median(best_iter + 1, axis=1).astype(np.int32)      # số cây, không phải chỉ số
    for j, cfg in enumerate(configs):
        t_test[j] = _xgb(n_jobs, random_state=0, n_estimators=int(n_trees[j]), **cfg).fit(Xtr, y_tr).predict(Xte)
    t_rmse = np.sqrt(np.mean((y_tr[None, :] - t_oof) ** 2, axis=1))
    jb = int(np.argmin(t_rmse))
    oof["rs_tuned"], test["rs_tuned"] = t_oof[jb].copy(), t_test[jb].copy()
    foldavg["rs_tuned"] = t_te_f[jb].copy()

    lo, hi = dl.tail_cutoffs(y_tr)
    meta = {"seed": int(seed), "feature_set": fset, "fake": True, "lo": lo, "hi": hi,
            "centers": list(oof), "trace_params": configs, "rs_tuned_index": jb,
            "n_features": int(Xtr.shape[1]), "max_trees": max_trees, "es_rounds": es_rounds,
            "fingerprint": fp, "provenance": provenance.stamp()}
    arrays = dict(idx_tr=tr, idx_te=te, y_tr=y_tr, y_te=y_te, fold_of=splits.fold_of(plan, len(tr)),
                  school_code=F.school_code, prov_code=F.prov_code, oof=oof, test=test,
                  test_foldavg=foldavg, trace_oof=t_oof, trace_test=t_test,
                  trace_test_foldavg=t_te_f, trace_best_iter=best_iter, trace_n_trees=n_trees,
                  trace_oof_rmse=t_rmse, meta=meta)
    print(f"  split {seed}: RMSE test " + " ".join(
        f"{c}={np.sqrt(np.mean((y_te - p) ** 2)):.3f}" for c, p in test.items())
        + f"; cutoffs {lo:g}/{hi:g}; rs_tuned = cfg {jb} ({time.time() - t0:.0f}s)", flush=True)
    return arrays


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--preds-dir", default=DEFAULT_OUT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--feature-set", default="F_dt", choices=features.FEATURE_SETS)
    ap.add_argument("--n-cfg", type=int, default=6)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--max-trees", type=int, default=300)
    ap.add_argument("--es-rounds", type=int, default=20)
    ap.add_argument("--on-mismatch", default="recompute", choices=["recompute", "raise"],
                    help="npz có sẵn khác dấu: tính lại (mặc định của fixture) hay báo lỗi")
    args = ap.parse_args(argv)
    if os.path.abspath(args.data) == os.path.abspath(os.path.join(ROOT, "data", "data_final.csv")):
        sys.exit("Không chạy trên dữ liệu thật ở máy này (AGENTS.md, ràng buộc 1).")
    provenance.print_versions()
    # Mọi thứ làm đổi số của một lần chia; không có danh sách seed và n_jobs.
    config = {"script": "make_fake_preds", "data_sha256": preds_io.file_sha256(args.data),
              "feature_set": args.feature_set, "n_cfg": args.n_cfg, "max_trees": args.max_trees,
              "es_rounds": args.es_rounds, "n_bag": N_BAG}
    fp = preds_io.fingerprint(config)
    F = features.load_frame(args.data)
    print(f"{args.data}: n={F.n}, {len(F.columns(args.feature_set))} cột {args.feature_set}", flush=True)
    paths = []
    for s in args.seeds:
        p = preds_io.split_path(args.preds_dir, s)
        if preds_io.load_or_none(p, fp, on_mismatch=args.on_mismatch) is not None:
            print(f"  split {s}: đã có, cùng dấu {fp[:12]} -> bỏ qua", flush=True)
            paths.append(p)
            continue
        d = run_split(F, args.feature_set, s, args.n_cfg, args.n_jobs, args.max_trees, args.es_rounds, fp)
        preds_io.save_split(p, **d)
        err = preds_io.validate_split(preds_io.load_split(p))
        assert not err, err
        paths.append(p)
    print("Đã ghi: " + ", ".join(paths))
    return paths


if __name__ == "__main__":
    main()
