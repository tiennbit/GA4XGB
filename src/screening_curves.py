# -*- coding: utf-8 -*-
"""Tầng 1 — mục 1.2 + 1.9: đánh giá sàng lọc trên ĐƯỜNG CONG, không tại một ngưỡng.

VÌ SAO PHẢI CÓ: Section IV-F so recall của hai mô hình tại CÙNG ngưỡng cứng 60,
trong khi chúng có độ chệch lệch nhau ~4,5 điểm. Đó là so hai classifier tại một
điểm tuỳ tiện. Con số "recall gấp 3,3 lần" có thể chỉ là hệ quả của việc mô hình
loss-weighted gắn cờ nhiều hơn 4,5 lần, chứ không phải phân biệt tốt hơn.

Ở đây mọi so sánh đều ở dạng bất biến với ngưỡng (AUC-PR, AUC-ROC) hoặc tại
NGÂN SÁCH GẮN CỜ BẰNG NHAU (cùng số học sinh được gọi lên tư vấn) — đó mới là
ràng buộc thật của một phòng tư vấn.

Kèm mục 1.9: độ chệch theo tỉnh báo cáo qua 5 seed thay vì 1, và mở rộng audit
sang giới tính / trường chuyên / khu vực tuyển sinh (các biến đã có sẵn trong
feature set nhưng chưa bao giờ được kiểm).

Chạy: PYTHONPATH=src python3 src/screening_curves.py [--n-jobs 2]
Kết quả: results_mseed/screening_curves.json
"""
import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import bins
import review_common as rc
from recalibration import oof_predictions, macro_mae, SCALE_GRID

FLAG_RATES = [0.05, 0.10, 0.15]
MIN_GROUP_N = 100
# (nhãn, tiền tố cột one-hot) hoặc (nhãn, tên cột nhị phân)
GROUPS_ONEHOT = {"province": "Tỉnh_", "region": "khuVuc_"}
GROUPS_BINARY = {"gender": "gioiTinh", "specialized_school": "truong_chuyen"}

METHODS = ["default", "ga_rmse", "ga_tail_a1", "ga_tail_a1_lw", "ga_rmse_recal"]


def at_budget(score, event, n_flag):
    """Gắn cờ đúng n_flag học sinh có score cao nhất. score: càng cao càng rủi ro."""
    if n_flag <= 0:
        return {"n_flag": 0, "recall": 0.0, "precision": float("nan")}
    thr = np.sort(score)[-n_flag]
    flag = score >= thr
    hit = int((flag & event).sum())
    return {"n_flag": int(flag.sum()), "recall": hit / int(event.sum()),
            "precision": hit / int(flag.sum())}


def tail_report(pred, y_true):
    """Báo cáo cho cả hai đuôi: đường cong + các điểm vận hành khớp ngân sách."""
    out = {}
    n = len(y_true)
    for tag, event, score, thr_abs in (
            ("low",  y_true < bins.LOW_TAIL_MAX,   -pred, -bins.LOW_TAIL_MAX),
            ("high", y_true >= bins.HIGH_TAIL_MIN,  pred,  bins.HIGH_TAIL_MIN)):
        d = {"n_event": int(event.sum()),
             "auc_pr": float(average_precision_score(event, score)),
             "auc_roc": float(roc_auc_score(event, score)),
             "prevalence": float(event.mean())}
        # điểm vận hành của bài: ngưỡng tuyệt đối trên điểm dự đoán
        flag = score >= thr_abs
        hit = int((flag & event).sum())
        d["at_paper_threshold"] = {
            "n_flag": int(flag.sum()),
            "recall": hit / int(event.sum()) if event.sum() else float("nan"),
            "precision": hit / int(flag.sum()) if flag.sum() else float("nan")}
        # điểm vận hành khớp ngân sách
        d["at_flag_rate"] = {f"{r:.0%}": at_budget(score, event, int(round(r * n)))
                             for r in FLAG_RATES}
        out[tag] = d
    return out


def group_bias(X_te, y_true, pred, cols_mask):
    rows = []
    for name, m in cols_mask:
        if m.sum() < MIN_GROUP_N:
            continue
        rows.append({"group": name, "n": int(m.sum()),
                     "bias": float((pred[m] - y_true[m]).mean()),
                     "mae": float(np.abs(pred[m] - y_true[m]).mean()),
                     "flag_rate_low": float((pred[m] < bins.LOW_TAIL_MAX).mean())})
    if not rows:
        return None
    b = np.array([r["bias"] for r in rows])
    fr = np.array([r["flag_rate_low"] for r in rows])
    return {"n_group": len(rows), "bias_min": float(b.min()), "bias_max": float(b.max()),
            "bias_sd": float(b.std(ddof=1)),
            "flag_rate_min": float(fr.min()), "flag_rate_max": float(fr.max()),
            "rows": sorted(rows, key=lambda r: r["bias"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=2)
    args = ap.parse_args()

    X_tr, X_te, y_tr, y_te = rc.get_data()
    ytr_v, yte_v = y_tr.to_numpy(float), y_te.to_numpy(float)
    mu = float(ytr_v.mean())

    masks = []
    for label, prefix in GROUPS_ONEHOT.items():
        cols = [c for c in X_te.columns if c.startswith(prefix)]
        masks.append((label, [(c[len(prefix):], X_te[c].to_numpy().astype(bool))
                              for c in cols]))
    for label, col in GROUPS_BINARY.items():
        v = X_te[col].to_numpy().astype(bool)
        masks.append((label, [("1", v), ("0", ~v)]))

    per_seed = {m: [] for m in METHODS}
    fairness = {m: {lab: [] for lab, _ in masks} for m in METHODS}

    for seed in rc.SEEDS:
        preds = {}
        preds["default"] = rc.fit_predict({}, None, X_tr, y_tr, X_te, args.n_jobs)
        for name in ("ga_rmse", "ga_tail_a1", "ga_tail_a1_lw"):
            xp, beta = rc.load_params(name, seed)
            preds[name] = rc.fit_predict(xp, beta, X_tr, y_tr, X_te, args.n_jobs)

        # hiệu chỉnh tuyến tính trên OOF của train — cùng quy trình với recalibration.py
        xp, beta = rc.load_params("ga_rmse", seed)
        oof = oof_predictions(xp, beta, X_tr, y_tr, args.n_jobs)
        s_star = float(SCALE_GRID[int(np.argmin(
            [macro_mae(ytr_v, mu + s * (oof - mu)) for s in SCALE_GRID]))])
        preds["ga_rmse_recal"] = mu + s_star * (preds["ga_rmse"] - mu)

        for m, p in preds.items():
            per_seed[m].append(tail_report(p, yte_v))
            for lab, cm in masks:
                fairness[m][lab].append(group_bias(X_te, yte_v, p, cm))
        print(f"[screen] seed {seed} xong (s*={s_star:.2f})", flush=True)

    # --- tổng hợp qua seed ---
    def agg(method):
        rows = per_seed[method]
        out = {}
        for tag in ("low", "high"):
            d = {"n_event": rows[0][tag]["n_event"],
                 "auc_pr": rc.ms([r[tag]["auc_pr"] for r in rows]),
                 "auc_roc": rc.ms([r[tag]["auc_roc"] for r in rows]),
                 "at_paper_threshold": {
                     k: rc.ms([r[tag]["at_paper_threshold"][k] for r in rows])
                     for k in ("n_flag", "recall", "precision")}}
            d["at_flag_rate"] = {
                rate: {k: rc.ms([r[tag]["at_flag_rate"][rate][k] for r in rows])
                       for k in ("recall", "precision")}
                for rate in rows[0][tag]["at_flag_rate"]}
            out[tag] = d
        return out

    summary = {m: agg(m) for m in METHODS}
    fair_sum = {m: {lab: {k: rc.ms([d[k] for d in fairness[m][lab] if d])
                          for k in ("bias_min", "bias_max", "bias_sd",
                                    "flag_rate_min", "flag_rate_max")}
                    for lab, _ in masks} for m in METHODS}

    with open(f"{rc.MSEED_DIR}/screening_curves.json", "w") as f:
        json.dump({"protocol": ("So sánh bất biến với ngưỡng (AUC-PR/ROC) và tại "
                                "ngân sách gắn cờ bằng nhau; 5 seed, mean ± sd."),
                   "summary": summary, "fairness": fair_sum,
                   "fairness_detail_seed42": {m: {lab: fairness[m][lab][0]
                                                  for lab, _ in masks} for m in METHODS}},
                  f, indent=2, ensure_ascii=False)

    for tag in ("low", "high"):
        print(f"\n=== đuôi {tag} (n_event={summary['ga_rmse'][tag]['n_event']}) ===")
        print(f"{'method':<16}{'AUC-PR':>16}{'recall@10%':>16}{'prec@10%':>16}"
              f"{'recall@thr':>16}")
        for m in METHODS:
            d = summary[m][tag]
            print(f"{m:<16}{d['auc_pr']['mean']:>10.4f} ±{d['auc_pr']['sd']:.4f}"
                  f"{d['at_flag_rate']['10%']['recall']['mean']:>10.3f} "
                  f"±{d['at_flag_rate']['10%']['recall']['sd']:.3f}"
                  f"{d['at_flag_rate']['10%']['precision']['mean']:>10.3f} "
                  f"±{d['at_flag_rate']['10%']['precision']['sd']:.3f}"
                  f"{d['at_paper_threshold']['recall']['mean']:>10.3f} "
                  f"±{d['at_paper_threshold']['recall']['sd']:.3f}")


if __name__ == "__main__":
    main()
