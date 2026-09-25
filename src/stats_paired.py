# -*- coding: utf-8 -*-
"""Thống kê cho phép so cặp theo lần chia và thước đo chi phí theo RMSE.

Vì sao Nadeau-Bengio mà không phải t thường hay Wilcoxon: 10 lần chia 80/20 trên
cùng 57.174 bản ghi có tập huấn luyện chồng nhau khoảng 80% và tập kiểm tra chồng
nhau, nên chênh cặp giữa các lần chia tương quan dương. t thường coi chúng độc lập
và cho SE quá nhỏ; Wilcoxon cũng giả định độc lập nên bị bỏ (khung bài, mục 6).
Nadeau và Bengio (2003) sửa phương sai thành (1/J + n_te/n_tr)·s², df = J - 1.

Bootstrap cụm theo trường bổ sung một góc khác: bất định do chọn thí sinh trong
MỘT tập kiểm tra khi các em cùng trường tương quan với nhau. Lấy lại trường có
hoàn lại, không lấy lại từng em.

Mọi thước đo theo RMSE (quy ước người dùng 2026-09-24):
    cost_K = sqrt( Σ w_i e_i² / Σ w_i ),  w_i = K_L nếu y_i < lo, K_H nếu y_i >= hi, 1 nếu không.
K_L = K_H = 1 là RMSE. Ngưỡng [lo, hi) theo quy ước đóng trái mở phải của src/bins.py.
Quy ước dấu cho mọi hiệu: âm nghĩa là vế trái tốt hơn; `wins` đếm số lần chia âm.

Holm: cổng trên ba phép so chính dùng primary_holm (họ cố định m = 3, phép so chưa
có coi là p = 1). holm(pvals) không có m chỉ dành cho bảng thứ cấp, nơi họ co theo
số ô có p.
"""
import numpy as np
from scipy import stats

from gates import ALPHA, CLUSTER_BOOT_B, NB_RATIO, PRIMARY_CONTRASTS, SESOI


# ---------------------------------------------------------------------------
# Thước đo
# ---------------------------------------------------------------------------
def step_weights(y, K_low, K_high, lo, hi):
    y = np.asarray(y, dtype=float)
    return np.where(y < lo, float(K_low), np.where(y >= hi, float(K_high), 1.0))


def cost_w(y, pred, w):
    """RMSE có trọng số với trọng số cho sẵn (mọi họ: bậc thang, prior, φ)."""
    y, pred, w = (np.asarray(a, dtype=float) for a in (y, pred, w))
    return float(np.sqrt(np.sum(w * (y - pred) ** 2) / np.sum(w)))


def cost_k(y, pred, K_low, K_high, lo, hi):
    """cost_K họ bậc thang. Mẫu số Σw không phụ thuộc dự đoán nên cực tiểu của
    cost_K chính là trung bình có trọng số (Bổ đề 1)."""
    return cost_w(y, pred, step_weights(y, K_low, K_high, lo, hi))


def region_rmse(y, pred, lo, hi):
    """RMSE theo vùng, cùng tên khoá với tail_prior.point_metrics để bảng cũ đọc được.
    Vùng rỗng cho NaN (không lỗi) vì lần chia nhỏ trong --smoke có thể thiếu đuôi."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    e2 = (y - pred) ** 2
    masks = {"Low tail": y < lo, "Middle": (y >= lo) & (y < hi), "High tail": y >= hi,
             "Tails": (y < lo) | (y >= hi), "All": np.ones(len(y), dtype=bool)}
    return {k: float(np.sqrt(e2[m].mean())) if m.any() else float("nan")
            for k, m in masks.items()}


def cost_from_regions(A, M, T, K):
    """cost_K họ bậc thang đối xứng suy từ RMSE toàn bộ A, vùng giữa M, đuôi gộp T.

    Tỉ lệ đuôi của tập kiểm tra suy ra đúng từ ba số: p_T = (A² - M²)/(T² - M²),
    rồi cost_K = sqrt(((1 - p_T)M² + K·p_T·T²) / ((1 - p_T) + K·p_T)). Dùng để tính
    lại cost_K từ JSON cũ chỉ lưu RMSE theo vùng (khung bài, mục 0.3). Nhận số
    hoặc mảng."""
    A, M, T = (np.asarray(v, dtype=float) for v in (A, M, T))
    pT = (A ** 2 - M ** 2) / (T ** 2 - M ** 2)
    out = np.sqrt(((1 - pT) * M ** 2 + K * pT * T ** 2) / ((1 - pT) + K * pT))
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Kiểm định theo lần chia
# ---------------------------------------------------------------------------
def _clean(diffs):
    d = np.asarray(diffs, dtype=float)
    return d[np.isfinite(d)]


def nb_ttest(diffs, ratio=NB_RATIO, alpha=ALPHA):
    """t hiệu chỉnh Nadeau-Bengio cho trung bình chênh cặp theo lần chia.

    var = (1/J + ratio)·s², s² là phương sai mẫu (ddof = 1) của J chênh; df = J - 1.
    Trả dict(mean, sd, se, t, df, p, ci_lo, ci_hi, n, wins); p hai phía; CI mức
    1 - alpha. Chênh NaN (ví dụ điểm đường đánh đổi nằm ngoài miền) bị bỏ, n là số
    chênh thật sự dùng."""
    d = _clean(diffs)
    J = len(d)
    out = {"n": int(J), "wins": int((d < 0).sum()), "ratio": float(ratio)}
    if J < 2:
        m = float(d.mean()) if J else float("nan")
        return out | {"mean": m, "sd": float("nan"), "se": float("nan"), "t": float("nan"),
                      "df": max(J - 1, 0), "p": float("nan"),
                      "ci_lo": float("nan"), "ci_hi": float("nan")}
    m, s2 = float(d.mean()), float(d.var(ddof=1))
    se = float(np.sqrt((1.0 / J + ratio) * s2))
    df = J - 1
    if se == 0.0:
        # Mọi chênh bằng nhau: không có bất định ước lượng được.
        t = 0.0 if m == 0 else float(np.sign(m) * np.inf)
        p = 1.0 if m == 0 else 0.0
    else:
        t = m / se
        p = float(2 * stats.t.sf(abs(t), df))
    q = float(stats.t.ppf(1 - alpha / 2, df))
    return out | {"mean": m, "sd": float(np.sqrt(s2)), "se": se, "t": float(t), "df": int(df),
                  "p": p, "ci_lo": m - q * se, "ci_hi": m + q * se}


def tost_nb(diffs, margin=SESOI, ratio=NB_RATIO, alpha=ALPHA):
    """TOST trên cùng thống kê Nadeau-Bengio, biên ±margin.

    H0₁: mean <= -margin, H0₂: mean >= margin; p = max(p₁, p₂). Qua khi p < alpha,
    tương đương với CI 1 - 2·alpha (90%) nằm trọn trong (-margin, margin)."""
    r = nb_ttest(diffs, ratio=ratio, alpha=2 * alpha)      # CI 90% cho TOST mức 5%
    m, se, df = r["mean"], r["se"], r["df"]
    if not np.isfinite(se) or df < 1:
        return {"p": float("nan"), "passed": False, "ci90": [float("nan")] * 2,
                "margin": float(margin), "mean": m, "se": se, "n": r["n"]}
    if se == 0.0:
        p1 = 0.0 if m > -margin else 1.0
        p2 = 0.0 if m < margin else 1.0
    else:
        p1 = float(stats.t.sf((m + margin) / se, df))      # bác mean <= -margin
        p2 = float(stats.t.cdf((m - margin) / se, df))     # bác mean >= margin
    p = max(p1, p2)
    return {"p": p, "p_lower": p1, "p_upper": p2, "passed": bool(p < alpha),
            "ci90": [r["ci_lo"], r["ci_hi"]], "margin": float(margin),
            "mean": m, "se": se, "n": r["n"]}


def holm(pvals, m=None):
    """p đã hiệu chỉnh Holm-Bonferroni, cùng thứ tự đầu vào.

    m = None (bảng thứ cấp): NaN giữ nguyên NaN và KHÔNG tính vào số phép so, nên
    họ phép so co lại theo số ô có p.
    m cho trước (họ chốt trước, ví dụ {C1; C2; C3}): số phép so cố định bằng m, và
    các phép so vắng mặt hoặc NaN được coi như p = 1 (đứng cuối thứ tự, không làm
    giảm hệ số của phép so nào). Như vậy một cổng tính giữa chừng, khi mới có một
    hay hai phép so, không qua dễ hơn cổng tính trên cả họ. p NaN vẫn trả NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    fam = len(ok) if m is None else int(m)
    if fam < len(ok):
        raise ValueError(f"m = {fam} nhỏ hơn số p hợp lệ ({len(ok)})")
    if len(ok) == 0:
        return out.tolist()
    order = ok[np.argsort(p[ok], kind="mergesort")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (fam - rank) * p[i]))
        out[i] = running
    return out.tolist()


def primary_holm(p_by_contrast, family=PRIMARY_CONTRASTS):
    """Holm trên họ phép so chính chốt trước (gates.PRIMARY_CONTRASTS = C1, C2, C3).

    Ba phép so đến từ ba thí nghiệm (E2b, E2, E1), nên cổng G1/G2 có thể được tính
    khi chưa đủ cả ba. Họ luôn có m = len(family): phép so thiếu hoặc p NaN được coi
    là p = 1 (bảo thủ) và gắn cờ missing. Trả {tên: {"p", "p_holm", "missing", "m"}}."""
    unknown = sorted(set(p_by_contrast) - set(family))
    if unknown:
        raise ValueError(f"phép so ngoài họ chính: {unknown}; họ là {list(family)}")
    raw = [p_by_contrast.get(c, float("nan")) for c in family]
    raw = [float("nan") if v is None else float(v) for v in raw]
    adj = holm(raw, m=len(family))
    return {c: {"p": r, "p_holm": (1.0 if not np.isfinite(r) else a),
                "missing": bool(not np.isfinite(r)), "m": len(family)}
            for c, r, a in zip(family, raw, adj)}


def paired(diffs, ratio=NB_RATIO, margin=SESOI, alpha=ALPHA):
    """Gói nb_ttest và TOST cho một ô bảng: {"nb": ..., "tost": ...}."""
    return {"nb": nb_ttest(diffs, ratio, alpha), "tost": tost_nb(diffs, margin, ratio, alpha)}


# ---------------------------------------------------------------------------
# Bootstrap cụm theo trường trong một tập kiểm tra
# ---------------------------------------------------------------------------
def cluster_boot_diff(y, pred1, pred2, clusters, K_low, K_high, lo, hi,
                      B=CLUSTER_BOOT_B, seed=0, alpha=ALPHA, weights=None):
    """CI phân vị của cost_K(pred1) - cost_K(pred2), lấy lại TRƯỜNG có hoàn lại.

    Mỗi vòng bốc G trường trong G trường (có hoàn lại), gộp mọi em của trường được
    bốc (một trường bốc hai lần thì đếm hai lần). Tính trên tổng theo cụm nên một
    vòng chỉ tốn một phép nhân ma trận: số lần bốc (B×G) nhân tổng (G×3).
    `weights` (tuỳ chọn) thay trọng số bậc thang bằng trọng số cho sẵn của họ khác.
    """
    y = np.asarray(y, dtype=float)
    e1 = (y - np.asarray(pred1, dtype=float)) ** 2
    e2 = (y - np.asarray(pred2, dtype=float)) ** 2
    w = step_weights(y, K_low, K_high, lo, hi) if weights is None else np.asarray(weights, float)
    _, cl = np.unique(np.asarray(clusters), return_inverse=True)
    G = int(cl.max()) + 1
    S = np.column_stack([np.bincount(cl, weights=w, minlength=G),
                         np.bincount(cl, weights=w * e1, minlength=G),
                         np.bincount(cl, weights=w * e2, minlength=G)])
    est = float(np.sqrt(S[:, 1].sum() / S[:, 0].sum()) - np.sqrt(S[:, 2].sum() / S[:, 0].sum()))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=int(B))
    tot = counts @ S
    boot = np.sqrt(tot[:, 1] / tot[:, 0]) - np.sqrt(tot[:, 2] / tot[:, 0])
    lo_ci, hi_ci = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"est": est, "ci_lo": float(lo_ci), "ci_hi": float(hi_ci),
            "se": float(boot.std(ddof=1)), "excludes_zero": bool(lo_ci > 0 or hi_ci < 0),
            "B": int(B), "n_clusters": G}
