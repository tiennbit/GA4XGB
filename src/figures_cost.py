# -*- coding: utf-8 -*-
"""Hình của bài cost-aware (khung bài 24/9, mục 7), vẽ từ results_cost/*.json.

Bản thảo v2 giữ 8 hình ở thân bài (fig01..fig08) và đưa 4 hình sang tài liệu bổ
sung (figS1..figS4, đánh số theo lần trích đầu tiên trong thân bài):
  S1 = kích thước bag và thời gian (trước là Hình 11), S2 = biên mô tả (Hình 9),
  S3 = biểu đồ Murphy (Hình 10), S4 = mô phỏng (Hình 12).
Bản v3 thêm S5 = lặp lại trên dữ liệu công khai (E11), đọc results_bench/paper_numbers_bench.json.

Chạy:  python3 src/figures_cost.py              (mọi hình)
       python3 src/figures_cost.py --fig 4 5 S1 (một số hình)
Xuất:  paper/figures_cost/figNN_<tên>.pdf|png, figSn_<tên>.pdf|png

Quy ước (giữ như src/figures.py để hai bộ hình nhìn như một):
- IEEE Access: cột đơn 3,5 in, cột đôi 7,16 in; PDF vector + PNG 600 dpi.
- Chữ nhỏ nhất 7 pt ở khổ in thật (nhãn trục 8 pt). Bản v1 có chú thích 5,4-6,2 pt,
  in ra không đọc được; đừng hạ xuống dưới FS_MIN để nhét chữ, hãy bớt chữ.
- Không có tiêu đề hay dòng ghi chú trong hình: IEEE đặt mọi giải thích (dải là gì,
  bao nhiêu lần chia) vào chú thích dưới hình. Trong hình chỉ còn nhãn ô (a), (b)...
- Nhãn trục có đơn vị: cost_K và RMSE tính bằng điểm HSA ("points").
- Bài tiếng Anh nên số trong hình dùng DẤU CHẤM thập phân (khác bản thảo tiếng Việt).
- Okabe-Ito cho người mù màu, và MỖI chuỗi còn có kiểu nét + marker riêng để in đen
  trắng vẫn tách được. Một quy tắc / chính sách giữ cùng màu-nét-marker ở mọi hình.

Vì sao hình không tự tính lại thống kê chính: CI, Holm, cổng đã có trong JSON. Ngoại
lệ duy nhất là Hình 4(b): dải CI của Rx - R1 trên lưới K dày không được lưu, nên tính
bằng stats_paired.nb_ttest từ chi phí từng lần chia (cùng hàm script thí nghiệm dùng).

Hình 3 chỉ dùng bảng đếm theo điểm trong data_audit.json (tổng hợp, không có dòng
dữ liệu); không đọc data/data_final.csv. Hình 1, 2 là sơ đồ, không có số liệu.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_paired import nb_ttest          # noqa: E402

RES = "results_cost"
FIGDIR = "paper/figures_cost"
W1, W2 = 3.5, 7.16                          # cột đơn, cột đôi (in)
SESOI = 0.10

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
     "grey": "#7F7F7F", "lgrey": "#BFBFBF"}

# Một kiểu cho mỗi quy tắc, dùng ở mọi hình.
RS = {
    "R0": dict(color=C["grey"], ls=":", marker="x", label="R0 raw"),
    "R1": dict(color=C["blue"], ls="-", marker="o", label="R1 Bayes rule"),
    "R1_1": dict(color=C["purple"], ls="-.", marker="v", label="R1$_1$ (w ≡ 1)"),
    "R2": dict(color=C["orange"], ls="--", marker="s", label="R2 weighted isotonic"),
    "R3": dict(color=C["sky"], ls=(0, (1, 1)), marker="<", label="R3 weighted histogram"),
    "R4": dict(color=C["purple"], ls=(0, (5, 1, 1, 1)), marker=">", label="R4 weighted linear"),
    "R5": dict(color=C["red"], ls="-.", marker="^", label="R5 two-sided stretch"),
    "R7": dict(color=C["yellow"], ls=":", marker="P", label="R7 normal reframing"),
    "R8*": dict(color=C["green"], ls=(0, (6, 2)), marker="D", label="R8* weighted training"),
}
# Một kiểu cho mỗi trung tâm (center). Màu trung tính để không lẫn với màu của quy
# tắc: v1 vẽ rs_tuned màu cam trùng R2 và default xám chấm trùng R0.
CEN = {"default": dict(color=C["lgrey"], ls="-", lw=1.5, label="default"),
       "sub1": dict(color=C["sky"], ls=(0, (1, 1)), lw=1.1, label="sub1"),
       "bag20": dict(color=C["blue"], ls="-", lw=1.1, label="bag20"),
       "rs_tuned": dict(color=C["black"], ls="--", lw=1.0, label="rs_tuned"),
       "rs_tuned_bag5": dict(color=C["black"], ls="-.", lw=1.0, label="rs_tuned_bag5")}
REG = {"Low tail": dict(color=C["blue"], marker="v", label="Low tail (y < 60)"),
       "Middle": dict(color=C["black"], marker="o", label="Middle"),
       "High tail": dict(color=C["red"], marker="^", label="High tail (y ≥ 100)")}
POL = {"a": dict(color=C["black"], ls="-", marker="o", label="(a) RMSE, then rule"),
       "b": dict(color=C["blue"], ls="--", marker="s", label="(b) cost$_K$ after rule"),
       "c": dict(color=C["red"], ls=":", marker="^", label="(c) raw cost$_K$, no rule"),
       "c+": dict(color=C["green"], ls="-.", marker="D", label="(c+) raw cost$_K$, then rule"),
       "o": dict(color=C["grey"], ls=(0, (1, 1)), marker="v", label="(o) oracle")}
PP = {"P0": dict(color=C["grey"], ls=":", marker="x", label="P0: ŷ vs c"),
      "P1": dict(color=C["blue"], ls="-", marker="o", label="P1: g$_K$(ŷ) vs c"),
      "P2": dict(color=C["red"], ls="--", marker="^", label="P2: P(event | x) ≥ 1/(1+K)")}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
    "lines.markersize": 3.5, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 2.2,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.formatter.use_locale": False,
})

FS_MIN = 7          # cỡ chữ nhỏ nhất cho mọi chú giải trong hình (pt)
PTS = "(points)"     # đơn vị của cost_K, RMSE, MAE: điểm trên thang HSA

_cache = {}


def J(name):
    if name not in _cache:
        with open(os.path.join(RES, f"{name}.json"), encoding="utf-8") as f:
            _cache[name] = json.load(f)
    return _cache[name]


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGDIR}/{name}.{ext}", metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    print(f"  -> {FIGDIR}/{name}.pdf|png")


def panel(ax, letter, x=-0.16, y=1.04, extra=""):
    ax.text(x, y, f"({letter}){extra}", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def logk_axis(ax, ks):
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k:g}" for k in ks])
    ax.minorticks_off()


def seeds():
    return [str(s) for s in J("decomp_rules")["summary"]["seeds"]]


# ---------------------------------------------------------------------------
# Hình 1: sơ đồ Thuật toán 1
# ---------------------------------------------------------------------------
def fig1():
    # Không có dòng tiêu đề trong hình (bản v1 có "One center fit serves every K"):
    # câu đó nằm ở chú thích Hình 1.
    fig, ax = plt.subplots(figsize=(W2, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(1.5, 29.5)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                           fc=fc, ec="#333333", lw=0.7)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=7.5, fontweight="bold")
        ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top", fontsize=FS_MIN, linespacing=1.2)

    def arrow(x0, y0, x1, y1, style="-|>", ls="-", color="#333333"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8,
                                     lw=0.8, ls=ls, color=color, shrinkA=0, shrinkB=0))

    y0, h, w = 11.0, 17.5, 17.6
    xs = [0.6, 20.9, 41.2, 61.5, 81.8]
    box(xs[0], y0, w, h, "1. Features",
        "pre-exam record:\ngrades 10, 11 and\n12 (term 1), school,\nprovince, area\n(F$_{dt\\text{-}cn}$, 160 cols)", "#F2F2F2")
    box(xs[1], y0, w, h, "2. Center $\\hat y$",
        "XGBoost, squared\nloss; bag of B* = 20\n(or one tuning\nby RMSE)", "#DCEAF6")
    box(xs[2], y0, w, h, "3. OOF pairs",
        "5-fold out-of-fold\n$(\\hat y_{oof}, y)$ on\ntraining set; refit\non all training rows", "#DCEAF6")
    box(xs[3], y0, w, h, "4. Decision layer",
        "for each K: fit $g_K$\non $(\\hat y_{oof}, y)$ with\nweights $w_K(y)$\n(R2 isotonic\nor R1 bins)", "#FBE3D6")
    box(xs[4], y0, w, h, "5. Outputs",
        "$g_K(\\hat y)$: decision\nscore under K;\n$P(Y \\geq c \\mid x)$\nfor a fixed\nthreshold c", "#E2F0E4")
    for i in range(4):
        arrow(xs[i] + w + 0.5, y0 + h / 2, xs[i + 1] - 0.5, y0 + h / 2)
    # nơi các cách làm thực tế tác động
    ax.text(xs[1] + w / 2, 4.2, "tail-weighted HPO fitness and\nweighted training (R8) act here",
            ha="center", va="center", fontsize=FS_MIN, style="italic", color="#444444")
    arrow(xs[1] + w / 2, 6.6, xs[1] + w / 2, y0 - 0.5, ls="--", color="#666666")
    ax.text(xs[3] + w / 2, 4.2, "output stretch (R5) and the other\nlayer estimators (R3, R4, R7) act here",
            ha="center", va="center", fontsize=FS_MIN, style="italic", color="#444444")
    arrow(xs[3] + w / 2, 6.6, xs[3] + w / 2, y0 - 0.5, ls="--", color="#666666")
    save(fig, "fig01_pipeline")


# ---------------------------------------------------------------------------
# Hình 2: hình học Mệnh đề 1 và 2
# ---------------------------------------------------------------------------
def fig2():
    """Sơ đồ: biên Bayes trong toạ độ tổng MSE (M, T) cho G = σ(X) và G = σ(ŷ).

    Biên dạng T = a/(M - m0) + t0 (lồi, giảm); tiếp tuyến độ dốc -1/K chạm tại
    M = m0 + sqrt(aK). Không phải số liệu: chỉ minh hoạ cấu trúc của Mệnh đề 1
    (khoảng hụt trung tâm giữa hai biên, khoảng hụt quyết định từ một quy tắc tới
    tiếp tuyến) và Mệnh đề 2 (họ μ_K^G là nhánh ưu tiên đuôi, K = 1 là điểm RMSE)."""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    pT, pM = 0.15, 0.85

    def frontier(a, m0, t0):
        M = np.linspace(m0 + 0.3, 9.5, 400)
        return M, a / (M - m0) + t0

    fx = dict(a=8.0, m0=-1.2, t0=0.4)     # G = σ(X)
    fg = dict(a=12.0, m0=-1.0, t0=1.0)    # G = σ(ŷ)
    for f, col, lab, ls in ((fx, C["grey"], r"Bayes frontier, $G=\sigma(X)$", "--"),
                            (fg, C["blue"], r"Bayes frontier, $G=\sigma(\hat y)$", "-")):
        M, T = frontier(**f)
        ok = T < 8
        ax.plot(M[ok], T[ok], color=col, ls=ls, lw=1.2, label=lab)

    def tangent_pt(f, K):
        Mk = f["m0"] + np.sqrt(f["a"] * K)
        return Mk, f["a"] / (Mk - f["m0"]) + f["t0"]

    for K, col in ((1, C["black"]), (3, C["red"])):
        Mk, Tk = tangent_pt(fg, K)
        xx = np.linspace(Mk - 2.2, Mk + (1.4 if K == 1 else 2.6), 2)
        ax.plot(xx, Tk - (xx - Mk) / K, color=col, lw=0.7, ls=":")
        ax.plot(Mk, Tk, "o", color=col, ms=4.5, zorder=5)
        ax.annotate(fr"$\mu^{{G}}_{{K={K}}}$" + ("\n(RMSE point)" if K == 1 else ""),
                    (Mk, Tk), xytext=(6, 6 if K == 3 else 4), textcoords="offset points", fontsize=FS_MIN, color=col)
        xe = xx[0] if K == 1 else xx[1]
        ax.text(xe, Tk - (xe - Mk) / K + (0.12 if K == 1 else -0.12), f"slope $-1/{K}$" if K > 1 else "slope $-1$",
                fontsize=FS_MIN, color=col, ha="left" if K == 1 else "right", va="bottom" if K == 1 else "top")
    # một quy tắc h(ŷ) nằm trên biên: khoảng hụt quyết định
    Mk3, Tk3 = tangent_pt(fg, 3)
    hM, hT = Mk3 + 1.5, Tk3 + 1.0
    ax.plot(hM, hT, "^", color=C["red"], mfc="white", ms=5, zorder=5)
    ax.annotate("rule $h(\\hat y)$, e.g. stretch", (hM, hT), xytext=(5, 3), textcoords="offset points",
                fontsize=FS_MIN, color=C["red"])
    line_T = Tk3 - (hM - Mk3) / 3
    ax.annotate("", xy=(hM, line_T), xytext=(hM, hT),
                arrowprops=dict(arrowstyle="<->", color=C["red"], lw=0.7, shrinkA=1, shrinkB=1))
    ax.text(hM + 0.15, (hT + line_T) / 2, "decision gap\n(level-K excess)", fontsize=FS_MIN, color=C["red"], va="center")
    # khoảng hụt trung tâm giữa hai biên ở K = 3
    MkX, TkX = tangent_pt(fx, 3)
    ax.plot(MkX, TkX, "o", color=C["grey"], ms=4, mfc="white", zorder=5)
    ax.annotate("", xy=(MkX, TkX), xytext=(Mk3, Tk3),
                arrowprops=dict(arrowstyle="->", color=C["grey"], lw=0.7, shrinkA=3, shrinkB=3))
    ax.text(MkX, TkX - 0.3, "center gap", fontsize=FS_MIN, color="#555555", ha="center", va="top")
    ax.set_xlabel(r"Middle-region MSE mass  $M = E[(Y-f)^2\,1\{Y \in \mathcal{M}\}]$")
    ax.set_ylabel(r"Tail MSE mass  $T$")
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0.8, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=FS_MIN)
    panel(ax, "a", x=-0.06)

    ax = axes[1]
    for f, col, ls in ((fx, C["grey"], "--"), (fg, C["blue"], "-")):
        M, T = frontier(**f)
        ok = (T < 8) & (M > 0.2)
        ax.plot(np.sqrt(M[ok] / pM), np.sqrt(T[ok] / pT), color=col, ls=ls, lw=1.2)
    for K, col in ((1, C["black"]), (3, C["red"])):
        Mk, Tk = tangent_pt(fg, K)
        ax.plot(np.sqrt(Mk / pM), np.sqrt(Tk / pT), "o", color=col, ms=4.5)
        ax.annotate(f"K = {K}", (np.sqrt(Mk / pM), np.sqrt(Tk / pT)), xytext=(5, 2),
                    textcoords="offset points", fontsize=FS_MIN, color=col)
    ax.plot(np.sqrt(hM / pM), np.sqrt(hT / pT), "^", color=C["red"], mfc="white", ms=5)
    ax.set_xlabel("Middle RMSE  $r_M$")
    ax.set_ylabel("Tail RMSE  $r_T$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.97, 0.95, r"$dr_T/dr_M = -\dfrac{p_M\, r_M}{K\, p_T\, r_T}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=7)
    panel(ax, "b", x=-0.08)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig02_geometry")


# ---------------------------------------------------------------------------
# Hình 3: phân bố điểm HSA (bảng đếm của data_audit)
# ---------------------------------------------------------------------------
def fig3():
    da = J("data_audit")
    h = {int(float(k)): v for k, v in da["target"]["hist_by_score"].items()}
    n = sum(h.values())
    xs = np.arange(min(h), max(h) + 1)
    ys = np.array([h.get(int(x), 0) for x in xs]) / n * 100
    lo, hi, ng = 60, 100, da["near_guess"]["threshold"]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    col = np.where(xs < lo, C["blue"], np.where(xs >= hi, C["red"], C["lgrey"]))
    ax.bar(xs, ys, width=1.0, color=col, edgecolor="none")
    near = xs <= ng
    # cột y ≤ 37 quá thấp để thấy gạch chéo: tô cả dải điểm thay vì từng cột
    ax.axvspan(xs[near].min() - 0.5, ng + 0.5, color="none", ec="#AAAAAA", hatch="///", lw=0.0, alpha=0.5)
    r = da["target"]["regions_60_100"]
    ax.axvline(lo, color=C["black"], lw=0.6, ls="--")
    ax.axvline(hi, color=C["black"], lw=0.6, ls="--")
    ymax = ys.max()
    ax.text(lo - 1, ymax * 0.98, f"low tail\n{100 * r['mass_low']:.2f}%", ha="right", va="top", fontsize=FS_MIN, color=C["blue"])
    ax.text(hi + 1, ymax * 0.98, f"high tail\n{100 * r['mass_high']:.2f}%", ha="left", va="top", fontsize=FS_MIN, color=C["red"])
    ax.annotate(f"y ≤ {ng}: {da['near_guess']['n']} records\n(near guessing)", xy=(33, ymax * 0.55),
                xytext=(28.5, ymax * 0.78), fontsize=FS_MIN, va="center",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"),
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#444444"))
    ax.set_xlabel("HSA score (points; scale 0–150)")
    ax.set_ylabel("Records (%)")
    ax.set_xlim(26, 132)
    t = da["target"]["desc"]
    ax.text(0.99, 0.66, f"n = {n:,}\nmean {t['mean']:.1f}\nSD {t['sd']:.1f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=FS_MIN)
    save(fig, "fig03_score_distribution")


# ---------------------------------------------------------------------------
# Hình 4: cost_K theo K
# ---------------------------------------------------------------------------
def _cost_series(center, rule, Ks):
    dr = J("decomp_rules")
    return np.array([[dr["per_split"][s][center][rule]["step"][K]["cost_K"] for K in Ks] for s in seeds()])


def _r8_series(Ks):
    wt = J("wtrain_tuned")
    return np.array([[wt["per_split"][s][K]["cost_K"]["R8_bag5"] for K in Ks] for s in seeds()])


def fig4():
    dr = J("decomp_rules")
    center = dr["summary"]["roles"]["primary"]
    Kd = [k for k in dr["meta"]["grids"]["step"] if k <= 20]
    Ks = [f"{k:g}" for k in Kd]
    Kr8 = ["2", "3", "5", "8"]
    xr8 = np.array([float(k) for k in Kr8])
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axes[0]
    for rule in ("R0", "R5", "R2", "R1"):
        Y = _cost_series(center, rule, Ks)
        m = Y.mean(0)
        st = RS[rule]
        ax.plot(Kd, m, color=st["color"], ls=st["ls"], marker=st["marker"], ms=2.6, label=st["label"])
        if rule in ("R0", "R1"):
            ax.fill_between(Kd, np.quantile(Y, 0.025, axis=0), np.quantile(Y, 0.975, axis=0),
                            color=st["color"], alpha=0.15, lw=0)
    Y8 = _r8_series(Kr8)
    st = RS["R8*"]
    ax.errorbar(xr8, Y8.mean(0), yerr=[Y8.mean(0) - np.quantile(Y8, 0.025, 0), np.quantile(Y8, 0.975, 0) - Y8.mean(0)],
                color=st["color"], marker=st["marker"], ls="none", ms=3.5, capsize=2, lw=0.8, label=st["label"])
    Yd = _cost_series("default", "R1", Ks)
    ax.plot(Kd, Yd.mean(0), color=CEN["default"]["color"], lw=CEN["default"]["lw"], ls="-", label="R1 on default center")
    logk_axis(ax, [1, 2, 3, 5, 8, 12, 20])
    ax.set_xlabel("Cost ratio K (log scale)")
    ax.set_ylabel(f"Test cost$_K$ {PTS}")
    ax.set_ylim(9.5, 15.7)
    ax.legend(loc="upper left", fontsize=FS_MIN, handlelength=2.4, ncol=2, columnspacing=1.0)
    panel(ax, "a", x=-0.13)

    ax = axes[1]
    R1 = _cost_series(center, "R1", Ks)
    for rule in ("R5", "R4", "R2"):
        D = _cost_series(center, rule, Ks) - R1
        st = [nb_ttest(D[:, j]) for j in range(len(Ks))]
        m = np.array([s["mean"] for s in st])
        lo = np.array([s["ci_lo"] for s in st])
        hi = np.array([s["ci_hi"] for s in st])
        sty = RS[rule]
        ax.plot(Kd, m, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=2.6, label=sty["label"].split(" ", 1)[0] + " − R1")
        ax.fill_between(Kd, lo, hi, color=sty["color"], alpha=0.15, lw=0)
    D8 = Y8 - _cost_series(center, "R1", Kr8)
    st8 = [nb_ttest(D8[:, j]) for j in range(len(Kr8))]
    sty = RS["R8*"]
    m8 = np.array([s["mean"] for s in st8])
    ax.errorbar(xr8, m8, yerr=[m8 - np.array([s["ci_lo"] for s in st8]), np.array([s["ci_hi"] for s in st8]) - m8],
                color=sty["color"], marker=sty["marker"], ls=sty["ls"], ms=3.5, capsize=2, lw=0.8, label="R8* − R1")
    ax.axhline(0, color=C["black"], lw=0.6)
    ax.axhspan(-SESOI, SESOI, color=C["lgrey"], alpha=0.25, lw=0)
    ax.text(1.05, -SESOI - 0.01, "±SESOI", fontsize=FS_MIN, color="#555555", va="top")
    ax.axvline(3, color=C["black"], lw=0.5, ls=":")
    logk_axis(ax, [1, 2, 3, 5, 8, 12, 20])
    ax.set_xlabel("Cost ratio K (log scale)")
    ax.set_ylabel(f"cost$_K$(Rx) − cost$_K$(R1) {PTS}")
    ax.legend(loc="upper left", fontsize=FS_MIN, handlelength=2.6)
    panel(ax, "b", x=-0.13)
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig04_cost_vs_K")


# ---------------------------------------------------------------------------
# Hình 5: thanh phân rã
# ---------------------------------------------------------------------------
def fig5():
    dc = J("decomp_rules")["summary"]["decomposition"]
    comps = [
        ("default_to_bag|R1", "default to bag20 (with R1)", C["lgrey"], None),
        ("i", "(i) center gain after bagging", C["lgrey"], None),
        ("ii_a", "(ii-a) recalibration, K = 1", RS["R1_1"]["color"], None),
        ("ii_b", "(ii-b) tilt toward cost K", C["blue"], None),
        # màu của từng quy tắc như ở mọi hình khác (RS); gạch chéo để in đen trắng
        ("iii|R2", "(iii) R2 isotonic − R1", RS["R2"]["color"], "////"),
        ("iii|R3", "(iii) R3 histogram − R1", RS["R3"]["color"], "...."),
        ("iii|R4", "(iii) R4 linear − R1", RS["R4"]["color"], "\\\\\\\\"),
        ("iii|R5", "(iii) R5 stretch − R1", RS["R5"]["color"], None),
        ("iii|R7", "(iii) R7 normal − R1", RS["R7"]["color"], "xxxx"),
        ("iii|R8*", "(iii) R8* weighted tr. − R1", C["green"], None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 3.0), sharey=True)
    ypos = np.arange(len(comps))[::-1]
    for j, (ax, K) in enumerate(zip(axes, ["2", "3", "5", "8"])):
        d = dc[K]
        for y, (key, lab, fc, hatch) in zip(ypos, comps):
            v = d.get(key)
            if v is None:
                continue
            m = v["mean"]
            if key == "i":
                ax.plot([0], [y], marker="|", color=C["black"], ms=6)
                ax.text(0.03 * max(1, d["ii"]["mean"]), y, "0 (by design)", va="center", fontsize=FS_MIN, color="#555555",
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
                continue
            lo, hi = v["nb"]["ci_lo"], v["nb"]["ci_hi"]
            ax.barh(y, m, height=0.68, color=fc, edgecolor=C["black"], lw=0.5, hatch=hatch)
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], color=C["black"], lw=0.7, capsize=1.6)
        ax.axvline(0, color=C["black"], lw=0.6)
        ax.axvline(SESOI, color=C["black"], lw=0.5, ls="--")
        # nhãn ô (a)-(d) thay cho tiêu đề "K = 2" của v1
        panel(ax, "abcd"[j], x=0.0 if j else -0.02, y=1.01, extra=f"  K = {K}")
        ax.grid(axis="y", visible=False)
        ax.axhline(5.5, color="#999999", lw=0.4)
        ax.set_xlabel(f"cost$_K$ difference {PTS}")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([c[1] for c in comps], fontsize=FS_MIN)
    axes[0].tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig05_decomposition")


# ---------------------------------------------------------------------------
# Hình 6: RMSE theo vùng theo K; bất đối xứng
# ---------------------------------------------------------------------------
def fig6():
    dr = J("decomp_rules")
    center = dr["summary"]["roles"]["primary"]
    t = dr["summary"]["table"][center]
    Kd = [k for k in dr["meta"]["grids"]["step"] if k <= 20]
    Ks = [f"{k:g}" for k in Kd]
    wt = J("wtrain_tuned")["summary"]["by_K"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax = axes[0]
    for reg, st in REG.items():
        m1 = [t["R1"]["step"][k][reg]["mean"] for k in Ks]
        m5 = [t["R5"]["step"][k][reg]["mean"] for k in Ks]
        ax.plot(Kd, m1, color=st["color"], ls="-", marker=st["marker"], ms=2.6)
        ax.plot(Kd, m5, color=st["color"], ls="--", lw=0.9)
        r8 = [wt[k]["R8_bag5"]["region_rmse"][reg]["mean"] for k in ("2", "3", "5", "8")]
        ax.plot([2, 3, 5, 8], r8, color=st["color"], ls="none", marker="D", mfc="white", ms=3.6)
        ax.text(20.8, m1[-1], st["label"].split(" (")[0], fontsize=FS_MIN, color=st["color"], va="center")
    logk_axis(ax, [1, 2, 3, 5, 8, 12, 20])
    ax.set_xlim(0.9, 34)
    ax.set_xlabel("Cost ratio K (log scale)")
    ax.set_ylabel(f"Test RMSE by true region {PTS}")
    h = [Line2D([], [], color="#333333", ls="-", marker="o", ms=2.6, label="R1 Bayes rule"),
         Line2D([], [], color="#333333", ls="--", label="R5 two-sided stretch"),
         Line2D([], [], color="#333333", ls="none", marker="D", mfc="white", label="R8* weighted training")]
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(0.0, 0.40), fontsize=FS_MIN)
    panel(ax, "a", x=-0.12)

    ax = axes[1]
    cells = [("1", "step", "symmetric\nK = 1"), ("5,1", "step_pair", "(K$_L$, K$_H$)\n= (5, 1)"),
             ("1,5", "step_pair", "(K$_L$, K$_H$)\n= (1, 5)"), ("5", "step", "symmetric\nK = 5")]
    x = np.arange(len(cells))
    wbar = 0.26
    for i, (reg, st) in enumerate(REG.items()):
        vals = [t["R1"][fam][p][reg]["mean"] for p, fam, _ in cells]
        ax.bar(x + (i - 1) * wbar, vals, width=wbar, color=st["color"], alpha=0.85 if reg != "Middle" else 0.55,
               edgecolor=C["black"], lw=0.4, hatch=["....", None, "///"][i],
               label=st["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in cells], fontsize=FS_MIN)
    ax.set_ylim(0, 17.5)
    ax.set_ylabel(f"Test RMSE with R1 {PTS}")
    ax.legend(loc="upper center", ncol=3, fontsize=FS_MIN, bbox_to_anchor=(0.5, 1.16), handlelength=1.2,
              columnspacing=0.6, handletextpad=0.4)
    ax.grid(axis="x", visible=False)
    panel(ax, "b", x=-0.18, y=1.12)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig06_region_rmse")


# ---------------------------------------------------------------------------
# Hình 7: chính sách HPO
# ---------------------------------------------------------------------------
def fig7():
    sp = J("select_policy")
    pol = sp["summary"]["policies"]
    con = sp["summary"]["contrasts"]
    Ks = list(pol.keys())
    Kx = np.array([float(k) for k in Ks])
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    ax = axes[0]
    for p in ("c", "c+", "b", "o", "a"):
        st = POL[p]
        ax.plot(Kx, [pol[k][p]["mean"] for k in Ks], color=st["color"], ls=st["ls"], marker=st["marker"], label=st["label"])
    logk_axis(ax, [1, 2, 3, 5, 8, 12, 20])
    ax.set_xlabel("Cost ratio K (log scale)")
    ax.set_ylabel(f"Test cost$_K$, selected config. {PTS}")
    ax.set_ylim(9.4, 15.6)
    ax.legend(loc="upper left", fontsize=FS_MIN)
    panel(ax, "a", x=-0.2)

    ax = axes[1]
    off = {"b-a": -0.03, "c+-a": 0.0, "o-a": 0.03}
    for name, p in (("b-a", "b"), ("c+-a", "c+"), ("o-a", "o")):
        st = POL[p]
        m = np.array([con[k][name]["nb"]["mean"] for k in Ks])
        lo = np.array([con[k][name]["nb"]["ci_lo"] for k in Ks])
        hi = np.array([con[k][name]["nb"]["ci_hi"] for k in Ks])
        xx = Kx * np.exp(off[name])
        ax.errorbar(xx, m, yerr=[m - lo, hi - m], color=st["color"], ls=st["ls"], marker=st["marker"],
                    capsize=1.5, lw=0.8, label=f"({p}) − (a)")
    ax.axhline(0, color=C["black"], lw=0.6)
    ax.axhspan(-SESOI, SESOI, color=C["lgrey"], alpha=0.25, lw=0)
    logk_axis(ax, [1, 2, 3, 5, 8, 12, 20])
    ax.set_ylim(-0.16, 0.22)
    ax.set_xlabel("Cost ratio K (log scale)")
    ax.set_ylabel(f"Difference to policy (a) {PTS}")
    ax.legend(loc="lower left", fontsize=FS_MIN)
    cK3 = con["3"]["c-a"]["nb"]["mean"]
    ax.text(0.03, 0.97, f"(c) − (a) off scale: +{cK3:.2f} at K = 3", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS_MIN, color=C["red"])
    panel(ax, "b", x=-0.2)

    ax = axes[2]
    det = sp["per_split_detail"]
    xs, ys = [], []
    for s, v in det.items():
        A = np.array(v["A"])
        B = np.array(v["by_K"]["3"]["B"])
        xs.append(A - A.min())
        ys.append(B - B.min())
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    ax.scatter(xs, ys, s=3, color=C["blue"], alpha=0.45, lw=0)
    rc = sp["rank_corr"]["3"]
    ax.text(0.04, 0.96, f"60 configs × 10 splits\nSpearman {rc['spearman_all']['mean']:.3f}\nKendall {rc['kendall_all']['mean']:.3f}",
            transform=ax.transAxes, va="top", fontsize=FS_MIN)
    ax.set_xlabel(f"OOF RMSE − split best {PTS}")
    ax.set_ylabel(f"OOF cost$_3$ after rule − best {PTS}")
    panel(ax, "c", x=-0.24)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig07_hpo_policies")


# ---------------------------------------------------------------------------
# Hình 8: hậu quả tại ngưỡng
# ---------------------------------------------------------------------------
def fig8():
    uv = J("use_validity")
    th = uv["summary"]["mean"]["thresholds"]
    Ks = ["1", "2", "3", "5", "8"]
    Kx = np.array([float(k) for k in Ks])
    fig, axes = plt.subplots(2, 3, figsize=(W2, 3.6), sharex=True)
    names = {"lt60": "event y < 60", "ge100": "event y ≥ 100"}
    for r, (tk, tv) in enumerate(th.items()):
        for c, (met, lab) in enumerate((("L", "Loss L$_K$ per 1,000 cand."), ("FNR", "False-negative rate"), ("FPR", "False-positive rate"))):
            ax = axes[r, c]
            for p, st in PP.items():
                m = np.array([tv["policies"][p][k][met]["mean"] for k in Ks])
                sd = np.array([tv["policies"][p][k][met]["sd"] for k in Ks])
                ax.errorbar(Kx * {"P0": 0.97, "P1": 1.0, "P2": 1.03}[p], m, yerr=sd, color=st["color"], ls=st["ls"],
                            marker=st["marker"], capsize=1.4, lw=0.9, label=st["label"])
            logk_axis(ax, [1, 2, 3, 5, 8])
            ax.set_xlim(0.85, 9.4)
            ax.set_ylabel(lab)
            if r == 1:
                ax.set_xlabel("Cost ratio K (log scale)")
            if c == 0:
                ax.text(0.03, 0.95, names[tk], transform=ax.transAxes, va="top", fontsize=FS_MIN, fontweight="bold")
            if met in ("FNR", "FPR"):
                ax.set_ylim(0, 1.0 if met == "FNR" else None)
            panel(ax, "abcdef"[3 * r + c], x=-0.27)
    # chú giải đặt trên đầu hình: trong ô (c) nó đè lên đường P2 ở K = 8
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=FS_MIN, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.6, w_pad=1.0, rect=(0, 0, 1, 0.95))
    save(fig, "fig08_threshold_consequences")


# ---------------------------------------------------------------------------
# Hình S2 (bổ sung; Hình 9 ở v1): biên đuôi/giữa mô tả
# ---------------------------------------------------------------------------
def figS2():
    dr = J("decomp_rules")["summary"]
    wt = J("wtrain_tuned")["summary"]["by_K"]
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    for rule in ("R5", "R2", "R1"):
        pts = [p for p in dr["frontier"]["bag20"][f"{rule}|step"]["points"] if float(p["param"]) <= 20]
        st = RS[rule]
        ax.plot([p["Middle"] for p in pts], [p["tails"] for p in pts], color=st["color"], ls=st["ls"],
                marker=st["marker"], ms=2.4, label=st["label"].split(" (")[0])
        if rule == "R1":
            for p in pts:
                if p["param"] in ("1", "2", "3", "5", "8", "20"):
                    ax.annotate(f"K={p['param']}", (p["Middle"], p["tails"]),
                                xytext=(-3, -7) if p["param"] != "20" else (-2, 5),
                                textcoords="offset points", fontsize=FS_MIN, color=st["color"], ha="right")
    pts = [p for p in dr["frontier"]["default"]["R1|step"]["points"] if float(p["param"]) <= 20]
    ax.plot([p["Middle"] for p in pts], [p["tails"] for p in pts], color=CEN["default"]["color"], lw=CEN["default"]["lw"],
            label="R1 on default center")
    st = RS["R8*"]
    ax.plot([wt[k]["R8_bag5"]["region_rmse"]["Middle"]["mean"] for k in ("2", "3", "5", "8")],
            [wt[k]["R8_bag5"]["region_rmse"]["Tails"]["mean"] for k in ("2", "3", "5", "8")],
            color=st["color"], ls="none", marker=st["marker"], mfc="white", ms=3.8, label="R8* (K = 2, 3, 5, 8)")
    ax.set_xlabel(f"Middle-region RMSE {PTS}")
    ax.set_ylabel(f"Pooled tail RMSE {PTS}")
    ax.legend(loc="upper right", fontsize=FS_MIN)
    save(fig, "figS2_frontier")


# ---------------------------------------------------------------------------
# Hình S3 (bổ sung; Hình 10 ở v1): biểu đồ Murphy
# ---------------------------------------------------------------------------
def figS3():
    ps = J("proper_scores")
    th = np.array(ps["meta"]["theta"], dtype=float)
    mm = ps["summary"]["murphy_mean"]
    lo, hi = 60, 100
    series = [("default|R0", "default", CEN["default"]["color"], "-"),
              ("sub1|R0", "sub1", CEN["sub1"]["color"], CEN["sub1"]["ls"]),
              ("bag20|R0", "bag20", CEN["bag20"]["color"], "-"),
              ("rs_tuned|R0", "rs_tuned", CEN["rs_tuned"]["color"], CEN["rs_tuned"]["ls"]),
              ("bag20|R1_1", "R1$_1$ on bag20", RS["R1_1"]["color"], RS["R1_1"]["ls"])]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))
    ax = axes[0]
    for k, lab, col, ls in series:
        ax.plot(th, mm[k], color=col, ls=ls, label=lab)
    ax.set_xlabel("Threshold θ (points)")
    ax.set_ylabel(f"Mean elementary score S$_θ$ {PTS}")
    ax.legend(loc="upper right", fontsize=FS_MIN)
    panel(ax, "a", x=-0.12)
    ax = axes[1]
    base = np.array(mm["default|R0"])
    for k, lab, col, ls in series[1:]:
        ax.plot(th, np.array(mm[k]) - base, color=col, ls=ls, label=f"{lab} − default")
    ax.axhline(0, color=C["black"], lw=0.6)
    for a, b_ in ((th.min(), lo), (hi, th.max())):
        ax.axvspan(a, b_, color=C["lgrey"], alpha=0.25, lw=0)
    ax.set_xlabel("Threshold θ (points; shaded: tails)")
    ax.set_ylabel(f"Difference to default {PTS}")
    # v1 gọi set_ylim(1, 2e5) chép nhầm từ Hình 11, làm ô (b) trống. Giới hạn trục
    # lấy theo dữ liệu, chừa chỗ phía trên cho chú giải.
    ax.autoscale(axis="y")
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    ax.legend(loc="upper left", fontsize=FS_MIN, ncol=2, columnspacing=0.8)
    panel(ax, "b", x=-0.14)
    fig.tight_layout(w_pad=1.6)
    save(fig, "figS3_murphy")


# ---------------------------------------------------------------------------
# Hình S1 (bổ sung; Hình 11 ở v1): kích thước bag và thời gian
# ---------------------------------------------------------------------------
def figS1():
    dc = J("decomp_centers")["summary"]
    cen = dc["centers"]["F_dt-cn"]
    Bs = [1, 2, 5, 10, 20, 40]
    m = np.array([cen[f"bag{b}"]["cost_K"]["R1"]["3"]["mean"] for b in Bs])
    sd = np.array([cen[f"bag{b}"]["cost_K"]["R1"]["3"]["sd"] for b in Bs])
    fit = np.array([cen[f"bag{b}"]["fit_s"]["mean"] for b in Bs])
    bstar = dc["b_star"]["B_star"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
    ax = axes[0]
    ax.errorbar(Bs, m, yerr=sd, color=C["blue"], marker="o", capsize=1.8, label="bag of B")
    for c in ("default", "rs_tuned", "rs_tuned_bag5"):
        st = CEN[c]
        ax.axhline(cen[c]["cost_K"]["R1"]["3"]["mean"], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
    ax.axvline(bstar, color=C["black"], lw=0.5, ls=":")
    ax.text(bstar * 0.93, 0.03, f"B* = {bstar}", fontsize=FS_MIN, ha="right", va="bottom",
            transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xticks(Bs)
    ax.set_xticklabels([str(b) for b in Bs])
    ax.minorticks_off()
    ax.set_xlabel("Bag size B (log scale)")
    ax.set_ylabel(f"Test cost$_3$ with R1 {PTS}")
    ax.set_ylim(10.55, 11.6)
    # chú giải chia hai cột ở phía trên, nơi không có dữ liệu (v1 đè lên đường default)
    ax.legend(loc="upper center", fontsize=FS_MIN, ncol=2, columnspacing=0.8, handlelength=2.2)
    panel(ax, "a", x=-0.13)
    ax = axes[1]
    ax.plot(Bs, fit, color=C["blue"], marker="o", label="bag of B, final fit")
    ct = J("select_policy")["compute_table"]["centers"]
    for c, mk in (("default", "x"), ("rs_tuned", "s"), ("rs_tuned_bag5", "D")):
        v = ct[c]
        col = CEN[c]["color"] if c != "default" else C["grey"]
        ax.plot([5 if c == "rs_tuned_bag5" else 1], [v["fit_s"]["mean"]], marker=mk, color=col, ls="none", ms=4,
                mfc="white" if c == "rs_tuned_bag5" else col, label=f"{c}, final fit")
        if v["tune_s"]["mean"] > 0 and c == "rs_tuned":
            ax.axhline(v["tune_s"]["mean"], color=col, ls="--", lw=0.9, label="random search (60 configs)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Bs)
    ax.set_xticklabels([str(b) for b in Bs])
    ax.minorticks_off()
    ax.set_xlabel("Bag size B (log scale)")
    ax.set_ylabel("Thread-seconds per split")
    ax.set_ylim(1, 2e6)
    ax.legend(loc="upper left", fontsize=FS_MIN, ncol=2, columnspacing=0.8)
    panel(ax, "b", x=-0.13)
    fig.tight_layout(w_pad=1.6)
    save(fig, "figS1_bag_size_compute")


# ---------------------------------------------------------------------------
# Hình S4 (bổ sung; Hình 12 ở v1): mô phỏng
# ---------------------------------------------------------------------------
def figS4():
    s = J("sim_decomp")["summary"]
    scen_m = {"S1": "o", "S2": "s", "S3": "^", "S4": "D"}
    rule_c = {"R0": RS["R0"]["color"], "R4": RS["R4"]["color"], "R5": RS["R5"]["color"], "R8": RS["R8*"]["color"]}
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
    ax = axes[0]
    for key, v in s.items():
        scen, n = key.split("|")
        if n != "57000":
            continue
        for c, cv in v["centers"].items():
            for K, kv in cv["by_K"].items():
                if K == "1":
                    continue
                for r, g in kv["decision_gap"].items():
                    if r not in rule_c or g["true"] <= 0 or g["est"] <= 0:
                        continue
                    ax.plot(g["true"], g["est"], marker=scen_m[scen], color=rule_c[r], ls="none", ms=3.2,
                            mfc=rule_c[r] if c == "bag10" else "white", mew=0.7, alpha=0.9)
    lim = [0.004, 2.0]
    ax.plot(lim, lim, color=C["black"], lw=0.6)
    xx = np.array(lim)
    ax.fill_between(xx, xx * 0.9, xx * 1.1, color=C["lgrey"], alpha=0.3, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"True decision gap, via μ$^G$ {PTS}")
    ax.set_ylabel(f"Estimated gap, via R1 {PTS}")
    h1 = [Line2D([], [], marker=m, color="#333333", ls="none", ms=3.2, label=sc) for sc, m in scen_m.items()]
    h2 = [Patch(color=col, label=r if r != "R8" else "R8 (not retuned)") for r, col in rule_c.items()]
    h3 = [Line2D([], [], marker="o", color="#333333", ls="none", ms=3.2, label="bag10"),
          Line2D([], [], marker="o", color="#333333", mfc="white", ls="none", ms=3.2, label="default")]
    ax.legend(handles=h1 + h2 + h3, loc="upper left", fontsize=FS_MIN, ncol=2, columnspacing=0.6, handlelength=1.2)
    panel(ax, "a", x=-0.13)

    ax = axes[1]
    for key, v in s.items():
        scen, n = key.split("|")
        for c, cv in v["centers"].items():
            if c != "bag10":
                continue
            Ks = [k for k in cv["by_K"]]
            y = [cv["by_K"][k]["R1_gap_to_muG"]["mean"] for k in Ks]
            ax.plot([float(k) for k in Ks], y, marker=scen_m[scen], color=C["blue"] if n == "57000" else C["orange"],
                    ls="-" if n == "57000" else "--", ms=3, lw=0.8)
    logk_axis(ax, [1, 2, 3, 5, 8])
    ax.set_xlabel("Cost ratio K (log scale)")
    ax.set_ylabel(f"cost$_K$(R1) − cost$_K$(μ$^G$) {PTS}")
    h = [Line2D([], [], color=C["blue"], ls="-", label="n = 57,000"),
         Line2D([], [], color=C["orange"], ls="--", label="n = 10,000")] + h1
    ax.legend(handles=h, loc="upper left", fontsize=FS_MIN, ncol=2)
    panel(ax, "b", x=-0.15)
    fig.tight_layout(w_pad=1.6)
    save(fig, "figS4_simulation")


# ---------------------------------------------------------------------------
# Hình S5: lặp lại trên dữ liệu công khai (E11)
# ---------------------------------------------------------------------------
BENCH_ROWS = [("hsa", "HSA (bag20)"), ("california_housing", "california_housing"),
              ("diamonds", "diamonds"), ("kings_county", "kings_county"),
              ("cps88wages", "cps88wages"), ("saber", "Saber 11 to Saber Pro"),
              ("student_performance_por", "student_perf._por (not gated)")]


def figS5():
    """Bốn phép so ở K = 3, tính bằng % cost_3(R1, trung tâm chính) của từng bộ.

    Vì sao phần trăm mà không phải điểm: giá nhà, đô la và điểm Saber không cùng thang,
    và SESOI 0,10 là điểm HSA nên không vẽ dải SESOI ở đây. Số đọc từ
    results_bench/paper_numbers_bench.json (E11) và results_cost/paper_numbers_cost.json
    (HSA), cả hai đã chứa CI Nadeau-Bengio; hình chỉ đổi đơn vị, không tính lại thống kê.
    """
    with open("results_bench/paper_numbers_bench.json", encoding="utf-8") as f:
        B = json.load(f)
    with open(os.path.join(RES, "paper_numbers_cost.json"), encoding="utf-8") as f:
        H = json.load(f)
    hbase = H["e1.center.F_dt-cn.bag20.cost3.R1"]
    cols = [("C1", "c1", "C1 = R1 − R8*"), ("C2", "c2", "C2 = R1 − R5"),
            ("C3", "c3", "C3 = rs_tuned − bag"), ("e4.c_minus_a", "e4.K3.c_minus_a", "E4: (c) − (a)")]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.3), sharey=True)
    ys = np.arange(len(BENCH_ROWS))[::-1]
    for j, (bk, hk, title) in enumerate(cols):
        ax = axes[j]
        pts = []
        for (ds, lab), y in zip(BENCH_ROWS, ys):
            if ds == "hsa":
                base, m, lo, hi = hbase, H[f"{hk}.mean"], H[f"{hk}.ci_lo"], H[f"{hk}.ci_hi"]
                st = dict(color=C["black"], marker="s", mfc=C["black"])
            else:
                k = f"bench.{ds}.{bk}"
                base = B[f"bench.{ds}.cost3_R1_bag10"]
                m, lo, hi = B[f"{k}.mean"], B[f"{k}.ci_lo"], B[f"{k}.ci_hi"]
                core = B[f"bench.{ds}.core"]
                st = dict(color=C["blue"] if core else C["grey"], marker="o",
                          mfc=C["blue"] if core else "white")
            m, lo, hi = (100 * v / base for v in (m, lo, hi))
            pts.append((ds, y, m, lo, hi, st))
        # Trục theo các bộ trong cổng và HSA: khoảng tin cậy của bộ phụ (649 dòng) rộng
        # gấp vài lần và nếu để nó quyết định trục thì mọi bộ khác dồn vào một điểm.
        # Phần vượt trục được cắt và đánh dấu bằng tam giác ở mép.
        core_ext = [v for ds, _, m, lo, hi, _ in pts
                    if ds == "hsa" or B[f"bench.{ds}.core"] for v in (lo, hi)]
        pad = 0.12 * (max(core_ext) - min(core_ext))
        xl, xr = min(min(core_ext), 0) - pad, max(max(core_ext), 0) + pad
        for ds, y, m, lo, hi, st in pts:
            clo, chi = max(lo, xl), min(hi, xr)
            for edge, cut in ((xl, lo < xl), (xr, hi > xr)):
                if cut:
                    ax.plot([edge], [y], marker="<" if edge == xl else ">", color=st["color"], ms=3.5)
            ax.errorbar([m], [y], xerr=[[m - clo], [chi - m]], color=st["color"], marker=st["marker"],
                        mfc=st["mfc"], ms=3.5, capsize=0, lw=0.8, ls="none")
        ax.set_xlim(xl, xr)
        ax.axvline(0, color=C["black"], lw=0.6)
        ax.axhline(ys[0] - 0.5, color=C["lgrey"], lw=0.6)
        ax.set_xlabel("% of cost$_3$(R1, center)")
        ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=FS_MIN)
        ax.grid(axis="y", visible=False)
        panel(ax, "abcd"[j], x=-0.08 if j else -0.95, y=1.02)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, lab in BENCH_ROWS])
    fig.tight_layout(w_pad=0.6)
    save(fig, "figS5_replication")


FIGS = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7, "8": fig8,
        "S1": figS1, "S2": figS2, "S3": figS3, "S4": figS4,
        "S5": figS5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="*", default=list(FIGS))
    a = ap.parse_args()
    for k in a.fig:
        print(f"Fig. {k}")
        FIGS[k]()


if __name__ == "__main__":
    main()
