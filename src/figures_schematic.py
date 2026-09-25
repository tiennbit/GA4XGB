# -*- coding: utf-8 -*-
"""Fig. 1 (lưu đồ thuật toán) và Fig. 2 (minh hoạ toán tử tiến hoá).

Hai hình này là hình khái niệm, không sinh từ dữ liệu -> tách khỏi figures.py.
Bám theo công thức của GA4RF: Fig. 1 là lưu đồ hộp bo góc / thoi rẽ nhánh
Yes-No; Fig. 2 biến khái niệm trừu tượng (lai ghép) thành thứ nhìn thấy được.
Khác biệt cần làm bật lên ở Fig. 2: nhiễm sắc thể của ta là SỐ THỰC, nên BLX-α
lấy mẫu trên một khoảng liên tục — không có bước lượng tử hoá như mã nhị phân.

Chạy: PYTHONPATH=src python3 src/figures_schematic.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Rectangle

FIGDIR = "paper/figures"
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGDIR}/{name}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  -> {FIGDIR}/{name}.pdf|png")


# ---------------------------------------------------------------- Fig. 1
def box(ax, xy, w, h, text, fc="white", ec="#333333", fs=7.0, bold=False, round_=0.12):
    x, y = xy
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={round_}",
                       linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.35)
    return (x, y, w, h)


def diamond(ax, xy, w, h, text, fc="#FFF6E0", fs=7.0):
    x, y = xy
    p = Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                closed=True, linewidth=0.9, edgecolor="#333333", facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, p0, p1, text=None, rad=0.0, fs=6.5):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                        linewidth=0.8, color="#333333", zorder=1,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + 0.06, my, text, fontsize=fs, ha="left", va="center",
                color="#333333")


def fig1_flowchart():
    """Lưu đồ GA4XGB. Một cột (3.5in).

    Toạ độ được nới lại sau khi nâng cỡ chữ lên >=6.5pt (yêu cầu in ấn):
    hộp rộng 6.0/cao 1.4+ để chữ 7pt không chạm viền; chú giải dời hẳn
    xuống dưới hộp cuối cùng.
    """
    fig, ax = plt.subplots(figsize=(3.45, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.6, 20.1)
    ax.axis("off")

    W = 6.0
    xc = 4.2

    box(ax, (xc, 19.5), 2.6, 0.85, "Begin", fc="#E8E8E8", round_=0.4, bold=True)

    box(ax, (xc, 17.85), W, 1.45,
        "Initialize population of $N$ = 24\nreal-coded chromosomes\n"
        "(7 genes, uniform in range)", fc="#E1F0FA", fs=6.8)

    box(ax, (xc, 15.35), W, 2.5,
        "Evaluate fitness of each individual\n"
        "(5-fold CV on the training set)\n"
        "$F_{tail}=\\sqrt{\\sum_b w_b MSE_b}$\n"
        "$w_b \\propto n_b^{\\,1-\\alpha}$",
        fc="#D6F0E4", fs=6.6)

    diamond(ax, (xc, 12.5), 4.7, 2.0,
            "Termination?\ngen = 30  or\n10 gens no gain", fs=6.8)

    box(ax, (8.55, 12.5), 2.5, 1.0, "Return best\nchromosome", fc="white", fs=6.8)
    box(ax, (8.55, 10.5), 2.5, 1.0, "Decode &\ntrain XGBoost", fc="white", fs=6.8)
    box(ax, (8.55, 8.7), 2.5, 0.85, "End", fc="#E8E8E8", round_=0.4, bold=True)

    box(ax, (xc, 10.15), W, 1.1, "Elitism: copy 2 best\nindividuals unchanged", fs=6.8)
    box(ax, (xc, 8.55), W, 0.9, "Tournament selection ($k$ = 3)", fs=6.8)
    box(ax, (xc, 6.95), W, 1.1, "BLX-α crossover\n(rate 0.9,  $\\alpha_x$ = 0.3)",
        fc="#E1F0FA", fs=6.8)
    box(ax, (xc, 5.35), W, 1.1,
        "Gaussian mutation\n(rate 0.15,  $\\sigma$ = 0.15 × range)",
        fc="#E1F0FA", fs=6.8)
    box(ax, (xc, 3.75), W, 1.1, "Clip genes to bounds;\nform new population", fs=6.8)

    arrow(ax, (xc, 19.07), (xc, 18.58))
    arrow(ax, (xc, 17.12), (xc, 16.60))
    arrow(ax, (xc, 14.10), (xc, 13.50))
    arrow(ax, (xc + 2.35, 12.5), (7.30, 12.5))
    ax.text(6.75, 12.78, "Yes", fontsize=6.8, ha="center", va="bottom")
    arrow(ax, (8.55, 12.0), (8.55, 11.0))
    arrow(ax, (8.55, 10.0), (8.55, 9.13))
    arrow(ax, (xc, 11.5), (xc, 10.7))
    ax.text(xc + 0.14, 11.1, "No", fontsize=6.8, ha="left", va="center")
    arrow(ax, (xc, 9.60), (xc, 9.0))
    arrow(ax, (xc, 8.10), (xc, 7.5))
    arrow(ax, (xc, 6.40), (xc, 5.9))
    arrow(ax, (xc, 4.80), (xc, 4.3))

    # vòng lặp quay lại đánh giá fitness
    ax.plot([xc - 3.0, 0.5, 0.5, xc - 3.0], [3.75, 3.75, 15.35, 15.35],
            color="#333333", linewidth=0.8, zorder=1)
    a = FancyArrowPatch((0.5, 15.35), (xc - 2.98, 15.35), arrowstyle="-|>",
                        mutation_scale=7, linewidth=0.8, color="#333333")
    ax.add_patch(a)
    ax.text(0.66, 9.6, "next generation", fontsize=6.5, rotation=90,
            va="center", ha="left", color="#333333")

    # chú giải màu — dưới hộp cuối cùng, không đè gì
    for yy, fc_, lab in [(2.55, "#D6F0E4", "proposed tail-weighted fitness"),
                         (1.85, "#E1F0FA", "real-coded representation and operators"),
                         (1.15, "white", "standard GA machinery")]:
        ax.add_patch(Rectangle((0.5, yy), 0.46, 0.46, fc=fc_, ec="#333333", lw=0.7))
        ax.text(1.2, yy + 0.23, lab, fontsize=6.8, va="center")

    save(fig, "fig1_flowchart")


# ---------------------------------------------------------------- Fig. 2
def fig2_crossover():
    """BLX-α + đột biến Gauss trên véc-tơ số thực. Một cột.

    Bản v1 vẽ dải BLX-α thành các hộp trôi nổi không trục -> không ai đọc được vị
    trí dọc nghĩa là gì. Bản này đặt panel giữa lên MỘT TRỤC THẬT (giá trị gene
    chuẩn hoá 0-1), nên ba thứ cùng đọc được trên một thước đo: hai cha mẹ (chấm),
    khoảng lấy mẫu (dải xanh), và con sinh ra (sao).

    Điểm phải bật lên: dải mở rộng α_x·d ra NGOÀI hai cha mẹ mỗi phía — đó là lý
    do BLX-α khám phá được chứ không chỉ nội suy — và mọi giá trị thực trong dải
    đều lấy được, không có bước lượng tử như mã nhị phân 22-bit.
    """
    genes = ["$n_{est}$", "depth", "$\\eta$", "sub-\nsample", "colsam-\nple",
             "min_ch\n_wt", "$\\lambda$"]
    p1 = np.array([0.62, 0.28, 0.45, 0.80, 0.35, 0.20, 0.70])
    p2 = np.array([0.24, 0.72, 0.18, 0.40, 0.78, 0.62, 0.30])
    rng = np.random.default_rng(7)
    axc = 0.3
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    d = hi - lo
    o1 = np.clip(rng.uniform(lo - axc * d, hi + axc * d), 0.02, 0.98)
    mut = o1.copy()
    mpos = [2, 5]
    mut[mpos] = np.clip(mut[mpos] + np.array([0.17, -0.14]), 0.02, 0.98)

    fig = plt.figure(figsize=(3.45, 4.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.75, 1.05], hspace=0.42,
                          left=0.20, right=0.87, top=0.95, bottom=0.05)
    axA, axB, axC = (fig.add_subplot(g) for g in gs)

    def vec(ax, rows):
        ax.set_xlim(-0.02, 7.02)
        ax.set_ylim(-0.05, len(rows) + 0.55)
        ax.axis("off")
        for k, (v, color, label, marks) in enumerate(rows):
            y = len(rows) - 1 - k
            for g in range(7):
                ax.add_patch(Rectangle((g + 0.04, y + 0.08), 0.92, 0.78,
                                       facecolor=color, edgecolor="#444444",
                                       linewidth=0.6, alpha=0.85))
                ax.text(g + 0.5, y + 0.47, f"{v[g]:.2f}", ha="center", va="center",
                        fontsize=6.2)
                if g in marks:
                    ax.add_patch(Rectangle((g + 0.04, y + 0.08), 0.92, 0.78,
                                           facecolor="none", edgecolor=C["red"],
                                           linewidth=1.4, zorder=5))
            ax.text(-0.12, y + 0.47, label, ha="right", va="center", fontsize=6.8)

    # --- (trên) hai cha mẹ
    vec(axA, [(p1, C["sky"], "Parent 1", ()), (p2, C["yellow"], "Parent 2", ())])
    for g in range(7):
        axA.text(g + 0.5, 2.18, genes[g], ha="center", va="center", fontsize=6.0,
                 linespacing=1.05)

    # --- (giữa) khoảng BLX-α trên trục thật
    for g in range(7):
        l, h = lo[g] - axc * d[g], hi[g] + axc * d[g]
        axB.add_patch(Rectangle((g + 0.12, l), 0.76, h - l, facecolor=C["green"],
                                alpha=0.25, edgecolor=C["green"], linewidth=0.7,
                                zorder=1))
        axB.plot([g + 0.12, g + 0.88], [lo[g]] * 2, color="#777777", lw=0.6, ls=":")
        axB.plot([g + 0.12, g + 0.88], [hi[g]] * 2, color="#777777", lw=0.6, ls=":")
        axB.plot(g + 0.42, p1[g], "o", ms=3.2, color=C["sky"], mec="#333", mew=0.4,
                 zorder=4)
        axB.plot(g + 0.42, p2[g], "o", ms=3.2, color=C["yellow"], mec="#333", mew=0.4,
                 zorder=4)
        axB.plot(g + 0.68, o1[g], "*", ms=6, color=C["orange"], mec="#333", mew=0.3,
                 zorder=5)
    axB.set_xlim(-0.02, 7.02)
    axB.set_ylim(-0.16, 1.16)
    axB.set_xticks([])
    axB.set_yticks([0, 0.5, 1])
    axB.set_ylabel("gene value, normalized to [0,1]\n(see Table 1 for actual ranges)",
                   fontsize=6.0, linespacing=1.2)
    axB.tick_params(labelsize=6.2, length=2, pad=1)
    for sp in ("top", "right", "bottom"):
        axB.spines[sp].set_visible(False)
    axB.spines["left"].set_linewidth(0.6)
    axB.grid(axis="y", alpha=0.25, lw=0.4)
    axB.set_axisbelow(True)

    # thước α_x·d ở gene cuối
    g = 6
    axB.annotate("", xy=(g + 0.96, hi[g] + axc * d[g]), xytext=(g + 0.96, hi[g]),
                 arrowprops=dict(arrowstyle="<->", lw=0.7, color=C["green"],
                                 shrinkA=0, shrinkB=0))
    axB.text(g + 1.02, hi[g] + 0.5 * axc * d[g], "$\\alpha_x d$", fontsize=6.5,
             color=C["green"], va="center", clip_on=False)

    h = [plt.Line2D([], [], ls="", marker="o", ms=3.2, color=C["sky"], mec="#333",
                    mew=0.4, label="parent 1"),
         plt.Line2D([], [], ls="", marker="o", ms=3.2, color=C["yellow"], mec="#333",
                    mew=0.4, label="parent 2"),
         plt.Line2D([], [], ls="", marker="*", ms=6, color=C["orange"], mec="#333",
                    mew=0.3, label="offspring"),
         Rectangle((0, 0), 1, 1, facecolor=C["green"], alpha=0.25, edgecolor=C["green"],
                   lw=0.7, label="BLX-α sampling interval")]
    axB.legend(handles=h, fontsize=6.1, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), frameon=False, handletextpad=0.4,
               columnspacing=1.0)

    # --- (dưới) con và đột biến
    vec(axC, [(o1, C["orange"], "Offspring 1", ()),
              (mut, C["purple"], "After mutation", mpos)])
    axC.text(7.06, 0.47, "perturbed by\n$N(0,\\sigma^2)$", fontsize=6.1,
             color=C["red"], va="center", ha="left", linespacing=1.2)
    axC.text(3.5, 2.30, "every real value inside the interval is reachable —\n"
                        "no discretization step, unlike a binary-coded gene",
             ha="center", va="center", fontsize=6.4, color="#333333", linespacing=1.3)

    save(fig, "fig2_crossover")


if __name__ == "__main__":
    fig1_flowchart()
    fig2_crossover()
