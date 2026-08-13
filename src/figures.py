# -*- coding: utf-8 -*-
"""Sinh hình cho bài báo IEEE Access (GA4XGB).

Quy ước IEEE Access:
- Cột đơn ~3.5 in, cột đôi ~7.16 in; font 8-9pt; xuất PDF vector + PNG 600dpi.
- Bảng màu an toàn cho người mù màu (Okabe-Ito), phân biệt được cả khi in đen trắng
  nhờ kết hợp màu + kiểu nét + marker.

Chạy:  PYTHONPATH=src python3 src/figures.py --fig 3 4 7
       (fig 5, 6 cần kết quả multi-seed / tail-GA -> chạy sau)
Xuất:  paper/figures/figN_*.pdf|png
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import bins as B

FIGDIR = "paper/figures"
# Nguồn duy nhất cho mép/nhãn bin — KHỚP với fitness (xem src/bins.py)
BINS = B.EDGES
LOW_TAIL_MAX = B.LOW_TAIL_MAX
HIGH_TAIL_MIN = B.HIGH_TAIL_MIN
BIN_LABELS = B.LABELS

# Okabe-Ito colorblind-safe
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # font nhúng dạng TrueType
})


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGDIR}/{name}.{ext}")
    plt.close(fig)
    print(f"  -> {FIGDIR}/{name}.pdf|png")


def get_data():
    from preprocess import load_and_preprocess
    return load_and_preprocess()


def _nice_labels(idx):
    """Đổi tên cột tiếng Việt sang nhãn tiếng Anh ngắn cho hình."""
    return (idx.str.replace("Tỉnh_Thành phố ", "Province: ", regex=False)
                     .str.replace("Tỉnh_Tỉnh ", "Province: ", regex=False)
                     .str.replace("khuVuc_", "Region: ", regex=False)
                     .str.replace("Môn ngoại ngữ_", "For.Lang: ", regex=False)
                     .str.replace("Điểm tổng kết", "GPA", regex=False)
                     .str.replace("Học lực", "Standing", regex=False)
                     .str.replace("Hạnh kiểm", "Conduct", regex=False)
                     .str.replace("Toán", "Math", regex=False)
                     .str.replace("Văn", "Literature", regex=False)
                     .str.replace("Vật lí", "Physics", regex=False)
                     .str.replace("Hóa học", "Chemistry", regex=False)
                     .str.replace("Sinh học", "Biology", regex=False)
                     .str.replace("Lịch sử", "History", regex=False)
                     .str.replace("Địa lí", "Geography", regex=False)
                     .str.replace("Ngoại ngữ", "For.Lang", regex=False)
                     .str.replace("truong_freq", "School frequency", regex=False)
                     .str.replace("truong_chuyen", "Specialized school", regex=False)
                     .str.replace("gioiTinh", "Gender", regex=False)
                     .str.replace("thangSinh", "Birth month", regex=False)
                     .str.replace(" HK I", " S1", regex=False)
                     .str.replace(" HK II", " S2", regex=False)
                     .str.replace(" CN", " Year", regex=False))


# ---------------------------------------------------------------- Fig. 3
def fig3_weight_profiles(y):
    """Profile trọng số w_b = n_b^(1-alpha)/sum theo alpha — analog Fig.3 của GA4RF."""
    counts = np.array([((y >= lo) & (y < hi)).sum() for lo, hi in zip(BINS[:-1], BINS[1:])],
                      dtype=float)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    # nền: silhouette histogram (tỷ lệ mẫu)
    ax.bar(range(len(counts)), counts / counts.sum(), color="0.88",
           edgecolor="0.7", linewidth=0.4, label="Score distribution", zorder=1)

    styles = [(0.0, C["blue"], "o", "-"), (0.5, C["orange"], "s", "--"),
              (1.0, C["red"], "^", "-.")]
    for a, col, mk, ls in styles:
        w = counts ** (1 - a)
        w = w / w.sum()
        ax.plot(range(len(w)), w, color=col, marker=mk, linestyle=ls,
                markersize=3.5, label=rf"$\alpha$ = {a:g}", zorder=3)

    ax.set_xticks(range(len(BIN_LABELS)))
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right")
    ax.set_xlabel("HSA score bin")
    ax.set_ylabel(r"Fitness weight $w_b$")
    ax.legend(frameon=False, loc="upper right")
    ax.annotate(r"$\alpha$=0: micro (= RMSE)", xy=(3, counts.max()/counts.sum()),
                xytext=(3.6, 0.30), fontsize=7.0, color=C["blue"],
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.6))
    ax.annotate(r"$\alpha$=1: macro (all bins equal)", xy=(0, 1/len(counts)),
                xytext=(0.3, 0.20), fontsize=7.0, color=C["red"],
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.6))
    save(fig, "fig3_weight_profiles")


# ---------------------------------------------------------------- Fig. 4
def fig4_score_histogram(y):
    """Histogram điểm HSA + đánh dấu vùng đuôi."""
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.hist(y, bins=np.arange(25, 135, 2), color=C["sky"], edgecolor="white",
            linewidth=0.3, zorder=2)
    ymax = ax.get_ylim()[1]

    # Vùng đuôi tô nhạt — % TÍNH TỪ DỮ LIỆU, không hardcode.
    # PHẢI dùng đúng ranh giới của bins.LOW_TAIL_MAX / HIGH_TAIL_MIN (60 và 100):
    # bản trước tô <50 và >=110 nhưng vẫn dán nhãn "Low/High tail", trong khi mọi
    # số "tail" ở Bảng 5 và Hình 6 lại tính trên <60 và >=100 — cùng một chữ, hai
    # nghĩa khác nhau trong cùng một bài.
    yv = np.asarray(y, dtype=float)
    pct_lo = 100 * (yv < LOW_TAIL_MAX).mean()
    pct_hi = 100 * (yv >= HIGH_TAIL_MIN).mean()
    for lo, hi, lab in [(25, LOW_TAIL_MAX, f"Low tail\n(<{LOW_TAIL_MAX})\n{pct_lo:.2f}%"),
                        (HIGH_TAIL_MIN, 135,
                         f"High tail\n(≥{HIGH_TAIL_MIN})\n{pct_hi:.2f}%")]:
        ax.add_patch(Rectangle((lo, 0), hi - lo, ymax, facecolor=C["red"],
                               alpha=0.10, zorder=1))
        ax.text((lo + hi) / 2, ymax * 0.80, lab, ha="center", fontsize=6.5,
                color=C["red"])

    for e in BINS[1:-1]:
        ax.axvline(e, color="0.55", linestyle=":", linewidth=0.5, zorder=3)

    m, s = float(np.mean(y)), float(np.std(y))
    ax.axvline(m, color=C["black"], linestyle="--", linewidth=0.9, zorder=4)
    ax.text(m + 2.0, ymax * 0.88, f"mean = {m:.1f}\nSD = {s:.1f}", fontsize=7.0,
            va="top")
    ax.set_xlabel("HSA score (150-point scale)")
    ax.set_ylabel("Number of students")
    ax.set_xlim(25, 135)
    ax.set_ylim(0, ymax * 1.05)
    save(fig, "fig4_score_histogram")


# ---------------------------------------------------------------- Fig. 7
def fig7_feature_importance(X, y):
    """Top-20 gain importance của mô hình GA4XGB tốt nhất hiện có."""
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    best = json.load(open("results/ga_rmse_best.json"))["best_params"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1,
                     importance_type="gain", **best)
    m.fit(X_tr, y_tr)

    imp = pd.Series(m.feature_importances_, index=X.columns).nlargest(20)[::-1]
    nice = _nice_labels(imp.index)

    # Tô màu theo NHÓM đặc trưng. Cả bài lập luận rằng điểm Toán và biến Tỉnh
    # chi phối mô hình; vẽ 20 thanh cùng một màu thì phát hiện đó vô hình, người
    # đọc phải tự dò từng nhãn. Màu hoá nhóm là cách để hình tự nói ra kết luận.
    def group_of(name):
        if name.startswith("Province: "):
            return "Province"
        if "Math" in name:
            return "Mathematics grades"
        if any(k in name for k in ("Physics", "Chemistry", "Biology", "History",
                                   "Geography", "Literature", "For.Lang", "GPA")):
            return "Other subject grades"
        return "Other"

    groups = [group_of(x) for x in nice]
    gcol = {"Mathematics grades": C["red"], "Province": C["blue"],
            "Other subject grades": C["sky"], "Other": "#BBBBBB"}

    # Tỷ trọng tính TRONG TOP-20, đúng phạm vi mà mục IV-J của bài đang nói
    # ("half of the top-twenty gain", "a third of the top-twenty gain"). Bản
    # trước tính trên cả 186 đặc trưng -> ra 26%/26%, đọc lên mâu thuẫn với chữ
    # trong bài dù cả hai đều đúng: chúng là hai đại lượng khác nhau.
    share = imp.groupby(pd.Series(groups, index=imp.index)).sum() / imp.sum()

    fig, ax = plt.subplots(figsize=(3.45, 3.5))
    ax.barh(range(len(imp)), imp.values, height=0.74,
            color=[gcol[g] for g in groups], edgecolor="none")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(nice, fontsize=6.2)
    ax.set_xlabel("Gain importance (normalized)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [Rectangle((0, 0), 1, 1, facecolor=gcol[g], edgecolor="none",
                         label=f"{g} ({share.get(g, 0) * 100:.0f}% of top-20 gain)")
               for g in ["Mathematics grades", "Province", "Other subject grades"]]
    ax.legend(handles=handles, fontsize=6.2, loc="lower right", frameon=True,
              framealpha=0.95, edgecolor="#CCCCCC", handlelength=1.1,
              handleheight=0.9, borderpad=0.5)

    n_prov = sum(1 for g in groups if g == "Province")
    ax.set_title(f"{n_prov} of the top 20 features are province indicators",
                 fontsize=7, pad=4)
    save(fig, "fig7_feature_importance")
    imp[::-1].to_csv(f"{FIGDIR}/fig7_data.csv", header=["gain"])


# ---------------------------------------------------------------- Fig. 6
def fig6_mae_by_bin():
    """Hình mang LUẬN ĐIỂM của bài — 3 panel.

    LỊCH SỬ THIẾT KẾ (giữ lại để khỏi lặp lỗi):
    - Bản 1 vẽ MAE thô theo bin đúng công thức Fig.8/9 của GA4RF -> THẤT BẠI: hình
      chữ U của bài toán (5.5->19 điểm) nuốt khác biệt giữa các phương pháp (~0.2).
      GA4RF vẽ được vì F1 của họ trải 59-100 và khác biệt tới 5-10 điểm.
    - Bản 2 dùng delta + CI bootstrap -> tốt, nhưng khi có nhánh +LW (delta tới -3.7)
      thì các cột nhỏ (+-0.3) bị bẹp.
    - Bản 3 (hiện tại): (a) profile lỗi thô -> cho thấy +LW làm PHẲNG đường cong;
      (b) delta có CI, thang symlog để chứa cả +-0.2 lẫn -3.7;
      (c) biên Pareto tổng thể-vs-đuôi -> thông điệp thật của bài: đây là đánh đổi
      có thể chọn, không phải bữa trưa miễn phí.
    """
    import bins as B
    d = json.load(open("results/paper_numbers.json"))
    bs = json.load(open("results/bootstrap_test.json")) if \
        os.path.exists("results/bootstrap_test.json") else None

    order = ["Default XGBoost", "Grid search", "Random search", "GA-RMSE",
             "GA4XGB (a=0.5)", "GA4XGB (a=1)", "GA4XGB (a=1, +LW)"]
    show = [m for m in order if m in d["mae_per_bin"]]
    style = {"Default XGBoost":  (C["black"], ":",  "o"),
             "Grid search":      (C["sky"],   ":",  "P"),
             "Random search":    (C["purple"], "--", "d"),
             "GA-RMSE":          (C["blue"],  "-",  "s"),
             "GA-MAE":           (C["yellow"], "--", "x"),
             "GA4XGB (a=0.5)":   (C["orange"], "--", "^"),
             "GA4XGB (a=1)":     (C["red"],   "-",  "v"),
             "GA4XGB (a=1, +LW)":(C["green"], "-",  "*")}
    nice = lambda m: m.replace("a=", r"$\alpha$=")

    fig = plt.figure(figsize=(7.16, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.72,
                      wspace=0.28)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # ---------- (a) profile lỗi thô ----------
    xs = np.arange(len(B.LABELS))
    for m in show:
        col, ls, mk = style[m]
        lw = 1.8 if "+LW" in m else 1.0
        z = 5 if "+LW" in m else 2
        ax1.plot(xs, [d["mae_per_bin"][m][l] for l in B.LABELS], color=col,
                 linestyle=ls, marker=mk, markersize=3.4, linewidth=lw,
                 label=nice(m), zorder=z)
    for x0, x1 in [(-0.45, 1.45), (5.55, 7.45)]:
        ax1.add_patch(Rectangle((x0, 4.7), x1-x0, 15.5, fill=False,
                                edgecolor=C["red"], linestyle="--", linewidth=0.8))
    ax1.text(0.5, 20.5, "Low tail", ha="center", fontsize=7.0, color=C["red"])
    ax1.text(6.5, 20.5, "High tail", ha="center", fontsize=7.0, color=C["red"])
    ax1.annotate("conventional methods:\n3.4$\\times$ U-shape", xy=(1.05, 12.3),
                 xytext=(2.35, 17.6), fontsize=6.8, color="0.3", ha="center",
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))
    ax1.annotate("GA4XGB+LW: near-flat\nacross the distribution", xy=(4.0, 8.1),
                 xytext=(5.05, 14.6), fontsize=6.8, color=C["green"], ha="center",
                 arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.7))
    ax1.set_xticks(xs); ax1.set_xticklabels(B.LABELS, fontsize=7.0)
    ax1.set_ylabel("MAE (HSA points)")
    ax1.set_xlabel("HSA score bin", labelpad=1)
    ax1.set_ylim(4.6, 21.8)
    ax1.set_title("(a) Error profile across the score distribution", fontsize=7.5)
    ax1.legend(frameon=False, fontsize=6.2, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.17), columnspacing=1.1, handlelength=2.0,
               handletextpad=0.5)
    ax1.grid(axis="x", visible=False)

    # ---------- (b) thay đổi TƯƠNG ĐỐI vs GA-RMSE ----------
    # Bản trước vẽ ΔMAE tuyệt đối, buộc phải dùng thang symlog để chứa cùng lúc
    # ±0.2 (các biến thể fitness) và −3.7 (+LW) — hệ quả là một trục có "vùng
    # tuyến tính" tô xám và các nấc -3/-1/-0.3/0/0.3/1, gần như không đọc được.
    # Đổi sang PHẦN TRĂM thì mọi thứ nằm gọn trong −30%..+20%: thang tuyến tính
    # thường là đủ. Cùng một sự thật, chỉ đổi đơn vị cho đúng bản chất câu hỏi
    # ("cải thiện bao nhiêu phần") thay vì bẻ cong trục.
    base = "GA-RMSE"
    regs = ["Low tail", "Middle", "High tail", "All"]
    keymap = {"Low tail": "LOW TAIL (<60)", "Middle": "MIDDLE (60-100)",
              "High tail": "HIGH TAIL (>=100)", "All": "ALL"}
    bskey = {"GA4XGB (a=1)": "GA-tail a=1", "GA4XGB (a=0.5)": "GA-tail a=0.5",
             "GA4XGB (a=1, +LW)": "GA-tail a=1 +LW", "Random search": "RandomSearch",
             "Grid search": "GridSearch"}
    cand = [m for m in show if m not in (base, "Default XGBoost", "Grid search")]
    w = 0.8 / max(len(cand), 1)
    for j, m in enumerate(cand):
        col = style[m][0]
        vals, los, his = [], [], []
        for r in regs:
            ref = d["mae_per_region"][base][r]
            dv = d["mae_per_region"][m][r] - ref
            pct = 100.0 * dv / ref
            vals.append(pct)
            c = bs["comparisons"].get(keymap[r], {}).get(bskey.get(m, m)) if bs else None
            if c:
                los.append(pct - 100.0 * c["ci95"][0] / ref)
                his.append(100.0 * c["ci95"][1] / ref - pct)
            else:
                los.append(0); his.append(0)
        pos = np.arange(len(regs)) + (j - (len(cand) - 1) / 2) * w
        ax2.bar(pos, vals, w * 0.88, color=col, label=nice(m), zorder=2)
        ax2.errorbar(pos, vals, yerr=[los, his], fmt="none", ecolor="0.2",
                     elinewidth=0.6, capsize=1.5, zorder=3)
    ax2.axhline(0, color="0.3", linewidth=0.9, zorder=1)
    ax2.set_xticks(np.arange(len(regs)))
    ax2.set_xticklabels(regs, fontsize=7.0)
    ax2.tick_params(axis="y", labelsize=6.5)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%".replace("+0%", "0"))
    ax2.set_ylabel("change in MAE vs GA-RMSE", fontsize=7)
    ax2.set_title("(b) Effect size, with 95% CI", fontsize=7.5)
    ax2.text(0.02, 0.04, "below 0 = better", transform=ax2.transAxes, fontsize=6.2,
             color="0.35", style="italic")
    ax2.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.1,
               loc="upper left", handlelength=1.4, borderpad=0.3,
               labelspacing=0.25)
    ax2.grid(axis="x", visible=False)

    # ---------- (c) biên Pareto ----------
    # Bản trước: nhãn "α=0" đè lên nhãn trục y, và mũi tên xanh dựng ngay mép
    # trái cũng đè nốt. Ở đây nới lề trục rồi mới đặt nhãn, và bỏ mũi tên —
    # tiêu đề panel đã nói ý đó rồi.
    tail_mae = lambda m: (d["mae_per_region"][m]["Low tail"] +
                          d["mae_per_region"][m]["High tail"]) / 2
    front = [m for m in ["GA-RMSE", "GA4XGB (a=0.5)", "GA4XGB (a=1)",
                         "GA4XGB (a=1, +LW)"] if m in d["mae_per_region"]]
    fx = [d["mae_per_region"][m]["All"] for m in front]
    fy = [tail_mae(m) for m in front]
    ax3.plot(fx, fy, color="0.6", linestyle="--", linewidth=0.9, zorder=1)
    # CHỈ vẽ các điểm trên biên + default. Bản trước vẽ cả grid/random search:
    # chúng nằm chồng lên cụm α=0/0.5/1 nên nhãn "α=0" trông như đang chỉ vào
    # viên kim cương hồng của Random search. Chúng không thuộc biên (α, β) nên
    # không có việc gì ở panel này.
    pts = front + ["Default XGBoost"]
    for m in pts:
        col, _, mk = style[m]
        ax3.scatter(d["mae_per_region"][m]["All"], tail_mae(m), color=col, marker=mk,
                    s=36, zorder=3, edgecolor="white", linewidth=0.4)

    xs_all = [d["mae_per_region"][m]["All"] for m in pts]
    ys_all = [tail_mae(m) for m in pts]
    padx = (max(xs_all) - min(xs_all)) * 0.22
    pady = (max(ys_all) - min(ys_all)) * 0.18
    ax3.set_xlim(min(xs_all) - padx, max(xs_all) + padx)
    ax3.set_ylim(min(ys_all) - pady, max(ys_all) + pady * 1.5)

    lab = {"GA-RMSE": r"$\alpha$=0", "GA4XGB (a=0.5)": r"$\alpha$=0.5",
           "GA4XGB (a=1)": r"$\alpha$=1", "GA4XGB (a=1, +LW)": r"$\alpha$=1, +LW"}
    offs = {"GA-RMSE": (-4, 10), "GA4XGB (a=0.5)": (-5, -12),
            "GA4XGB (a=1)": (9, -3), "GA4XGB (a=1, +LW)": (-7, 9)}
    for m, x, yv in zip(front, fx, fy):
        ax3.annotate(lab[m], (x, yv), textcoords="offset points", xytext=offs[m],
                     fontsize=6.5, color="0.2",
                     ha="right" if offs[m][0] < 0 else "left",
                     arrowprops=dict(arrowstyle="-", lw=0.4, color="0.6",
                                     shrinkA=0, shrinkB=2))
    dx = d["mae_per_region"]["Default XGBoost"]
    ax3.annotate("default", (dx["All"], (dx["Low tail"] + dx["High tail"]) / 2),
                 textcoords="offset points", xytext=(7, -2), fontsize=6.5, color="0.35")
    ax3.set_xlabel("Overall MAE (all students)", fontsize=7)
    ax3.set_ylabel("Mean tail MAE\n(lower = fairer)", fontsize=7, linespacing=1.2)
    ax3.set_title("(c) The trade-off is a frontier, not a winner", fontsize=7.5)
    ax3.tick_params(labelsize=6.5)
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)
    save(fig, "fig6_mae_by_bin")

    rows = {m: {**{l: d["mae_per_bin"][m][l] for l in B.LABELS},
                **{k: d["mae_per_region"][m][k] for k in regs}} for m in show}
    pd.DataFrame(rows).T.round(2).to_csv(f"{FIGDIR}/fig6_table.csv")
    print(pd.DataFrame(rows).T.round(2).to_string())


# ---------------------------------------------------------------- Fig. 5
def fig5_convergence():
    """Đường hội tụ: fitness tốt nhất theo thế hệ cho các biến thể.

    Theo GA4RF Fig.4/5 (chỉ vẽ đoạn có thông tin), nhưng vẽ nhiều biến thể
    trên thang chuẩn hoá vì các fitness khác đơn vị nhau."""
    runs = [("ga_rmse", "GA-RMSE (RMSE fitness)", C["blue"], "-"),
            ("ga_tail_a0.5", r"GA4XGB ($\alpha$=0.5)", C["orange"], "--"),
            ("ga_tail_a1", r"GA4XGB ($\alpha$=1)", C["red"], "-"),
            ("ga_tail_a1_lw", r"GA4XGB ($\alpha$=1, +LW)", C["green"], "-.")]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    for tag, lab, col, ls in runs:
        f = f"results/{tag}_log.jsonl"
        if not os.path.exists(f):
            continue
        recs = [json.loads(l) for l in open(f)]
        g = [r["gen"] for r in recs]
        # Run GA-RMSE đầu tiên ghi log theo schema cũ: 'best_rmse' + không có
        # 'best_cv_scores'. Chấp nhận cả hai để khỏi phải chạy lại 2.3 giờ.
        fit = np.array([r.get("best_fitness", r.get("best_rmse")) for r in recs],
                       dtype=float)
        axes[0].plot(g, (fit-fit.min())/(fit[0]-fit.min()+1e-12), color=col,
                     linestyle=ls, linewidth=1.2, label=lab)
        cv = [r["best_cv_scores"]["rmse"] if "best_cv_scores" in r
              else r.get("best_rmse") for r in recs]
        axes[1].plot(g, cv, color=col, linestyle=ls, linewidth=1.2, label=lab)

    axes[0].set_ylabel("Normalized best fitness")
    axes[0].set_xlabel("Generation")
    axes[0].set_title("(a) Convergence of each objective", fontsize=7.5)
    axes[1].axhline(9.9661, color="0.45", linestyle=":", linewidth=0.9)
    axes[1].text(29.5, 9.985, "default XGBoost", fontsize=6.5, color="0.45",
                 ha="right", va="bottom")
    # Chú thích hiện tượng phản trực giác nhưng đúng thiết kế: run +LW có CV-RMSE
    # TĂNG dần vì nó đang tối ưu mục tiêu khác.
    # Mũi tên cũ trỏ tới gen 21 -> cắt ngang chú giải ở góc trên phải. Trỏ vào
    # đoạn gen ~9 (bên trái, trống) và đẩy chú giải xuống giữa-phải.
    axes[1].annotate("+LW trades aggregate RMSE\nfor tail accuracy by design",
                     xy=(8.4, 10.72), xytext=(13.5, 10.30), fontsize=6.4,
                     color=C["green"], ha="center",
                     arrowprops=dict(arrowstyle="->", color=C["green"], lw=0.6))
    axes[1].set_ylabel("Best CV-RMSE (HSA points)")
    axes[1].set_xlabel("Generation")
    axes[1].set_title("(b) Cross-validated RMSE of the incumbent", fontsize=7.5)
    axes[1].legend(frameon=False, fontsize=6.5, loc="center right",
                   bbox_to_anchor=(1.0, 0.62))
    save(fig, "fig5_convergence")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="+", default=["3", "4", "5", "6", "7"])
    args = ap.parse_args()

    need_data = any(f in args.fig for f in ["3", "4", "7"])
    X = y = None
    if need_data:
        X, y = get_data()
    if "3" in args.fig:
        print("Fig 3: weight profiles"); fig3_weight_profiles(y)
    if "4" in args.fig:
        print("Fig 4: score histogram"); fig4_score_histogram(y)
    if "5" in args.fig:
        print("Fig 5: convergence"); fig5_convergence()
    if "6" in args.fig:
        print("Fig 6: MAE by bin (hình ăn tiền)"); fig6_mae_by_bin()
    if "7" in args.fig:
        print("Fig 7: feature importance"); fig7_feature_importance(X, y)
