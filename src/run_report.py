# -*- coding: utf-8 -*-
"""Tổng hợp trạng thái và kết quả thí nghiệm GA4XGB.

Ba chế độ, tương ứng ba câu hỏi hay phải trả lời:

  --status   "Đang chạy tới đâu rồi?"      -> vài dòng, rẻ, gọi được liên tục
  --summary  "Kết quả ra sao?"             -> bảng so sánh mọi phương pháp
  --paper    "Phải sửa gì trong bài?"      -> số mới + vị trí theo checklist

Chạy được cả trên máy cá nhân (sau khi `sync_server.local.sh pull`) lẫn trên
server. Không import gì ngoài thư viện chuẩn, nên gọi bằng python3 hệ thống cũng
được — cố tình như vậy để `--status` không phụ thuộc vào venv có dựng xong hay chưa.
"""
import argparse
import json
import os
import re
import sys
import time

R = "results"

# Nhãn hiển thị -> tên file kết quả. Thứ tự ở đây là thứ tự in ra.
RUNS = [
    ("Default XGBoost",     "baseline_xgb_default.json"),
    ("Grid search",         "grid_search_best.json"),
    ("Random search",       "random_search_best.json"),
    ("GA-RMSE",             "ga_rmse_best.json"),
    ("GA-MAE",              "ga_mae_best.json"),
    ("GA-R2",               "ga_r2_best.json"),
    ("GA4XGB (a=0.25)",     "ga_tail_a0.25_best.json"),
    ("GA4XGB (a=0.5)",      "ga_tail_a0.5_best.json"),
    ("GA4XGB (a=0.75)",     "ga_tail_a0.75_best.json"),
    ("GA4XGB (a=1)",        "ga_tail_a1_best.json"),
    ("GA-RMSE (+LW)",       "ga_rmse_lw_best.json"),
    ("GA4XGB (a=1, +LW)",   "ga_tail_a1_lw_best.json"),
]

DERIVED = [("paper_numbers.json", "số cho bảng"),
           ("bootstrap_test.json", "khoảng tin cậy"),
           ("c1_c2_analysis.json", "recall + audit tỉnh")]


def _load(fn):
    p = os.path.join(R, fn)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return "CORRUPT"


def _evals(fn, d):
    """Số lần đánh giá THỰC SỰ tiêu thụ.

    Bẫy: grid/random search ghi khoá `budget` = ngân sách DANH NGHĨA được giao,
    không phải số cấu hình thật sự chạy. Lưới của grid không chia hết nên nó chỉ
    chạy 432/632 (68%). Bài báo phải dùng số thật — đếm dòng trong *_log.jsonl.
    """
    if "total_evals" in d:
        return d["total_evals"]
    log = os.path.join(R, fn.replace("_best.json", "_log.jsonl"))
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as f:
            return sum(1 for l in f if l.strip())
    return None


def _age(path):
    """Tuổi file tính bằng giờ, hoặc None nếu không có."""
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 3600.0


# ---------------------------------------------------------------- status ----
def status(logname="run_5fold.log"):
    """Trạng thái tiến trình. KHÔNG kết luận 'đã chết' chỉ vì log im —
    trên máy 8 nhân một bước GA có thể chạy 6-8 tiếng mà không in gì thêm."""
    done = [(lab, fn) for lab, fn in RUNS if os.path.exists(os.path.join(R, fn))]
    todo = [(lab, fn) for lab, fn in RUNS if not os.path.exists(os.path.join(R, fn))]

    print(f"Kết quả: {len(done)}/{len(RUNS)} thí nghiệm có file")
    if todo:
        print("  Còn thiếu: " + ", ".join(lab for lab, _ in todo))

    log = os.path.join(R, logname)
    if not os.path.exists(log):
        print(f"  (không thấy {log} — chưa chạy, hoặc log nằm trên server)")
    else:
        with open(log, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f if l.strip()]
        steps = [l for l in lines if "=====" in l]
        if steps:
            print(f"  Bước gần nhất: {steps[-1]}")
        # Dòng tiến độ GA có dạng "[tail] Gen 12 | best fitness ..."
        gens = [l for l in lines if re.search(r"Gen\s+\d+", l)]
        if gens:
            print(f"  Thế hệ gần nhất: {gens[-1][:110]}")
        finished = any("HOÀN TẤT" in l for l in lines)
        h = _age(log)
        print(f"  Log cập nhật cách đây {h:.1f} giờ"
              + ("  (đã chạy xong)" if finished else ""))
        # Chỉ cảnh báo khi log CHƯA có dòng kết thúc. Một bước GA trên máy 8 nhân
        # có thể im 6-8 tiếng mà vẫn khoẻ, nên ngưỡng để rộng.
        if not finished and h > 12:
            print("  ^ im quá lâu mà chưa thấy HOÀN TẤT — kiểm tra PID trên server.")
            print("    Lưu ý: không kết nối được != job đã chết (VPN hay rớt).")

    # Lỗi trong log là thứ duy nhất nên báo động ngay
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as f:
            bad = [l.rstrip() for l in f
                   if re.search(r"Traceback|Error|FAILED|Killed|MemoryError", l)]
        if bad:
            print(f"  !! {len(bad)} dòng lỗi trong log, gần nhất: {bad[-1][:110]}")

    missing_derived = [d for d, _ in DERIVED if not os.path.exists(os.path.join(R, d))]
    if not todo and missing_derived:
        print("  Thí nghiệm xong nhưng chưa tổng hợp: " + ", ".join(missing_derived))
    elif not todo and not missing_derived:
        print("  => Đủ cả thí nghiệm lẫn số tổng hợp.")
    return 0


# --------------------------------------------------------------- summary ----
def summary():
    """Bảng so sánh. Cột Evals/Giờ lấy từ chính file kết quả nên phản ánh
    đúng lần chạy đang có, không phải con số chép tay trong bài."""
    rows = []
    for lab, fn in RUNS:
        d = _load(fn)
        if d is None:
            continue
        if d == "CORRUPT":
            rows.append((lab, "FILE HỎNG", "", "", "", ""))
            continue
        t = d.get("test", {})
        ev = _evals(fn, d)
        ev = "" if ev is None else ev
        sec = d.get("total_seconds")
        rows.append((lab,
                     f"{t.get('rmse', float('nan')):.4f}",
                     f"{t.get('mae', float('nan')):.4f}",
                     f"{t.get('r2', float('nan')):.4f}",
                     str(ev),
                     f"{sec/3600:.2f}" if isinstance(sec, (int, float)) else ""))
    if not rows:
        print("Chưa có kết quả nào trong results/")
        return 1

    hdr = ("Phương pháp", "RMSE", "MAE", "R2", "Evals", "Giờ")
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(6)]
    line = lambda r: "  ".join(str(r[i]).ljust(w[i]) if i == 0 else str(r[i]).rjust(w[i])
                               for i in range(6))
    print(line(hdr))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print(line(r))

    # Ngân sách — con số phải đồng bộ khắp bài, nên in riêng cho nổi
    ga = _load("ga_rmse_best.json")
    if ga and ga != "CORRUPT":
        print(f"\nNgân sách đánh giá (từ GA-RMSE) = {ga['total_evals']}"
              f"  |  CV_FOLDS = {ga['ga_config']['cv_folds']}")
    return 0


# ----------------------------------------------------------------- paper ----
def paper():
    """Số mới + chỗ phải sửa. Bản đồ chi tiết ở notes/checklist-cap-nhat-sau-5fold.md;
    hàm này chỉ lo phần dễ sai nhất: những con số lặp lại ở nhiều nơi."""
    ga = _load("ga_rmse_best.json")
    if not ga or ga == "CORRUPT":
        print("Chưa có ga_rmse_best.json — chưa xác định được ngân sách.")
        return 1

    budget = ga["total_evals"]
    print(f"NGÂN SÁCH ĐÁNH GIÁ = {budget}   (CV_FOLDS = {ga['ga_config']['cv_folds']})")
    print("  Sửa ở 6 chỗ: Section IV-B (3 câu), Bảng 3 dòng 'Evaluation budget',")
    print("  Bảng 4 + Bảng 6 cột Evaluations, Section V.")

    for lab, fn in [("Random search", "random_search_best.json"),
                    ("Grid search", "grid_search_best.json")]:
        d = _load(fn)
        if not d or d == "CORRUPT":
            continue
        n = _evals(fn, d)
        if n is None:
            continue
        pct = 100.0 * n / budget
        nom = d.get("budget")
        note = ""
        if nom is not None and n != nom:
            note = f"  (ngân sách danh nghĩa {nom} — lưới không chia hết)"
        print(f"  {lab}: {n} lần đánh giá thực = {pct:.1f}% ngân sách{note}")

    pn = _load("paper_numbers.json")
    if not pn or pn == "CORRUPT":
        print("\nChưa có paper_numbers.json — chạy `python src/paper_numbers.py`.")
        return 1

    # Kiểm tra tính đơn điệu của quét alpha. Bài đang khẳng định 'monotonically'
    # dựa trên 3 mốc; có 5 mốc rồi thì phải kiểm lại, và nếu KHÔNG đơn điệu thì
    # đó là kết quả đáng giá hơn chứ không phải hỏng.
    alphas = [("0", "GA-RMSE"), ("0.25", "GA4XGB (a=0.25)"), ("0.5", "GA4XGB (a=0.5)"),
              ("0.75", "GA4XGB (a=0.75)"), ("1", "GA4XGB (a=1)")]
    mpr = pn.get("mae_per_region", {})
    have = [(a, m) for a, m in alphas if m in mpr]
    if len(have) >= 3:
        print(f"\nQUÉT ALPHA — {len(have)}/5 mốc có kết quả")
        print(f"  {'alpha':>6}  {'MAE tổng':>9}  {'MAE đuôi':>9}")
        tails = []
        for a, m in have:
            allv = mpr[m]["All"]
            tail = (mpr[m]["Low tail"] + mpr[m]["High tail"]) / 2
            tails.append(tail)
            print(f"  {a:>6}  {allv:9.4f}  {tail:9.4f}")
        deltas = [tails[i + 1] - tails[i] for i in range(len(tails) - 1)]
        if all(d < 0 for d in deltas):
            print("  => MAE đuôi giảm đơn điệu. Chữ 'monotonically' ở IV-F giữ nguyên được.")
        else:
            up = [have[i + 1][0] for i, d in enumerate(deltas) if d >= 0]
            print(f"  => KHÔNG đơn điệu — đảo chiều tại alpha = {', '.join(up)}.")
            print("     PHẢI sửa chữ 'monotonically' ở IV-F. Đây là phát hiện,")
            print("     không phải lỗi: nó cho thấy có điểm bão hoà/gãy.")

    print("\nChi tiết còn lại: notes/checklist-cap-nhat-sau-5fold.md")
    return 0


# ----------------------------------------------------------------- mseed ----
def mseed(outdir="results_mseed"):
    """Tổng hợp chiến dịch multi-seed: mean +- SD theo từng cấu hình.

    Cột SD mới là thứ bài báo cần: nếu SD giữa các seed lớn hơn khoảng cách
    giữa hai cấu hình thì hai cấu hình đó KHÔNG phân biệt được, dù bảng một-seed
    trông như có thắng thua rõ ràng.
    """
    import glob
    import re as _re
    if not os.path.isdir(outdir):
        print(f"Chưa có {outdir}/ — chiến dịch multi-seed chưa chạy hoặc chưa kéo về.")
        return 1

    runs = {}
    for f in sorted(glob.glob(os.path.join(outdir, "*_best.json"))):
        base = os.path.basename(f)[: -len("_best.json")]
        m = _re.search(r"_seed(\d+)$", base)
        seed = int(m.group(1)) if m else 42
        cfg = base[: m.start()] if m else base
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        runs.setdefault(cfg, {})[seed] = d

    if not runs:
        print(f"{outdir}/ chưa có file *_best.json nào.")
        return 1

    def stat(xs):
        n = len(xs)
        mu = sum(xs) / n
        if n < 2:
            return mu, float("nan")
        var = sum((x - mu) ** 2 for x in xs) / (n - 1)
        return mu, var ** 0.5

    print(f"{'cấu hình':<22} {'seed':>4}  {'RMSE mean±SD':>20}  {'MAE mean±SD':>20}")
    print("-" * 72)
    order = sorted(runs, key=lambda c: (("tail" in c), c))
    for cfg in order:
        seeds = runs[cfg]
        rm = [d["test"]["rmse"] for d in seeds.values() if "test" in d]
        ma = [d["test"]["mae"] for d in seeds.values() if "test" in d]
        if not rm:
            continue
        r_mu, r_sd = stat(rm)
        m_mu, m_sd = stat(ma)
        print(f"{cfg:<22} {len(seeds):>4}  {r_mu:>10.4f} ± {r_sd:<7.4f}  "
              f"{m_mu:>10.4f} ± {m_sd:<7.4f}")

    # So sánh từng cấu hình với GA-RMSE bằng Welch t-test.
    #
    # BẪY ĐÃ MẮC: bản đầu lấy SD của RIÊNG ga_rmse làm thước đo nhiễu cho mọi
    # so sánh. GA-RMSE hội tụ rất ổn định (SD ~0.0004) nên thước đo đó bé xíu,
    # và mọi cấu hình khác đều bị tuyên là "phân biệt được" — kể cả những cái
    # có SD của chính nó lớn gấp 50 lần. Phải dùng SD của CẢ HAI phía.
    base = "ga_rmse"
    if base in runs and len(runs[base]) >= 2:
        try:
            from scipy import stats
        except ImportError:
            stats = None
        b = [d["test"]["rmse"] for d in runs[base].values() if "test" in d]
        print(f"\nSo với {base} (test RMSE, Welch t-test):")
        print(f"  {'cấu hình':<20}{'Δ mean':>9}{'p':>9}  kết luận")
        for cfg in order:
            if cfg == base:
                continue
            xs = [d["test"]["rmse"] for d in runs[cfg].values() if "test" in d]
            if len(xs) < 2:
                continue
            diff = sum(xs) / len(xs) - sum(b) / len(b)
            if stats is not None:
                _, pv = stats.ttest_ind(xs, b, equal_var=False)
            else:
                pv = float("nan")
            # Khoảng giá trị có chồng nhau không — thông tin mộc, không giả định gì
            overlap = not (max(xs) < min(b) or min(xs) > max(b))
            if pv == pv and pv < 0.05 and not overlap:
                verd = "khác biệt rõ"
            elif overlap:
                verd = "KHOẢNG GIÁ TRỊ CHỒNG NHAU — không kết luận được"
            else:
                verd = "tách nhau nhưng p chưa đủ nhỏ (ít seed)"
            print(f"  {cfg:<20}{diff:>+9.4f}{pv:>9.4f}  {verd}")
        print("  Lưu ý: n = 3–5 nên lực kiểm định thấp. 'Không kết luận được'")
        print("  nghĩa là CHƯA ĐỦ BẰNG CHỨNG, không phải 'đã chứng minh là như nhau'.")

    n_missing = 40 - sum(len(v) for v in runs.values())
    if n_missing > 0:
        print(f"\n(còn {n_missing} lượt chạy chưa xong)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true", help="tiến độ chạy")
    g.add_argument("--summary", action="store_true", help="bảng so sánh kết quả")
    g.add_argument("--paper", action="store_true", help="số mới + chỗ cần sửa trong bài")
    g.add_argument("--mseed", action="store_true", help="mean±SD của chiến dịch multi-seed")
    ap.add_argument("--log", default="run_5fold.log", help="tên file log trong results/")
    a = ap.parse_args()

    if a.summary:
        return summary()
    if a.paper:
        return paper()
    if a.mseed:
        return mseed()
    return status(a.log)


if __name__ == "__main__":
    sys.exit(main())
