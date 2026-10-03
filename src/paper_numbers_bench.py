# -*- coding: utf-8 -*-
"""Gom các con số E11 (lặp lại trên dữ liệu công khai) mà bài sẽ trích.

Đầu vào: results_bench/bench_decomp.json (E11) và results_cost/paper_numbers_cost.json
(chỉ để lấy số tham chiếu HSA đã dùng trong bài). Đầu ra:
    results_bench/paper_numbers_bench.json   từ điển PHẲNG khoá -> giá trị (+ "_meta")
    results_bench/paper_numbers_bench.md     cùng số đó, dạng bảng, nhãn tiếng Anh

Vì sao file riêng thay vì nối vào paper_numbers_cost.py: E11 chạy ở commit khác
(7522d7c, dirty vì hai file chưa theo dõi) và ghi vào results_bench/, không phải
results_cost/. Gộp vào một sổ sẽ làm "_meta.run_commit" của sổ kia sai cho một phần số.

Giống paper_numbers_cost.py, script KHÔNG tính lại p, CI, Holm hay cổng: mọi thống kê
lấy nguyên từ JSON thí nghiệm (tính một lần trên server từ code đã verify). Ngoại lệ
duy nhất là khoá "derived.*": tổng trung bình C1 + C3 theo từng lần chia, bằng
cost_3(R1, rs_tuned) - cost_3(R8_bag5). Đó là phép cộng hai hiệu ghép cặp trên CÙNG
lần chia nên trung bình là chính xác, nhưng không có p (ghi rõ là mô tả). Lý do cần số
này: trên HSA, C1 đổi dấu theo trung tâm (R1 thua R8* trên default và sub1), nên khi
C1 trên dữ liệu công khai dương, câu hỏi tự nhiên là phần nào do bag10 chưa dò.

Quy ước dấu như gates.CONTRASTS: âm = vế trái tốt hơn. Cỡ hiệu ứng ở hai thang không
phụ thuộc đơn vị: chia SD(y) của từng bộ, và phần trăm cost_3(R1, bag10) của bộ đó.
SESOI 0,10 là điểm HSA nên không áp cho bộ khác (bench_decomp lựa chọn 7).

Không đọc data/data_final.csv. Chạy:  python3 src/paper_numbers_bench.py
"""
import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_numbers_cost import Book, fmt   # noqa: E402  (cùng sổ, cùng định dạng)

SRC = "results_bench/bench_decomp.json"
HSA = "results_cost/paper_numbers_cost.json"
OUT_JSON = "results_bench/paper_numbers_bench.json"
OUT_MD = "results_bench/paper_numbers_bench.md"

CONTRASTS = ["C1", "C2", "C3"]
E4 = [("b-a", "b_minus_a"), ("c+-a", "cplus_minus_a"), ("c-a", "c_minus_a"), ("o-a", "o_minus_a")]
SECONDARY = ["R2-R1", "ii", "ii_a", "ii_b", "bag10-default", "bag10-sub1"]
LABEL = {
    "california_housing": "california_housing", "diamonds": "diamonds",
    "kings_county": "kings_county", "cps88wages": "cps88wages",
    "saber": "Saber 11 to Saber Pro", "student_performance_por": "student_performance_por",
}


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def add_effect(b, key, x, label):
    """Một phép so: trung bình, CI, p (và Holm nếu có), hai cỡ chuẩn hoá, đếm lần chia."""
    nb = x["nb"]
    b.nb(key, nb, label)
    b.add(f"{key}.sd_units", x.get("std_by_sd_y"), f"{label}: mean / SD(y)")
    b.add(f"{key}.pct", x.get("pct_of_cost3_R1_bag10"), f"{label}: mean as % of cost_3(R1, bag10)")
    if "n_pos" in x:
        b.add(f"{key}.n_pos", x["n_pos"], f"{label}: splits with positive difference (of {nb['n']})")
    holm = x.get("p_holm_within_dataset", x.get("p_holm"))
    if holm is not None:
        b.add(f"{key}.p_holm", holm, f"{label}: Holm p within the dataset's family")
    if x.get("boot_n"):
        b.add(f"{key}.boot_excl0", x.get("boot_splits_excluding_zero"),
              f"{label}: splits whose cluster bootstrap CI excludes zero (of {x['boot_n']})")


def add_dataset(b, ds, v):
    s, info = v["summary"], v["info"]
    lab = LABEL[ds]
    b.section(f"E11 dataset: {lab}")
    p = f"bench.{ds}"
    b.add(f"{p}.core", bool(info.get("core")), f"{lab}: counts toward the 4/5 gate")
    b.add(f"{p}.domain", info.get("domain"), f"{lab}: domain")
    b.add(f"{p}.n", info["n"], f"{lab}: records analyzed")
    b.add(f"{p}.n_features", info["n_features"], f"{lab}: features")
    b.add(f"{p}.n_clusters", info.get("n_clusters"), f"{lab}: bootstrap clusters (None: rows)")
    b.add(f"{p}.y_sd", s["sd_y"], f"{lab}: SD(y)")
    b.add(f"{p}.y_mean", info.get("y_mean"), f"{lab}: mean(y)")
    b.add(f"{p}.n_splits", s["n_splits"], f"{lab}: complete splits")
    b.ms(f"{p}.tail_lo", s["tails"]["lo"], f"{lab}: low-tail cutoff")
    b.ms(f"{p}.tail_hi", s["tails"]["hi"], f"{lab}: high-tail cutoff")
    b.add(f"{p}.cost3_R1_bag10", s["cost3_R1_bag10"], f"{lab}: cost_3(R1, bag10), mean over splits")
    for cen in ("default", "bag10", "rs_tuned"):
        c = s["centers"][cen]
        b.ms(f"{p}.{cen}.r2", c["r2"], f"{lab}: test R² of {cen}")
        b.ms(f"{p}.{cen}.rmse", c["all_rmse"], f"{lab}: test RMSE of {cen}")
        b.ms(f"{p}.{cen}.calib_slope", c["calib_slope"], f"{lab}: calibration slope of {cen}")
        b.ms(f"{p}.{cen}.sd_ratio", c["sd_ratio"], f"{lab}: SD(ŷ)/SD(y) of {cen}")
    b.add(f"{p}.r5_edge_K3", s.get("r5_edge_K3"), f"{lab}: R5 stretch factor at grid edge, K = 3")
    b.ms(f"{p}.r8.kish", s["r8"]["kish_ratio"], f"{lab}: Kish ratio of R8 weights")
    b.add(f"{p}.r8.capped_fold_fits", s["r8"].get("capped_fold_fits"), f"{lab}: R8 fold fits hitting the tree cap")
    for c in CONTRASTS:
        add_effect(b, f"{p}.{c}", s["contrasts"][c], f"{lab} {c}")
    for raw, key in E4:
        add_effect(b, f"{p}.e4.{key}", s["e4"]["contrasts"][raw], f"{lab} E4 ({raw[:-2]})−(a)")
    b.add(f"{p}.e4.policy_a", s["e4"]["policies"]["a"]["mean"], f"{lab}: test cost_3 of policy (a)")
    b.add(f"{p}.e4.kendall_all", s["e4"]["rank_corr"]["kendall_all"]["mean"],
          f"{lab}: Kendall τ, OOF RMSE vs post-rule OOF cost_3 rankings")
    for k in SECONDARY:
        add_effect(b, f"{p}.sec.{k.replace('-', '_minus_')}", s["secondary"][k], f"{lab} {k}")

    # Mô tả, không có p: R1 trên trung tâm đã dò so với R8* (= C1 + C3 theo từng lần chia).
    d1, d3 = s["contrasts"]["C1"]["diffs"], s["contrasts"]["C3"]["diffs"]
    if len(d1) == len(d3):
        dd = [a + c for a, c in zip(d1, d3)]
        m = sum(dd) / len(dd)
        b.add(f"{p}.derived.R1rs_minus_R8star.mean", m,
              f"{lab}: cost_3(R1, rs_tuned) − cost_3(R8*) = C1 + C3, mean (descriptive, no p)")
        b.add(f"{p}.derived.R1rs_minus_R8star.pct", 100 * m / s["cost3_R1_bag10"],
              f"{lab}: same, % of cost_3(R1, bag10)")
        b.add(f"{p}.derived.R1rs_minus_R8star.n_neg", sum(x < 0 for x in dd),
              f"{lab}: same, splits where R1 on rs_tuned is better (of {len(dd)})")
    else:
        b.note(f"{ds}: C1 và C3 có số lần chia khác nhau, bỏ khoá derived")


def add_signs(b, d):
    b.section("E11 sign agreement with HSA and the gate")
    for c, x in d["sign_counts"].items():
        key = "sign." + c.replace("E4 ", "e4.").replace("-", "_minus_").replace("+", "plus")
        b.add(f"{key}.hsa_mean", x["hsa_mean"], f"{c}: HSA mean (reference)")
        b.add(f"{key}.hsa_sign", x["hsa_sign"], f"{c}: HSA sign")
        b.add(f"{key}.core_same", x["core_complete"]["same_sign"],
              f"{c}: core datasets with the HSA sign (of {x['core_complete']['n']})")
        b.add(f"{key}.core_same_p05", x["core_complete"]["same_sign_p05"],
              f"{c}: core datasets with the HSA sign and p < 0.05")
        b.add(f"{key}.all_same", x["all_run"]["same_sign"],
              f"{c}: all datasets with the HSA sign (of {x['all_run']['n']})")
        # Cùng dấu mà p < 0,05 chỉ đếm khi cùng dấu; đếm thêm số bộ NGƯỢC dấu có p < 0,05.
        opp = [ds for ds, r in x["per_dataset"].items()
               if r["core"] and not r["same_sign"] and r["p"] is not None and r["p"] < 0.05]
        b.add(f"{key}.core_opposite_p05", len(opp), f"{c}: core datasets with the opposite sign and p < 0.05")
        b.add(f"{key}.core_opposite_p05_names", ", ".join(opp) or "none", f"{c}: their names")
    g = d["gate"]
    b.add("gate.complete", g["complete"], "E11 gate: all core datasets complete")
    b.add("gate.threshold", g["threshold"], "E11 gate: threshold")
    for c in CONTRASTS:
        b.add(f"gate.count.{c}", g["counts"][c], f"E11 gate: core datasets sharing the HSA sign, {c}")
        b.add(f"gate.not_replicating.{c}", ", ".join(g["not_replicating"][c]) or "none",
              f"E11 gate: core datasets not sharing the HSA sign, {c}")
    b.add("gate.replicates", g["replicates"], "E11 gate: ordering replicated")
    b.add("gate.verdict", "appendix: boundary-condition table" if not g["replicates"] else "replicated",
          f"E11 gate verdict (JSON: {g['verdict']})")


def add_hsa(b, h):
    """Số HSA dùng làm hàng tham chiếu trong bảng E11, ở cùng hai thang chuẩn hoá."""
    b.section("HSA reference on the E11 scales (from paper_numbers_cost.json)")
    sd = h["data.y_sd"]
    base = h["e1.center.F_dt-cn.bag20.cost3.R1"]
    b.add("hsa.y_sd", sd, "HSA: SD(y)")
    b.add("hsa.cost3_R1_bag20", base, "HSA: cost_3(R1, bag20), primary center")
    b.add("hsa.r2_bag20", h.get("e1.center.F_dt-cn.bag20.r2"), "HSA: test R² of bag20")
    for c in ("c1", "c2", "c3"):
        m = h[f"{c}.mean"]
        b.add(f"hsa.{c}.mean", m, f"HSA {c.upper()}: mean")
        b.add(f"hsa.{c}.p", h[f"{c}.p"], f"HSA {c.upper()}: p (NB)")
        b.add(f"hsa.{c}.sd_units", m / sd, f"HSA {c.upper()}: mean / SD(y)")
        b.add(f"hsa.{c}.pct", 100 * m / base, f"HSA {c.upper()}: % of cost_3(R1, bag20)")
    for _, key in E4:
        m = h[f"e4.K3.{key}.mean"]
        b.add(f"hsa.e4.{key}.mean", m, f"HSA E4 {key}: mean")
        b.add(f"hsa.e4.{key}.sd_units", m / sd, f"HSA E4 {key}: mean / SD(y)")
        b.add(f"hsa.e4.{key}.pct", 100 * m / base, f"HSA E4 {key}: % of cost_3(R1, bag20)")
    m = h["c1.mean"] + h["c3.mean"]
    b.add("hsa.derived.R1rs_minus_R8star.mean", m, "HSA: C1 + C3 = cost_3(R1, rs_tuned) − cost_3(R8*), descriptive")
    b.add("hsa.derived.R1rs_minus_R8star.pct", 100 * m / base, "HSA: same, % of cost_3(R1, bag20)")


def check(b, d, h):
    """Ghi lại chỗ số HSA mà bench_decomp đọc lệch với số bài đang dùng."""
    ref = d["hsa_reference"]["means"]
    pairs = {"C1": "c1.mean", "C2": "c2.mean", "C3": "c3.mean", "E4 b-a": "e4.K3.b_minus_a.mean",
             "E4 c+-a": "e4.K3.cplus_minus_a.mean", "E4 c-a": "e4.K3.c_minus_a.mean",
             "E4 o-a": "e4.K3.o_minus_a.mean"}
    for c, k in pairs.items():
        if abs(ref[c] - h[k]) > 1e-9:
            b.note(f"HSA {c}: bench_decomp đọc {ref[c]:.6f}, paper_numbers_cost có {h[k]:.6f}")
    prov = d["meta"].get("provenance", {})
    if prov.get("dirty"):
        b.note(f"E11 chạy ở commit {prov.get('commit', '?')[:7]} có cờ dirty "
               f"(diff_sha256 rỗng; file chưa theo dõi: {', '.join(prov.get('untracked_sha256', {}))})")
    if not prov.get("code_matches_stamp", True) or prov.get("code_changed_since_start"):
        b.note("E11: mã đổi so với dấu khởi chạy")


def write_md(b, meta):
    V = b.vals
    core = [ds for ds in LABEL if V.get(f"bench.{ds}.core")]
    rows = ["| Dataset | n | R² bag10 | C1 | C2 | C3 | (b)−(a) | (c+)−(a) | (c)−(a) | (o)−(a) | C1+C3 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(k):
        pct, p = V.get(f"{k}.pct"), V.get(f"{k}.p")
        return f"{pct:+.2f}% (p {fmt(p, 'x.p')})" if pct is not None else "n/a"
    for ds in LABEL:
        p = f"bench.{ds}"
        rows.append(f"| {LABEL[ds]}{'' if V[p + '.core'] else ' (secondary)'} | {V[p + '.n']:,} | "
                    f"{V[p + '.bag10.r2']:.3f} | " + " | ".join(cell(f"{p}.{c}") for c in CONTRASTS) + " | "
                    + " | ".join(cell(f"{p}.e4.{k}") for _, k in E4)
                    + f" | {V[p + '.derived.R1rs_minus_R8star.pct']:+.2f}% |")
    lines = ["# Paper numbers, E11 replication on public data", "",
             f"Generated {meta['generated']} by `src/paper_numbers_bench.py` from `{SRC}` "
             f"(run commit {meta['run_commit']}). Means over 10 outer 80/20 splits (seeds 100-109); "
             "p: Nadeau-Bengio corrected t (df = 9), not pooled across datasets. Effects as % of "
             "cost_3(R1, bag10) of each dataset; negative favors the left side. Core datasets: "
             + ", ".join(core) + ".", "", "## Overview (K = 3)", ""] + rows + [""]
    if b.notes:
        lines += ["## Consistency notes", ""] + [f"- {n}" for n in b.notes] + [""]
    lines += ["## All numbers", ""]
    for sec in b.sections:
        keys = [k for k in b.vals if b.section_of[k] == sec]
        lines += [f"### {sec}", "", "| Quantity | Value | Key |", "|---|---|---|"]
        for k in keys:
            lines.append(f"| {b.labels[k].replace('|', '/')} | {fmt(b.vals[k], k).replace('|', '/')} | `{k}` |")
        lines.append("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    d, sha = load(SRC)
    h, sha_h = load(HSA)
    b = Book()
    for ds in LABEL:
        add_dataset(b, ds, d["datasets"][ds])
    add_signs(b, d)
    add_hsa(b, h)
    check(b, d, h)
    prov = d["meta"].get("provenance", {})
    meta = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "script": "src/paper_numbers_bench.py",
        "run_commit": prov.get("commit", "?")[:7] + (" (dirty)" if prov.get("dirty") else ""),
        "run_launched": prov.get("launched"),
        "source_sha256": {SRC: sha, HSA: sha_h},
        "consistency_notes": b.notes,
        "n_numbers": len(b.vals),
    }
    out = {"_meta": meta}
    out.update(b.vals)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    write_md(b, meta)
    print(f"{len(b.vals)} numbers -> {OUT_JSON}, {OUT_MD}")
    with open(OUT_MD, encoding="utf-8") as f:
        md = f.read()
    print(md[md.index("## Overview"):md.index("## All numbers")])


if __name__ == "__main__":
    sys.exit(main())
