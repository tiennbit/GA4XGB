# -*- coding: utf-8 -*-
"""Gom mọi con số bài cost-aware sẽ trích từ các JSON khẳng định (E0b đến E10).

Đầu vào: results_cost/{data_audit, bench_eval, decomp_centers, decomp_rules,
wtrain_tuned, select_policy, proper_scores, sensitivity, sim_decomp, use_validity,
group_audit}.json. Đầu ra:
    results_cost/paper_numbers_cost.json   từ điển PHẲNG khoá -> giá trị (+ "_meta")
    results_cost/paper_numbers_cost.md     cùng số đó, dạng bảng, nhãn tiếng Anh
và in một bản tóm tắt ngắn ra màn hình.

Vì sao một từ điển phẳng: bài trích số theo tên (vd. "c1.mean", "decomp.K3.ii_b.mean")
và docs/ledger.tsv nối mỗi tên với JSON nguồn. Khi số trong bài phải truy về nguồn,
một khoá phẳng tra được bằng grep; một cây lồng nhau thì không.

Vì sao script không tự tính lại thống kê: mọi p, CI, Holm và cổng đã được tính MỘT
lần trong script thí nghiệm, trên server, từ code đã verify. Tính lại ở đây (dù cùng
công thức) sẽ tạo bản sao thứ hai có thể lệch. Script này chỉ đọc, đổi tên, làm tròn
cho bảng md, và ghi lại chỗ các JSON không khớp nhau (mục "consistency").

Hai chỗ phải chọn nguồn vì JSON trước được ghi khi họ phép so chính chưa đủ:
- Holm của C1, C2, C3 lấy từ decomp_rules.summary.contrasts.primary_holm (lượt
  decomp_rules_c1, sau E2b). decomp_centers.summary.C3.primary_holm_provisional coi
  C1, C2 là p = 1 nên lớn hơn (0,019 so với 0,006); số đó không dùng.
- C1 ở K = 3 lấy từ decomp_rules.summary.contrasts.C1. wtrain_tuned.summary.C1 tính
  cho mọi trung tâm khi chưa biết trung tâm chính; chỉ dùng cho K thứ cấp và bảng phụ.

Không đọc data/data_final.csv. Chạy:  python3 src/paper_numbers_cost.py
"""
import datetime
import hashlib
import json
import math
import os
import sys

RES = "results_cost"
OUT_JSON = os.path.join(RES, "paper_numbers_cost.json")
OUT_MD = os.path.join(RES, "paper_numbers_cost.md")

SOURCES = ["data_audit", "bench_eval", "decomp_centers", "decomp_rules",
           "wtrain_tuned", "select_policy", "proper_scores", "sensitivity",
           "sim_decomp", "use_validity", "group_audit"]

K_REPORT = ["2", "3", "5", "8"]
K_ALL = ["1", "2", "3", "5", "8"]

CENTER_LABEL = {
    "default": "XGBoost default", "sub1": "single subsampled model (sub1)",
    "bag1": "bag of 1", "bag2": "bag of 2", "bag5": "bag of 5", "bag10": "bag of 10",
    "bag20": "bag of 20 (B*)", "bag40": "bag of 40",
    "rs_tuned": "random-search tuned (60 configs)", "rs_tuned_bag5": "tuned, bag of 5",
}
RULE_LABEL = {
    "R0": "raw prediction", "R1": "Bayes rule on OOF residual bins",
    "R1_1": "R1 with w=1 (recalibration only)", "R2": "weighted isotonic",
    "R3": "weighted histogram", "R4": "weighted linear", "R5": "two-sided stretch",
    "R6": "quantile mapping", "R7": "normal reframing",
    "R8": "weighted training, retuned", "R8_bag5": "weighted training, retuned, bag of 5",
    "R8*": "weighted training, retuned, bag of 5",
}
VERDICT_EN = {
    "trong biên": "within margin", "chưa xác định": "undetermined",
    "vượt biên": "exceeds margin", "có chênh, vượt biên": "differs, exceeds margin",
    "có chênh, trong biên": "differs, within margin",
}
REGION_EN = {"Low tail": "low", "Middle": "mid", "High tail": "high",
             "Tails": "tails", "All": "all"}


def load(name):
    path = os.path.join(RES, f"{name}.json")
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def fnum(x):
    """NaN/inf -> None để JSON hợp lệ; số nguyên giữ nguyên."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


class Book:
    """Sổ số liệu: khoá phẳng -> giá trị, kèm nhãn tiếng Anh và mục cho bảng md."""

    def __init__(self):
        self.vals, self.labels, self.section_of = {}, {}, {}
        self.sections = []
        self.cur = None
        self.notes = []          # các chỗ không khớp giữa JSON / log

    def section(self, title):
        self.cur = title
        if title not in self.sections:
            self.sections.append(title)

    def add(self, key, value, label):
        if key in self.vals:
            raise KeyError(f"khoá trùng: {key}")
        self.vals[key] = fnum(value)
        self.labels[key] = label
        self.section_of[key] = self.cur

    def nb(self, key, d, label):
        """Thống kê Nadeau-Bengio: trung bình, CI 95%, p, số lần chia thắng."""
        if not d:
            return
        self.add(f"{key}.mean", d.get("mean"), f"{label}: mean paired difference")
        self.add(f"{key}.ci_lo", d.get("ci_lo"), f"{label}: 95% CI lower (NB-corrected)")
        self.add(f"{key}.ci_hi", d.get("ci_hi"), f"{label}: 95% CI upper (NB-corrected)")
        self.add(f"{key}.p", d.get("p"), f"{label}: p (NB-corrected t, df={d.get('df')})")
        self.add(f"{key}.wins", d.get("wins"), f"{label}: splits with negative difference (of {d.get('n')})")

    def tost(self, key, d, label):
        if not d:
            return
        self.add(f"{key}.tost_p", d.get("p"), f"{label}: TOST p (margin ±{d.get('margin')})")
        self.add(f"{key}.tost_passed", d.get("passed"), f"{label}: TOST equivalence passed")
        ci90 = d.get("ci90") or [None, None]
        self.add(f"{key}.ci90_lo", ci90[0], f"{label}: 90% CI lower (TOST)")
        self.add(f"{key}.ci90_hi", ci90[1], f"{label}: 90% CI upper (TOST)")

    def ms(self, key, d, label):
        """Trung bình (và SD) qua lần chia của một dict {mean, sd, n}."""
        if d is None:
            return
        self.add(f"{key}", d.get("mean"), f"{label} (mean over splits)")
        if d.get("sd") is not None:
            self.add(f"{key}_sd", d.get("sd"), f"{label} (SD over splits)")

    def note(self, text):
        self.notes.append(text)


# ---------------------------------------------------------------------------
# E0b, E0: dữ liệu, luồng mẫu, đo thời gian
# ---------------------------------------------------------------------------
def add_data(b, da, be, dc):
    b.section("Dataset and sample flow (E0b, Table I)")
    t = da["target"]
    b.add("data.n_records", t["n"], "Records in the file (one HSA score each)")
    b.add("data.n_removed", da["sample_flow"]["n_removed"], "Records removed by preprocessing")
    b.add("data.n_final", da["sample_flow"]["n_final"], "Records analysed")
    b.add("data.y_mean", t["desc"]["mean"], "HSA score: mean")
    b.add("data.y_sd", t["desc"]["sd"], "HSA score: SD")
    b.add("data.y_min", t["min"], "HSA score: minimum")
    b.add("data.y_max", t["max"], "HSA score: maximum")
    for q in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"):
        b.add(f"data.y_{q}", t["desc"]["quantiles"][q], f"HSA score: {q[1:]}th percentile")
    r = t["regions_60_100"]
    b.add("data.n_low_tail", r["n_low"], "Records with y < 60 (low tail)")
    b.add("data.n_high_tail", r["n_high"], "Records with y >= 100 (high tail)")
    b.add("data.mass_low_tail", r["mass_low"], "Low-tail mass (share of records)")
    b.add("data.mass_high_tail", r["mass_high"], "High-tail mass (share of records)")
    b.add("data.mass_tails", r["mass_low"] + r["mass_high"], "Combined tail mass")
    b.add("data.mass_cutoffs_equal_60_100", t["mass_cutoffs_all"]["equals_60_100"],
          "Mass-defined tails (9.25%/5.72%) coincide with 60/100 on the full cohort")
    si = dc["splits_info"]
    lows = [v["mass_low_tr"] for v in si.values()]
    highs = [v["mass_high_tr"] for v in si.values()]
    b.add("data.n_train", si["100"]["n_tr"], "Training records per split (80%)")
    b.add("data.n_test", si["100"]["n_te"], "Test records per split (20%)")
    b.add("data.n_splits", len(si), "Outer splits (seeds 100-109)")
    b.add("data.train_mass_low_min", min(lows), "Low-tail mass in training sets: min over splits")
    b.add("data.train_mass_low_max", max(lows), "Low-tail mass in training sets: max over splits")
    b.add("data.train_mass_high_min", min(highs), "High-tail mass in training sets: min over splits")
    b.add("data.train_mass_high_max", max(highs), "High-tail mass in training sets: max over splits")
    for rel, sem in t["sem_assumed"]["by_reliability"].items():
        b.add(f"data.sem_rel{rel}", sem, f"SEM if reliability = {rel} (assumed)")
    b.add("data.sem_assumed", t["sem_assumed"]["sem_gates"], "SEM used for % SEM conversions (assumed)")

    d = da["duplicates"]
    for k in ("exact", "near", "full_row"):
        b.add(f"data.dup_{k}_groups", d[k]["n_groups"], f"Duplicate groups ({k.replace('_', ' ')} key)")
    b.add("data.stt_is_1_to_n", d["stt"]["is_1_to_n"], "Row index STT runs 1..n without resets")
    ng = da["near_guess"]
    b.add("data.near_guess_threshold", ng["threshold"], "Near-guessing region: y <= this score")
    b.add("data.n_near_guess", ng["n"], "Records with y <= 37 (near guessing)")
    b.add("data.frac_near_guess", ng["frac"], "Share of records with y <= 37")
    for kv, v in ng["by_region"].items():
        b.add(f"data.near_guess_rate_{kv}", v["rate"], f"Near-guess rate in priority area {kv}")
    b.add("data.near_guess_gpa11_mean", ng["gpa_11_cn"]["near_guess"]["mean"],
          "Grade-11 GPA of near-guess records: mean")
    b.add("data.all_gpa11_mean", ng["gpa_11_cn"]["all"]["mean"], "Grade-11 GPA of all records: mean")

    sf = da["sample_flow"]
    for fs, n in sf["n_features"].items():
        b.add(f"data.n_features.{fs}", n, f"Number of features in {fs}")
    for st in sf["steps"]:
        if "out of" in st["step"] or "ngoài [0; 10]" in st["step"]:
            b.add("data.n_gradebook_out_of_range", st["n_affected"],
                  "Gradebook cells outside [0, 10] (kept)")
    s = sf["schools"]
    b.add("data.n_schools", s["n_school_key_prov_name"], "Schools (key: province, name)")
    b.add("data.n_school_names", s["n_school_names_bare"], "Distinct bare school names")
    b.add("data.n_provinces", s["n_provinces"], "Provinces")
    b.add("data.n_priority_areas", s["n_region_labels"], "Priority-area labels (khuVuc)")
    b.add("data.school_size_median", s["school_size"]["desc"]["quantiles"]["p50"], "School size: median records")
    b.add("data.school_size_max", s["school_size"]["desc"]["quantiles"]["p100"], "School size: max records")
    b.add("data.n_schools_size1", s["school_size"]["n_size_1"], "Schools with a single record")
    b.add("data.n_specialized_schools", s["chuyen"]["n_schools"], "Specialized (chuyen) schools, string match")
    b.add("data.n_specialized_rows", s["chuyen"]["n_rows"], "Records from specialized schools")
    by = sf["birth_year"]
    b.add("data.birth_2006_share", by.get("2006", 0) / t["n"], "Share born in 2006")
    hdr = da["header"]
    for k in ("has_candidate_id", "has_round", "has_exam_date", "has_test_form", "has_section_scores"):
        b.add(f"data.header_{k}", hdr[k], f"Header contains {k[4:].replace('_', ' ')}")
    g = da["gates"]
    b.add("gate.E0b.group_split_required", g["group_split_required"], "E0b gate: grouped splitting required")
    b.add("gate.E0b.stop_E1", g["stop_E1_until_IDT"], "E0b gate: stop until IDT answers")

    b.section("Compute benchmark (E0)")
    s = be["summary"]
    b.add("bench.fit_machine_s", s["fit_machine_s"], "Machine-seconds per in-fold fit with early stopping")
    b.add("bench.fit_threshold_s", be["gate"]["threshold_s"], "Threshold that would cut the tuning budget")
    b.add("bench.tune_budget", be["gate"]["tune_budget"], "Tuning budget kept (configurations)")
    b.add("bench.trees_best_median", s["trees_best"]["median"], "Best iteration: median over benchmark fits")
    b.add("bench.frac_hit_cap", s["frac_hit_cap"], "Share of benchmark fits hitting 3,000 trees")


# ---------------------------------------------------------------------------
# E1: tập đặc trưng, trung tâm, B*, C3, G1
# ---------------------------------------------------------------------------
def add_centers(b, dc, dr):
    s = dc["summary"]
    b.section("Feature-set gate and timing constraint (E1)")
    fg = s["feature_gate"]
    b.add("e1.primary_feature_set", fg["primary"], "Primary feature set chosen by the gate")
    for st in fg["steps"]:
        tag = "cn_vs_dt" if st["compare"].startswith("F_dt-cn -") else "cnbc_vs_cn"
        lab = f"RMSE(bag10) {st['compare']}"
        b.nb(f"e1.gate.{tag}", st["nb"], lab)
        b.tost(f"e1.gate.{tag}", st["tost"], lab)
    for c, v in s["timing_constraint"].items():
        if not isinstance(v, dict):
            continue
        b.nb(f"e1.timing.{c}.drmse_Fdt_minus_Ffull", v["d_rmse_Fdt_minus_Ffull"],
             f"RMSE F_dt - F_full ({c})")
        b.ms(f"e1.timing.{c}.r2_loss", v["r2_loss_Ffull_minus_Fdt"], f"R2 lost by dropping late grade-12 columns ({c})")
    b.add("e1.timing.report_both", s["timing_constraint"]["report_both"],
          "R2 loss >= 0.02, so Table III must report F_full and F_dt")

    b.section("Centers (E1, Table III)")
    for fs, cents in s["centers"].items():
        for c, v in cents.items():
            if fs != "F_dt-cn" and c not in ("default", "bag10", "bag20", "sub1"):
                continue
            k = f"e1.center.{fs}.{c}"
            lab = f"{CENTER_LABEL.get(c, c)} on {fs}"
            for reg in ("Low tail", "Middle", "High tail", "Tails"):
                b.ms(f"{k}.rmse_{REGION_EN[reg]}", v[f"rmse_{reg}"], f"{lab}: RMSE {REGION_EN[reg]}")
            b.ms(f"{k}.rmse_all", v["all_rmse"], f"{lab}: RMSE overall")
            b.ms(f"{k}.r2", v["r2"], f"{lab}: R2")
            b.ms(f"{k}.sd_ratio", v["sd_ratio"], f"{lab}: SD(pred)/SD(y)")
            b.ms(f"{k}.calib_slope", v["calib_slope"], f"{lab}: calibration slope (y on pred)")
            b.ms(f"{k}.oof_rmse", v["oof_rmse"], f"{lab}: OOF RMSE")
            b.ms(f"{k}.fit_s", v["fit_s"], f"{lab}: fit time, thread-seconds")
            if v["tune_s"]["mean"]:
                b.ms(f"{k}.tune_s", v["tune_s"], f"{lab}: tuning time, thread-seconds")
            for rule in ("R0", "R1", "R1_1"):
                for K in K_ALL:
                    if K == "1" and rule != "R0":
                        continue
                    b.add(f"{k}.cost{K}.{rule}", v["cost_K"][rule][K]["mean"],
                          f"{lab}: cost_{K} with {rule} ({RULE_LABEL[rule]})")

    b.section("Bag size and B* (E1)")
    bs = s["b_star"]
    b.add("e1.b_star", bs["B_star"], "B*: smallest bag with |cost_3(R1, bag B) - cost_3(R1, bag 40)| <= tol")
    b.add("e1.b_star_tol", bs["tol"], "Tolerance for B*")
    b.add("e1.b_star_feature_set", bs["feature_set"], "Feature set on which B* was determined")
    for B, v in bs["by_B"].items():
        b.add(f"e1.b_star.dev_B{B}", v["mean_abs_diff"], f"Mean |cost_3(R1, bag {B}) - bag 40| (F_dt)")

    b.section("Table III secondary contrasts (E1, cost_3 with R1)")
    for name, v in s["table_III"]["contrasts"].items():
        key = "e1.t3." + name.replace(" - ", "_minus_").replace(" ", "").replace("(", "_").replace(")", "")
        b.nb(key, v["nb"], f"cost_3(R1) {name}")
        b.add(f"{key}.p_holm", v.get("p_holm"), f"cost_3(R1) {name}: Holm p (Table III family)")
        b.add(f"{key}.tost_passed", v["tost"]["passed"], f"cost_3(R1) {name}: TOST ±0.10 passed")
        b.add(f"{key}.boot_excl0", v["cluster_boot"]["n_excl_zero"],
              f"cost_3(R1) {name}: splits whose school-cluster bootstrap CI excludes 0")

    b.section("Gate G1 (primary center)")
    g1 = s["G1"]
    b.add("gate.G1.primary_center", g1["primary_center"], "Primary center")
    b.add("gate.G1.rs_tuned_wins_C3", g1["rs_tuned_wins_C3"], "Tuning beats bag B* by >= SESOI (C3 gate)")
    b.add("gate.G1.rs_bag_wins_both", g1["rs_bag_wins_both"], "Tuned bag of 5 beats both by >= SESOI")
    for name, v in g1["rs_bag_contrasts"].items():
        key = "e1.g1." + name.replace(" - ", "_minus_")
        b.nb(key, v["nb"], f"cost_3(R1) {name}")
        b.add(f"{key}.tost_passed", v["tost"]["passed"], f"cost_3(R1) {name}: TOST ±0.10 passed")
    tr = s["trace"]
    b.add("e1.trace.n_capped_mean", tr["n_capped"]["mean"], "Configs (of 60) whose fold fits hit 3,000 trees: mean per split")
    b.add("e1.trace.tune_s_mean", tr["tune_s"]["mean"], "Tuning time for 60 configs, thread-seconds per split")
    b.add("e1.trace.refit_all_s_mean", tr["refit_all_s"]["mean"], "Refit of all 60 configs, thread-seconds per split")
    if g1.get("provisional"):
        b.note("decomp_centers.json: summary.G1.provisional = true and C3 Holm there is provisional "
               f"({s['C3']['primary_holm_provisional']['p_holm']:.4f}, C1 and C2 counted as p = 1). "
               "Final Holm for C3 is in decomp_rules.json "
               f"({dr['summary']['contrasts']['primary_holm']['C3']['p_holm']:.4f}); this file uses the final value.")


# ---------------------------------------------------------------------------
# Ba phép so chính
# ---------------------------------------------------------------------------
def add_primary(b, dr, wt):
    b.section("Primary contrasts C1-C3 at K = 3 (Table IV)")
    c = dr["summary"]["contrasts"]
    ph = c["primary_holm"]
    desc = {
        "C1": "C1 = cost_3(R1) - cost_3(R8*) on bag20 (post-hoc rule vs retuned weighted training, bag of 5)",
        "C2": "C2 = cost_3(R1) - cost_3(R5) on bag20 (Bayes rule vs two-sided stretch)",
        "C3": "C3 = cost_3(R1 on rs_tuned) - cost_3(R1 on bag20) (tuning after bagging)",
    }
    for name in ("C1", "C2", "C3"):
        v = c[name]["3"] if name == "C2" else c[name]
        k = name.lower()
        b.add(f"{k}.definition", desc[name], "Definition")
        b.nb(k, v["nb"], name)
        b.add(f"{k}.se", v["nb"]["se"], f"{name}: NB-corrected SE")
        b.add(f"{k}.p_holm", ph[name]["p_holm"], f"{name}: Holm p over {{C1, C2, C3}} (m = 3)")
        b.tost(k, v["tost"], name)
        if v.get("boot"):
            b.add(f"{k}.boot_excl0", v["boot"]["n_excl0"], f"{name}: splits whose school-cluster bootstrap CI excludes 0")
        else:
            b.add(f"{k}.boot_excl0", None, f"{name}: school-cluster bootstrap (not available: no R8 test predictions)")
        if v.get("pct_sem"):
            b.add(f"{k}.pct_sem", v["pct_sem"]["assumed_4.3"], f"{name}: as % of assumed SEM 4.3")
            b.add(f"{k}.pct_sem_rel085", v["pct_sem"]["rel_0.85"], f"{name}: as % of SEM at reliability 0.85")
            b.add(f"{k}.pct_sem_rel095", v["pct_sem"]["rel_0.95"], f"{name}: as % of SEM at reliability 0.95")
        if v.get("flag_per_1000"):
            b.add(f"{k}.flag_per_1000_lt60", v["flag_per_1000"]["60"], f"{name}: candidates per 1,000 changing flag at y<60")
            b.add(f"{k}.flag_per_1000_ge100", v["flag_per_1000"]["100"], f"{name}: candidates per 1,000 changing flag at y>=100")
    # cổng E2b cho C1
    c1 = c["C1"]["nb"]
    b.add("gate.C1.verdict", "post_hoc_better" if (c1["mean"] <= -0.10 and ph["C1"]["p_holm"] < 0.05)
          else ("equivalent" if c["C1"]["tost"]["passed"] else "inconclusive"),
          "E2b gate on C1 (mean <= -0.10 and Holm p < 0.05 -> post-hoc better)")
    b.add("gate.C1.ci_hi_above_minus_sesoi", c1["ci_hi"] > -0.10,
          "C1 95% CI upper bound lies above -0.10 (effect not shown to exceed SESOI)")
    if c1["ci_hi"] > -0.10:
        b.note(f"C1 = {c1['mean']:.3f} passes the E2b gate on the point estimate (<= -0.10, Holm p "
               f"{ph['C1']['p_holm']:.4f}) but its 95% CI [{c1['ci_lo']:.3f}, {c1['ci_hi']:.3f}] reaches "
               "above -0.10 and TOST fails: the data show post-hoc < retuned weighted training, not "
               "that the gap exceeds SESOI. No cluster bootstrap for C1 (R8 test predictions not saved).")

    b.section("C1 and C2 at secondary K (Holm within table)")
    for K in ("2", "5", "8"):
        v = c["C2"][K]
        b.nb(f"c2.K{K}", v["nb"], f"C2 at K = {K}")
        b.add(f"c2.K{K}.p_holm", v.get("p_holm"), f"C2 at K = {K}: Holm p (Table IV family)")
        b.add(f"c2.K{K}.tost_passed", v["tost"]["passed"], f"C2 at K = {K}: TOST ±0.10 passed")
    for K in ("2", "5", "8"):
        v = wt["summary"]["C1"]["bag20"][K]
        b.nb(f"c1.K{K}", v["nb"], f"C1 at K = {K}")
        b.add(f"c1.K{K}.p_holm", v.get("p_holm_table"), f"C1 at K = {K}: Holm p (E2b table)")
        b.add(f"c1.K{K}.tost_passed", v["tost"]["passed"], f"C1 at K = {K}: TOST ±0.10 passed")


# ---------------------------------------------------------------------------
# E2: phân rã, bảng quy tắc, cổng G2
# ---------------------------------------------------------------------------
DECOMP_LABEL = {
    "default_to_bag|R0": "default -> bag20, raw (baseline repair)",
    "default_to_bag|R1": "default -> bag20, with R1",
    "ii": "(ii) decision layer total = cost(R0) - cost(R1)",
    "ii_a": "(ii-a) recalibration at K = 1 = cost(R0) - cost(R1_1)",
    "ii_b": "(ii-b) tilt by K = cost(R1_1) - cost(R1)",
    "iii|R2": "(iii) R2 weighted isotonic - R1", "iii|R3": "(iii) R3 weighted histogram - R1",
    "iii|R4": "(iii) R4 weighted linear - R1", "iii|R5": "(iii) R5 two-sided stretch - R1",
    "iii|R7": "(iii) R7 normal reframing - R1", "iii|R8*": "(iii) R8* retuned weighted training - R1",
}


def add_decomp(b, dr, wt):
    s = dr["summary"]
    b.section("Decomposition on bag20 (E2, Table IV; positive = cost reduction or excess over R1)")
    for K in K_REPORT:
        dK = s["decomposition"][K]
        b.add(f"decomp.K{K}.i", dK["i"]["mean"], f"K={K}: (i) center gain after bagging (0 by definition: primary = bag B*)")
        for comp, lab in DECOMP_LABEL.items():
            v = dK.get(comp)
            if v is None:
                continue
            key = f"decomp.K{K}.{comp.replace('|', '_').replace('*', 'star')}"
            L = f"K={K}: {lab}"
            b.nb(key, v["nb"], L)
            b.add(f"{key}.p_holm", v.get("p_holm"), f"{L}: Holm p (Table IV family)")
            b.add(f"{key}.tost_passed", v["tost"]["passed"], f"{L}: TOST ±0.10 passed")
            if v.get("boot"):
                b.add(f"{key}.boot_excl0", v["boot"]["n_excl0"], f"{L}: splits with cluster-bootstrap CI excluding 0")
            if v.get("pct_sem"):
                b.add(f"{key}.pct_sem", v["pct_sem"]["assumed_4.3"], f"{L}: % of assumed SEM")
            if v.get("flag_per_1000"):
                b.add(f"{key}.flag_per_1000_lt60", v["flag_per_1000"]["60"], f"{L}: flag changes per 1,000 at y<60")
                b.add(f"{key}.flag_per_1000_ge100", v["flag_per_1000"]["100"], f"{L}: flag changes per 1,000 at y>=100")
        b.add(f"decomp.K{K}.ii_a_share", dK["ii_a"]["mean"] / dK["ii"]["mean"],
              f"K={K}: share of decision-layer gain that is K=1 recalibration")

    b.section("cost_K by estimator and K on bag20 (E2/E2b, Table V)")
    t = s["table"]["bag20"]
    for K in K_ALL:
        for rule in ("R0", "R1", "R1_1", "R2", "R3", "R4", "R5", "R7", "R6"):
            if rule == "R6" and K != "1":
                continue
            v = t[rule]["step"][K]
            b.add(f"rules.bag20.K{K}.{rule}.cost", v["cost_K"]["mean"], f"bag20, K={K}, {rule} ({RULE_LABEL[rule]}): cost_K")
            b.add(f"rules.bag20.K{K}.{rule}.cost_q025", v["cost_K"]["q025"], f"bag20, K={K}, {rule}: cost_K 2.5% over splits")
            b.add(f"rules.bag20.K{K}.{rule}.cost_q975", v["cost_K"]["q975"], f"bag20, K={K}, {rule}: cost_K 97.5% over splits")
            for reg, short in (("Low tail", "low"), ("Middle", "mid"), ("High tail", "high"), ("all", "all")):
                b.add(f"rules.bag20.K{K}.{rule}.rmse_{short}", v[reg]["mean"], f"bag20, K={K}, {rule}: RMSE {short}")
        if K != "1":
            byK = wt["summary"]["by_K"][K]
            for r8 in ("R8", "R8_bag5"):
                b.add(f"rules.bag20.K{K}.{r8}.cost", byK[r8]["cost_K"]["mean"], f"K={K}, {r8} ({RULE_LABEL[r8]}): cost_K")
                for reg, short in (("Low tail", "low"), ("Middle", "mid"), ("High tail", "high"), ("All", "all")):
                    b.add(f"rules.bag20.K{K}.{r8}.rmse_{short}", byK[r8]["region_rmse"][reg]["mean"],
                          f"K={K}, {r8}: RMSE {short}")
    for c in ("default", "rs_tuned"):
        for K in K_ALL:
            for rule in ("R0", "R1"):
                b.add(f"rules.{c}.K{K}.{rule}.cost", s["table"][c][rule]["step"][K]["cost_K"]["mean"],
                      f"{CENTER_LABEL[c]}, K={K}, {rule}: cost_K")

    b.section("Equivalence of estimators to R1 and gate G2")
    for rule, byK in s["tost_vs_R1"].items():
        for K, v in byK.items():
            b.add(f"tost_vs_R1.{rule}.K{K}.mean", v["mean"], f"{rule} - R1 at K={K}: mean")
            b.add(f"tost_vs_R1.{rule}.K{K}.passed", v["passed"], f"{rule} - R1 at K={K}: TOST ±0.10 passed")
    g2 = s["gate_G2"]
    b.add("gate.G2.verdict", g2["verdict"], "G2 verdict on C2")
    b.add("gate.G2.k_stretch_breaks", g2["k_stretch_breaks"], "Smallest K in {5, 8} where R1 - R5 <= -0.10")
    b.add("gate.G2.R2_equiv_R1_all_K", g2["R2_equiv_R1_all_K"], "R2 equivalent to R1 at all four K (recommend R2 as default)")
    b.add("gate.G2.R4_equiv_R1_all_K", g2["R4_equiv_R1_all_K"], "R4 equivalent to R1 at all four K")
    b.add("gate.G2.ii_a_share_K3", g2["ii_a_share_K3"], "(ii-a)/(ii) at K = 3")
    b.add("gate.G2.ii_a_majority", g2["ii_a_majority"], "(ii-a) more than half of (ii) at K = 3")
    for c, v in s["calibration"].items():
        b.ms(f"e2.calib.{c}.slope_oof", v["slope_oof"], f"{CENTER_LABEL[c]}: calibration slope on OOF")
        b.ms(f"e2.calib.{c}.slope_test", v["slope_test"], f"{CENTER_LABEL[c]}: calibration slope on test")
    se = s["stretch_edges"]["bag20"]["step"]
    b.add("e2.stretch_edge_hit_frac", se["edge_hit_frac"], "R5: share of (split, K) cells where s hits the grid edge")

    b.section("Asymmetric costs (E2, secondary)")
    for pair in ("5,1", "1,5", "3,1", "1,3"):
        for rule in ("R1", "R5"):
            v = t[rule]["step_pair"][pair]
            for reg, short in (("Low tail", "low"), ("Middle", "mid"), ("High tail", "high")):
                b.add(f"asym.{pair.replace(',', '_')}.{rule}.rmse_{short}", v[reg]["mean"],
                      f"(K_L,K_H)=({pair}), {rule}: RMSE {short}")


# ---------------------------------------------------------------------------
# E2b
# ---------------------------------------------------------------------------
def add_wtrain(b, wt):
    s = wt["summary"]
    b.section("Weighted training, normalisation and retuning (E2b)")
    for K in K_REPORT:
        v = s["by_K"][K]
        b.add(f"e2b.K{K}.kish_ratio", v["kish_ratio"]["mean"], f"K={K}: Kish effective sample size / n")
        b.add(f"e2b.K{K}.n_trees_mean", v["n_trees"]["mean"], f"K={K}: trees of selected R8 config, mean")
        b.add(f"e2b.K{K}.capped_share", v["capped_share"], f"K={K}: share of fold fits hitting 3,000 trees")
        b.add(f"e2b.K{K}.tune_s", v["tune_s"]["mean"], f"K={K}: retuning wall-seconds per split")
        b.nb(f"e2b.K{K}.bag5_minus_single", v["bag_minus_single"], f"K={K}: R8_bag5 - R8")
        a = s["attribution"][K]["R8_bag5"]
        for part in ("normalization", "retuning", "total"):
            b.nb(f"e2b.K{K}.attr.{part}", a[part], f"K={K}, R8_bag5: {part} effect on cost_K")
        for arm in ("step_raw", "step_norm", "R8_retuned"):
            b.add(f"e2b.K{K}.attr.cost_{arm}", a["cost"][arm]["mean"], f"K={K}, bag of 5: cost_K of {arm}")
    sig = [(K, part, s["attribution"][K]["R8_bag5"][part]["p"]) for K in K_REPORT
           for part in ("normalization", "retuning")]
    if not any(p < 0.05 for _, _, p in sig):
        b.note("E2b attribution (R8_bag5): neither normalisation nor retuning reaches p < 0.05 at any K "
               "(" + ", ".join(f"K={K} {part[:4]}. p={p:.2f}" for K, part, p in sig) + "). Retuning is the "
               "larger component at K >= 5 (point estimate), so write 'retuning accounts for most of the "
               "change, normalisation almost none' as a description, not as a tested effect.")
    cp = s["compute"]
    b.add("e2b.wall_h", cp["wall_h"], "E2b total wall-clock hours (8 cores)")
    b.add("e2b.cpu_h", cp["cpu_h"], "E2b total CPU hours")
    b.section("C1 on other centers at K = 3 (E2b, secondary)")
    for c, byK in s["C1"].items():
        if c in ("bag1",):          # bag1 trùng sub1
            continue
        v = byK["3"]
        b.add(f"e2b.C1_by_center.{c}.mean", v["nb"]["mean"], f"cost_3(R1 on {c}) - cost_3(R8*): mean")
        b.add(f"e2b.C1_by_center.{c}.p", v["nb"]["p"], f"cost_3(R1 on {c}) - cost_3(R8*): NB p")


# ---------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------
def add_proper(b, ps):
    s = ps["summary"]
    b.section("Proper scores (E3)")
    g = s["gates"]["center_forecast_gain"]
    b.add("e3.dominance_frac_informative", g["frac_theta"],
          "Share of informative theta where bag20 elementary score <= default in >= 9/10 splits")
    b.add("e3.dominance_n_theta_informative", g["n_theta_informative"], "Informative theta (non-tied)")
    b.add("e3.dominance_frac_all_theta", g["frac_theta_all_grid"], "Same share over all 101 theta (ties count as dominated)")
    b.add("gate.E3.center_forecast_gain", g["passed"], "E3 gate: bag20 dominates default at >= 90% theta")
    if not g["passed"] and g["frac_theta_all_grid"] >= g["threshold"]:
        b.note(f"E3 gate: bag20 dominates default on {g['frac_theta']:.3f} of the {g['n_theta_informative']} "
               f"informative theta (gate fails, < {g['threshold']}), but on {g['frac_theta_all_grid']:.3f} of all "
               "101 theta if ties count. The gate as coded uses informative theta; state that definition "
               "when citing 89.7%.")
    b.add("gate.E3.tilt_alarm", s["gates"]["tilt_check"]["alarm"], "E3 check: R1 at K > 1 beats R1_1 on Taggart score")
    for c in ("default", "sub1", "bag10", "bag20", "bag40", "rs_tuned", "rs_tuned_bag5"):
        b.add(f"e3.taggart.{c}", s["scores"][f"{c}|R0"]["taggart"]["mean"], f"Tail-emphasis (Taggart) score, {c}, raw")
    for K in ("2", "3", "5", "8"):
        b.add(f"e3.taggart.bag20_R1_K{K}", s["scores"][f"bag20|R1@K{K}"]["taggart"]["mean"], f"Taggart score, bag20 with R1 at K={K}")
    b.add("e3.taggart.bag20_R1_1", s["scores"]["bag20|R1_1"]["taggart"]["mean"], "Taggart score, bag20 with R1_1")
    for pair, v in s["pairs"].items():
        if not pair.startswith("bag20|R0 - ") and not pair.startswith("bag20|R1@K3") and pair != "rs_tuned|R0 - bag20|R0":
            continue
        key = "e3.pair." + pair.replace("|", "_").replace(" - ", "_minus_").replace("@", "_")
        b.nb(key, v["taggart"], f"Taggart {pair}")
        b.add(f"{key}.p_holm", v["taggart"].get("p_holm"), f"Taggart {pair}: Holm p")
        if v.get("dominance"):
            b.add(f"{key}.dominance_informative", v["dominance"]["frac_theta_informative"],
                  f"{pair}: share of informative theta dominated")
    for c in ("default", "bag20", "rs_tuned"):
        b.add(f"e3.crps.{c}", s["dist"][c]["crps"]["mean"], f"CRPS of R1 distribution, {c}")
        b.add(f"e3.twcrps.{c}", s["dist"][c]["twcrps"]["mean"], f"Tail-weighted CRPS, {c}")
    b.nb("e3.crps.bag20_minus_default", s["dist_pairs"]["bag20 - default"]["crps"], "CRPS bag20 - default")


# ---------------------------------------------------------------------------
# E4
# ---------------------------------------------------------------------------
def add_policy(b, sp):
    s = sp["summary"]
    b.section("HPO policies (E4, Table VI)")
    for K, pol in s["policies"].items():
        for p, v in pol.items():
            b.add(f"e4.K{K}.policy_{p}.cost", v["mean"], f"K={K}: test cost_K of policy ({p})")
            b.add(f"e4.K{K}.policy_{p}.same_j_as_a", v["same_j_as_a"], f"K={K}: splits where ({p}) picks the same config as (a)")
    for K, cons in s["contrasts"].items():
        for name, v in cons.items():
            key = f"e4.K{K}.{name.replace('+', 'plus').replace('-', '_minus_')}"
            b.nb(key, v["nb"], f"K={K}: ({name.replace('-', ') - (')})")
            b.add(f"{key}.p_holm", v["p_holm"], f"K={K}: {name}: Holm p (m = {s['holm_family_size']})")
            b.add(f"{key}.tost_passed", v["tost"]["passed"], f"K={K}: {name}: TOST ±0.10 passed")
    b.section("Rank correlations (E4)")
    for K, v in sp["rank_corr"].items():
        for m in ("spearman_all", "kendall_all", "spearman_top", "kendall_top", "top_overlap",
                  "kendall_centers_oof", "kendall_centers_test"):
            b.add(f"e4.rank.K{K}.{m}", v[m]["mean"], f"K={K}: {m.replace('_', ' ')} (mean over splits)")
            if m.startswith("kendall_centers") or m == "kendall_top":
                b.add(f"e4.rank.K{K}.{m}_min", v[m]["min"], f"K={K}: {m.replace('_', ' ')} (min over splits)")
    g = sp["gates"]
    b.add("gate.E4.one_tuning_suffices", g["one_tuning_suffices"], "E4 gate: one RMSE tuning suffices (TOST (b)-(a), (c+)-(a) at K in {2,3,5,8})")
    b.add("gate.E4.selection_buys_nothing", g["selection_buys_nothing"], "E4 gate: oracle - (a) > -0.10 at all K")
    b.add("gate.E4.h7_holds", g["h7_holds"], "Hypothesis 7: Kendall tau >= 0.9 at all K")
    b.section("Compute (E4)")
    ct = sp["compute_table"]
    for c, v in ct["centers"].items():
        nm = v["center"]
        b.add(f"e4.compute.{nm}.cost3_raw", v["cost3_raw"]["mean"], f"{nm}: cost_3 raw")
        b.add(f"e4.compute.{nm}.cost3_R1", v["cost3_R1"]["mean"], f"{nm}: cost_3 with R1")
        b.add(f"e4.compute.{nm}.fit_s", v["fit_s"]["mean"], f"{nm}: final fit, thread-seconds")
        b.add(f"e4.compute.{nm}.tune_s", v["tune_s"]["mean"], f"{nm}: tuning, thread-seconds")
    b.add("e4.compute.rule_fit_s_per_K", ct["e4_seconds"]["rule_all_cfg_per_K_mean"]["3"],
          "Cross-fitting R1 for 60 configs at one K, seconds per split")


# ---------------------------------------------------------------------------
# E5
# ---------------------------------------------------------------------------
def add_sensitivity(b, se):
    s = se["summary"]
    b.section("Sensitivity (E5, Table VIII)")
    for cell, byc in s["sign_keep"].items():
        for con, v in byc.items():
            if "all" not in v:      # C1 với họ prior: không có R8 tương ứng
                continue
            a = v["all"]
            key = f"e5.{cell.replace(':', '_K' if cell.startswith('step') else '_lambda')}.{con.replace('-', '_minus_')}"
            b.add(f"{key}.sign_keep", f"{a['n_keep']}/{a['n_settings']}", f"{cell} {con}: settings keeping the baseline sign")
            b.add(f"{key}.unconditional", a["unconditional"], f"{cell} {con}: sign kept in >= 90% of settings")
            if a["flips"]:
                b.add(f"{key}.flips", "; ".join(f"{f['setting']} ({f['mean']:+.4f}, p={f['p']:.2f})" for f in a["flips"]),
                      f"{cell} {con}: settings that flip the sign (mean, p)")
    st = s["settings"]
    for name in ("test_src=foldavg", "tree_factor=1.2", "tail=mass5%/5%", "tail=fixed50/110",
                 "tail=fixed41/113", "B=80,count", "S=25", "drop_near_dup"):
        cs = st[name]["contrasts"].get("step:3", {})
        for con in ("C1", "C2", "C3", "R2-R1"):
            v = cs.get(con)
            if v and v.get("valid"):
                b.add(f"e5.setting.{name}.K3.{con.replace('-', '_minus_')}", v["mean"], f"{name}: {con} at K=3")
    g = s["gates"]
    b.add("gate.E5.c2_tail_limit_claim", g["c2_tail_definition"]["limit_claim"],
          "E5 gate: a tail definition flips C2 (evaluated at K=3)")
    b.add("gate.E5.stretch_old_grid_blocked", g["stretch_old_grid"]["blocked"], "Old stretch grid hits its edge in > 10% of cells")
    b.add("e5.stretch_old_grid_frac_hit", g["stretch_old_grid"]["frac_hit_K_GRID"], "Old stretch grid: share of cells at the edge")
    fi = s["frontier_interp"]["rules"]["R1"]
    for mb, v in fi.items():
        b.add(f"e5.frontier_interp.R1.mid{mb}", v["old_minus_dense"]["mean"],
              f"Frontier tail RMSE at middle budget {mb}: old 8-point grid minus dense grid (R1)")
    k2 = s["sign_keep"]["step:2"]["C2"]["all"]["flips"]
    if k2 and not g["c2_tail_definition"]["limit_claim"]:
        b.note("E5: the tail-definition gate on C2 is evaluated at K = 3 only (no flip). At K = 2, C2 flips sign "
               "(to about 0) under " + ", ".join(f["setting"] for f in k2) +
               "; worth a sentence in Table VIII.")


# ---------------------------------------------------------------------------
# E6
# ---------------------------------------------------------------------------
def add_sim(b, sd):
    s = sd["summary"]
    b.section("Simulation (E6)")
    gaps = []
    for sc, v in s.items():
        scen, n = sc.split("|")
        for c, cv in v["centers"].items():
            for K, kv in cv["by_K"].items():
                if K not in ("3",) and not (n == "57000" and K in K_REPORT):
                    continue
                key = f"e6.{scen}.n{n}.{c}.K{K}"
                b.add(f"{key}.R1_gap_to_muG", kv["R1_gap_to_muG"]["mean"], f"{scen}, n={n}, {c}, K={K}: cost(R1) - cost(mu^G)")
                if n == "57000" and K == "3":
                    gaps.append(kv["R1_gap_to_muG"]["mean"])
                if K == "3":
                    b.add(f"{key}.center_gap", kv["center_gap"]["mean"], f"{scen}, n={n}, {c}, K=3: cost(mu^G) - cost(mu^X)")
                    for r, g in kv["decision_gap"].items():
                        b.add(f"{key}.gap_{r}.est", g["est"], f"{scen}, n={n}, {c}, K=3: decision gap of {r}, estimated via R1")
                        b.add(f"{key}.gap_{r}.true", g["true"], f"{scen}, n={n}, {c}, K=3: decision gap of {r}, true via mu^G")
                        b.add(f"{key}.gap_{r}.rel_err", g["rel_err"], f"{scen}, n={n}, {c}, K=3: relative error (None if true gap small)")
                    if "R8_minus_R1" in kv:
                        b.add(f"{key}.R8_minus_R1", kv["R8_minus_R1"]["mean"], f"{scen}, n={n}, {c}, K=3: cost(R8) - cost(R1)")
    b.add("e6.R1_gap_to_muG_K3_n57000_min", min(gaps), "K=3, n=57,000: smallest cost(R1) - cost(mu^G) over scenarios and centers")
    b.add("e6.R1_gap_to_muG_K3_n57000_max", max(gaps), "K=3, n=57,000: largest cost(R1) - cost(mu^G)")
    g = sd["gates"]
    b.add("gate.E6.quantitative_fig5", g["quantitative_fig5"]["pass"], "E6 gate: relative error <= 10% in S1 and S3 at n = 57,000")
    w = g["quantitative_fig5"]["worst"]
    b.add("gate.E6.worst_rel_err", w["rel_err"], f"E6: worst relative error ({w['scen']}, {w['center']}, {w['rule']})")
    for c, v in g["single_index_needed"].items():
        b.add(f"gate.E6.S2_R8_minus_R1.{c}", v["R8_minus_R1"], f"E6 S2 n=57,000, {c}: R8 - R1 at K=3 (gate: R8 wins by >= 0.10)")
        b.add(f"gate.E6.single_index_needed.{c}", v["pass"], f"E6 S2 gate passed ({c})")
    for c, v in g["guessing_flips_R5_R1"].items():
        b.add(f"gate.E6.guessing_flip.{c}", v["flip"], f"E6 S4: guessing flips R5 - R1 ({c})")


# ---------------------------------------------------------------------------
# E9
# ---------------------------------------------------------------------------
def add_use(b, uv):
    s = uv["summary"]
    m = s["mean"]
    b.section("Fixed thresholds (E9, Table VII)")
    for th, tv in m["thresholds"].items():
        b.add(f"e9.{th}.prevalence_test", tv["prevalence_te"]["mean"], f"{th}: event prevalence in test")
        for pol, byK in tv["policies"].items():
            for K, v in byK.items():
                if pol == "P0" and K != "3":
                    b.add(f"e9.{th}.P0.K{K}.L", v["L"]["mean"], f"{th}, P0, K={K}: loss per 1,000")
                    continue
                key = f"e9.{th}.{pol}.K{K}"
                for met in ("L", "flag_rate", "FNR", "FPR", "PPV", "FP_per_TP"):
                    b.add(f"{key}.{met}", v[met]["mean"], f"{th}, {pol}, K={K}: {met}")
        for K, v in tv["flip_per_1000"]["P1_vs_P0"].items():
            b.add(f"e9.{th}.flip_P1_vs_P0.K{K}", v["mean"], f"{th}, K={K}: candidates per 1,000 whose flag differs P1 vs P0")
        for bud, v in tv["budget"].items():
            b.add(f"e9.{th}.budget{bud}.recall_yhat", v["recall_yhat"]["mean"], f"{th}: recall at {bud} screening budget, ranking by yhat")
            b.add(f"e9.{th}.budget{bud}.recall_g3", v["recall_g"]["3"]["mean"], f"{th}: recall at {bud} budget, ranking by g_3")
        for kind, pv in tv["prob"].items():
            if isinstance(pv, dict) and "brier" in pv:
                b.add(f"e9.{th}.prob_{kind}.brier", pv["brier"]["mean"], f"{th}: Brier score of P(event|x), {kind}")
                if "ece" in pv:
                    b.add(f"e9.{th}.prob_{kind}.ece", pv["ece"]["mean"], f"{th}: ECE of P(event|x), {kind}")
                if "brier_skill" in pv:
                    b.add(f"e9.{th}.prob_{kind}.brier_skill", pv["brier_skill"]["mean"],
                          f"{th}: Brier skill vs climatology, {kind}")
    b.section("Loss tests P2 vs P1 (E9)")
    for th, byK in s["L_tests"].items():
        for K, v in byK.items():
            key = f"e9.{th}.K{K}.P2_minus_P1"
            b.nb(key, v["P2_minus_P1"], f"{th}, K={K}: L(P2) - L(P1)")
            b.add(f"{key}.p_holm", v["P2_minus_P1"].get("p_holm"), f"{th}, K={K}: L(P2) - L(P1), Holm p")
            b.add(f"{key}.tost_passed", v["tost_P2_P1"]["passed"], f"{th}, K={K}: TOST ±5 per 1,000 passed")
            b.nb(f"e9.{th}.K{K}.P1_minus_P0", v["P1_minus_P0"], f"{th}, K={K}: L(P1) - L(P0)")
    g = s["gates"]
    for th, v in g["P2_recommended"].items():
        b.add(f"gate.E9.P2_recommended.{th}", v["passed"], f"E9 gate: P2 beats P1 by >= 5/1,000 (Holm) at >= 3 of 4 K, {th}")
        b.add(f"gate.E9.P2_wins_K.{th}", ",".join(v["K_wins"]), f"{th}: K at which P2 wins")
    b.add("gate.E9.spearman_min", g["spearman"]["min"], "Min Spearman(yhat, g_K) over K and splits")
    b.add("gate.E9.spearman_passed", g["spearman"]["passed"], "Spearman >= 0.99 at all K (budget screening unchanged)")
    for K, v in g["spearman"]["min_by_K"].items():
        b.add(f"e9.spearman_min.K{K}", v, f"Min Spearman(yhat, g_K) at K={K}")
    b.add("gate.E9.pi90_individual_use", g["pi90_individual_use"]["passed"], "PI90 fit for individual advice (decile >= 0.85, tails >= 0.80)")
    b.section("PI90 coverage (E9)")
    pi = m["pi90"]
    b.add("e9.pi90.cover", pi["overall"]["cover"]["mean"], "PI90 coverage overall")
    b.add("e9.pi90.width", pi["overall"]["width"]["mean"], "PI90 width overall (points)")
    b.add("e9.pi90.width_unconditional", 2 * 1.6448536 * m_sd_y(uv), "Width of an unconditional 90% interval (+-1.645 SD(y))")
    for reg, v in pi["by_region"].items():
        b.add(f"e9.pi90.cover_{REGION_EN[reg]}", v["cover"]["mean"], f"PI90 coverage, true region {REGION_EN[reg]}")
        b.add(f"e9.pi90.width_{REGION_EN[reg]}", v["width"]["mean"], f"PI90 width, true region {REGION_EN[reg]}")
    dec = [d["cover"]["mean"] for d in pi["by_yhat_decile"]]
    b.add("e9.pi90.cover_decile_min", min(dec), "PI90 coverage: lowest over yhat deciles")
    b.add("e9.pi90.cover_decile_max", max(dec), "PI90 coverage: highest over yhat deciles")
    gname = {"gender": "gender", "region": "priority area", "chuyen": "specialized school", "prov_size_t3": "province-size tercile"}
    for grp, lv in pi["by_group"].items():
        for lev, v in lv.items():
            b.add(f"e9.pi90.cover.{grp}_{lev}", v["cover"]["mean"], f"PI90 coverage, {gname.get(grp, grp)} = {lev} (n~{v['n']['mean']:.0f})")
    b.section("Near-guessing exclusion (E9 d)")
    ng = s["near_guess_tests"]
    b.add("e9.near_guess.n_removed_train", m["near_guess"]["n_removed_tr"]["mean"], "Records y <= 37 removed from training (mean per split)")
    b.add("e9.near_guess.n_removed_test", m["near_guess"]["n_removed_te"]["mean"], "Records y <= 37 removed from test (mean per split)")
    for var in ("full", "excl"):
        for con in ("C1", "C2", "C3"):
            v = ng[var][con]
            b.add(f"e9.near_guess.{var}.{con}.mean", v["mean"], f"{con}, {'all records (E9 refit)' if var == 'full' else 'y <= 37 removed'}: mean")
            b.add(f"e9.near_guess.{var}.{con}.p", v["p"], f"{con}, {var}: NB p")
    for con, v in ng["excl_primary_holm"].items():
        b.add(f"e9.near_guess.excl.{con}.p_holm", v["p_holm"], f"{con}, y <= 37 removed: Holm p")
    for con, v in g["near_guess_sign_change"].items():
        b.add(f"gate.E9.near_guess_sign_change.{con}", v, f"Removing y <= 37 changes the sign of {con}")
    c3_full = ng["full"]["C3"]["mean"]
    c3_npz = ng["npz_full"]["C3"]["mean"]
    if abs(c3_full - c3_npz) > 0.001:
        b.note(f"E9 near-guess block: C3 on all records after the E9 refit is {c3_full:.4f}, versus {c3_npz:.4f} "
               "from the E1 predictions (rs_tuned refit differs slightly). C1 and C2 agree to 1e-4. Cite C3 from E1/E2; "
               "use the E9 refit values only for the full-vs-excluded comparison.")


def m_sd_y(uv):
    # SD của y toàn khoá (data_audit) dùng cho khoảng không điều kiện; đọc ở đây để
    # tránh truyền thêm tham số. Không có trong use_validity nên đọc data_audit.
    da, _ = load("data_audit")
    return da["target"]["desc"]["sd"]


# ---------------------------------------------------------------------------
# E10
# ---------------------------------------------------------------------------
def add_group(b, ga):
    b.section("Group audit (E10, Table VII)")
    mg = ga["meta"]["margins"]
    for k, v in mg.items():
        b.add(f"e10.margin.{k}", v, f"Equivalence margin for {k}")
    b.add("e10.n_perm", ga["meta"]["n_perm"], "Permutations of province labels within priority area")
    tau_outside = []
    for mk, M in ga["models"].items():
        mkey = mk.replace("|", "_")
        b.add(f"e10.{mkey}.rmse", M["rmse"], f"{mk}: cross-fitted RMSE (2 x 5-fold, all records)")
        P = M["province"]
        b.add(f"e10.{mkey}.n_provinces", P["n_provinces"], f"{mk}: provinces audited")
        for pred, v in P["by_pred"].items():
            t = v["tau_reml"]
            k = f"e10.{mkey}.tau.{pred}"
            lab = f"{mk}, {pred}"
            b.add(f"{k}.est", t["est"], f"{lab}: province SD tau (REML)")
            b.add(f"{k}.ci_lo", t["ci_lo"], f"{lab}: tau 95% CI lower (Q-profile)")
            b.add(f"{k}.ci_hi", t["ci_hi"], f"{lab}: tau 95% CI upper (Q-profile)")
            b.add(f"{k}.verdict", VERDICT_EN.get(t["verdict"], t["verdict"]), f"{lab}: tau verdict vs margin 1.0")
            b.add(f"{k}.dl", v["tau_dl"], f"{lab}: tau (DerSimonian-Laird)")
            b.add(f"{k}.null_q95", P["permutation"]["tau_null_q95"][pred], f"{lab}: tau under permutation, 95th percentile")
            b.add(f"{k}.n_prov_gt_margin", v["n_prov_abs_mean_gt_margin"], f"{lab}: provinces with |mean residual| > 1")
            if t["ci_lo"] > t["est"] + 1e-9:
                tau_outside.append(f"{mk} {pred}: REML {t['est']:.3f}, CI [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]")
        for pred, v in P["gate_tail_weight_increases_province_bias"].items():
            k = f"e10.{mkey}.tau_increase.{pred}"
            b.add(f"{k}.delta", v["delta_tau_vs_K1"], f"{mk}: tau({pred}) - tau(K=1)")
            b.add(f"{k}.null_q95", v["null_q95"], f"{mk}: same difference under permutation, 95th pct")
            b.add(f"{k}.increase", v["increase"], f"{mk}: tail weighting increases province bias at {pred}")
        for pred, v in P["rmse_by_province_size_tertile"].items():
            for tt, r in v.items():
                b.add(f"e10.{mkey}.rmse_prov_t{tt}.{pred}", r, f"{mk}: RMSE in province-size tercile {tt}, {pred}")
        # Cleary
        n_out = []
        for grp, G in M["cleary"].items():
            for lev, L in G["levels"].items():
                for par in ("a_g", "b_g"):
                    x = L[par]
                    k = f"e10.{mkey}.cleary.{grp}_{lev}.{par}"
                    if mk == "bag20|F_dt-cn" or x["verdict"] != "trong biên":
                        b.add(f"{k}.est", x["est"], f"{mk}: Cleary {par} for {grp} = {lev}")
                        b.add(f"{k}.ci_lo", x["ci_lo"], f"{mk}: Cleary {par} for {grp} = {lev}, 95% CI lower")
                        b.add(f"{k}.ci_hi", x["ci_hi"], f"{mk}: Cleary {par} for {grp} = {lev}, 95% CI upper")
                        b.add(f"{k}.verdict", VERDICT_EN.get(x["verdict"], x["verdict"]), f"{mk}: Cleary {par} for {grp} = {lev}, verdict")
                        b.add(f"{k}.p_holm", x.get("p_holm"), f"{mk}: Cleary {par} for {grp} = {lev}, Holm p")
                    if x["verdict"] != "trong biên":
                        n_out.append(f"{grp}={lev}:{par}({VERDICT_EN.get(x['verdict'], x['verdict'])})")
        b.add(f"e10.{mkey}.cleary_not_within", "; ".join(n_out) if n_out else "none",
              f"{mk}: Cleary cells not within margin")
        if mk == "bag20|F_dt-cn" and n_out:
            hp = []
            for grp, G in M["cleary"].items():
                for lev, L in G["levels"].items():
                    for par in ("a_g", "b_g"):
                        if L[par]["verdict"] != "trong biên":
                            hp.append(f"{grp}={lev} {par} {L[par]['est']:+.4f} "
                                      f"[{L[par]['ci_lo']:+.4f}, {L[par]['ci_hi']:+.4f}], "
                                      f"{VERDICT_EN.get(L[par]['verdict'])}, Holm p {L[par]['p_holm']:.2f}")
            b.note("E10 Cleary on the primary model: cells outside 'within margin' are " + "; ".join(hp) +
                   ". The specialised-school slope CI excludes 0 and crosses the 0.05 margin but its Holm p is "
                   "large; the largest school-size quintile is undetermined (CI contains 0), not a shown deviation.")
        # cờ theo nhóm
        for th, pols in M["flags"].items():
            for pol, pv in pols.items():
                if pol not in ("P0", "P1@K3", "P2@K3", "P1@K8", "P2@K8"):
                    continue
                k = f"e10.{mkey}.flags.{th}.{pol.replace('@', '_')}"
                for met in ("flag_rate", "FNR", "FPR", "PPV"):
                    b.add(f"{k}.{met}", pv["overall"][met], f"{mk}, {th}, {pol}: overall {met}")
                cnt = {"exceeds": 0, "undetermined": 0}
                for grp, G in pv["groups"].items():
                    for lev, L in G["levels"].items():
                        for d in ("dFNR", "dFPR"):
                            vd = L[d]["verdict"]
                            if "vượt" in vd:
                                cnt["exceeds"] += 1
                            elif vd == "chưa xác định":
                                cnt["undetermined"] += 1
                b.add(f"{k}.n_cells_exceed", cnt["exceeds"], f"{mk}, {th}, {pol}: group cells (dFNR, dFPR) exceeding margin")
                b.add(f"{k}.n_cells_undetermined", cnt["undetermined"], f"{mk}, {th}, {pol}: group cells undetermined")
    if tau_outside:
        # q_profile đảo hàm Q tổng quát (kiểu Paule-Mandel), còn điểm ước lượng là REML:
        # hai ước lượng khác nhau nên CI không nhất thiết chứa điểm.
        b.note("E10: the REML point estimate of province tau falls below its Q-profile 95% CI in "
               f"{len(tau_outside)} cells (" + "; ".join(tau_outside) + "). group_audit.q_profile inverts the "
               "generalised Q statistic (Paule-Mandel type) while tau_reml maximises the REML likelihood, so "
               "estimate and interval come from different estimators. Report one consistent pair (e.g. "
               "Paule-Mandel tau with Q-profile CI) or flag it in the table note.")
    b.section("Sensitive attributes (E10 d)")
    base = ga["models"]["bag10|F_dt"]
    for mk in ("bag10|F_dt-cn", "bag10|F_dt-cn-bc"):
        M = ga["models"][mk]
        b.add(f"e10.sens.{mk.replace('|', '_')}.rmse_minus_Fdt", M["rmse"] - base["rmse"], f"{mk}: RMSE minus bag10|F_dt")
    g = ga["models"]["bag20|F_dt-cn"]["cleary"]["gender"]["levels"]
    gb = base["cleary"]["gender"]["levels"]
    b.add("e10.sens.gender_intercept_gap_Fdt", gb["1"]["a_g"]["est"] - gb["0"]["a_g"]["est"],
          "Cleary intercept gap between gender groups, with gender as a feature (bag10|F_dt)")
    b.add("e10.sens.gender_intercept_gap_primary", g["1"]["a_g"]["est"] - g["0"]["a_g"]["est"],
          "Cleary intercept gap between gender groups, gender removed (bag20|F_dt-cn)")


# ---------------------------------------------------------------------------
# md
# ---------------------------------------------------------------------------
def fmt(v, key=""):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if key.endswith((".p", "p_holm", "tost_p")) or ".p_" in key:
            return f"{v:.2g}" if v < 0.001 else f"{v:.4f}"
        a = abs(v)
        if a == 0:
            return "0"
        if a >= 1000:
            return f"{v:,.0f}"
        if a >= 100:
            return f"{v:.1f}"
        if a >= 1:
            return f"{v:.3f}"
        return f"{v:.4f}"
    return str(v)


def curated_tables(b):
    V = b.vals
    out = []

    def g(k):
        return fmt(V.get(k), k)

    out.append("## Key tables\n")
    # Table III
    out.append("### Table III. Centers on F_dt-cn (mean over 10 splits)\n")
    out.append("| Center | RMSE low | RMSE mid | RMSE high | RMSE all | R2 | SD ratio | Calib. slope | cost_3 R0 | cost_3 R1 | Fit (thread-s) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in ("default", "sub1", "bag10", "bag20", "bag40", "rs_tuned", "rs_tuned_bag5"):
        k = f"e1.center.F_dt-cn.{c}"
        out.append(f"| {CENTER_LABEL[c]} | {g(k + '.rmse_low')} | {g(k + '.rmse_mid')} | {g(k + '.rmse_high')} | "
                   f"{g(k + '.rmse_all')} | {g(k + '.r2')} | {g(k + '.sd_ratio')} | {g(k + '.calib_slope')} | "
                   f"{g(k + '.cost3.R0')} | {g(k + '.cost3.R1')} | {g(k + '.fit_s')} |")
    out.append("")
    # Primary contrasts
    out.append("### Primary contrasts at K = 3 (negative = left side better)\n")
    out.append("| Contrast | Mean | 95% CI | p | Holm p | TOST ±0.10 | Splits won | % SEM |")
    out.append("|---|---|---|---|---|---|---|---|")
    for c in ("c1", "c2", "c3"):
        out.append(f"| {c.upper()} | {g(c + '.mean')} | [{g(c + '.ci_lo')}, {g(c + '.ci_hi')}] | {g(c + '.p')} | "
                   f"{g(c + '.p_holm')} | {g(c + '.tost_passed')} | {g(c + '.wins')}/10 | {g(c + '.pct_sem')} |")
    out.append("")
    # Decomposition
    out.append("### Table IV. Decomposition on bag20 (mean [95% CI]; positive = cost reduction, or excess of Rx over R1)\n")
    out.append("| Component | K = 2 | K = 3 | K = 5 | K = 8 |")
    out.append("|---|---|---|---|---|")
    comps = [("default_to_bag_R0", "default -> bag20 (raw)"), ("default_to_bag_R1", "default -> bag20 (with R1)"),
             ("i", "(i) center gain after bagging"), ("ii", "(ii) decision layer, total"),
             ("ii_a", "(ii-a) recalibration at K = 1"), ("ii_b", "(ii-b) tilt by K"),
             ("iii_R2", "(iii) R2 isotonic - R1"), ("iii_R3", "(iii) R3 histogram - R1"),
             ("iii_R4", "(iii) R4 linear - R1"), ("iii_R5", "(iii) R5 stretch - R1 (= -C2)"),
             ("iii_R7", "(iii) R7 normal - R1"), ("iii_R8star", "(iii) R8* - R1 (= -C1)")]
    for ck, lab in comps:
        row = [lab]
        for K in K_REPORT:
            if ck == "i":
                row.append(g(f"decomp.K{K}.i"))
                continue
            k = f"decomp.K{K}.{ck}"
            row.append(f"{g(k + '.mean')} [{g(k + '.ci_lo')}, {g(k + '.ci_hi')}]")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    # Table V
    out.append("### Table V. cost_K by estimator on bag20\n")
    out.append("| Estimator | K = 1 | K = 2 | K = 3 | K = 5 | K = 8 |")
    out.append("|---|---|---|---|---|---|")
    for r in ("R0", "R1", "R1_1", "R2", "R3", "R4", "R5", "R7", "R8", "R8_bag5"):
        row = [f"{r} {RULE_LABEL[r]}"]
        for K in K_ALL:
            row.append(g(f"rules.bag20.K{K}.{r}.cost"))
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    # Table VI
    out.append("### Table VI. HPO policies: test cost_K, and difference to (a) with Holm p\n")
    out.append("| K | (a) | (b) | (c) | (c+) | (o) | (b)-(a) | (c+)-(a) | (c)-(a) | (o)-(a) | Kendall tau (all) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for K in ("1", "1.5", "2", "3", "5", "8", "12", "20"):
        row = [K] + [g(f"e4.K{K}.policy_{p}.cost") for p in ("a", "b", "c", "c+", "o")]
        for nm in ("b_minus_a", "cplus_minus_a", "c_minus_a", "o_minus_a"):
            row.append(f"{g(f'e4.K{K}.{nm}.mean')} ({g(f'e4.K{K}.{nm}.p_holm')})")
        row.append(g(f"e4.rank.K{K}.kendall_all"))
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    # Table VII (E9)
    out.append("### Table VII (part). Fixed thresholds: loss per 1,000 and error rates\n")
    out.append("| Threshold | K | L(P0) | L(P1) | L(P2) | P2-P1 [Holm p] | FNR P1 | FNR P2 | FPR P1 | FPR P2 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for th in ("lt60", "ge100"):
        for K in K_ALL:
            p0 = f"e9.{th}.P0.K{K}.L"
            out.append(f"| {th} | {K} | {g(p0)} | {g(f'e9.{th}.P1.K{K}.L')} | {g(f'e9.{th}.P2.K{K}.L')} | "
                       f"{g(f'e9.{th}.K{K}.P2_minus_P1.mean')} [{g(f'e9.{th}.K{K}.P2_minus_P1.p_holm')}] | "
                       f"{g(f'e9.{th}.P1.K{K}.FNR')} | {g(f'e9.{th}.P2.K{K}.FNR')} | {g(f'e9.{th}.P1.K{K}.FPR')} | {g(f'e9.{th}.P2.K{K}.FPR')} |")
    out.append("")
    # E10
    out.append("### Province SD tau (REML, Q-profile 95% CI) by K\n")
    out.append("| Model | yhat | g_1 | g_2 | g_3 | g_5 | g_8 |")
    out.append("|---|---|---|---|---|---|---|")
    for mk in ("bag20_F_dt-cn", "bag10_F_dt", "bag10_F_dt-cn", "bag10_F_dt-cn-bc"):
        row = [mk]
        for pr in ("yhat", "g_1", "g_2", "g_3", "g_5", "g_8"):
            k = f"e10.{mk}.tau.{pr}"
            row.append(f"{g(k + '.est')} [{g(k + '.ci_lo')}, {g(k + '.ci_hi')}] (null {g(k + '.null_q95')})")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return out


def write_md(b, meta):
    lines = ["# Paper numbers, cost-aware HSA study",
             "",
             f"Generated {meta['generated']} by `src/paper_numbers_cost.py` from `results_cost/*.json` "
             f"(commit of the confirmatory runs: {meta['run_commit']}). Means are over 10 outer splits "
             "(seeds 100-109); p values use the Nadeau-Bengio corrected t (df = 9); SESOI = 0.10. "
             "Differences are left minus right; for cost differences, negative means the left side is better.",
             ""]
    lines += curated_tables(b)
    if b.notes:
        lines += ["## Consistency notes", ""]
        lines += [f"- {n}" for n in b.notes]
        lines.append("")
    lines += ["## All numbers", ""]
    for sec in b.sections:
        keys = [k for k in b.vals if b.section_of[k] == sec]
        lines += [f"### {sec}", "", "| Quantity | Value | Key |", "|---|---|---|"]
        for k in keys:
            lab = b.labels[k].replace("|", "/")
            val = fmt(b.vals[k], k).replace("|", "/")
            lines.append(f"| {lab} | {val} | `{k}` |")
        lines.append("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def summary_text(b):
    V = b.vals

    def g(k, nd=3):
        v = V.get(k)
        if v is None:
            return "n/a"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.{nd}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"
        return str(v)

    L = []
    L.append(f"Data: n = {V['data.n_records']:,}, tails {V['data.mass_low_tail']:.4f} + {V['data.mass_high_tail']:.4f}, "
             f"near-guess y<=37: {V['data.n_near_guess']}, exact duplicates: {V['data.dup_exact_groups']}")
    L.append(f"E1: feature set {V['e1.primary_feature_set']} (F_dt-cn - F_dt RMSE {g('e1.gate.cn_vs_dt.mean')}, TOST {g('e1.gate.cn_vs_dt.tost_passed')}; "
             f"-cn-bc {g('e1.gate.cnbc_vs_cn.mean')}, TOST {g('e1.gate.cnbc_vs_cn.tost_passed')}); B* = bag{V['e1.b_star']}; "
             f"primary center {V['gate.G1.primary_center']}; bag10 - default cost_3 {g('e1.t3.bag10_minus_default.mean')}; "
             f"sub1 - default {g('e1.t3.sub1_minus_default.mean')} (p {g('e1.t3.sub1_minus_default.p')})")
    for c in ("c1", "c2", "c3"):
        L.append(f"{c.upper()}: {g(c + '.mean')} [{g(c + '.ci_lo')}, {g(c + '.ci_hi')}], p {g(c + '.p', 5)}, Holm {g(c + '.p_holm', 5)}, "
                 f"TOST {g(c + '.tost_passed')}, wins {V[c + '.wins']}/10")
    L.append("Decomposition K=3 (bag20): default->bag(R1) {0}, (ii) {1}, (ii-a) {2}, (ii-b) {3}; (iii) R2 {4}, R3 {5}, R4 {6}, R5 {7}, R7 {8}, R8* {9}".format(
        *[g(f"decomp.K3.{x}.mean") for x in ("default_to_bag_R1", "ii", "ii_a", "ii_b", "iii_R2", "iii_R3", "iii_R4", "iii_R5", "iii_R7", "iii_R8star")]))
    L.append(f"G2: {V['gate.G2.verdict']}, stretch breaks at K={V['gate.G2.k_stretch_breaks']}; R2 equiv R1 all K: {g('gate.G2.R2_equiv_R1_all_K')}; "
             f"(ii-a)/(ii) = {g('gate.G2.ii_a_share_K3')}")
    L.append(f"E2b K=3: normalisation {g('e2b.K3.attr.normalization.mean')} (p {g('e2b.K3.attr.normalization.p')}), "
             f"retuning {g('e2b.K3.attr.retuning.mean')} (p {g('e2b.K3.attr.retuning.p')}); K=8 retuning {g('e2b.K8.attr.retuning.mean')} (p {g('e2b.K8.attr.retuning.p')})")
    L.append(f"E3: dominance {g('e3.dominance_frac_informative')} of informative theta (all-grid {g('e3.dominance_frac_all_theta')}); gate {g('gate.E3.center_forecast_gain')}; "
             f"Taggart bag20 - default {g('e3.pair.bag20_R0_minus_default_R0.mean', 4)}")
    L.append(f"E4 K=3: (a) {g('e4.K3.policy_a.cost')}, (b)-(a) {g('e4.K3.b_minus_a.mean')}, (c+)-(a) {g('e4.K3.cplus_minus_a.mean')}, "
             f"(c)-(a) {g('e4.K3.c_minus_a.mean')} (Holm {g('e4.K3.c_minus_a.p_holm')}), (o)-(a) {g('e4.K3.o_minus_a.mean')}; "
             f"one tuning suffices: {g('gate.E4.one_tuning_suffices')}; Kendall tau all K=3 {g('e4.rank.K3.kendall_all')}")
    L.append(f"E5: C1/C2/C3 sign kept at K=3 in {V['e5.step_K3.C1.sign_keep']}, {V['e5.step_K3.C2.sign_keep']}, {V['e5.step_K3.C3.sign_keep']} settings; "
             f"R2-R1 {V['e5.step_K3.R2_minus_R1.sign_keep']}")
    L.append(f"E6: R1 - mu^G at K=3, n=57,000: {g('e6.R1_gap_to_muG_K3_n57000_min')} to {g('e6.R1_gap_to_muG_K3_n57000_max')}; "
             f"quantitative Fig. 5 gate {g('gate.E6.quantitative_fig5')} (worst rel. err {g('gate.E6.worst_rel_err')})")
    L.append(f"E9: P2 recommended lt60 {g('gate.E9.P2_recommended.lt60')} (K {V['gate.E9.P2_wins_K.lt60']}), ge100 {g('gate.E9.P2_recommended.ge100')} (K {V['gate.E9.P2_wins_K.ge100']}); "
             f"Spearman min {g('gate.E9.spearman_min', 4)}; PI90 cover {g('e9.pi90.cover')}, low {g('e9.pi90.cover_low')}, high {g('e9.pi90.cover_high')}, "
             f"width {g('e9.pi90.width', 1)}; near-guess C1/C2/C3 excl {g('e9.near_guess.excl.C1.mean')}/{g('e9.near_guess.excl.C2.mean')}/{g('e9.near_guess.excl.C3.mean')}")
    taus = ", ".join(f"{p} {g(f'e10.bag20_F_dt-cn.tau.{p}.est', 2)}" for p in ("yhat", "g_1", "g_2", "g_3", "g_5", "g_8"))
    L.append(f"E10 (bag20|F_dt-cn) tau: {taus}; null q95 g_3 {g('e10.bag20_F_dt-cn.tau.g_3.null_q95', 2)}; "
             f"g_8 verdict {V['e10.bag20_F_dt-cn.tau.g_8.verdict']}; F_dt-cn-bc tau yhat {g('e10.bag10_F_dt-cn-bc.tau.yhat.est', 2)}")
    L.append(f"E10 Cleary not within margin (primary): {V['e10.bag20_F_dt-cn.cleary_not_within']}")
    return "\n".join(L)


def main():
    data, sha = {}, {}
    for n in SOURCES:
        data[n], sha[n] = load(n)
    b = Book()
    add_data(b, data["data_audit"], data["bench_eval"], data["decomp_centers"])
    add_centers(b, data["decomp_centers"], data["decomp_rules"])
    add_primary(b, data["decomp_rules"], data["wtrain_tuned"])
    add_decomp(b, data["decomp_rules"], data["wtrain_tuned"])
    add_wtrain(b, data["wtrain_tuned"])
    add_proper(b, data["proper_scores"])
    add_policy(b, data["select_policy"])
    add_sensitivity(b, data["sensitivity"])
    add_sim(b, data["sim_decomp"])
    add_use(b, data["use_validity"])
    add_group(b, data["group_audit"])

    prov = data["decomp_rules"]["meta"].get("provenance", {})
    meta = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "script": "src/paper_numbers_cost.py",
        "run_commit": prov.get("commit", "?")[:7] + (" (dirty)" if prov.get("dirty") else ""),
        "source_sha256": {f"{RES}/{n}.json": sha[n] for n in SOURCES},
        "consistency_notes": b.notes,
        "n_numbers": len(b.vals),
    }
    out = {"_meta": meta}
    out.update(b.vals)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    write_md(b, meta)
    print(f"{len(b.vals)} numbers -> {OUT_JSON}, {OUT_MD}\n")
    print(summary_text(b))
    if b.notes:
        print("\nConsistency notes:")
        for n in b.notes:
            print(" -", n)


if __name__ == "__main__":
    sys.exit(main())
