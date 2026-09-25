# -*- coding: utf-8 -*-
"""E2 (khung bài 24/9, mục 6.5): các cách ước lượng tầng quyết định và bảng phân rã.

Hậu kỳ thuần trên npz của E1: không khớp XGBoost nào. Với mỗi lần chia và mỗi trung
tâm (trung tâm chính, bag B*, default), khớp R0..R7 của decision_layer trên (ŷ OOF,
y huấn luyện) rồi áp cho ŷ test của mô hình khớp lại, và chấm trên tập kiểm tra.

Vì sao hậu kỳ trên npz mà không dựng lại từ final_compare.run_split: E2, E3, E4, E5,
E9 phải đọc CÙNG một ŷ. final_compare tự khớp trung tâm trong từng lần chạy, nên hai
thí nghiệm có thể lệch nhau ở trung tâm chứ không ở quy tắc. Ở đây chỉ phần đường
đánh đổi (along_middle) và cách lưu điểm theo tham số là lấy từ final_compare.

Lưới (mục 5.6, gates.py):
  step      K ∈ K_DENSE (17 điểm)        R0, R1, R1₁, R2, R3, R4, R5, R7; R6 chỉ ở K = 1
  prior     λ ∈ LAMBDA_DENSE (21 điểm)   R0, R1, R2, R5 (thứ cấp)
  phi       K ∈ K_DENSE                  R0, R1, R2, R5 (thứ cấp)
  step_pair (K_L,K_H) ∈ ASYM_PAIRS       R0, R1, R1₁, R2, R5 (thứ cấp)
R0 có mặt ở mọi họ vì nó là mốc của mọi đường cong; R0, R1₁, R6 không phụ thuộc w nên
dự đoán tính một lần, chỉ cost_K đổi theo tham số.

Đầu ra results_cost/decomp_rules.json:
  per_split[seed][center][rule][family][param] = {region_rmse{Low tail, Middle,
      High tail}, all, tails, cost_K, macro_bin, sera[, info]}
    cost_K là RMSE có trọng số theo ĐÚNG họ và tham số của ô đó: bậc thang K (hoặc
    cặp K_L,K_H), p̂(y)^(-λ), hay 1 + (K-1)φ(y), trọng số tính trên y test với mật
    độ và φ khớp trên y huấn luyện. Mỗi họ là quy tắc Bayes cho thước đo của chính
    nó (Nhận xét 6), nên chỉ so trong cùng họ. info: tham số R5 (s, chạm biên lưới)
    và R4 (a, b), để E5 đếm số lần chạm biên mà không phải chạy lại.
  split_info[seed]: lo/hi (khối lượng trên y huấn luyện), khối lượng đuôi, độ dốc
    hiệu chỉnh của trung tâm, SD của y.
  boot[seed][K][ô]: bootstrap cụm theo trường (stats_paired.cluster_boot_diff) và số
    thí sinh trên 1.000 đổi trạng thái cờ ở các ngưỡng E9.
  inputs[seed]: đường dẫn và sha256 của npz đã đọc.
  summary: table (mô tả), frontier (mô tả), decomposition[K], contrasts, tost_vs_R1,
    secondary (họ prior/φ/bất đối xứng), gate_G2, r8, stretch_edges, calibration.

Bảng phân rã (Bảng IV), K ∈ {2; 3; 5; 8}, họ bậc thang đối xứng. Quy ước dấu đúng như
khung bài, nên (i), (ii-a), (ii-b) dương là CÓ lợi, (iii) dương là R1 tốt hơn:
  default_to_bag|R0  cost(R0, default) - cost(R0, bag B*)   dòng riêng "sửa mốc quá phân tán"
  i                  cost(R1, bag B*) - cost(R1, chính)    bỏ (bằng 0) khi chính là bag B*
  ii_a               cost(R0) - cost(R1₁) trên trung tâm chính  (hiệu chỉnh ở K = 1)
  ii_b               cost(R1₁) - cost(R1)                        (nghiêng theo K)
  iii|Rx             cost(Rx) - cost(R1), x ∈ {R2, R3, R4, R5, R7, R8*}
Thêm ba dòng mô tả, không vào Holm: ii = ii_a + ii_b (cho cổng "(ii-a) quá nửa (ii)"),
i_R0 = cost(R0, bag B*) - cost(R0, chính) để cột Hình 5 cộng khít
cost(R0, bag B*) - cost(R1, chính) = i_R0 + ii_a + ii_b, và default_to_bag|R1.
C2 = cost(R1) - cost(R5) trên trung tâm chính (âm = R1 tốt hơn, như gates.CONTRASTS).
Trong JSON, nb.wins đếm chênh ÂM theo quy ước của stats_paired; n_pos đếm chênh dương.

Các lựa chọn khi khung bài chưa nói rõ (chọn cách đơn giản, đúng):
 1. Tên trung tâm: --primary/--bstar nhận tên trong npz, hoặc "auto" để đọc từ
    results_cost/decomp_centers.json. Trung tâm chính CHỈ lấy từ summary.G1.primary_center
    (không dò khoá "primary" theo chiều sâu: summary.feature_gate.primary là tên TẬP
    ĐẶC TRƯNG, từng bị lấy nhầm làm tên trung tâm). B* lấy từ G1.bag_Bstar, rồi mới dò
    bstar_center/B_star (B_star là số thì tên là bag<B>). "auto" cũng dừng khi G1 ghi
    splits_complete = false (G1 tính trên thiếu lần chia), hoặc khi JSON của E1 là
    lượt khói mà E2 không chạy --smoke.
 2. Họ Holm thứ cấp của Bảng IV/V: mọi ô i, ii_a, ii_b, iii|Rx ở cả bốn K, TRỪ các ô
    trùng một phép so chính (iii|R5 ở K = 3 là -C2, iii|R8* ở K = 3 là -C1, i ở K = 3
    là -C3 khi trung tâm chính là rs_tuned). Khung bài ghi "ở K ∈ {2; 5; 8}" nhưng các
    ô K = 3 còn lại (ii_a, iii|R2...) không phải phép so chính, bỏ chúng khỏi mọi họ
    Holm sẽ lỏng hơn; gộp vào thì bảo thủ hơn. C2 ở K ≠ 3 dùng chính p_holm của
    iii|R5 cùng K (cùng một phép thử, không đếm hai lần). default_to_bag thuộc Bảng
    III của E1 nên chỉ báo p thô.
 3. Họ Holm chính {C1; C2; C3} (stats_paired.primary_holm, m = 3): C2 tính ở đây; C1 =
    -(iii|R8*) ở K = 3 khi có wtrain_tuned.json; C3 = cost_3(R1, rs_tuned) -
    cost_3(R1, bag B*) tính từ chính npz khi có trung tâm rs_tuned. Phép so thiếu
    được coi p = 1 nên cổng G2 khi thiếu là bảo thủ và gắn cờ provisional.
    Đủ lần chia: C1, C2, C3 chỉ vào Holm chính khi n = len(gates.SEEDS) (--smoke: số lần
    chia khói). nb_ttest bỏ NaN nên n tự co khi wtrain_tuned.json hay .partial thiếu lần
    chia; một phép so tính trên 6/10 lần chia không được đứng ngang phép so đủ. Thiếu thì
    phép so được báo (per_split, nb) nhưng coi là vắng trong Holm (incomplete), và G2
    gắn provisional kèm lý do.
 4. R8* (huấn luyện có trọng số đã dò lại, E2b) chỉ có cost_K theo lần chia trong
    wtrain_tuned.json, không có dự đoán, nên ô iii|R8* có NB và TOST nhưng không có
    bootstrap cụm hay số đổi cờ. Đọc ở bước tóm tắt: E2b xong sau E2 thì chạy lại
    script này, mọi lần chia lấy từ .partial và chỉ phần tóm tắt được tính lại.
    Không có file hoặc không tìm thấy giá trị thì ô để null kèm lý do.
    Trước khi tính, mọi e1_npz_sha256 mà wtrain_tuned.json ghi cho các lần chia đang
    chạy phải trùng sha256 của npz E1 mà E2 đọc (inputs[seed]); lệch thì dừng: E2b đã
    khớp trên một bản dự đoán E1 khác, C1 sẽ trừ hai số của hai bản mã.
 5. Bootstrap cụm: cùng seed (seed của lần chia) cho mọi ô trong một lần chia, tức
    cùng các lần bốc trường (common random numbers), nên hai ô cùng lần chia so
    được với nhau. --smoke dùng B = 200.
 6. Chạy tiếp: fingerprint gồm cấu hình, code_sha256 và DẤU CỦA E1 (meta.fingerprint
    của npz theo từng tag), KHÔNG gồm danh sách seed hay sha256 của npz (thêm seed vẫn
    chạy tiếp được). Mỗi lần chia ghi sha256 npz của nó vào inputs[seed]; chạy tiếp mà
    npz đã đổi (E1 chạy lại) thì báo lỗi, hoặc tính lại lần chia đó với --on-mismatch
    recompute.
    Nhất quán của E1 (trước mọi tính toán): npz của mọi lần chia phải cùng
    meta.fingerprint, cùng cờ smoke và cùng data_sha256 (theo từng tag), và khớp
    meta.fingerprints.phase1/phase2 của --centers-json khi file đó có. Vì sao: E1 chạy
    lại với --on-mismatch recompute trên một phần --seeds để lại thư mục có hai bản mã;
    E2 đọc cả thư mục sẽ trộn chúng mà không lỗi gì. npz khói (smoke = true) bị từ chối
    khi E2 không chạy --smoke. npz không ghi smoke/data_sha256 (E1 trước 25/9, npz giả
    của tests/make_fake_preds.py) chỉ được so dấu.
 7. --data chỉ ghi vào meta, không mở: mọi thứ E2 cần (y, mã trường) có trong npz.
 8. --tag nhận nhiều tag: nếu E1 để trung tâm ở nhiều file (ví dụ giai đoạn 1 và 2),
    các file được gộp sau khi kiểm idx_tr, idx_te, y khớp nhau.
 9. frontier: điểm (RMSE giữa, RMSE đuôi) trung bình theo tham số và
    final_compare.along_middle với mốc là RMSE giữa của R0 trên cùng trung tâm.
    Chỉ mô tả, không kiểm định (mục 5.6).
10. Đổi đơn vị: pct_sem = trung bình / SEM × 100 với SEM giả định 4,3 và SEM =
    SD(y)·sqrt(1 - độ tin cậy) ở 0,85 / 0,90 / 0,95; flag_per_1000 = số thí sinh
    trên 1.000 đổi trạng thái {ŷ >= c} giữa hai vế, c ∈ gates.E9_THRESHOLDS.
11. JSON chặt: NaN/inf ghi thành null.
12. --save-preds DIR (tuỳ chọn): lưu dự đoán test của mọi quy tắc họ bậc thang ở
    K ∈ {1} ∪ K_GRID cho E3/E9 đọc đúng số của E2, khoá "<trung tâm>|<quy tắc>|<K>"
    (R0, R1_1, R6 chỉ ở K = 1). Không có thì E3 tự tính lại bằng decision_layer trên
    npz của E1 (cùng kết quả). Lần chia lấy từ .partial không được lưu lại.
13. Cổng "(ii-a) quá nửa (ii)" chỉ có nghĩa khi (ii) > 0 (tầng quyết định CÓ lợi):
    điều kiện là ii > 0 và ii_a > 0,5·ii. Tỉ số ii_a/ii với ii ≤ 0 đổi dấu vô nghĩa (dữ
    liệu giả: ii = -0,079, ii_a = -0,424 cho tỉ số 5,4 và câu "phần lớn lợi ích là hiệu
    chỉnh" dù tầng quyết định làm tệ đi); trường hợp đó báo riêng (ii_nonpositive).
14. OOF của rs_tuned, rs_tuned_bag5 trong npz E1 là OOF khớp lại fold trên fit ∪ es
    (decomp_centers, lựa chọn 12), không phải OOF dừng sớm; E2 dùng d["oof"] như cũ.
15. Ghi chú thiết kế (không phải lỗi mã): trung tâm chính đọc từ G1 TẠM của E1 (C1, C2
    lúc đó coi p = 1). C2 ở đây tính trên trung tâm đó; khi đủ họ, Holm cuối có thể
    lật G1 và khi ấy phải chạy lại E2 trên trung tâm mới. B* và cổng tập đặc trưng
    của E1 chọn theo tập kiểm tra rồi cùng tập kiểm tra dùng cho suy luận (khung bài
    cho phép).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import decision_layer as dl
import preds_io
import provenance
import stats_paired as sp
from bins import bin_index
from final_compare import METRICS as FRONTIER_METRICS
from final_compare import along_middle
from gates import (ALPHA, ASYM_PAIRS, CLUSTER_BOOT_B, E9_THRESHOLDS, K_DENSE, K_GRID,
                   LAMBDA_DENSE, NB_RATIO, PRIMARY_CONTRASTS, PRIMARY_K, SEEDS, SEM_ASSUMED,
                   SESOI, r8_star)
from preprocess import DATA_PATH
from tail_prior import MID_BUDGETS, Relevance, sera

EXPERIMENT = "decomp_rules"
FAMILY_RULES = {
    "step": ["R0", "R1", "R1_1", "R2", "R3", "R4", "R5", "R7"],
    "prior": ["R0", "R1", "R2", "R5"],
    "phi": ["R0", "R1", "R2", "R5"],
    "step_pair": ["R0", "R1", "R1_1", "R2", "R5"],
}
K_FREE = ("R0", "R1_1", "R6")            # dự đoán không phụ thuộc w
INFO_RULES = ("R4", "R5")                # ghi tham số đã chọn vào per_split
ITER_RULES = ["R2", "R3", "R4", "R5", "R7"]     # ô (iii) có dự đoán
TOST_RULES = ["R2", "R3", "R4"]                 # TOST ±SESOI so với R1 (cổng G2)
SECONDARY_VS_R1 = ["R2", "R5"]                  # so với R1 ở prior/φ/bất đối xứng
SAVE_KS = [1] + list(K_GRID)
RELIABILITY = [0.85, 0.90, 0.95]
SMOKE_BOOT_B = 200
C3_CENTER = "rs_tuned"                   # vế trái của C3 (gates.CONTRASTS)
REGIONS = ("Low tail", "Middle", "High tail")

assert all(k in K_DENSE for k in K_GRID), "K_GRID phải nằm trong K_DENSE"
assert PRIMARY_K in K_GRID


def pkey(v):
    """Khoá chuỗi cho tham số: 1.25 -> '1.25', 3 -> '3', 0.05 -> '0.05', '5,1' giữ nguyên."""
    if isinstance(v, str):
        return v
    return f"{float(v):g}"


def _f(x):
    return float("nan") if x is None else float(x)


def _clean_json(obj):
    """NaN/inf thành None để JSON chặt (json của Python mặc định ghi NaN, nhiều parser từ chối)."""
    obj = preds_io.to_jsonable(obj)
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def dump(obj, path):
    preds_io.dump_json_atomic(_clean_json(obj), path)


# ---------------------------------------------------------------------------
# Đọc npz của E1
# ---------------------------------------------------------------------------
def npz_paths(preds_dir, seed, tags):
    return [preds_io.split_path(preds_dir, seed, t) for t in (tags or [None])]


def load_merged(paths):
    """Gộp các npz của cùng một lần chia (nhiều tag). Kiểm lần chia và y khớp nhau:
    gộp nhầm hai lần chia sẽ cho R1 khớp trên OOF của lần chia này rồi áp cho ŷ test
    của lần chia khác mà không lỗi gì."""
    d = None
    for p in paths:
        x = preds_io.load_split(p)
        err = preds_io.validate_split(x)
        if err:
            raise ValueError(f"{p}: npz không hợp lệ: {err}")
        if d is None:
            d = x
            d["meta_by_file"] = {p: x.get("meta", {})}
            continue
        for k in ("idx_tr", "idx_te", "y_tr", "y_te"):
            if not np.array_equal(np.asarray(d[k]), np.asarray(x[k])):
                raise ValueError(f"{p}: {k} khác file đầu, không cùng lần chia")
        for part in ("oof", "test"):
            for c, a in x[part].items():
                if c in d[part] and not np.array_equal(d[part][c], a):
                    raise ValueError(f"{p}: trung tâm '{c}' có ở hai file với dự đoán khác nhau")
                d[part][c] = a
        d["meta_by_file"][p] = x.get("meta", {})
    return d


def _find_key(obj, names):
    """Giá trị đầu tiên (duyệt theo chiều sâu) có khoá thuộc names và là chuỗi hoặc số."""
    if isinstance(obj, dict):
        for k in names:
            v = obj.get(k)
            if isinstance(v, (str, int)) and not isinstance(v, bool):
                return v
        for v in obj.values():
            r = _find_key(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, names)
            if r is not None:
                return r
    return None


def resolve_roles(args):
    """{primary, bag_Bstar, default} là tên trung tâm trong npz; kèm nguồn của từng tên.

    "auto": đọc summary.G1 của decomp_centers.json (E1 ghi primary_center, bag_Bstar,
    provisional, splits_complete ở đó). Trung tâm chính chỉ lấy từ G1.primary_center
    (lựa chọn 1). B* thiếu trong G1 thì tìm theo chiều sâu các tên khoá của B*; B* là
    số thì tên trung tâm là bag<B>, theo cách E1 đặt tên."""
    roles, src = {}, {}
    cj = g1 = None
    for role, val, g1key, keys in (
            # Trung tâm chính: KHÔNG có danh sách dò theo chiều sâu (lựa chọn 1)
            ("primary", args.primary, "primary_center", []),
            ("bag_Bstar", args.bstar, "bag_Bstar", ["bag_Bstar", "bstar_center", "B_star_center",
                                                    "B_star", "Bstar"])):
        if val != "auto":
            roles[role], src[role] = val, "argv"
            continue
        if cj is None:
            cj = preds_io.load_json(args.centers_json)
            if cj is None:
                sys.exit(f"--{'primary' if role == 'primary' else 'bstar'} auto cần {args.centers_json} "
                         "(E1); chưa có thì truyền tên trung tâm, ví dụ --primary bag10 --bstar bag10")
            g1 = ((cj.get("summary") or {}).get("G1") or {})
            e1_smoke = (cj.get("meta") or {}).get("smoke")
            if e1_smoke and not args.smoke:
                sys.exit(f"{args.centers_json} là lượt KHÓI của E1 (meta.smoke = true) mà E2 không chạy "
                         "--smoke: không đọc vai trò từ đó; trỏ --centers-json tới JSON của lượt thật.")
            if g1.get("splits_complete") is False:
                sys.exit(f"summary.G1 trong {args.centers_json} tính trên thiếu lần chia "
                         f"({g1.get('incomplete')}, cần {g1.get('n_ref')}): trung tâm chính chưa chốt. "
                         "Chạy E1 đủ lần chia, hoặc truyền --primary/--bstar tay (ghi lý do).")
            if "primary_center" in g1 and "splits_complete" not in g1:
                print(f"  [cảnh báo] G1 trong {args.centers_json} không ghi splits_complete (E1 trước "
                      "25/9): không kiểm được G1 có đủ lần chia", flush=True)
        v = g1.get(g1key) if isinstance(g1.get(g1key), (str, int)) else None
        where = "summary.G1"
        if v is None and role == "primary":
            sys.exit(f"summary.G1 trong {args.centers_json} không ghi primary_center (G1: "
                     f"{g1.get('status', 'không có')}): E1 chưa chọn trung tâm chính. Chạy xong E1 "
                     "giai đoạn 2, hoặc truyền --primary <tên trung tâm trong npz> tay.")
        if v is None:
            v, where = _find_key(cj, keys), "tìm theo khoá"
        if v is None:
            sys.exit(f"Không tìm thấy {g1key} trong {args.centers_json} (summary.G1: "
                     f"{g1.get('status', 'không có')}); truyền tên trung tâm tay")
        roles[role] = f"bag{v}" if isinstance(v, int) else str(v)
        src[role] = f"{args.centers_json} {where}={v}"
    if g1:
        src["G1"] = {k: g1.get(k) for k in ("provisional", "primary_center", "bag_Bstar", "feature_set",
                                             "splits_complete", "n_ref")}
    roles["default"], src["default"] = args.default_center, "argv"
    return roles, src


# ---------------------------------------------------------------------------
# Nhất quán của đầu vào E1 và E2b (lựa chọn 4, 6)
# ---------------------------------------------------------------------------
E1_SIG_KEYS = ("fingerprint", "smoke", "data_sha256")


def npz_meta(path):
    """meta của npz mà không nạp mảng (10 file × vài chục MB chỉ để đọc dấu)."""
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z[preds_io.META_KEY])) if preds_io.META_KEY in z.files else {}


def check_e1_inputs(preds_dir, seeds, tags, centers_json, smoke):
    """Dấu E1 chung cho mọi lần chia, theo từng tag: {tag: {fingerprint, smoke, data_sha256}}.

    Dừng (sys.exit) khi: hai lần chia có dấu/cờ khói/băm dữ liệu khác nhau ở cùng tag;
    npz là lượt khói mà E2 không chạy --smoke; hoặc --centers-json có mặt và dấu npz
    không phải meta.fingerprints.phase<1|2> của nó. Trả kèm nguồn đối chiếu để ghi meta."""
    sig, first = {}, None
    for s in seeds:
        cur = {}
        for t, p in zip(tags or [None], npz_paths(preds_dir, s, tags)):
            m = npz_meta(p)
            cur[str(t)] = {k: m.get(k) for k in E1_SIG_KEYS} | {"phase": m.get("phase")}
        if first is None:
            first, sig = s, cur
            continue
        for t, v in cur.items():
            bad = [k for k in E1_SIG_KEYS if v.get(k) != sig[t].get(k)]
            if bad:
                sys.exit(f"npz E1 lệch giữa lần chia {first} và {s} (tag {t}) ở {bad}: "
                         + "; ".join(f"{k}: {str(sig[t].get(k))[:12]} vs {str(v.get(k))[:12]}" for k in bad)
                         + ". Thư mục có dự đoán của hai lượt E1 (mã, cấu hình, lượt khói hay file dữ "
                           "liệu khác nhau): chạy lại E1 cho đủ mọi lần chia bằng cùng một mã.")
    for t, v in sig.items():
        if v.get("smoke") and not smoke:
            sys.exit(f"npz E1 (tag {t}) là lượt KHÓI (meta.smoke = true) mà E2 không chạy --smoke.")
    check = {"centers_json": None}
    cj = preds_io.load_json(centers_json) if centers_json and os.path.exists(centers_json) else None
    if cj is not None:
        fps = (cj.get("meta") or {}).get("fingerprints") or {}
        if not fps:
            print(f"  [cảnh báo] {centers_json} không có meta.fingerprints: không đối chiếu được dấu npz",
                  flush=True)
            check["centers_json"] = "no_fingerprints"
        else:
            for t, v in sig.items():
                ph = v.get("phase")
                want = fps.get(f"phase{ph}") if ph in (1, 2) else None
                ok = (v.get("fingerprint") == want) if want else (v.get("fingerprint") in set(fps.values()))
                if not ok:
                    sys.exit(f"npz E1 (tag {t}, giai đoạn {ph}) có dấu {str(v.get('fingerprint'))[:12]} khác "
                             f"{centers_json} meta.fingerprints {({k: str(x)[:12] for k, x in fps.items()})}: "
                             "vai trò/G1 đọc từ JSON này không mô tả các npz đang đọc.")
            check["centers_json"] = {"path": os.path.abspath(centers_json), "fingerprints": fps}
    return {t: {k: v.get(k) for k in E1_SIG_KEYS} for t, v in sig.items()}, check


def _collect(obj, key, out):
    """Mọi giá trị của khoá `key` ở mọi độ sâu (E2b ghi e1_npz_sha256 ở từng mục K và ở
    mục thứ cấp, lược đồ có thể lồng thêm)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, str):
                out.append(v)
            else:
                _collect(v, key, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect(v, key, out)
    return out


def check_e2b_inputs(wj, inputs, seeds):
    """So e1_npz_sha256 mà E2b ghi với sha256 của npz E1 mà E2 đọc, từng lần chia.
    Lệch thì dừng (lựa chọn 4). Trả số mục đã khớp và số lần chia E2b không ghi băm."""
    if wj is None:
        return {"status": "no_wtrain_json"}
    bad, n_ok, no_sha = [], 0, []
    for s in map(str, seeds):
        ps = (wj.get("per_split") or {}).get(s)
        if not isinstance(ps, dict):
            continue
        shas = _collect(ps, "e1_npz_sha256", [])
        if not shas:
            no_sha.append(s)
            continue
        mine = set(inputs[s].values())
        for h in shas:
            if h in mine:
                n_ok += 1
            else:
                bad.append((s, h))
    if bad:
        s, h = bad[0]
        sys.exit(f"wtrain_tuned.json (E2b) khớp trên npz E1 khác npz E2 đang đọc: lần chia {s}, "
                 f"e1_npz_sha256 {h[:12]} không thuộc {[x[:12] for x in inputs[s].values()]} "
                 f"({len(bad)} mục lệch). E1 đã chạy lại sau E2b: chạy lại E2b trên npz hiện tại.")
    if no_sha:
        print(f"  [cảnh báo] wtrain_tuned.json không ghi e1_npz_sha256 ở lần chia {no_sha}: không "
              "kiểm được E2b đọc cùng npz E1", flush=True)
    return {"status": "checked", "n_matched": n_ok, "splits_without_sha": no_sha}


# ---------------------------------------------------------------------------
# Một lần chia
# ---------------------------------------------------------------------------
def point_metrics(y, pred, lo, hi, phi_y):
    """Thước đo mô tả, không phụ thuộc họ trọng số (cost_K tính riêng)."""
    rr = sp.region_rmse(y, pred, lo, hi)
    e2 = (y - pred) ** 2
    b = bin_index(y)                    # 8 bin cố định của src/bins.py
    macro = float(np.mean([np.sqrt(e2[b == i].mean()) for i in np.unique(b)]))
    return {"region_rmse": {k: rr[k] for k in REGIONS}, "all": rr["All"], "tails": rr["Tails"],
            "macro_bin": macro, "sera": sera(y, pred, phi_y)}


def calib_slope(yhat, y):
    """Hệ số của y theo ŷ (OLS): < 1 là trung tâm quá phân tán."""
    yhat, y = np.asarray(yhat, float), np.asarray(y, float)
    v = np.var(yhat)
    return float(np.cov(yhat, y, bias=True)[0, 1] / v) if v > 0 else float("nan")


def eval_center(p_oof, y_tr, p_te, y_te, lo, hi, wgrid, w_te, phi_te, families, keep_ks):
    """(bảng kết quả của trung tâm, {(rule, K): dự đoán test họ bậc thang ở keep_ks}).

    R1 và R7 khớp MỘT lần (phân vị/mômen phần dư theo bin không phụ thuộc w) rồi
    predict với từng wfun; R2..R5 khớp lại cho từng tham số vì w vào chính phép khớp."""
    fixed = {"R0": np.asarray(p_te, float).copy(),
             "R1_1": dl.make_rule("R1_1").fit(p_oof, y_tr).predict(p_te),
             "R6": dl.make_rule("R6").fit(p_oof, y_tr).predict(p_te)}
    fixed_m = {r: point_metrics(y_te, p, lo, hi, phi_te) for r, p in fixed.items()}
    r1 = dl.make_rule("R1").fit(p_oof, y_tr)
    r7 = dl.make_rule("R7").fit(p_oof, y_tr)
    keep_keys = {pkey(k) for k in keep_ks}
    out, kept = {}, {}
    for fam, rules in families.items():
        for prm, wf in wgrid[fam]:
            key = pkey(prm)
            w = w_te[fam][key]
            for rule in rules:
                info = None
                if rule in fixed:
                    pred, m = fixed[rule], fixed_m[rule]
                else:
                    if rule == "R1":
                        pred = r1.predict(p_te, wfun=wf)
                    elif rule == "R7":
                        pred = r7.predict(p_te, wfun=wf)
                    else:
                        r = dl.make_rule(rule).fit(p_oof, y_tr, wf)
                        pred = r.predict(p_te)
                        info = r.info() if rule in INFO_RULES else None
                    m = point_metrics(y_te, pred, lo, hi, phi_te)
                rec = dict(m, cost_K=sp.cost_w(y_te, pred, w))
                if info:
                    rec["info"] = info
                out.setdefault(rule, {}).setdefault(fam, {})[key] = rec
                if fam == "step" and key in keep_keys:
                    kept[(rule, key)] = pred
    if "step" in families:
        # R6 không phụ thuộc K: chỉ là tham chiếu ở K = 1 (mục 5.4)
        out["R6"] = {"step": {"1": dict(fixed_m["R6"], cost_K=sp.cost_w(y_te, fixed["R6"], np.ones(len(y_te))))}}
        kept[("R6", "1")] = fixed["R6"]
    return out, kept


def cell_specs(roles, c3_center):
    """[(tên ô, (trung tâm, quy tắc) vế trái, vế phải)]: giá trị = cost(trái) - cost(phải)."""
    P, B, D = roles["primary"], roles["bag_Bstar"], roles["default"]
    cells = [("default_to_bag|R0", (D, "R0"), (B, "R0")),
             ("default_to_bag|R1", (D, "R1"), (B, "R1")),
             ("i", (B, "R1"), (P, "R1")),
             ("i_R0", (B, "R0"), (P, "R0")),
             ("ii", (P, "R0"), (P, "R1")),
             ("ii_a", (P, "R0"), (P, "R1_1")),
             ("ii_b", (P, "R1_1"), (P, "R1"))]
    cells += [(f"iii|{r}", (P, r), (P, "R1")) for r in ITER_RULES]
    cells += [("C2", (P, "R1"), (P, "R5"))]
    if c3_center:
        cells += [("C3", (c3_center, "R1"), (B, "R1"))]
    return [c for c in cells if c[1] != c[2]]           # hai vế trùng: chênh bằng 0, bỏ


def run_split(seed, paths, roles, cfg):
    """Mọi trung tâm, quy tắc, họ trọng số và bootstrap cụm cho một lần chia."""
    t0 = time.time()
    d = load_merged(paths)
    y_tr, y_te = np.asarray(d["y_tr"], float), np.asarray(d["y_te"], float)
    lo, hi = dl.tail_cutoffs(y_tr)
    school_te = preds_io.rows(d, "te", "school_code")
    avail = list(d["oof"])
    need = sorted(set(roles.values()))
    miss = [c for c in need if c not in d["oof"] or c not in d["test"]]
    if miss:
        raise KeyError(f"lần chia {seed}: npz thiếu trung tâm {miss}; có {avail}")
    c3 = cfg["c3_center"] if cfg["c3_center"] in d["oof"] and cfg["c3_center"] in d["test"] else None

    wgrid = dl.weight_grid(y_tr, lo, hi, step_Ks=cfg["k_dense"], lambdas=cfg["lambdas"],
                           phi_Ks=cfg["k_dense"], asym_pairs=cfg["asym_pairs"])
    w_te = {fam: {pkey(p): np.asarray(wf(y_te), float) for p, wf in params}
            for fam, params in wgrid.items()}
    phi_te = Relevance(y_tr)(y_te)

    res, preds = {}, {}
    for c in need + ([c3] if c3 and c3 not in need else []):
        fams = FAMILY_RULES if c in need else {"step": ["R0", "R1"]}   # C3: chỉ cần R1
        res[c], kept = eval_center(d["oof"][c], y_tr, d["test"][c], y_te, lo, hi, wgrid, w_te,
                                   phi_te, fams, SAVE_KS)
        for (rule, key), p in kept.items():
            preds[(c, rule, key)] = p
        # R1 ở K = 1 phải trùng R1₁ (w ≡ 1): kiểm cài đặt ngay trên số thật
        if "R1_1" in res[c]:
            a, b = res[c]["R1"]["step"]["1"]["cost_K"], res[c]["R1_1"]["step"]["1"]["cost_K"]
            assert abs(a - b) < 1e-9, f"R1(K=1) {a} khác R1_1 {b}"

    boot = {}
    for K in cfg["k_grid"]:
        k = pkey(K)
        boot[k] = {}
        for name, (cl, rl), (cr, rr) in cell_specs(roles, c3):
            p1, p2 = preds[(cl, rl, k)], preds[(cr, rr, k)]
            b = sp.cluster_boot_diff(y_te, p1, p2, school_te, K, K, lo, hi, B=cfg["boot_B"], seed=int(seed))
            b["flag_per_1000"] = {str(c): float(np.mean((p1 >= c) != (p2 >= c)) * 1000) for c in E9_THRESHOLDS}
            boot[k][name] = b

    y_all = np.concatenate([y_tr, y_te])
    info = {"lo": lo, "hi": hi, "n_tr": int(len(y_tr)), "n_te": int(len(y_te)),
            "mass_tr": dl.tail_masses(y_tr, lo, hi), "mass_te": dl.tail_masses(y_te, lo, hi),
            "npz_lo_hi": {p: [m.get("lo"), m.get("hi")] for p, m in d["meta_by_file"].items()},
            "n_schools_te": int(len(np.unique(school_te))), "sd_y": float(np.std(y_all, ddof=1)),
            "centers_available": avail, "c3_center": c3,
            "calib": {c: {"slope_oof": calib_slope(d["oof"][c], y_tr),
                          "slope_test": calib_slope(d["test"][c], y_te),
                          "sd_ratio_test": float(np.std(d["test"][c]) / np.std(y_te))}
                      for c in res},
            "seconds": None}
    for p, (a, b) in info["npz_lo_hi"].items():
        if a is not None and (float(a), float(b)) != (lo, hi):
            print(f"  [cảnh báo] lần chia {seed}: lo/hi trong {p} = {a}/{b}, tính lại = {lo:g}/{hi:g}",
                  flush=True)

    if cfg["save_preds"]:
        out_p = preds_io.split_path(cfg["save_preds"], seed)
        preds_io.save_split(out_p, idx_te=d["idx_te"], y_te=y_te,
                            # quy tắc không phụ thuộc K chỉ lưu một bản (khoá K = 1)
                            test={f"{c}|{r}|{k}": p for (c, r, k), p in preds.items()
                                  if r not in K_FREE or k == "1"},
                            meta={"seed": int(seed), "lo": lo, "hi": hi, "roles": roles,
                                  "fingerprint": cfg["fingerprint"], "source": paths})
    info["seconds"] = round(time.time() - t0, 1)
    P = roles["primary"]
    k3 = pkey(PRIMARY_K)
    print(f"  split {seed}: cost_{k3} trên {P}: " + " ".join(
        f"{r}={res[P][r]['step'][k3]['cost_K']:.3f}" for r in ["R0", "R1_1", "R1", "R2", "R3", "R4", "R5", "R7"])
        + f"; lo/hi {lo:g}/{hi:g} ({info['seconds']}s)", flush=True)
    return seed, {"per_split": res, "split_info": info, "boot": boot}


# ---------------------------------------------------------------------------
# Đọc R8* của E2b (wtrain_tuned.json), chịu nhiều lược đồ
# ---------------------------------------------------------------------------
R8_ALIASES = {"R8": ["R8", "single", "wtrain"], "R8_bag5": ["R8_bag5", "bag5", "R8bag5", "wtrain_bag5"]}
COST_KEYS = ["cost_K", "cost", "test_cost", "cost_k"]


def _num(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x):
        return float(x)
    return None


def _kkeys(K):
    k = pkey(K)
    return [k, str(K), f"{float(K)}", f"K{k}", f"K={k}"]


def _cost_at(obj, K):
    """cost_K dạng số, hoặc dict theo K."""
    v = _num(obj)
    if v is not None:
        return v
    if isinstance(obj, dict):
        for k in _kkeys(K):
            if k in obj and _num(obj[k]) is not None:
                return _num(obj[k])
    return None


def _pick(d, keys):
    return next((d[k] for k in keys if isinstance(d, dict) and k in d), None)


def r8_cost(wj, seed, K, which):
    """(cost_K test của R8 hoặc R8_bag5, đường dẫn khoá đã dùng) từ wtrain_tuned.json.

    Lược đồ chính là của src/wtrain_tuned.py (per_split[seed][K].cost_K[biến thể]).
    Khung bài chỉ ghi per_split[seed][K] = {..., cost_K, ...}, nên khi lược đồ đó đổi,
    hàm thử thêm vài chỗ hợp lý và GHI LẠI khoá đã dùng để người đọc kiểm."""
    ps = _pick((wj or {}).get("per_split", {}), [str(seed)])
    if not isinstance(ps, dict):
        return None, f"wtrain_tuned.json không có lần chia {seed}"
    cands = []
    entry = _pick(ps, _kkeys(K))
    if isinstance(entry, dict):
        # Lược đồ của src/wtrain_tuned.py: cost_K = {"R8": x, "R8_bag5": y}, test[biến thể].cost_K
        ck = entry.get("cost_K")
        v = _num(ck.get(which)) if isinstance(ck, dict) else None
        if v is not None:
            return v, f"per_split[{seed}][{pkey(K)}][cost_K][{which}]"
        t = entry.get("test")
        v = _num(t[which].get("cost_K")) if isinstance(t, dict) and isinstance(t.get(which), dict) else None
        if v is not None:
            return v, f"per_split[{seed}][{pkey(K)}][test][{which}][cost_K]"
        # Lược đồ khác (dự phòng): thử các chỗ hợp lý và ghi lại khoá đã dùng
        for a in R8_ALIASES[which]:
            if isinstance(entry.get(a), dict):
                cands.append((f"per_split[{seed}][{pkey(K)}][{a}]", entry[a]))
        if which == "R8":
            cands.append((f"per_split[{seed}][{pkey(K)}]", entry))
    for a in R8_ALIASES[which]:
        sub = ps.get(a)
        e2 = _pick(sub, _kkeys(K)) if isinstance(sub, dict) else None
        if isinstance(e2, dict):
            cands.append((f"per_split[{seed}][{a}][{pkey(K)}]", e2))
    for path, obj in cands:
        for scope, o in (("", obj), ("[test]", obj.get("test"))):
            if not isinstance(o, dict):
                continue
            for ck in COST_KEYS:
                if ck in o:
                    v = _cost_at(o[ck], K)
                    if v is not None:
                        return v, f"{path}{scope}[{ck}]"
    return None, f"không tìm thấy cost_K của {which} ở K = {pkey(K)}, lần chia {seed}"


def check_vs_e2b(wj, per, seeds, P, which, r8_values, k_grid):
    """Đối chiếu với C1 mà E2b tự tính (per_split[seed][K].centers[P]): E2b khớp R1
    bằng cùng decision_layer trên cùng npz E1 nên R1 và C1 phải trùng tới sai số làm
    tròn. Lệch nghĩa là hai thí nghiệm đọc hai bản npz, hay hai định nghĩa đuôi khác
    nhau, và ô R8* trong Bảng IV không đáng tin."""
    d_r1, d_c1, star_ok = [], [], []
    for K in k_grid:
        k = pkey(K)
        for s in seeds:
            ps = _pick((wj or {}).get("per_split", {}), [str(s)])
            entry = _pick(ps, _kkeys(K)) if isinstance(ps, dict) else None
            c = ((entry.get("centers") or {}).get(P)) if isinstance(entry, dict) else None
            if not isinstance(c, dict):
                continue
            mine = _cost(per, s, P, "R1", "step", k)
            d_r1.append(abs(_f(c.get("R1")) - mine))
            if r8_values[k][s] is not None:
                d_c1.append(abs(_f(c.get("C1")) - (mine - r8_values[k][s])))
            star_ok.append(c.get("r8_star") in (None, which))
    out = {"n": len(d_r1), "max_abs_R1": max(d_r1) if d_r1 else None,
           "max_abs_C1": max(d_c1) if d_c1 else None, "r8_star_match": all(star_ok) if star_ok else None}
    bad = [v for v in (out["max_abs_R1"], out["max_abs_C1"]) if v is not None and not v <= 1e-6]
    if bad or out["r8_star_match"] is False:
        print(f"  [cảnh báo] R1/C1 lệch với wtrain_tuned.json: {out}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Tóm tắt
# ---------------------------------------------------------------------------
def ms(vals):
    v = np.asarray([_f(x) for x in vals], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0, "q025": float("nan"), "q975": float("nan")}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v)),
            "q025": float(np.quantile(v, 0.025)), "q975": float(np.quantile(v, 0.975))}


def _cost(per, s, center, rule, fam, key):
    try:
        return _f(per[s][center][rule][fam][key]["cost_K"])
    except (KeyError, TypeError):
        return float("nan")


def _flat(rec):
    return {**{k: rec["region_rmse"][k] for k in REGIONS}, "all": rec["all"], "tails": rec["tails"],
            "cost_K": rec["cost_K"], "macro_bin": rec["macro_bin"], "sera": rec["sera"]}


def describe(per, seeds):
    """table[center][rule][family][param][thước đo] = trung bình, SD, dải 95% qua lần chia."""
    first = per[seeds[0]]
    out = {}
    for c, rules in first.items():
        for r, fams in rules.items():
            for fam, params in fams.items():
                for key in params:
                    recs = [_flat(per[s][c][r][fam][key]) for s in seeds
                            if key in per[s].get(c, {}).get(r, {}).get(fam, {})]
                    out.setdefault(c, {}).setdefault(r, {}).setdefault(fam, {})[key] = {
                        m: ms([x[m] for x in recs]) for m in recs[0]}
    return out


def frontier(per, seeds, table):
    """Mô tả (mục 5.6): điểm trung bình (RMSE giữa, RMSE đuôi) theo tham số, và giá trị
    thước đo khi RMSE giữa tăng Δ so với R0 của cùng trung tâm (along_middle)."""
    out = {}
    for c, rules in table.items():
        for r, fams in rules.items():
            if r in K_FREE:
                continue
            for fam, params in fams.items():
                if fam == "step_pair":
                    continue
                keys = list(params)
                pts = [{"param": k, **{m: params[k][m]["mean"] for m in ("Middle", "tails", "Low tail",
                                                                         "High tail", "cost_K")}}
                       for k in keys]
                along = {m: [] for m in FRONTIER_METRICS}
                for s in seeds:
                    base = _f(per[s][c]["R0"]["step"]["1"]["region_rmse"]["Middle"])
                    sp_pts = [{"Middle": _f(per[s][c][r][fam][k]["region_rmse"]["Middle"]),
                               "Tails": _f(per[s][c][r][fam][k]["tails"]),
                               "macro_bin": _f(per[s][c][r][fam][k]["macro_bin"]),
                               "sera": _f(per[s][c][r][fam][k]["sera"])} for k in keys]
                    for m in FRONTIER_METRICS:
                        along[m].append(along_middle(sp_pts, base, m))
                out.setdefault(c, {})[f"{r}|{fam}"] = {
                    "points": pts,
                    "along_middle": {m: {str(dd): ms([a[str(dd)] for a in along[m]]) for dd in MID_BUDGETS}
                                     for m in FRONTIER_METRICS}}
    return out


def _sem_units(mean, sd_y):
    out = {"assumed_4.3": mean / SEM_ASSUMED * 100}
    for rel in RELIABILITY:
        out[f"rel_{rel:.2f}"] = mean / (sd_y * np.sqrt(1 - rel)) * 100
    return out


def _cell_stats(diffs, boots, sd_y):
    """Một ô: nb (Nadeau-Bengio), tost (±SESOI), bootstrap cụm theo lần chia, đổi đơn vị."""
    v = np.asarray([_f(x) for x in diffs], dtype=float)
    fin = v[np.isfinite(v)]
    out = {"per_split": v.tolist(), "mean": float(fin.mean()) if len(fin) else float("nan"),
           "n": int(len(fin)), "n_pos": int((fin > 0).sum()),
           "nb": sp.nb_ttest(v, ratio=NB_RATIO), "tost": sp.tost_nb(v, margin=SESOI, ratio=NB_RATIO)}
    bs = [b for b in boots if b is not None]
    if bs:
        out["boot"] = {"n_splits": len(bs), "n_excl0": int(sum(bool(b["excludes_zero"]) for b in bs)),
                       "ci_lo": ms([b["ci_lo"] for b in bs])["mean"],
                       "ci_hi": ms([b["ci_hi"] for b in bs])["mean"]}
        out["flag_per_1000"] = {c: ms([b["flag_per_1000"][c] for b in bs])["mean"]
                                for c in bs[0]["flag_per_1000"]}
    else:
        out["boot"], out["flag_per_1000"] = None, None
    out["pct_sem"] = _sem_units(out["mean"], sd_y) if np.isfinite(out["mean"]) else None
    return out


def summarize(res, seeds, roles, cfg, wj, wsrc):
    per, boot, info = res["per_split"], res["boot"], res["split_info"]
    P, B = roles["primary"], roles["bag_Bstar"]
    sd_y = float(np.mean([_f(info[s]["sd_y"]) for s in seeds]))
    c3 = info[seeds[0]].get("c3_center")
    table = describe(per, seeds)

    # R8* (E2b) theo lần chia
    which = r8_star(P)
    r8 = {"which": which, "source": wsrc, "values": {}, "paths": {}, "missing": {}}
    for K in cfg["k_grid"]:
        k = pkey(K)
        for s in seeds:
            v, path = r8_cost(wj, s, K, which) if wj is not None else (None, "không có wtrain_tuned.json")
            r8["values"].setdefault(k, {})[s] = v
            (r8["paths"] if v is not None else r8["missing"]).setdefault(k, {})[s] = path
    r8["check_vs_e2b"] = check_vs_e2b(wj, per, seeds, P, which, r8["values"], cfg["k_grid"])

    dec = {}
    for K in cfg["k_grid"]:
        k = pkey(K)
        dec[k] = {}
        for name, (cl, rl), (cr, rr) in cell_specs(roles, c3):
            diffs = [_cost(per, s, cl, rl, "step", k) - _cost(per, s, cr, rr, "step", k) for s in seeds]
            boots = [boot[s][k].get(name) for s in seeds]
            dec[k][name] = _cell_stats(diffs, boots, sd_y) | {"left": [cl, rl], "right": [cr, rr]}
        if P == B:
            dec[k]["i"] = {"mean": 0.0, "note": "trung tâm chính là bag B*: (i) = 0 theo định nghĩa"}
        diffs = [_f(r8["values"][k][s]) - _cost(per, s, P, "R1", "step", k) for s in seeds]
        dec[k]["iii|R8*"] = _cell_stats(diffs, [None] * len(seeds), sd_y) | {
            "left": [P, which], "right": [P, "R1"],
            "note": "cost_K của R8* lấy từ wtrain_tuned.json; không có dự đoán nên không có bootstrap"}

    # Họ Holm thứ cấp của Bảng IV/V (lựa chọn 2 trong docstring)
    k3 = pkey(PRIMARY_K)
    dup = {(k3, "iii|R5"), (k3, "iii|R8*")} | ({(k3, "i")} if c3 is not None and P == c3 else set())
    fam_cells = [(pkey(K), n) for K in cfg["k_grid"]
                 for n in ["i", "ii_a", "ii_b"] + [f"iii|{r}" for r in ITER_RULES] + ["iii|R8*"]
                 if (pkey(K), n) not in dup and "nb" in dec[pkey(K)].get(n, {})]
    adj = sp.holm([dec[k][n]["nb"]["p"] for k, n in fam_cells])
    for (k, n), a in zip(fam_cells, adj):
        dec[k][n]["p_holm"] = a
        dec[k][n]["holm_family"] = "table_iv"
    for k, n in dup:
        if n in dec[k]:
            dec[k][n]["p_holm"] = None
            dec[k][n]["holm_family"] = "primary (xem contrasts)"

    # Phép so chính: C2 ở đây, C1 = -(iii|R8*), C3 từ npz
    contrasts = {"C2": {}}
    for K in cfg["k_grid"]:
        k = pkey(K)
        c2 = dec[k]["C2"]
        c2["p_holm"] = None if k == k3 else dec[k]["iii|R5"].get("p_holm")
        c2["holm_family"] = "primary" if k == k3 else "table_iv (qua iii|R5)"
        contrasts["C2"][k] = c2
    c1_diffs = [-x if np.isfinite(x) else float("nan") for x in dec[k3]["iii|R8*"]["per_split"]]
    contrasts["C1"] = _cell_stats(c1_diffs, [None] * len(seeds), sd_y) | {
        "source": "tính trong decomp_rules: cost_3(R1) - cost_3(R8*) với R8* từ wtrain_tuned.json"}
    if "C3" in dec[k3]:
        contrasts["C3"] = dec[k3]["C3"] | {"source": f"tính trong decomp_rules từ npz ({c3} và {B})"}
    # Lựa chọn 3: chỉ phép so ĐỦ n_ref lần chia vào Holm chính; thiếu thì coi là vắng
    # (p = 1) và ghi incomplete, không để nb_ttest lặng lẽ tính trên phần lần chia còn lại.
    n_ref = cfg["n_ref"]
    p_prim, incomplete = {}, []
    for c, cell in (("C1", contrasts["C1"]), ("C2", dec[k3]["C2"]), ("C3", contrasts.get("C3"))):
        if cell is None:
            continue
        final = bool(cell["n"] == n_ref)
        cell["n_ref"], cell["final"] = n_ref, final
        cell["status"] = "final" if final else f"provisional: {cell['n']}/{n_ref} lần chia"
        if final:
            p_prim[c] = cell["nb"]["p"]
        elif cell["n"] > 0:
            incomplete.append((c, cell["status"]))
    contrasts["primary_holm"] = sp.primary_holm({c: p for c, p in p_prim.items() if c in PRIMARY_CONTRASTS})
    for c, st in incomplete:
        contrasts["primary_holm"][c] |= {"incomplete": True, "status": st}
    contrasts["incomplete"] = [c for c, _ in incomplete]
    contrasts["n_ref"], contrasts["n_splits_done"] = n_ref, len(seeds)
    contrasts["C2"][k3]["p_holm"] = contrasts["primary_holm"]["C2"]["p_holm"]

    tost = {r: {pkey(K): dec[pkey(K)][f"iii|{r}"]["tost"] for K in cfg["k_grid"]} for r in TOST_RULES}

    # Thứ cấp: R2 - R1 và R5 - R1 ở họ prior, φ, bất đối xứng; Holm trong từng họ
    secondary = {}
    for fam in ("prior", "phi", "step_pair"):
        cells, ps = [], []
        for key in table[P]["R1"][fam]:
            for r in SECONDARY_VS_R1:
                diffs = [_cost(per, s, P, r, fam, key) - _cost(per, s, P, "R1", fam, key) for s in seeds]
                st = {"per_split": diffs, "nb": sp.nb_ttest(diffs, ratio=NB_RATIO),
                      "tost": sp.tost_nb(diffs, margin=SESOI, ratio=NB_RATIO)}
                cells.append((f"{r}-R1|{key}", st))
                ps.append(st["nb"]["p"])
        for (name, st), a in zip(cells, sp.holm(ps)):
            st["p_holm"] = a
        secondary[fam] = dict(cells)

    # R5: số lần s chạm biên lưới mới, và số lần nằm ngoài lưới cũ [0,80; 2,50]
    edges = {}
    for c in table:
        for fam in table[c].get("R5", {}):
            hits = out_old = tot = 0
            for s in seeds:
                for rec in per[s][c]["R5"][fam].values():
                    inf = rec.get("info")
                    if not inf:
                        continue
                    tot += 1
                    hits += bool(inf.get("edge_low")) or bool(inf.get("edge_high"))
                    out_old += any(not (0.80 <= _f(inf.get(x)) <= 2.50) for x in ("s_low", "s_high"))
            edges.setdefault(c, {})[fam] = {"n": tot, "edge_hit_frac": hits / tot if tot else None,
                                            "outside_old_grid_frac": out_old / tot if tot else None}

    calib = {c: {m: ms([info[s]["calib"][c][m] for s in seeds if c in info[s]["calib"]])
                 for m in ("slope_oof", "slope_test", "sd_ratio_test")} for c in table}

    return {"roles": roles, "seeds": seeds, "sd_y": sd_y, "table": table,
            "frontier": frontier(per, seeds, table), "decomposition": dec, "contrasts": contrasts,
            "tost_vs_R1": tost, "secondary": secondary, "stretch_edges": edges, "calibration": calib,
            "r8": r8, "gate_G2": gate_g2(dec, contrasts, calib, P, cfg["k_grid"])}


def gate_g2(dec, contrasts, calib, P, k_grid):
    """Cổng G2 (mục 6.5). provisional = True khi họ chính còn thiếu C1, C2 hoặc C3 (vắng
    hay chưa đủ lần chia), hoặc khi một ô mà G2 đọc (C2, TOST của R2/R4, ii, ii_a) tính
    trên ít hơn n_ref lần chia (lựa chọn 3)."""
    k3 = pkey(PRIMARY_K)
    c2 = dec[k3]["C2"]
    ph = contrasts["primary_holm"]
    n_ref = contrasts["n_ref"]
    missing = [c for c in PRIMARY_CONTRASTS if ph[c]["missing"]]
    used = [(k3, "C2"), (k3, "ii"), (k3, "ii_a")] + [(pkey(K), "C2") for K in (5, 8) if pkey(K) in dec]
    used += [(pkey(K), f"iii|{r}") for K in k_grid for r in ("R2", "R4")]
    short = [f"{n}@K{k}" for k, n in used if dec[k][n].get("n") != n_ref]
    m = c2["mean"]
    out = {"provisional": bool(missing or short), "primary_missing": missing,
           "primary_incomplete": list(contrasts.get("incomplete", [])), "cells_incomplete": short,
           "n_ref": n_ref, "C2_mean": m, "C2_n": c2["n"],
           "C2_p": c2["nb"]["p"], "C2_p_holm": ph["C2"]["p_holm"], "C2_tost": c2["tost"]["passed"]}
    if np.isfinite(m) and m <= -SESOI and ph["C2"]["p_holm"] < ALPHA:
        out["verdict"] = "bayes_better"
        out["text"] = "ở K = 3, quy tắc Bayes hơn giãn hai phía"
    elif c2["tost"]["passed"]:
        out["verdict"] = "stretch_sufficient"
        ks = [K for K in (5, 8) if pkey(K) in dec and dec[pkey(K)]["C2"]["mean"] <= -SESOI]
        out["k_stretch_breaks"] = ks[0] if ks else None
        out["text"] = ("ở K = 3 giãn hai phía đủ dùng" +
                       (f"; từ K = {ks[0]} chênh thứ cấp <= -{SESOI:g}" if ks else ""))
    else:
        out["verdict"] = "inconclusive"
        out["text"] = "C2 không qua ngưỡng hiệu ứng, cũng không qua TOST"
    ks = [pkey(K) for K in k_grid]
    for r, flag in (("R2", "R2_equiv_R1_all_K"), ("R4", "R4_equiv_R1_all_K")):
        out[flag] = all(dec[k][f"iii|{r}"]["tost"]["passed"] for k in ks)
    if out["R2_equiv_R1_all_K"]:
        out["text_R2"] = "R2 tương đương R1 ở cả bốn K: khuyến nghị R2 làm mặc định, R1 vào phụ lục"
    if out["R4_equiv_R1_all_K"]:
        out["text_R4"] = "R4 tương đương R1: hiệu chỉnh tuyến tính có trọng số đã đủ"
    # Lựa chọn 13: "(ii-a) quá nửa (ii)" chỉ khi tầng quyết định CÓ lợi (ii > 0); tỉ số
    # với ii ≤ 0 đổi dấu và vô nghĩa.
    ii, iia = dec[k3]["ii"]["mean"], dec[k3]["ii_a"]["mean"]
    pos = bool(np.isfinite(ii) and ii > 0)
    out["ii_K3"], out["ii_a_K3"] = ii, iia
    out["ii_a_share_K3"] = iia / ii if pos else float("nan")
    out["ii_a_majority"] = bool(pos and np.isfinite(iia) and iia > 0.5 * ii)
    out["ii_nonpositive"] = bool(np.isfinite(ii) and ii <= 0)
    out["calib_slope_primary"] = calib[P]["slope_test"]["mean"]
    if out["ii_a_majority"]:
        out["text_ii_a"] = ("phần lớn lợi ích quyết định ở K nhỏ là hiệu chỉnh lại trung tâm "
                            f"(độ dốc hiệu chỉnh test {out['calib_slope_primary']:.3f})")
    if out["ii_nonpositive"]:
        out["text_ii_nonpositive"] = (f"ở K = 3 tầng quyết định không có lợi trên trung tâm chính "
                                      f"((ii) = {ii:+.3f} ≤ 0; (ii-a) = {iia:+.3f}): không tách (ii) thành "
                                      "hiệu chỉnh và nghiêng, cổng (ii-a)/(ii) không áp dụng")
    return out


def print_summary(S):
    P = S["roles"]["primary"]
    print(f"\n=== Bảng IV (trung tâm chính {P}, bag B* {S['roles']['bag_Bstar']}, "
          f"default {S['roles']['default']}; {len(S['seeds'])} lần chia) ===")
    print("  ô                    K   trung bình  CI95 NB              p       p_holm  boot loại 0  dương")
    for k, cells in S["decomposition"].items():
        for n, c in cells.items():
            if "nb" not in c:
                print(f"  {n:20s} {k:>2}   {c.get('mean', float('nan')):+.4f}  ({c.get('note', '')})")
                continue
            nb, b = c["nb"], c.get("boot") or {}
            ph = c.get("p_holm")
            print(f"  {n:20s} {k:>2}   {c['mean']:+.4f}   [{nb['ci_lo']:+.4f}, {nb['ci_hi']:+.4f}]  "
                  f"{nb['p']:.4f}  {'  -   ' if ph is None else f'{ph:.4f}'}  "
                  f"{b.get('n_excl0', '-')}/{b.get('n_splits', '-')}        {c['n_pos']}/{c['n']}")
    ph = S["contrasts"]["primary_holm"]
    ct = S["contrasts"]
    print(f"  Họ chính (m = 3; cuối khi đủ {ct['n_ref']} lần chia): " + ", ".join(
        f"{c}: p={v['p']:.4f} p_holm={v['p_holm']:.4f}"
        + (" (CHƯA ĐỦ lần chia: " + v["status"] + ")" if v.get("incomplete")
           else " (thiếu)" if v["missing"] else "")
        for c, v in ph.items()))
    print("  TOST ±%.2f so với R1: " % SESOI + "; ".join(
        f"{r}: " + " ".join(f"K{k}={'qua' if t['passed'] else 'không'}" for k, t in v.items())
        for r, v in S["tost_vs_R1"].items()))
    g = S["gate_G2"]
    why = [f"thiếu {','.join(g['primary_missing'])}"] if g["primary_missing"] else []
    why += [f"chưa đủ {g['n_ref']} lần chia ở {','.join(g['primary_incomplete'] + g['cells_incomplete'])}"] \
        if g["primary_incomplete"] or g["cells_incomplete"] else []
    print(f"  G2: {g['verdict']} ({g['text']})" + (f" [TẠM: {'; '.join(why)}]" if g["provisional"] else " [cuối]"))
    for t in ("text_R2", "text_R4", "text_ii_a", "text_ii_nonpositive"):
        if t in g:
            print(f"      {g[t]}")
    print(f"  (ii-a)/(ii) ở K = 3: {g['ii_a_share_K3']:.3f}; R8* = {S['r8']['which']}, nguồn {S['r8']['source']}")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="E2: tầng quyết định R0..R7 và bảng phân rã (hậu kỳ trên npz E1)")
    ap.add_argument("--data", default=DATA_PATH, help="chỉ ghi vào meta; E2 không đọc CSV")
    ap.add_argument("--out", default=f"results_cost/{EXPERIMENT}.json")
    ap.add_argument("--preds-dir", default="preds/decomp", help="thư mục npz của E1 (đầu vào)")
    ap.add_argument("--tag", nargs="*", default=None,
                    help="tag của npz E1 (split<seed>_<tag>.npz); nhiều tag thì gộp các file")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--smoke", action="store_true", help=f"1 lần chia, bootstrap B = {SMOKE_BOOT_B}")
    ap.add_argument("--workers", type=int, default=1, help="số tiến trình joblib (mỗi tiến trình một lần chia)")
    ap.add_argument("--primary", default="auto", help="tên trung tâm chính trong npz, hoặc auto (đọc E1)")
    ap.add_argument("--bstar", default="auto", help="tên trung tâm bag B* trong npz, hoặc auto")
    ap.add_argument("--default-center", default="default")
    ap.add_argument("--c3-center", default=C3_CENTER, help="vế trái của C3; không có trong npz thì bỏ C3")
    ap.add_argument("--centers-json", default="results_cost/decomp_centers.json")
    ap.add_argument("--wtrain-json", default="results_cost/wtrain_tuned.json")
    ap.add_argument("--boot-B", type=int, default=CLUSTER_BOOT_B)
    ap.add_argument("--save-preds", default=None, help="thư mục lưu dự đoán test theo quy tắc (tuỳ chọn)")
    ap.add_argument("--on-mismatch", default="raise", choices=["raise", "recompute"])
    args = ap.parse_args(argv)

    provenance.print_versions()
    seeds = args.seeds[:1] if args.smoke else args.seeds
    boot_B = SMOKE_BOOT_B if args.smoke else args.boot_B
    roles, roles_src = resolve_roles(args)
    for s in seeds:
        miss = [p for p in npz_paths(args.preds_dir, s, args.tag) if not os.path.exists(p)]
        if miss:
            sys.exit(f"Thiếu npz của E1: {miss} (E1 chưa xong, hoặc sai --preds-dir/--tag)")
    # Lựa chọn 6: mọi npz cùng một lượt E1 (và khớp --centers-json nếu có) trước khi tính
    e1_sig, e1_check = check_e1_inputs(args.preds_dir, seeds, args.tag, args.centers_json, args.smoke)

    cfg = {"script": EXPERIMENT, "smoke": bool(args.smoke), "roles": roles, "c3_center": args.c3_center,
           "tag": args.tag, "k_dense": list(K_DENSE), "lambdas": list(LAMBDA_DENSE),
           "asym_pairs": [list(p) for p in ASYM_PAIRS], "k_grid": list(K_GRID), "boot_B": boot_B,
           "family_rules": FAMILY_RULES,
           # Dấu của E1 vào dấu của E2: .partial tính trên một lượt E1 không được nối tiếp
           # bằng npz của lượt E1 khác, kể cả khi sha256 từng file được kiểm riêng.
           "e1_fingerprints": {t: v["fingerprint"] for t, v in e1_sig.items()}}
    # --save-preds không đổi số nào nên không vào dấu: bật/tắt nó không làm hỏng .partial
    fp = preds_io.fingerprint(cfg)
    # n_ref (lựa chọn 3) chỉ đổi nhãn cuối/tạm của phần tóm tắt, không đổi số của lần chia
    cfg = cfg | {"fingerprint": fp, "save_preds": args.save_preds,
                 "asym_pairs": [tuple(p) for p in ASYM_PAIRS],
                 "n_ref": len(seeds) if args.smoke else len(SEEDS)}
    res = preds_io.load_partial(args.out + ".partial", fp,
                                {"per_split": {}, "split_info": {}, "boot": {}, "inputs": {}},
                                on_mismatch=args.on_mismatch)
    inputs = {str(s): {p: preds_io.file_sha256(p) for p in npz_paths(args.preds_dir, s, args.tag)}
              for s in seeds}
    # Lựa chọn 4: E2b phải khớp trên đúng các npz này; kiểm TRƯỚC khi tính để lệch thì
    # dừng sớm thay vì sau cả lượt chạy
    wj = preds_io.load_json(args.wtrain_json) if os.path.exists(args.wtrain_json) else None
    e2b_check = check_e2b_inputs(wj, inputs, seeds)
    todo = []
    for s in map(str, seeds):
        if s in res["per_split"]:
            if res["inputs"].get(s) == inputs[s]:
                print(f"  split {s}: đã có trong .partial, cùng npz -> bỏ qua", flush=True)
                continue
            msg = f"lần chia {s}: npz đã đổi so với .partial (E1 chạy lại?)"
            if args.on_mismatch == "raise":
                raise preds_io.FingerprintMismatch(msg + "; --on-mismatch recompute để tính lại")
            print(f"  {msg} -> tính lại", flush=True)
            for part in ("per_split", "split_info", "boot", "inputs"):
                res[part].pop(s, None)
        todo.append(s)
    res["meta"].update({"experiment": "E2", "roles": roles, "roles_source": roles_src,
                        "preds_dir": args.preds_dir, "tag": args.tag, "smoke": bool(args.smoke)})
    print(f"E2 {EXPERIMENT}: vai trò {roles}; {len(todo)} lần chia cần tính, "
          f"bootstrap B = {boot_B}, workers = {args.workers}", flush=True)

    jobs = [(int(s), npz_paths(args.preds_dir, s, args.tag)) for s in todo]
    if args.workers > 1 and len(jobs) > 1:
        from joblib import Parallel, delayed
        gen = Parallel(n_jobs=min(args.workers, len(jobs)), return_as="generator_unordered")(
            delayed(run_split)(s, p, roles, cfg) for s, p in jobs)
    else:
        gen = (run_split(s, p, roles, cfg) for s, p in jobs)
    for seed, out in gen:
        s = str(seed)
        for part in ("per_split", "split_info", "boot"):
            res[part][s] = out[part]
        res["inputs"][s] = inputs[s]
        dump(res, args.out + ".partial")          # ghi dần: ngắt giữa chừng vẫn giữ lần chia đã xong

    done = [str(s) for s in seeds if str(s) in res["per_split"]]
    wsrc = None if wj is None else {"path": args.wtrain_json, "sha256": preds_io.file_sha256(args.wtrain_json),
                                    "commit": ((wj.get("meta") or {}).get("provenance") or {}).get("commit")}
    summary = summarize(res, done, roles, cfg, wj, wsrc)
    print_summary(summary)
    final = {k: v for k, v in res.items() if k != "meta"}
    final["summary"] = summary
    final["meta"] = res["meta"] | {
        "protocol": ("Hậu kỳ trên npz E1: quy tắc khớp trên (ŷ OOF, y huấn luyện), áp cho ŷ test của mô "
                     "hình khớp lại; đuôi theo khối lượng 9,25%/5,72% của y huấn luyện; mọi thước đo RMSE; "
                     "NB t (J lần chia, ratio 0,25), TOST ±SESOI, Holm; bootstrap cụm theo trường"),
        "seeds": [int(s) for s in done], "data": args.data, "data_read": False,
        "grids": {"step": list(K_DENSE), "prior": list(LAMBDA_DENSE), "phi": list(K_DENSE),
                  "step_pair": [list(p) for p in ASYM_PAIRS], "decomposition_K": list(K_GRID)},
        "family_rules": FAMILY_RULES, "boot_B": boot_B, "sesoi": SESOI, "nb_ratio": NB_RATIO,
        "alpha": ALPHA, "primary_k": PRIMARY_K, "flag_thresholds": list(E9_THRESHOLDS),
        "centers_json": args.centers_json, "wtrain_json": wsrc, "save_preds": args.save_preds,
        "e1_inputs": {"signature": e1_sig, "check": e1_check}, "e2b_inputs_check": e2b_check,
        "n_ref": cfg["n_ref"],
        "provenance": provenance.stamp()}
    dump(final, args.out)
    print(f"\nĐã ghi {args.out}")
    return final


if __name__ == "__main__":
    main()
