# -*- coding: utf-8 -*-
"""Hằng số chốt trước lượt chạy khẳng định: K chính, phép so chính, ngưỡng, cổng.

Vì sao có file này (khung bài 24/9, mục 6, quy ước chung): K chính, ba phép so
chính và mọi ngưỡng quyết định phải được viết ra và commit TRƯỚC E1. Nếu mỗi script
tự khai hằng số của mình thì một lần sửa "cho đẹp số" ở một chỗ sẽ không để lại dấu
vết, và câu "quy tắc quyết định cố định trước lượt chạy khẳng định" trong bài không
còn đúng. Mọi script thí nghiệm import từ đây; đừng chép lại giá trị.

File này chỉ chứa dữ liệu (không import gì nặng), để đọc được cả ở máy không có
xgboost. Giá trị lấy từ Bảng II (mục 5.8) và các cổng ở mục 6.2 đến 6.13.
"""

# ---------------------------------------------------------------------------
# Lần chia và thống kê cặp
# ---------------------------------------------------------------------------
# 10 lần chia 80/20. Seed 42 và 1..4 đã dùng trong thăm dò nên seed khẳng định
# bắt đầu từ 100 để không trùng lần chia nào đã nhìn.
SEEDS = list(range(100, 110))
TEST_SIZE = 0.2
# Tỉ lệ n_te/n_tr cho hiệu chỉnh Nadeau-Bengio: 0,2/0,8.
NB_RATIO = 0.25
ALPHA = 0.05
# Bootstrap cụm theo trường trong từng tập kiểm tra.
CLUSTER_BOOT_B = 2000

# Ngưỡng hiệu ứng tối thiểu (điểm cost_K hoặc RMSE). Đặt SAU khi xem thăm dò, bài
# nói thẳng như vậy: khoảng 1% RMSE, khoảng 2,3% SEM giả định 4,3, gấp khoảng 5
# lần SE Nadeau-Bengio của chênh cặp giữa hai trung tâm gần nhau.
SESOI = 0.10
SEM_ASSUMED = 4.3

# ---------------------------------------------------------------------------
# Chi phí và họ trọng số
# ---------------------------------------------------------------------------
PRIMARY_K = 3
PRIMARY_FAMILY = "step"            # bậc thang đối xứng K_L = K_H = K
K_GRID = [2, 3, 5, 8]              # lưới báo cáo (K = 3 là chính, còn lại thứ cấp)
E1_KS = [1, 2, 3, 5, 8]            # E1 áp R0, R1, R1₁ ở các K này
E4_KS = [1, 1.5, 2, 3, 5, 8, 12, 20]
# Lưới dày cho biên (mục 5.6): trên lưới cũ 8 điểm, dây cung nội suy nằm trên biên
# lồi và phạt cách làm nhảy bước lớn; lưới dày làm độ lệch này nhỏ (E5 đo).
K_DENSE = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50]
LAMBDA_DENSE = [round(0.05 * i, 2) for i in range(21)]          # 0; 0,05; ...; 1
ASYM_PAIRS = [(5, 1), (1, 5), (3, 1), (1, 3)]                  # (K_L, K_H), thứ cấp

# Đuôi theo KHỐI LƯỢNG của y huấn luyện, không theo điểm cố định: định nghĩa này
# mang sang được các bộ dữ liệu E11. Trên khoá HSA 2024 nó trùng 60/100 của
# src/bins.py (chỉ lệch do điểm trùng). Cắt bằng decision_layer.tail_cutoffs.
TAIL_MASS_LOW = 0.0925
TAIL_MASS_HIGH = 0.0572
# Vùng gần đoán mò: kỳ vọng đoán mò (~28) cộng 2 SD (~4,6), giả định cấu trúc đề.
NEAR_GUESS_MAX = 37
Y_MIN, Y_MAX = 0.0, 150.0          # thang HSA, dùng để cắt điểm mẫu của R7

# Họ prior (thứ cấp): giữ như thăm dò (tail_prior.KDE_SIGMA, DENS_FLOOR).
KDE_SIGMA = 2.0
DENS_FLOOR = 0.01

# ---------------------------------------------------------------------------
# Tầng quyết định (mục 5.4) và độ nhạy (E5)
# ---------------------------------------------------------------------------
N_BINS = 20                        # B: ~2.290 phần dư mỗi bin với 45.739 dòng huấn luyện
N_SAMPLES = 50                     # S: độ phân giải xác suất 2%
# Lưới s của giãn hai phía. Lưới cũ [0,80; 2,50] có thể chặn ở K lớn nên mở rộng;
# R5 ghi lại việc s chạm biên để E5 đếm.
STRETCH_S_MIN, STRETCH_S_MAX, STRETCH_S_STEP = 0.50, 4.00, 0.01
STRETCH_S_OLD = (0.80, 2.50)
RULE_CROSSFIT_FOLDS = 5            # E4: khớp chéo quy tắc ngay trong OOF

E5_N_BINS = [10, 20, 40, 80]
E5_N_SAMPLES = [25, 50, 100]
E5_BINNING = ["count", "width"]
E5_TAIL_DEFS = [("mass", 0.0925, 0.0572), ("mass", 0.05, 0.05),
                ("fixed", 50, 110), ("fixed", 41, 113)]      # 41/113: relevance boxplot
E5_KDE_SIGMA = [1.0, 2.0, 4.0]
E5_DENS_FLOOR = [0.001, 0.01, 0.05]
E5_TREE_FACTOR = [1.0, 1.2]
E5_SIGN_KEEP = 0.90                # tuyên bố không điều kiện khi dấu giữ ở >= 90% thiết lập
E5_EDGE_HIT_MAX = 0.10             # lưới s cũ chạm biên ở > 10% ô -> ghi rõ trong bài

# ---------------------------------------------------------------------------
# Trung tâm và dò siêu tham số (E1, E2b, E4)
# ---------------------------------------------------------------------------
BAG_SUBSAMPLE = 0.8                # subsample = colsample_bytree của bag
BAG_SIZES = [1, 2, 5, 10, 20, 40]  # trung bình tiền tố của bag40
B_STAR_TOL = 0.02                  # B* nhỏ nhất có |cost_3(bag B) - cost_3(bag 40)| <= 0,02
TUNE_BUDGET = 60                   # cấu hình cho trung tâm và cho mỗi K của R8
TUNE_BUDGET_FALLBACK = 40          # nếu thời gian fit trung bình > FIT_TIME_MAX_S (E0)
FIT_TIME_MAX_S = 5.0
EARLY_STOP_ROUNDS = 50
ES_FRAC = 0.1                      # tập dừng sớm: 10% fold huấn luyện trong
MAX_TREES = 3000                   # 29/40 lượt GA cũ chạm trần 800
R8_BAG = 5                         # R8_bag5 và rs_tuned_bag5: random_state 0..4

# Kiểm tra lại sau khi sửa preprocess (E0): chạy lại base và enc_school trên lần
# chia 42; lệch quá 0,02 so với số cũ chỉ ghi vào ledger.
E0_RECHECK = {"base": 9.908, "enc_school": 9.736, "tol": 0.02}

# ---------------------------------------------------------------------------
# Ba phép so chính (Holm trên {C1; C2; C3} ở ALPHA). Âm = vế trái tốt hơn.
# Mọi phép so khác là thứ cấp, có Holm trong từng bảng và nhãn "thứ cấp".
# Họ chính luôn có m = 3, kể cả khi cổng được tính lúc mới có một hai phép so (ba
# phép so đến từ ba thí nghiệm): dùng stats_paired.primary_holm, không dùng holm()
# trên danh sách đang có, vì họ co lại sẽ cho qua ở p giữa 0,017 và 0,05.
# ---------------------------------------------------------------------------
CONTRASTS = {
    "C1": {"desc": "cost_3(R1) - cost_3(R8*) trên trung tâm chính; R8* = R8_bag5 nếu "
                   "trung tâm chính là trung tâm bag, ngược lại R8",
           "left": {"rule": "R1", "center": "primary"},
           "right": {"rule": "R8*", "center": "primary"},
           "K": PRIMARY_K, "family": PRIMARY_FAMILY, "experiment": "E2b"},
    "C2": {"desc": "cost_3(R1) - cost_3(R5) trên trung tâm chính",
           "left": {"rule": "R1", "center": "primary"},
           "right": {"rule": "R5", "center": "primary"},
           "K": PRIMARY_K, "family": PRIMARY_FAMILY, "experiment": "E2"},
    "C3": {"desc": "cost_3(R1 trên rs_tuned) - cost_3(R1 trên bag B*)",
           "left": {"rule": "R1", "center": "rs_tuned"},
           "right": {"rule": "R1", "center": "bag_Bstar"},
           "K": PRIMARY_K, "family": PRIMARY_FAMILY, "experiment": "E1"},
}
PRIMARY_CONTRASTS = ["C1", "C2", "C3"]


def r8_star(primary_center):
    """Tên nhánh huấn luyện có trọng số dùng trong C1.

    Trung tâm bag (bag B*, rs_tuned_bag5) được so với R8_bag5 để hai vế cùng hưởng
    lợi ích giảm phương sai của bagging; trung tâm một mô hình so với R8."""
    return "R8_bag5" if "bag" in str(primary_center) else "R8"


# ---------------------------------------------------------------------------
# Cổng từng thí nghiệm
# ---------------------------------------------------------------------------
E1_FEATURE_TOST_MARGIN = SESOI     # TOST RMSE bag10 giữa các tập đặc trưng
E1_R2_LOSS_REPORT = 0.02           # F_dt mất >= 0,02 R² so với F_full thì báo cả hai
E3_THETA = list(range(30, 131))    # điểm sơ cấp Ehm et al. 2016
E3_DOMINANCE = 0.90                # trung tâm trội ở >= 90% θ
E3_SPLITS_MIN = 9                  # ... trong ít nhất 9/10 lần chia
E4_KENDALL_MIN = 0.9               # Giả thuyết 7 viết thành câu thực nghiệm nếu τ >= 0,9
E4_TOP = 10                        # độ trùng top-10
E6_REL_ERR_MAX = 0.10
E6_R8_WIN = SESOI

# E9: ngưỡng cố định, khoảng dự đoán, gần đoán mò.
E9_THRESHOLDS = [60, 100]          # thay bằng ngưỡng sàn có nguồn khi người dùng cung cấp
E9_KS = [1, 2, 3, 5, 8]
E9_FLAG_EFFECT = 5.0               # đơn vị L_K trên 1.000 thí sinh
E9_MIN_K_WINS = 3                  # P2 hơn P1 ở >= 3 trong 4 giá trị K > 1
E9_SPEARMAN_MIN = 0.99
E9_PI_COVER_DECILE_MIN = 0.85
E9_PI_COVER_TAIL_MIN = 0.80
E9_PI_LEVEL = 0.90

# E10: biên tương đương nhóm (Bảng II). FNR/FPR theo tỉ lệ (0,05 = 5 điểm %).
# 1 điểm ≈ 0,23 SEM giả định; 2 điểm % FPR ≈ 20 em gắn cờ nhầm trên 1.000; FNR
# lỏng hơn vì tỉ lệ nền nhỏ.
E10_MARGINS = {"a_g": 1.0, "b_g": 0.05, "tau": 1.0, "dFNR": 0.05, "dFPR": 0.02}
E10_GROUPS = ["gender", "region", "chuyen", "school_size_q", "birth_quarter"]
E10_PERMUTATIONS = 500
E10_CROSSFIT = (2, 5)              # 2 lần lặp x 5 fold trên toàn bộ bản ghi
E10_CONFIG_SEED = 100              # cấu hình trung tâm lấy từ lần chia 100 (ghi rõ)
