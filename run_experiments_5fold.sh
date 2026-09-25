#!/usr/bin/env bash
# =============================================================================
# Chạy lại TOÀN BỘ thí nghiệm dưới 5-fold CV (CV_FOLDS = 5 trong src/ga_xgb.py).
#
#   nohup ./run_experiments_5fold.sh > results/run_5fold.log 2>&1 &
#   tail -f results/run_5fold.log
#
# Ước tính ~45–50 giờ trên máy 10 nhân. Script BỎ QUA thí nghiệm đã có kết quả,
# nên ngắt giữa chừng rồi chạy lại là an toàn.
#
# THỨ TỰ CÓ CHỦ Ý: GA-RMSE chạy TRƯỚC vì ngân sách đánh giá dùng chung cho
# random/grid search được lấy bằng đúng số lần đánh giá mà nó tiêu thụ — đó là
# giao kèo "ngân sách như nhau" mà Section IV-B dựa vào. Ở 3-fold con số này là
# 622; dưới 5-fold nó sẽ khác, nên KHÔNG được hardcode lại 622.
# =============================================================================
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
LOG() { echo "[$(date '+%F %T')] $*"; }

run_if_missing() {
  local out="$1"; shift
  if [[ -f "$out" ]]; then
    LOG "SKIP (đã có $out)"
  else
    LOG "RUN: $*"
    python3 "$@" 2>&1 | grep -v -i warning
  fi
}

LOG "CV_FOLDS = $(python3 -c 'import sys;sys.path.insert(0,"src");from ga_xgb import CV_FOLDS;print(CV_FOLDS)')"

LOG "===== BƯỚC 1: GA-RMSE (đặt ngân sách cho mọi chiến lược khác) ====="
run_if_missing results/ga_rmse_best.json src/ga_xgb.py --metric rmse

BUDGET=$(python3 -c "import json;print(json.load(open('results/ga_rmse_best.json'))['total_evals'])")
LOG "Ngân sách đánh giá dùng chung = $BUDGET (từ lần chạy GA-RMSE)"

LOG "===== BƯỚC 2: baseline mặc định + random/grid cùng ngân sách ====="
run_if_missing results/baseline_xgb_default.json src/baseline.py
run_if_missing results/random_search_best.json src/search_baselines.py --strategy random --budget "$BUDGET"
run_if_missing results/grid_search_best.json   src/search_baselines.py --strategy grid   --budget "$BUDGET"

LOG "===== BƯỚC 3: GA với các fitness quy ước ====="
run_if_missing results/ga_mae_best.json src/ga_xgb.py --metric mae
run_if_missing results/ga_r2_best.json  src/ga_xgb.py --metric r2

LOG "===== BƯỚC 4: quét alpha — 5 điểm (alpha=0 chính là GA-RMSE) ====="
run_if_missing results/ga_tail_a0.25_best.json src/ga_xgb.py --metric tail --alpha 0.25
run_if_missing results/ga_tail_a0.5_best.json  src/ga_xgb.py --metric tail --alpha 0.5
run_if_missing results/ga_tail_a0.75_best.json src/ga_xgb.py --metric tail --alpha 0.75
run_if_missing results/ga_tail_a1_best.json    src/ga_xgb.py --metric tail --alpha 1.0

LOG "===== BƯỚC 5: gene loss-weighting (beta) ====="
# RMSE + gene beta: chứng minh mục tiêu quy ước TỪ CHỐI loss weighting
run_if_missing results/ga_rmse_lw_best.json src/ga_xgb.py --metric rmse --loss-weight
# tail alpha=1 + gene beta: cấu hình đuôi tốt nhất
run_if_missing results/ga_tail_a1_lw_best.json src/ga_xgb.py --metric tail --alpha 1.0 --loss-weight

LOG "===== BƯỚC 6: tổng hợp số liệu cho bài ====="
python3 src/paper_numbers.py 2>&1 | grep -v -i warning
python3 src/bootstrap_test.py 2>&1 | grep -v -i warning
python3 src/decision_fairness.py 2>&1 | grep -v -i warning

LOG "===== BƯỚC 7: vẽ lại hình ====="
python3 src/figures.py --fig 3 4 5 6 7 2>&1 | grep -v -i warning

LOG "===== HOÀN TẤT ====="
LOG "Ngân sách đánh giá = $BUDGET — CẦN cập nhật con số này trong bài (Section IV-B, Bảng 3, Bảng 4, Bảng 6)."
