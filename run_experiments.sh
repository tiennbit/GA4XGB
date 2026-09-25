#!/usr/bin/env bash
# =============================================================================
# Chạy toàn bộ thí nghiệm cho bài báo — KHÔNG CẦN CLAUDE.
# Cách dùng (từ thư mục gốc dự án):
#   chmod +x run_experiments.sh
#   nohup ./run_experiments.sh > results/run_all.log 2>&1 &
# Theo dõi:  tail -f results/run_all.log
# Ước tính tổng: ~1.5–2 ngày trên máy 10 nhân (phần lớn là multi-seed ở Bước 4;
# có thể Ctrl-C sau Bước 3 nếu chỉ cần kết quả chính).
# Script TỰ ĐỘNG BỎ QUA các thí nghiệm đã có kết quả (an toàn khi chạy lại).
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

# --- Chờ 2 run GA đang chạy dở (nếu có) kết thúc để tránh nghẽn CPU ---
while pgrep -f "ga_xgb.py --metric (mae|r2)" >/dev/null 2>&1; do
  LOG "Đang chờ run GA-MAE/GA-R2 của phiên trước kết thúc..."
  sleep 300
done

LOG "===== BƯỚC 1: GA tail-weighted fitness (novelty chính) ====="
run_if_missing results/ga_tail_a1_best.json    src/ga_xgb.py --metric tail --alpha 1.0
run_if_missing results/ga_tail_a0.5_best.json  src/ga_xgb.py --metric tail --alpha 0.5

LOG "===== BƯỚC 2: RandomSearch + GridSearch cùng ngân sách 622 ====="
run_if_missing results/random_search_best.json src/search_baselines.py --strategy random --budget 622
run_if_missing results/grid_search_best.json   src/search_baselines.py --strategy grid   --budget 622

LOG "===== BƯỚC 3: (đảm bảo đủ) GA rmse/mae/r2 seed 42 ====="
run_if_missing results/ga_rmse_best.json src/ga_xgb.py --metric rmse
run_if_missing results/ga_mae_best.json  src/ga_xgb.py --metric mae
run_if_missing results/ga_r2_best.json   src/ga_xgb.py --metric r2

LOG "===== BƯỚC 4: Multi-seed (thống kê mean±std, Wilcoxon) — PHẦN NẶNG ====="
# 9 seed nữa cho 2 phương án chính (RMSE và tail-a1) + random search
for SEED in 1 2 3 4 5 6 7 8 9; do
  run_if_missing "results/ga_rmse_seed${SEED}_best.json" \
    src/ga_xgb.py --metric rmse --seed $SEED
  run_if_missing "results/ga_tail_a1_seed${SEED}_best.json" \
    src/ga_xgb.py --metric tail --alpha 1.0 --seed $SEED
  run_if_missing "results/random_search_seed${SEED}_best.json" \
    src/search_baselines.py --strategy random --budget 622 --seed $SEED
done

LOG "===== HOÀN TẤT ====="
LOG "Kết quả trong results/*.json — phiên Claude sau chỉ cần đọc các file này."
