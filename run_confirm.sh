#!/usr/bin/env bash
# Chuỗi lượt chạy khẳng định (khung bài 24/9, mục 6.1), chạy qua đêm không cần trông:
#   E1 (decomp_centers) -> E2 (decomp_rules) -> E4 (select_policy) -> E2b (wtrain_tuned)
#   -> E2 lần hai (C1 cần kết quả E2b; các lần chia đã xong được bỏ qua) -> E6 (sim_decomp)
# E3, E5, E9, E10 chạy sau, khi đã review xong và có trung tâm chính.
#
# set -e: một bước lỗi hoặc một cổng từ chối (ví dụ G1 chưa đủ lần chia, E2b lệch mã
# băm E1) thì cả chuỗi dừng ở đó, không chạy tiếp trên đầu vào sai.
# Mỗi bước ghi log riêng results_cost/run_<bước>.log; log của chuỗi chỉ có mốc thời gian.
# RUN_STAMP (do sync_server.local.sh sh tạo) được truyền cho mọi bước.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
export RUN_STAMP="${RUN_STAMP:-}"
# Thư mục tạm của joblib trên đĩa (còn ~80 GB), không ở /dev/shm (RAM, 7,9 GB): lượt
# 25/9 hỏng vì mảng gửi sang tiến trình con tích luỹ làm đầy /dev/shm.
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-$PWD/.joblib_tmp}"
mkdir -p "$JOBLIB_TEMP_FOLDER"
step() {
  local name="$1"; shift
  echo "[$(date -Is)] BẮT ĐẦU $name"
  if "$@" > "results_cost/run_${name}.log" 2>&1; then
    echo "[$(date -Is)] XONG $name"
  else
    echo "[$(date -Is)] LỖI $name (xem results_cost/run_${name}.log)"
    exit 1
  fi
}
step decomp_centers bash run_decomp_centers.sh
step decomp_rules bash run_decomp_rules.sh
step select_policy bash run_select_policy.sh
step wtrain_tuned bash run_wtrain_tuned.sh
step decomp_rules_c1 bash run_decomp_rules.sh
step sim_decomp bash run_sim_decomp.sh
touch results_cost/_DONE_confirm
echo "[$(date -Is)] HOÀN TẤT chuỗi E1 -> E6"
