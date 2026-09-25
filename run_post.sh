#!/usr/bin/env bash
# Chuỗi thứ hai của lượt chạy khẳng định: E3 -> E5 -> E9 -> E10. Cần kết quả của
# run_confirm.sh (E1, E2, E2b), nên nó CHỜ results_cost/_DONE_confirm; nếu chuỗi đầu báo
# LỖI thì dừng luôn thay vì chờ mãi. Mỗi bước log riêng, dừng ở lỗi đầu tiên.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
export RUN_STAMP="${RUN_STAMP:-}"
until [ -f results_cost/_DONE_confirm ]; do
  if grep -q "LỖI" results_cost/run_confirm.log 2>/dev/null; then
    echo "[$(date -Is)] run_confirm.sh báo LỖI: không chạy E3 -> E10"; exit 1
  fi
  sleep 300
done
step() {
  local name="$1"; shift
  echo "[$(date -Is)] BẮT ĐẦU $name"
  if "$@" > "results_cost/run_${name}.log" 2>&1; then
    echo "[$(date -Is)] XONG $name"
  else
    echo "[$(date -Is)] LỖI $name (xem results_cost/run_${name}.log)"; exit 1
  fi
}
step proper_scores bash run_proper_scores.sh
step sensitivity bash run_sensitivity.sh
step use_validity bash run_use_validity.sh
step group_audit bash run_group_audit.sh
touch results_cost/_DONE_post
echo "[$(date -Is)] HOÀN TẤT chuỗi E3 -> E10"
