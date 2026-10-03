#!/usr/bin/env bash
# E11 (khung bài 24/9, mục 6.13): lặp lại E1, E2, E2b, E4 trên dữ liệu công khai.
# Năm bộ chính (california_housing, diamonds, kings_county, cps88wages, saber) và một
# bộ phụ (student_performance_por). Mỗi bộ, mỗi lần chia (gates.SEEDS, 10 lần 80/20):
# default, bag10, rs_tuned (30 cấu hình, dừng sớm như E1); R0, R1, R2, R5; R8 dò lại ở
# K = 3 (30 cấu hình); chính sách E4 ở K = 3; C1, C2, C3 với Nadeau-Bengio từng bộ và
# đếm số bộ cùng dấu với HSA (đọc results_cost/*.json). Mọi thước đo RMSE.
# Chạy lại an toàn: npz trong $PREDS_DIR/<bộ>/ và results_bench/bench_decomp.json.partial
# cùng dấu thì được dùng lại (theo bộ và theo lần chia).
#
# Dữ liệu: data_public/ (công khai, không có PII). saber.csv.gz do src/convert_saber.py
# sinh ở Mac (server không có openpyxl) và được `sync_server.local.sh code` đẩy lên;
# sha256 mọi file được đối chiếu với data_public/SHA256SUMS trước khi chạy.
#
# Biến môi trường (tuỳ chọn):
#   WORKERS=8        số tiến trình joblib; mỗi XGBoost dùng cpu_count // WORKERS luồng
#   N_CONFIGS=30     cấu hình cho rs_tuned và R8 (khung bài 6.13)
#   DATASETS=        tên bộ cách nhau bằng dấu cách; rỗng là cả sáu bộ
#   PREDS_DIR=preds/bench   ngoài src/ (mirror --delete) và ngoài results_bench/ (pulldir)
#   RUN_STAMP=       stamp JSON do script đồng bộ tạo ở Mac; provenance.stamp đọc nó
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_bench
PY=".venv/bin/python"
WORKERS="${WORKERS:-8}"
N_CONFIGS="${N_CONFIGS:-30}"
PREDS_DIR="${PREDS_DIR:-preds/bench}"
# Thư mục tạm của joblib trên đĩa, không ở /dev/shm (RAM 7,9 GB): lượt 25/9 hỏng vì mảng
# gửi sang tiến trình con tích luỹ làm đầy /dev/shm (xem run_confirm.sh).
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-$PWD/.joblib_tmp}"
mkdir -p "$JOBLIB_TEMP_FOLDER"
ARGS=(--workers "$WORKERS" --n-configs "$N_CONFIGS" --preds-dir "$PREDS_DIR"
      --out results_bench/bench_decomp.json)
if [ -n "${DATASETS:-}" ]; then
  read -r -a DS <<< "$DATASETS"
  ARGS+=(--datasets "${DS[@]}")
fi
if [ -n "${RUN_STAMP:-}" ]; then
  export RUN_STAMP
  echo "RUN_STAMP=$RUN_STAMP"
else
  echo "RUN_STAMP chưa đặt: meta.provenance lấy commit từ git (server không có git -> unknown); code_sha256 vẫn ghi"
fi
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
PYTHONPATH=src $PY -W ignore src/bench_decomp.py "${ARGS[@]}"
touch results_bench/_DONE_bench_decomp
