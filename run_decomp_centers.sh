#!/usr/bin/env bash
# E1 (khung bài 24/9, mục 6.4): tập đặc trưng, trung tâm và vết 60 cấu hình.
# Giai đoạn 1 (default, bag10 trên 4 tập đặc trưng; sub1, bag40 trên F_dt) -> cổng tập
# đặc trưng + B* -> giai đoạn 2 trên tập chính (rs_tuned từ 60 cấu hình dừng sớm, khớp
# lại cả 60 cấu hình cho E4, rs_tuned_bag5). Mọi thước đo RMSE.
# Ước lượng khoảng 2,5 giờ trên 16 nhân (8 tiến trình x 2 luồng); xem docstring
# src/decomp_centers.py. Chạy lại an toàn: npz trong $PREDS_DIR và
# results_cost/decomp_centers.json.phase{1,2}.partial cùng dấu thì được dùng lại.
#
# Biến môi trường (tuỳ chọn):
#   WORKERS=8        số tiến trình joblib; mỗi XGBoost dùng cpu_count // WORKERS luồng
#   PHASE=all        1 | 2 | all
#   FEATURE_SET=     đặt tập chính, bỏ qua cổng (F_full | F_dt | F_dt-cn | F_dt-cn-bc)
#   N_CONFIGS=60     40 nếu E0 đo thời gian fit trung bình > 5 s (gates.TUNE_BUDGET_FALLBACK)
#   PREDS_DIR=preds/decomp   ngoài src/ (mirror --delete) và ngoài results_cost/ (pulldir)
#   RUN_STAMP=       đường dẫn stamp JSON do script đồng bộ tạo ở Mac; truyền nguyên
#                    cho Python (provenance.stamp đọc nó), không có thì provenance dùng git/unknown
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
WORKERS="${WORKERS:-8}"
PHASE="${PHASE:-all}"
N_CONFIGS="${N_CONFIGS:-60}"
PREDS_DIR="${PREDS_DIR:-preds/decomp}"
ARGS=(--phase "$PHASE" --workers "$WORKERS" --n-configs "$N_CONFIGS"
      --preds-dir "$PREDS_DIR" --out results_cost/decomp_centers.json)
if [ -n "${FEATURE_SET:-}" ]; then
  ARGS+=(--feature-set "$FEATURE_SET")
fi
if [ -n "${RUN_STAMP:-}" ]; then
  export RUN_STAMP
  echo "RUN_STAMP=$RUN_STAMP"
else
  echo "RUN_STAMP chưa đặt: meta.provenance lấy commit từ git (server không có git -> unknown); code_sha256 vẫn ghi"
fi
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
PYTHONPATH=src $PY -W ignore src/decomp_centers.py "${ARGS[@]}"
# Chỉ giai đoạn 1 thì đánh dấu riêng: _DONE_decomp_centers nghĩa là đã có vết cấu hình cho E2, E4
if [ "$PHASE" = "1" ]; then
  touch results_cost/_DONE_decomp_centers_phase1
else
  touch results_cost/_DONE_decomp_centers
fi
