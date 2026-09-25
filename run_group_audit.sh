#!/usr/bin/env bash
# E10 (khung bài 24/9, mục 6.12): kiểm toán nhóm. Cross-fit 2 x 5-fold trên toàn bộ bản
# ghi cho trung tâm chính (cổng G1 của E1) và bag10 trên F_dt, F_dt-cn, F_dt-cn-bc.
# Chạy SAU E1 (cần decomp_centers.json và, nếu trung tâm chính là rs_tuned*, npz lần
# chia 100). Dự đoán từng dòng ở preds/group_audit/ (ở lại server); JSON chỉ có tổng hợp.
#   CENTER, FSET: ghi đè trung tâm và tập đặc trưng chính (mặc định auto, đọc G1)
#   CUTOFFS: ngưỡng sàn xét tuyển thêm, ví dụ "ge:85 ge:90"
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
WORKERS="${WORKERS:-8}"
export RUN_STAMP="${RUN_STAMP:-}"
echo "RUN_STAMP=${RUN_STAMP:-<không có>}"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
EXTRA=()
[[ -n "${CUTOFFS:-}" ]] && { read -r -a C <<< "$CUTOFFS"; EXTRA+=(--cutoffs "${C[@]}"); }
PYTHONPATH=src $PY -W ignore src/group_audit.py --workers "$WORKERS" \
    --center "${CENTER:-auto}" --fset "${FSET:-auto}" \
    --out results_cost/group_audit.json --preds-dir preds/group_audit \
    ${EXTRA[@]+"${EXTRA[@]}"}
touch results_cost/_DONE_group_audit
