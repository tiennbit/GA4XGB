#!/usr/bin/env bash
# E9 (khung bài 24/9, mục 6.11): ngưỡng cố định, khoảng dự đoán, gần đoán mò. CUTOFFS thêm ngưỡng sàn có nguồn, ví dụ "ge:85"; CUTOFFS_SOURCE ghi URL và ngày truy cập.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
export RUN_STAMP="${RUN_STAMP:-}"
echo "RUN_STAMP=${RUN_STAMP:-<không có>}"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
EXTRA=(); [[ -n "${CUTOFFS:-}" ]] && { read -r -a C <<< "$CUTOFFS"; EXTRA+=(--cutoffs "${C[@]}" --cutoffs-source "${CUTOFFS_SOURCE:-}"); }
PYTHONPATH=src $PY -W ignore src/use_validity.py --workers "${WORKERS:-4}" --near-guess --out results_cost/use_validity.json ${EXTRA[@]+"${EXTRA[@]}"}
touch results_cost/_DONE_use_validity
