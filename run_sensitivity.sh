#!/usr/bin/env bash
# E5 (khung bài 24/9, mục 6.9): độ nhạy của từng tham số; cần npz E1 và E2b (preds/wtrain). --refit khớp lại hệ số số cây 1,0/1,2.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
export RUN_STAMP="${RUN_STAMP:-}"
echo "RUN_STAMP=${RUN_STAMP:-<không có>}"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
PYTHONPATH=src $PY -W ignore src/sensitivity.py --workers "${WORKERS:-4}" --e2b-dir preds/wtrain --refit --out results_cost/sensitivity.json
touch results_cost/_DONE_sensitivity
