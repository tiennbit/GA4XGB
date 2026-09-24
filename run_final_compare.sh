#!/usr/bin/env bash
# So sánh chốt với trung tâm tốt (tuned GA-RMSE seed 42, bag10): raw / stretch /
# huấn luyện có trọng số (cùng cấu hình) / quy tắc Bayes; ba kiểu trọng số; ba thước đo.
# Mọi thước đo RMSE. Khoảng 40 phút trên 8 nhân.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__)"
PYTHONPATH=src $PY -W ignore src/final_compare.py --out results_cost/final_compare.json
touch results_cost/_DONE_final_compare
