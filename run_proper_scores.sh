#!/usr/bin/env bash
# E3 (khung bài 24/9, mục 6.7): điểm hợp thức hậu kỳ trên npz của E1; trung tâm chính và B* đọc từ summary.G1 của E1.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
export RUN_STAMP="${RUN_STAMP:-}"
echo "RUN_STAMP=${RUN_STAMP:-<không có>}"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
PYTHONPATH=src $PY -W ignore src/proper_scores.py --workers "${WORKERS:-4}" --out results_cost/proper_scores.json
touch results_cost/_DONE_proper_scores
