#!/usr/bin/env bash
# Trọng số riêng cho hai đầu (K_thấp, K_cao) + ablation tỉnh/trường.
# XGBoost mặc định, CHƯA chạy GA, mọi thước đo RMSE. Kết quả vào results_cost/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"
PYTHONPATH=src $PY src/tail_weights.py --out results_cost/tail_weights.json
touch results_cost/_DONE_tail_weights
