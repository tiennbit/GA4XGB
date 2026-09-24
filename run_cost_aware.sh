#!/usr/bin/env bash
# Lượt thử khả thi hướng cost-aware: XGBoost mặc định, CHƯA chạy GA.
# Vài phút trên server. Kết quả vào results_cost/, tách khỏi results_mseed/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"
PYTHONPATH=src $PY src/cost_aware.py --out results_cost/cost_aware.json
touch results_cost/_DONE
