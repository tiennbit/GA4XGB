#!/usr/bin/env bash
# Hai tầng: thông tin p(y|x) (bin / quantile XGBoost / qshape) × trọng số w(y)
# (prior λ / relevance φ / bậc thang K). XGBoost mặc định, CHƯA GA, mọi thước đo RMSE.
# 5 lần chia train/test; mỗi lần 12 lượt fit quantile 50 mức -> khoảng 1,5–2 giờ trên 8 nhân.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__)"
PYTHONPATH=src $PY -W ignore src/tail_prior.py --out results_cost/tail_prior.json
touch results_cost/_DONE_tail_prior
