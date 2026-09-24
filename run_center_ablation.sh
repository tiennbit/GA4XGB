#!/usr/bin/env bash
# Vì sao trung bình 50 phân vị là trung tâm tốt hơn: so default / bag10 / tuned (GA-RMSE
# seed 42, không chạy GA mới) / qmean / qmedian. Mọi thước đo RMSE. Khoảng 1 giờ trên 8 nhân.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__)"
PYTHONPATH=src $PY -W ignore src/center_ablation.py --out results_cost/center_ablation.json
touch results_cost/_DONE_center_ablation
