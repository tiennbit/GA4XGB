#!/usr/bin/env bash
# E6 (khung bài 24/9, mục 6.10): mô phỏng với p(y|x) biết trước. Không dùng dữ liệu thật.
# 4 kịch bản x 2 cỡ huấn luyện x 50 lần lặp, trung tâm default và bag10. Khoảng 1,5 giờ
# trên server IDT với WORKERS=8 (mỗi tiến trình giữ tập Monte Carlo 2 triệu dòng của một
# kịch bản, khoảng 0,4 GB). Ngắt giữa chừng rồi chạy lại là an toàn (.partial có dấu).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
WORKERS="${WORKERS:-8}"
export RUN_STAMP="${RUN_STAMP:-}"
echo "RUN_STAMP=${RUN_STAMP:-<không có>}"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
PYTHONPATH=src $PY -W ignore src/sim_decomp.py --workers "$WORKERS" --out results_cost/sim_decomp.json
touch results_cost/_DONE_sim_decomp
