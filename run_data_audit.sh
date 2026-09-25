#!/usr/bin/env bash
# E0b (khung bài 24/9, mục 6.3): kiểm dữ liệu trên data/data_final.csv tại chỗ, chỉ ghi
# số đếm và phân vị (dòng trùng chính xác/gần, gần đoán mò, luồng mẫu, câu hỏi IDT).
# Sau đó E0 bước (8): đo thời gian fit 20 cấu hình ngẫu nhiên trên F_dt có early
# stopping để áp cổng ngân sách 60 -> 40 cho E1 và E2b.
# Khoảng 5 đến 20 phút trên server IDT (16 luồng; E0b dưới 1 phút, phép đo 20 cấu hình phần còn lại).
#
# WORKERS: số tiến trình của phép đo (mỗi tiến trình cpu/WORKERS luồng XGBoost). Đặt
# bằng --workers mà E1 và E2b sẽ dùng, để cổng phản ánh đúng cách chạy của chúng.
# RUN_STAMP: đường dẫn stamp JSON do script đồng bộ tạo ở Mac (provenance.py); nếu có
# thì được truyền nguyên cho Python và nhúng vào meta.provenance.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
WORKERS="${WORKERS:-8}"   # = mặc định của run_decomp_centers.sh và run_wtrain_tuned.sh, để cổng đo đúng cách chạy thật
export RUN_STAMP="${RUN_STAMP:-}"
echo "RUN_STAMP=${RUN_STAMP:-<không có>}"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
PYTHONPATH=src $PY -W ignore src/data_audit.py --out results_cost/data_audit.json
PYTHONPATH=src $PY -W ignore src/bench_eval.py --workers "$WORKERS" --out results_cost/bench_eval.json
touch results_cost/_DONE_data_audit
