#!/usr/bin/env bash
# E2b (khung bài 24/9, mục 6.6): huấn luyện có trọng số dò lại với CÙNG bảng cấu hình,
# lần chia, fold và tập dừng sớm của E1; K ∈ {2; 3; 5; 8}, trọng số chuẩn hoá trung
# bình 1, dừng sớm theo RMSE có trọng số, R8 và R8_bag5; so C1 với R1 trên các trung
# tâm của E1. Cần npz của E1 ở preds/decomp (chạy sau run_decomp_centers.sh).
# Khoảng 7 giờ trên 16 nhân với WORKERS=8 (8 tiến trình x 2 luồng XGBoost); máy 8 nhân
# thì đặt WORKERS=4, khoảng 13 giờ. Ngắt giữa chừng rồi chạy lại là an toàn: đơn vị
# (lần chia, K) đã xong được bỏ qua.
#
# Biến môi trường (tuỳ chọn):
#   WORKERS         số tiến trình joblib (mặc định 8; hết RAM thì giảm xuống 4)
#   E1_PREDS, E1_TAG  thư mục và tag npz của E1 (mặc định preds/decomp, không tag)
#   PRIMARY_CENTER  trung tâm chính sau cổng G1 (ví dụ rs_tuned_bag5), để đánh dấu C1 chính
#   SECONDARY_FROM  rs_tuned (mặc định) hoặc bag, siêu tham số cho họ prior/φ thứ cấp
#   RUN_STAMP       stamp do sync_server.local.sh tạo ở Mac; truyền nguyên cho provenance
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
WORKERS="${WORKERS:-8}"
E1_PREDS="${E1_PREDS:-preds/decomp}"
SECONDARY_FROM="${SECONDARY_FROM:-rs_tuned}"
export RUN_STAMP="${RUN_STAMP:-}"
echo "RUN_STAMP=${RUN_STAMP:-<không có, provenance thử git>}"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
EXTRA=()
[[ -n "${E1_TAG:-}" ]] && EXTRA+=(--e1-tag "$E1_TAG")
[[ -n "${PRIMARY_CENTER:-}" ]] && EXTRA+=(--primary-center "$PRIMARY_CENTER")
PYTHONPATH=src $PY -W ignore src/wtrain_tuned.py \
    --out results_cost/wtrain_tuned.json \
    --preds-dir preds/wtrain \
    --e1-preds-dir "$E1_PREDS" \
    --secondary-from "$SECONDARY_FROM" \
    --workers "$WORKERS" \
    ${EXTRA[@]+"${EXTRA[@]}"}
touch results_cost/_DONE_wtrain_tuned
