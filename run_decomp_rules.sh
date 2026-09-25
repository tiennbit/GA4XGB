#!/usr/bin/env bash
# E2 (khung bài 24/9, mục 6.5): các cách ước lượng tầng quyết định R0..R7 và bảng phân
# rã, hậu kỳ trên npz của E1 (preds/decomp/). Không khớp XGBoost nào: khoảng 11 s một
# lần chia ở cỡ thật, vài phút cho 10 lần chia.
#
# Thứ tự: cần E1 xong (npz, và results_cost/decomp_centers.json khi PRIMARY/BSTAR=auto).
# Chạy lại sau khi E2b xong (results_cost/wtrain_tuned.json): mọi lần chia lấy từ
# .partial, chỉ phần tóm tắt được tính lại để điền ô R8* và C1.
#
# Biến môi trường (tuỳ chọn):
#   PRIMARY, BSTAR  tên trung tâm trong npz; mặc định auto (đọc decomp_centers.json)
#   TAG             tag npz của E1 (split<seed>_<tag>.npz); nhiều tag cách nhau bằng dấu cách
#   PREDS           thư mục npz của E1, mặc định preds/decomp (ngoài src/ và results_cost/)
#   SAVE_PREDS      nơi lưu dự đoán theo quy tắc cho E3/E9, mặc định preds/decomp_rules;
#                   đặt rỗng (SAVE_PREDS=) để không lưu
#   WORKERS         số tiến trình, mặc định 2 (E2b có thể đang chiếm phần lớn số nhân)
#   RUN_STAMP       stamp do sync_server.local.sh sh ghi; provenance.py nhúng vào JSON
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
if [ -n "${RUN_STAMP:-}" ]; then
    export RUN_STAMP
    echo "RUN_STAMP=$RUN_STAMP"
else
    echo "RUN_STAMP chưa đặt: provenance thử git (server không có git nên commit sẽ là unknown)"
fi

ARGS=(--out results_cost/decomp_rules.json --preds-dir "${PREDS:-preds/decomp}"
      --primary "${PRIMARY:-auto}" --bstar "${BSTAR:-auto}" --workers "${WORKERS:-2}")
if [ -n "${TAG:-}" ]; then
    read -r -a TAGS <<< "$TAG"
    ARGS+=(--tag "${TAGS[@]}")
fi
SAVE="${SAVE_PREDS-preds/decomp_rules}"
if [ -n "$SAVE" ]; then
    ARGS+=(--save-preds "$SAVE")
fi
PYTHONPATH=src $PY -W ignore src/decomp_rules.py "${ARGS[@]}"
touch results_cost/_DONE_decomp_rules
