#!/usr/bin/env bash
# E4 (khung bài 24/9, mục 6.8): HPO nên tối ưu gì. Năm chính sách chọn cấu hình (a), (b),
# (c), (c+), (o) trên vết 60 cấu hình của E1, lưới K {1; 1,5; 2; 3; 5; 8; 12; 20}, tương
# quan hạng (Giả thuyết 7) và bảng chi phí. Hậu kỳ thuần trên npz của E1 (preds/decomp),
# không khớp XGBoost nào; chạy SAU E1 (và sau E2 nếu cổng G2 đổi quy tắc sang R2:
# RULE=R2). Khoảng 5 phút trên 16 nhân với WORKERS=5 (đo trên Mac: 40 s và 0,5 GB mỗi lần chia).
#
# Biến môi trường (tuỳ chọn):
#   RUN_STAMP   stamp JSON của lượt chạy (sync_server.local.sh ghi), nhúng vào meta.provenance
#   E1_DIR      npz của E1 (mặc định preds/decomp)
#   E1_TAGS     tag npz của E1 cần gộp, cách nhau bằng dấu cách (mặc định: split<seed>.npz)
#   BSTAR       bag B* của E1 (tên trung tâm hoặc số); mặc định đọc decomp_centers.json
#   RULE        R1 (mặc định) hoặc R2 nếu G2 chọn R2
#   WORKERS     số tiến trình joblib, mỗi tiến trình một lần chia (mặc định 5)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results_cost
PY=".venv/bin/python"
export RUN_STAMP="${RUN_STAMP:-}"
echo "RUN_STAMP=${RUN_STAMP:-<không có, provenance thử git>}"
# In phiên bản trước khi chạy: lệch bản thư viện là lệch số (AGENTS.md, ràng buộc 2)
PYTHONPATH=src $PY -c "import xgboost,sklearn,numpy,pandas,scipy,joblib; print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'joblib',joblib.__version__)"
EXTRA=()
if [[ -n "${E1_TAGS:-}" ]]; then
  read -r -a TAG_LIST <<< "$E1_TAGS"
  EXTRA+=(--e1-tags "${TAG_LIST[@]}")
fi
if [[ -n "${BSTAR:-}" ]]; then
  EXTRA+=(--bstar "$BSTAR")
fi
PYTHONPATH=src $PY -W ignore src/select_policy.py \
  --out results_cost/select_policy.json \
  --e1-dir "${E1_DIR:-preds/decomp}" \
  --preds-dir preds/select_policy \
  --centers-json results_cost/decomp_centers.json \
  --rule "${RULE:-R1}" \
  --workers "${WORKERS:-5}" \
  ${EXTRA[@]+"${EXTRA[@]}"} "$@"
touch results_cost/_DONE_select_policy
