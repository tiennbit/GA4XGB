#!/usr/bin/env bash
# =============================================================================
# TẦNG 1 của Revision Roadmap sau vòng phản biện (notes/peer-review-2026-08-29.md).
#
# Hai nhóm, chạy song song vì nhóm LONG chiếm máy rất lâu còn nhóm CHEAP thì không:
#
#   LONG  (1.3)  arm control coupling: beta=1 ÁP CỐ ĐỊNH vào loss + tái tối ưu
#                hyperparameter dưới fitness RMSE thuần. 5 seed × ~10h.
#                ĐÂY LÀ THÍ NGHIỆM QUYẾT ĐỊNH: nếu nó cũng chạm profile
#                10.2/9.4 thì alpha không đóng góp gì cho kết quả headline.
#
#   CHEAP (1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.10) — không chạy lại search nào,
#                chỉ fit lại mô hình từ cấu hình đã lưu. Tổng ~1-2h.
#
#   PAR=5 ./run_review_tier1.sh            # mặc định: 5 job LONG song song
#   ONLY=cheap ./run_review_tier1.sh       # chỉ nhóm rẻ, có kết quả ngay
#   ONLY=long  ./run_review_tier1.sh       # chỉ arm control
#
# An toàn khi ngắt: job nào đã có file kết quả thì bỏ qua.
# =============================================================================
set -u
cd "$(dirname "$0")"
ROOT=$PWD          # tuyệt đối: các subshell có cd sang thư mục làm việc tạm

PAR=${PAR:-5}
NJOBS=${NJOBS:-2}          # luồng mỗi job LONG — benchmark: 8×2 tối ưu thông lượng
CHEAP_NJOBS=${CHEAP_NJOBS:-6}
SEEDS=${SEEDS:-"42 1 2 3 4"}
OUT=${OUT:-results_mseed}
ONLY=${ONLY:-all}
BETA=${BETA:-1.0}
PY=.venv/bin/python
export PYTHONPATH=src

mkdir -p "$OUT" logs
[ -x "$PY" ] || PY=python3

# Ép mọi script ghi vào $OUT: ga_xgb.py hardcode tiền tố "results/". Chạy trong
# một thư mục làm việc mà "results" TRỎ TỚI $OUT — không phải sửa mã khoa học.
WORK=$(mktemp -d)
for d in src data .venv; do [ -e "$PWD/$d" ] && ln -s "$PWD/$d" "$WORK/$d"; done
ln -s "$PWD/$OUT" "$WORK/results"
trap 'rm -rf "$WORK"' EXIT

# ---------------------------------------------------------------- LONG (1.3)
if [ "$ONLY" != "cheap" ]; then
  JOBS=$(mktemp)
  for s in $SEEDS; do
    sfx=""; [ "$s" != "42" ] && sfx="_seed$s"
    f="$OUT/ga_rmse_fb${BETA%.0}${sfx}_best.json"
    [ -f "$f" ] || echo "$PY src/ga_xgb.py --metric rmse --fixed-beta $BETA --seed $s --n-jobs $NJOBS" >> "$JOBS"
  done
  N=$(wc -l < "$JOBS" | tr -d ' ')
  echo "[$(date '+%F %T')] LONG: $N job (beta=$BETA cố định, fitness=RMSE) | $PAR song song × $NJOBS luồng"
  if [ "$N" -gt 0 ]; then
    ( cd "$WORK" && xargs -P "$PAR" -I CMD sh -c 'CMD' < "$JOBS" \
        > "$ROOT/logs/tier1_long.log" 2>&1
      echo "[$(date '+%F %T')] LONG xong" >> "$ROOT/logs/tier1_long.log" ) &
    LONG_PID=$!
    echo "[$(date '+%F %T')] LONG chạy nền (pid $LONG_PID) -> logs/tier1_long.log"
  fi
  rm -f "$JOBS"
fi

# --------------------------------------------------------------- CHEAP (rest)
if [ "$ONLY" != "long" ]; then
  # CHÚ Ý: nhóm CHEAP chạy ở $ROOT, KHÔNG phải $WORK. Các script này đọc/ghi
  # thẳng results_mseed/ qua review_common.MSEED_DIR, nên chúng không cần — và
  # không được — thư mục tạm với symlink "results" (thứ chỉ ga_xgb.py mới cần
  # vì nó hardcode tiền tố "results/").
  run_cheap() {   # $1 = file kết quả kỳ vọng, $2 = nhãn, $3... = lệnh
    local f="$OUT/$1" tag="$2"; shift 2
    if [ -f "$f" ]; then echo "  [bỏ qua] $tag — đã có $1"; return; fi
    echo "  [chạy]  $tag"
    ( cd "$ROOT" && "$@" ) > "$ROOT/logs/tier1_${tag}.log" 2>&1 \
      && echo "  [xong]  $tag -> $1" \
      || echo "  [LỖI]   $tag — xem logs/tier1_${tag}.log"
  }
  echo "[$(date '+%F %T')] CHEAP: bắt đầu"
  # Thứ tự cố ý: những cái không cần huấn luyện chạy trước để có số ngay.
  run_cheap boundary_audit.json     boundary     $PY src/boundary_audit.py
  run_cheap multiplicity.json       multiplicity $PY src/multiplicity.py
  run_cheap landscape.json          landscape    $PY src/landscape.py --walk-steps 200 --n-jobs $CHEAP_NJOBS
  run_cheap recalibration.json      recal        $PY src/recalibration.py --n-jobs $CHEAP_NJOBS
  run_cheap calibration_table.json  calibration  $PY src/calibration_table.py --n-jobs $CHEAP_NJOBS
  run_cheap screening_curves.json   screening    $PY src/screening_curves.py --n-jobs $CHEAP_NJOBS
  run_cheap simple_baselines.json   baselines    $PY src/simple_baselines.py --n-jobs $CHEAP_NJOBS
  echo "[$(date '+%F %T')] CHEAP xong"
fi

if [ -n "${LONG_PID:-}" ]; then
  echo "[$(date '+%F %T')] chờ LONG (pid $LONG_PID)..."
  wait "$LONG_PID"
fi
echo "[$(date '+%F %T')] TẦNG 1 hoàn tất — $OUT/"
ls -1 "$OUT"/{boundary_audit,multiplicity,landscape,recalibration,calibration_table,screening_curves,simple_baselines}.json 2>/dev/null
ls -1 "$OUT"/ga_rmse_fb*_best.json 2>/dev/null
