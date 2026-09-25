#!/usr/bin/env bash
# =============================================================================
# Chiến dịch multi-seed cho GA4XGB.
#
# VÌ SAO CẦN: mọi số hiện có đến từ MỘT lần chạy mỗi cấu hình. Ở ngân sách 632,
# khác biệt giữa các mục tiêu tổng hợp nhỏ hơn dao động giữa các lần chạy —
# bằng chứng là GA-MAE tìm được cấu hình RMSE tốt hơn cả lượt tối ưu RMSE.
# Không có mean±SD thì không phát biểu được gì chắc chắn về GA.
#
# CHẠY VÀO THƯ MỤC RIÊNG: seed 42 ghi đè đúng tên file mà bộ kết quả hiện tại
# đang dùng. Campaign này phải nằm tách biệt cho tới khi chốt chuyển sang nó.
#
#   PAR=8 ./run_multiseed.sh            # 8 job song song, mỗi job 2 luồng
#   PAR=6 NJOBS=2 ./run_multiseed.sh    # nhẹ hơn nếu RAM căng
#
# An toàn khi ngắt giữa chừng: job nào đã có file kết quả thì bỏ qua.
# =============================================================================
set -u
cd "$(dirname "$0")"

PAR=${PAR:-8}            # số job chạy đồng thời
NJOBS=${NJOBS:-2}        # luồng mỗi job — benchmark: 8x2 cho thông lượng cao nhất
SEEDS=${SEEDS:-"42 1 2 3 4"}
OUT=${OUT:-results_mseed}
PY=.venv/bin/python
export PYTHONPATH=src

mkdir -p "$OUT"
JOBS=$(mktemp)

# Ngân sách cho random search phải bằng số eval GA-RMSE tiêu thụ. Lấy từ bộ kết
# quả hiện có; nếu campaign này đổi con số đó thì phải chạy lại random search.
BUDGET=$($PY -c "import json;print(json.load(open('results/ga_rmse_best.json'))['total_evals'])" 2>/dev/null || echo 632)

# GA-R2 CỐ TÌNH không có trong danh sách: R² là biến đổi đơn điệu của MSE nên
# nó tối ưu cùng thứ tự ứng viên với RMSE — đẳng thức đúng theo định nghĩa,
# không phải kết quả thực nghiệm cần nhiều seed để xác nhận.
for s in $SEEDS; do
  sfx=""; [ "$s" != "42" ] && sfx="_seed$s"
  add() {  # $1 = tên file kết quả, $2... = lệnh
    local f="$OUT/$1"; shift
    [ -f "$f" ] || echo "$*" >> "$JOBS"
  }
  add "ga_rmse${sfx}_best.json"        $PY src/ga_xgb.py --metric rmse --seed $s --n-jobs $NJOBS
  add "ga_mae${sfx}_best.json"         $PY src/ga_xgb.py --metric mae  --seed $s --n-jobs $NJOBS
  add "ga_tail_a0.25${sfx}_best.json"  $PY src/ga_xgb.py --metric tail --alpha 0.25 --seed $s --n-jobs $NJOBS
  add "ga_tail_a0.5${sfx}_best.json"   $PY src/ga_xgb.py --metric tail --alpha 0.5  --seed $s --n-jobs $NJOBS
  add "ga_tail_a0.75${sfx}_best.json"  $PY src/ga_xgb.py --metric tail --alpha 0.75 --seed $s --n-jobs $NJOBS
  add "ga_tail_a1${sfx}_best.json"     $PY src/ga_xgb.py --metric tail --alpha 1.0  --seed $s --n-jobs $NJOBS
  add "ga_tail_a1_lw${sfx}_best.json"  $PY src/ga_xgb.py --metric tail --alpha 1.0 --loss-weight --seed $s --n-jobs $NJOBS
  add "random_search${sfx}_best.json"  $PY src/search_baselines.py --strategy random --budget $BUDGET --seed $s --n-jobs $NJOBS
done

N=$(wc -l < "$JOBS" | tr -d ' ')
echo "[$(date '+%F %T')] $N job cần chạy | $PAR song song × $NJOBS luồng | ngân sách random = $BUDGET"
echo "[$(date '+%F %T')] kết quả -> $OUT/"
[ "$N" -eq 0 ] && { echo "Không còn gì để chạy."; rm -f "$JOBS"; exit 0; }

# Ép mọi script ghi vào $OUT: chúng hardcode tiền tố "results/", nên chạy trong
# một thư mục làm việc mà "results" TRỎ TỚI $OUT. Không phải sửa mã khoa học.
WORK=$(mktemp -d)
ln -s "$PWD/src" "$WORK/src"
ln -s "$PWD/data" "$WORK/data"
ln -s "$PWD/.venv" "$WORK/.venv"
ln -s "$PWD/$OUT" "$WORK/results"

cd "$WORK"
# xargs -P giới hạn số job đồng thời; mỗi dòng là một lệnh đầy đủ.
xargs -P "$PAR" -I CMD sh -c 'CMD > /dev/null 2>&1' < "$JOBS"
rc=$?
cd - >/dev/null
rm -rf "$WORK"; rm -f "$JOBS"

DONE=$(ls -1 "$OUT"/*_best.json 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date '+%F %T')] xong (rc=$rc) — $OUT/ nay có $DONE file *_best.json"
