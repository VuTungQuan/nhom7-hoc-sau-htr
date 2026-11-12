#!/bin/bash
# Theo dõi huấn luyện trực tiếp trên máy chủ, tự cập nhật mỗi 10 giây.
# Dùng: bash /mnt/data/hs-htr/bin/theo_doi.sh [tên_run] [tổng_epoch]   (Ctrl+C để thoát, KHÔNG ảnh hưởng việc train)
RUN=${1:-crnn_line_20k}
TONG=${2:-202}
D=/mnt/data/hs-htr/runs
PY=~/anaconda3/envs/ocr/bin/python
export TERM=${TERM:-xterm}
while true; do
  clear
  echo "=== HUẤN LUYỆN $RUN   $(date +%H:%M:%S)   (Ctrl+C để thoát) ==="
  if pgrep -u "$USER" -f "[t]rain.py .*runs/$RUN" >/dev/null; then echo "Trạng thái: ĐANG CHẠY"; else echo "Trạng thái: ĐÃ DỪNG / XONG"; fi
  "$PY" - "$D/$RUN/log.csv" "$TONG" <<'PY'
import csv, sys
try:
    r = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
except FileNotFoundError:
    print("Chưa có log.csv"); sys.exit()
if not r:
    print("Chưa xong epoch nào"); sys.exit()
tong = int(sys.argv[2]); ep = int(r[-1]["epoch"])
g = sum(float(x["giay"]) for x in r) / len(r)
b = min(r, key=lambda x: float(x["val_CER"]))
con = (tong - ep) * g
print(f"Epoch {ep}/{tong} ({ep / tong:.0%})  |  ~{g:.0f}s/epoch  |  còn ~{con // 3600:.0f}h{con % 3600 // 60:02.0f}m")
print(f"Tốt nhất: epoch {b['epoch']}  CER {float(b['val_CER']):.2%}  WER {float(b['val_WER']):.2%}   (mô hình cũ: CER 3,02%)")
print("\n epoch    loss   CER val   WER val   lỗi dấu   lr")
for x in r[-8:]:
    print(f" {x['epoch']:>5} {float(x['loss']):7.3f} {float(x['val_CER']):8.2%} {float(x['val_WER']):9.2%} {float(x['val_loi_dau']):8.2%}   {x['lr']}")
PY
  echo; echo "--- Mẫu đọc gần nhất (pd = mô hình đọc, gt = đúng) ---"
  grep -A2 "^ep" "$D/$RUN.log" 2>/dev/null | tail -2
  echo
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader \
    | awk -F", " '{print "GPU: " $1 " | VRAM " $2 " / " $3 " | " $4 "°C"}'
  sleep 10
done
