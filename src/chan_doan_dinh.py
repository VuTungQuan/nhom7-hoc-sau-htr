"""Chẩn đoán tìm đỉnh dòng trên các đoạn val lệch số dòng: khoảng cách giữa các đỉnh (chia trung vị),
độ nổi mỗi đỉnh (chia max), mực nằm ngoài đỉnh đầu/cuối."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import find_peaks
import htr

d = json.load(open("../runs/crnn_line/thu_tach_dong_val.json", encoding="utf-8"))["chi_tiet"]
for p, r in d.items():
    lech = r["so_dong_cc"] - r["so_dong_gt"]
    if not lech:
        continue
    gray = htr.can_bang_nen(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ps = np.convolve(bw.sum(1).astype(float), np.ones(9) / 9, "same")
    pk, pr = find_peaks(ps, distance=25, prominence=0.10 * ps.max())
    g = np.diff(pk); kd = np.median(g)
    hang = np.nonzero(ps > 0.05 * ps.max())[0]
    print(f"{Path(p).parent.name}/{Path(p).name} gt {r['so_dong_gt']} dinh {len(pk)} | kd {kd:.0f} | gap/kd {np.round(g / kd, 2).tolist()} | "
          f"prom {np.round(pr['prominences'] / ps.max(), 2).tolist()} | muc truoc/sau dinh {round((pk[0] - hang[0]) / kd, 2)}/{round((hang[-1] - pk[-1]) / kd, 2)} "
          f"| CER cc {round(100 * __import__('editdistance').eval(r['cc'], r['gt']) / len(r['gt']), 1)}")
