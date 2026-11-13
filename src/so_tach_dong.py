"""So tách dòng mới với cũ (mức đoạn, ghép cặp theo đoạn): CER, KTC 95 % bootstrap theo cụm người viết, đếm người viết.
python so_tach_dong.py <cũ.json> <mới.json> [cấu hình ...]"""
import json
import sys
from collections import defaultdict
import editdistance
import numpy as np

cu, moi = (json.load(open(f, encoding="utf-8"))["chi_tiet"] for f in sys.argv[1:3])
for ch in sys.argv[3:] or ["a", "t@1.01"]:
    W = defaultdict(lambda: np.zeros(3))  # lỗi cũ, lỗi mới, độ dài
    for x, y in zip(cu, moi):
        assert x["anh"] == y["anh"]
        w = x["anh"].split("/")[-2]
        W[w] += [editdistance.eval(x[ch], x["gt"]), editdistance.eval(y[ch], y["gt"]), len(x["gt"])]
    A = np.array(list(W.values()))
    tong = A.sum(0)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(5000):
        s = A[rng.integers(0, len(A), len(A))].sum(0)
        bs.append(100 * (s[1] - s[0]) / s[2])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    tot = int((A[:, 1] < A[:, 0]).sum()); xau = int((A[:, 1] > A[:, 0]).sum())
    print(f"{ch}: cũ {100 * tong[0] / tong[2]:.2f} % -> mới {100 * tong[1] / tong[2]:.2f} % | Δ {100 * (tong[1] - tong[0]) / tong[2]:+.2f} "
          f"[{lo:+.2f}; {hi:+.2f}] | người viết tốt lên/xấu đi/như cũ {tot}/{xau}/{len(A) - tot - xau} | "
          + " ".join(f"{w}:{100 * v[0] / v[2]:.1f}->{100 * v[1] / v[2]:.1f}" for w, v in sorted(W.items(), key=lambda kv: int(kv[0]))))
