"""Thống kê bộ UIT-HWDB (word/line/paragraph) đọc thẳng từ zip, không giải nén.

Chạy: python -X utf8 scripts/khao_sat_du_lieu.py <thư mục chứa các zip> [số ảnh mẫu đo kích thước]
"""
import collections
import io
import json
import statistics as st
import sys
import unicodedata
import zipfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

# Ký tự chữ cái tiếng Việt có dấu (sau NFC): khác chữ ASCII cơ bản sau khi bỏ dấu
def co_dau(c):
    return c.isalpha() and unicodedata.normalize("NFD", c)[0] != c


def khao_sat(zp, n_anh):
    z = zipfile.ZipFile(zp)
    names = set(z.namelist())
    out = {}
    for split in ("train_data", "test_data"):
        labels, missing, nfd, writers = {}, 0, 0, set()
        for m in sorted(n for n in names if f"/{split}/" in n and n.endswith("label.json")):
            folder = m.rsplit("/", 1)[0]
            writers.add(folder.rsplit("/", 1)[1])
            for img, txt in json.loads(z.read(m).decode("utf-8")).items():
                if f"{folder}/{img}" not in names:
                    missing += 1
                if txt != unicodedata.normalize("NFC", txt):
                    nfd += 1
                labels[f"{folder}/{img}"] = unicodedata.normalize("NFC", txt)
        imgs = [n for n in names if f"/{split}/" in n and n.lower().endswith(".jpg")]
        txt = list(labels.values())
        chars = collections.Counter("".join(txt))
        letters = sum(v for c, v in chars.items() if c.isalpha())
        sizes = []
        if Image:
            for n in sorted(imgs)[: n_anh]:
                with Image.open(io.BytesIO(z.read(n))) as im:
                    sizes.append(im.size)
        out[split] = {
            "nguoi_viet": len(writers),
            "anh": len(imgs),
            "nhan": len(labels),
            "nhan_thieu_anh": missing,
            "anh_thieu_nhan": len(imgs) - (len(labels) - missing),
            "nhan_NFD": nfd,
            "ky_tu_tb": round(st.mean(map(len, txt)), 1),
            "ky_tu_max": max(map(len, txt)),
            "tu_tb": round(st.mean(len(t.split()) for t in txt), 1),
            "bo_ky_tu": len(chars),
            "ti_le_chu_co_dau": round(sum(v for c, v in chars.items() if co_dau(c)) / letters, 3),
            "kich_thuoc_anh_mau(w,h)_median": (st.median(s[0] for s in sizes), st.median(s[1] for s in sizes)) if sizes else None,
            "_writers": writers,
            "_chars": chars,
        }
    out["nguoi_viet_trung_train_test"] = len(out["train_data"]["_writers"] & out["test_data"]["_writers"])
    out["ky_tu_test_khong_co_trong_train"] = "".join(sorted(set(out["test_data"]["_chars"]) - set(out["train_data"]["_chars"])))
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    n_anh = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    tat_ca = collections.Counter()
    for zp in sorted(root.glob("UIT_HWDB_*.zip")):
        r = khao_sat(zp, n_anh)
        print(f"\n=== {zp.name}")
        for split in ("train_data", "test_data"):
            tat_ca.update(r[split]["_chars"])
            print(split, {k: v for k, v in r[split].items() if not k.startswith("_")})
        print("nguoi_viet_trung_train_test:", r["nguoi_viet_trung_train_test"])
        print("ky_tu_test_khong_co_trong_train:", repr(r["ky_tu_test_khong_co_trong_train"]))
    print("\nBo ky tu chung (", len(tat_ca), "):", "".join(sorted(tat_ca)))
