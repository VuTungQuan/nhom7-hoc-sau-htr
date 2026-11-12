"""Hình từ dữ liệu (chạy trên máy chủ có dữ liệu): Hình 2.3 (mẫu UIT-HWDB), 4.2 (so sánh tách dòng), 4.8 (tiền xử lý ảnh chụp).
Chạy: python ve_hinh_du_lieu.py <thư mục ra> <thư mục data> <ảnh chụp lời giải toán>
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import htr

OUT, DATA, ANH_CHUP = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
plt.rcParams["font.family"] = "DejaVu Sans"
MAU = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14), (23, 190, 207), (140, 86, 75), (227, 119, 194)]


def luu(fig, ten):
    fig.savefig(OUT / ten, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig); print("ok", ten)


def h2_3_mau(nguoi="1"):
    nhan = {m: json.loads((DATA / f"UIT_HWDB_{m}/train_data/{nguoi}/label.json").read_text(encoding="utf-8")) for m in ["word", "line", "paragraph"]}
    tu = [k for k in sorted(nhan["word"], key=lambda s: int(s.split(".")[0]))[:6]]
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 0.8, 3.2], hspace=0.55)
    for i, k in enumerate(tu):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(cv2.imread(str(DATA / f"UIT_HWDB_word/train_data/{nguoi}/{k}"), 0), cmap="gray"); ax.axis("off")
        ax.set_title(f"“{nhan['word'][k]}”", fontsize=9)
    fig.text(0.12, 0.905, "a) Mức từ", fontsize=10, weight="bold")
    ax = fig.add_subplot(gs[1, :]); ax.imshow(cv2.imread(str(DATA / f"UIT_HWDB_line/train_data/{nguoi}/1.jpg"), 0), cmap="gray"); ax.axis("off")
    ax.set_title("b) Mức dòng – nhãn: “" + nhan["line"]["1.jpg"][:70] + "…”", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[2, :]); ax.imshow(cv2.imread(str(DATA / f"UIT_HWDB_paragraph/train_data/{nguoi}/1.jpg"), 0), cmap="gray"); ax.axis("off")
    ax.set_title("c) Mức đoạn – nhãn: “" + nhan["paragraph"]["1.jpg"][:70] + "…”", fontsize=9, loc="left")
    fig.text(0.5, 0.06, f"Ảnh tổng hợp của UIT-HWDB (tập huấn luyện, người viết {nguoi})", ha="center", fontsize=8.5, style="italic")
    luu(fig, "h2-3-mau-uit-hwdb.png")


def cat_ngang(gray, le=0.1):
    """Cách cũ: chiếu ngang -> đỉnh -> cắt tại chỗ trũng giữa hai đỉnh, nới `le` × khoảng dòng."""
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ps = np.convolve(bw.sum(1).astype(float), np.ones(9) / 9, "same")
    pk, _ = find_peaks(ps, distance=25, prominence=0.10 * ps.max())
    cat = [0] + [int(a + np.argmin(ps[a:b])) for a, b in zip(pk[:-1], pk[1:])] + [gray.shape[0]]
    nho = int(le * np.median(np.diff(pk)))
    return [(max(0, a - nho), min(gray.shape[0], b + nho)) for a, b in zip(cat[:-1], cat[1:])], pk


def to_mau_thanh_phan(gray):
    """Cách mới: tô mỗi thành phần liên thông theo dòng được gán (giống htr.tach_dong)."""
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ps = np.convolve(bw.sum(1).astype(float), np.ones(9) / 9, "same")
    pk, _ = find_peaks(ps, distance=25, prominence=0.10 * ps.max())
    _, nhan, _, tam = cv2.connectedComponentsWithStats(bw, connectivity=8)
    thuoc = np.abs(tam[1:, 1:2] - pk[None, :]).argmin(1)
    rgb = np.full((*gray.shape, 3), 255, np.uint8)
    for k in range(len(pk)):
        rgb[np.isin(nhan, np.where(thuoc == k)[0] + 1)] = MAU[k % len(MAU)]
    return rgb, pk


def h4_2_tach_dong(duong_dan):
    g = htr.can_bang_nen(cv2.imread(str(duong_dan), 0))
    doan, pk = cat_ngang(g, 0.1)
    rgb_cu = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    for k, (a, b) in enumerate(doan):
        lop = rgb_cu.copy(); lop[a:b] = (0.75 * lop[a:b] + 0.25 * np.array(MAU[k % len(MAU)])).astype(np.uint8)
        rgb_cu = lop
        cv2.line(rgb_cu, (0, a), (rgb_cu.shape[1], a), MAU[k % len(MAU)], 2)
    rgb_moi, _ = to_mau_thanh_phan(g)
    k = 2  # dòng thứ 3
    a, b = doan[k]
    dong_cu = g[a:b]
    dong_moi = htr.tach_dong(g)[k]
    cao = [rgb_cu.shape[0], rgb_moi.shape[0], dong_cu.shape[0] * 1.6, dong_moi.shape[0] * 1.6]
    fig = plt.figure(figsize=(10, 7.4))
    gs = fig.add_gridspec(4, 1, height_ratios=cao, hspace=0.22)
    muc = [(rgb_cu, None, "a) Cắt ngang tại chỗ trũng (lề 0,1 × khoảng dòng): các dải chồng lên nhau, mỗi dải chứa nét của dòng kề"),
           (rgb_moi, None, "b) Gán thành phần liên thông về dòng có tâm gần nhất: mỗi màu là một dòng, nét không bị chia đôi"),
           (dong_cu, "gray", "c) Ảnh dòng 3 theo cách (a): lẫn chân chữ của dòng trên và dấu của dòng dưới"),
           (dong_moi, "gray", "d) Ảnh dòng 3 theo cách (b): hết nét chồng lấn của dòng kề, nhưng vì dòng viết nghiêng, cuối dòng 2 (“Majestic,”)\n"
                              "    bị gán vào dòng 3 và một phần dòng 3 rơi sang dòng 4 – giới hạn của phương pháp dựa trên chiếu ngang")]
    for i, (im, cm, t) in enumerate(muc):
        ax = fig.add_subplot(gs[i]); ax.imshow(im, cmap=cm); ax.axis("off"); ax.set_title(t, fontsize=8.5, loc="left")
    fig.text(0.5, 0.06, f"Đoạn {duong_dan.parent.name}/{duong_dan.name} của tập kiểm định (dòng viết sát nhau)", ha="center", fontsize=8.5, style="italic")
    luu(fig, "h4-2-tach-dong.png")


def h4_8_tien_xu_ly():
    goc = cv2.imread(ANH_CHUP)
    g = cv2.cvtColor(goc, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nen = htr.can_bang_nen(g)
    dong = htr.tach_dong(nen)
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1.25], hspace=0.28, wspace=0.06)
    for j, (im, t, cm) in enumerate([(cv2.cvtColor(goc, cv2.COLOR_BGR2RGB), "a) Ảnh chụp điện thoại (nền bàn,\nbóng đổ, bút bi xanh)", None),
                                     (otsu, "b) Nhị phân Otsu trực tiếp: ~20 % điểm ảnh\nthành “mực” → cả trang thành 1 dòng", "gray"),
                                     (nen, "c) Sau cân bằng nền (chia cho nền\nước lượng bằng phép giãn)", "gray")]):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(im, cmap=cm); ax.axis("off"); ax.set_title(t, fontsize=8.5)
    sub = gs[1, :].subgridspec(min(4, len(dong)), 1, hspace=0.25)
    for i, im in enumerate(dong[:4]):
        ax = fig.add_subplot(sub[i]); ax.imshow(htr.keo_tuong_phan(im), cmap="gray", aspect="auto"); ax.axis("off")
        if i == 0:
            ax.set_title(f"d) {len(dong)} ảnh dòng sau tách dòng (tự co giãn ×≈2, kéo tương phản) – 4 dòng đầu", fontsize=8.5)
    luu(fig, "h4-8-tien-xu-ly.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    h2_3_mau()
    ch = json.load(open("../runs/crnn_line/cau_hinh.json"))
    assert "143" in ch["val_writers"], "đoạn minh hoạ phải thuộc tập kiểm định"
    h4_2_tach_dong(DATA / "UIT_HWDB_paragraph/train_data/143/1.jpg")
    h4_8_tien_xu_ly()
