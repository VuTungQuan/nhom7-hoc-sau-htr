"""CRNN + CTC cho chữ viết tay tiếng Việt (UIT-HWDB): dữ liệu, mô hình, giải mã, độ đo."""
import json
import random
import unicodedata
from pathlib import Path

import cv2
import editdistance
import numpy as np
import torch
import torch.nn as nn

NHIEU = set("ƒ")  # ký tự nhiễu gán nhãn (xem khao-sat-du-lieu.md)


def nfc(s):
    return unicodedata.normalize("NFC", s)


def bo_dau(s):
    """Bỏ dấu thanh + dấu mũ/móc, giữ đ→d: dùng để tách lỗi dấu khỏi lỗi chữ cái."""
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# ---------------- dữ liệu ----------------
def doc_mau(goc, split):
    """Trả về [(đường dẫn ảnh, nhãn, mã người viết)] của một split; bỏ ảnh thiếu nhãn và nhãn có ký tự nhiễu."""
    out = []
    for lj in sorted(Path(goc, split).glob("*/label.json")):
        w = lj.parent.name
        for img, txt in json.loads(lj.read_text(encoding="utf-8")).items():
            p = lj.parent / img
            txt = nfc(txt).strip()
            if p.exists() and txt and not (set(txt) & NHIEU):
                out.append((str(p), txt, w))
    return out


def chia_val(mau, n_val_writers=25, seed=0):
    ws = sorted({w for _, _, w in mau}, key=int)
    random.Random(seed).shuffle(ws)
    val = set(ws[:n_val_writers])
    return [m for m in mau if m[2] not in val], [m for m in mau if m[2] in val], sorted(val, key=int)


class BoKyTu:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.idx = {c: i + 1 for i, c in enumerate(self.chars)}  # 0 = blank CTC

    def ma_hoa(self, s):
        return [self.idx[c] for c in s if c in self.idx]

    def __len__(self):
        return len(self.chars) + 1


def doc_anh(path, h):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return chuan_hoa_anh(im, h)


def chuan_hoa_anh(im, h, w_max=2400):
    """Ảnh xám -> resize theo chiều cao h, giữ tỉ lệ; nền trắng."""
    r = h / im.shape[0]
    w = max(h, min(w_max, int(round(im.shape[1] * r))))
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)


class TapDong(torch.utils.data.Dataset):
    def __init__(self, mau, bkt, h=64, aug=None):
        self.mau, self.bkt, self.h, self.aug = mau, bkt, h, aug

    def __len__(self):
        return len(self.mau)

    def __getitem__(self, i):
        p, txt, _ = self.mau[i]
        im = doc_anh(p, self.h)
        if self.aug is not None:
            im = self.aug(image=im)["image"]
        return im, txt


def gop_lo(bkt):
    def f(batch):
        ims, txts = zip(*batch)
        W = max(im.shape[1] for im in ims)
        x = np.full((len(ims), 1, ims[0].shape[0], W), 255, np.uint8)
        for i, im in enumerate(ims):
            x[i, 0, :, : im.shape[1]] = im
        x = torch.from_numpy(x).float().div_(255).sub_(0.5).div_(0.5)
        widths = torch.tensor([im.shape[1] for im in ims])
        tg = [torch.tensor(bkt.ma_hoa(t)) for t in txts]
        return x, widths, torch.cat(tg), torch.tensor([len(t) for t in tg]), list(txts)
    return f


def tang_cuong():
    import albumentations as A
    # albumentations 1.4.2 (env ocr trên máy chủ): Affine dùng cval/mode
    return A.Compose([
        A.Affine(shear=(-8, 8), scale=(0.9, 1.05), rotate=(-2, 2), cval=255, mode=cv2.BORDER_CONSTANT, p=0.7),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=3)], p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    ])


# ---------------- mô hình ----------------
def _cbr(i, o):
    return [nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """CNN kiểu VGG (Shi và cộng sự, 2017) -> BiLSTM -> CTC. Chiều rộng giảm 4 lần: T = W/4."""

    def __init__(self, n_lop, h=64, rnn=256):
        super().__init__()
        self.cnn = nn.Sequential(
            *_cbr(1, 64), nn.MaxPool2d(2, 2),                       # h/2, w/2
            *_cbr(64, 128), nn.MaxPool2d(2, 2),                     # h/4, w/4
            *_cbr(128, 256), *_cbr(256, 256), nn.MaxPool2d((2, 1)),  # h/8
            *_cbr(256, 512), *_cbr(512, 512), nn.MaxPool2d((2, 1)),  # h/16
            *_cbr(512, 512), nn.Dropout2d(0.1),
        )
        self.rnn = nn.LSTM(512 * (h // 16), rnn, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(2 * rnn, n_lop)

    def forward(self, x):
        f = self.cnn(x)                       # B,C,H',T
        b, c, hh, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * hh)
        return self.fc(self.rnn(f)[0])        # B,T,n_lop (logits)


# ---------------- tiền xử lý ảnh chụp ----------------
def can_bang_nen(gray):
    """Chia ảnh cho nền ước lượng bằng phép giãn (lọc max) -> bóng đổ, ánh sáng không đều, nền bàn thành trắng,
    chỉ nét chữ tối hơn nền xung quanh còn lại. Với ảnh scan nền trắng (UIT-HWDB) gần như không đổi."""
    k = max(15, min(gray.shape) // 25) | 1
    nen = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    nen = cv2.GaussianBlur(nen, (k, k), 0)
    return np.clip(gray.astype(np.float32) / np.maximum(nen, 1) * 255, 0, 255).astype(np.uint8)


def keo_tuong_phan(im):
    """Kéo giãn để nét đậm nhất về gần đen (mực xanh/bút chì nhạt -> giống mực đen trong dữ liệu huấn luyện)."""
    toi = im[im < 200]
    if toi.size < 20:
        return im
    lo = float(np.percentile(toi, 5))
    return np.clip((im.astype(np.float32) - lo) * 255 / max(255 - lo, 1), 0, 255).astype(np.uint8)


# ---------------- tách dòng (chiếu ngang + gán thành phần liên thông) ----------------
KHOANG_DONG_CHUAN = 77  # px, trung vị khoảng cách dòng của ảnh đoạn UIT-HWDB (p10 59, p90 91)


def tach_dong(gray, _da_co_gian=False):
    """Ảnh trang/đoạn xám -> list ảnh dòng (trên xuống).
    0) Nếu khoảng cách dòng lệch xa dữ liệu huấn luyện (ảnh chụp nhỏ/quá lớn) -> co giãn ảnh về KHOANG_DONG_CHUAN
       rồi tách lại, để bước lọc nhiễu không xoá mất nét mảnh và ảnh dòng có tỉ lệ nét giống lúc huấn luyện.
    1) Chiếu ngang ảnh nhị phân, mỗi đỉnh nổi bật của biên dạng là tâm một dòng (Likforman-Sulem và cộng sự, 2007).
    2) Mỗi thành phần liên thông (nét chữ, dấu) gán về dòng có tâm gần nó nhất -> ảnh dòng chỉ chứa nét của dòng đó,
       không lẫn chân chữ/dấu của dòng kề. Tham số chọn trên tập val (CER mức đoạn 18,0 % -> 12,0 %).
    ponytail: giả định trang 1 cột, dòng gần nằm ngang; trang nghiêng/nhiều cột cần mô hình phát hiện dòng
    (Doc-UFCN, Boillet và cộng sự, 2022)."""
    from scipy.signal import find_peaks
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # bỏ đốm nhiễu
    H, W = bw.shape
    # xoá dòng kẻ vở / lề: nét thẳng ngang dài >= W/4 hoặc dọc dài >= H/3 (chữ viết tay không có nét thẳng dài vậy);
    # không xoá thì dòng kẻ đứt khúc thành các mảnh nhỏ, CRNN đọc thành chuỗi "...."
    # chỉ xoá khi có >= 3 đường ngang dài (giấy kẻ dòng); 1-2 đường thường là mép giấy/bóng, để bước lọc dưới lo
    ngang = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, max(25, W // 4)), np.uint8))
    if cv2.connectedComponents(ngang)[0] - 1 >= 3:
        ke = ngang | cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((max(25, H // 3), 1), np.uint8))
        bw = bw & (1 - ke)
    # bỏ thành phần không phải chữ: chạm mép ảnh hoặc dài quá nửa ảnh (mép giấy, bóng, nếp gấp)
    n0, nhan0, st0, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    x, y, w, h = st0[:, 0], st0[:, 1], st0[:, 2], st0[:, 3]
    bo = (x == 0) | (y == 0) | (x + w >= W) | (y + h >= H) | (w > W / 2) | (h > H / 2)
    bo[0] = True  # nhãn 0 là nền
    bw = (~bo[nhan0]).astype(np.uint8)
    ps = np.convolve(bw.sum(1).astype(float), np.ones(9) / 9, "same")
    if ps.max() == 0:
        return []
    pk, _ = find_peaks(ps, distance=25, prominence=0.10 * ps.max())
    if len(pk) <= 1:
        return [gray]
    if len(pk) > 2 and not _da_co_gian:
        k = float(np.median(np.diff(pk)))
        if not 55 <= k <= 110:
            f = min(4.0, max(0.25, KHOANG_DONG_CHUAN / k))
            return tach_dong(cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA), True)
    _, nhan, st, tam = cv2.connectedComponentsWithStats(bw, connectivity=8)
    kc = np.abs(tam[1:, 1:2] - pk[None, :])
    thuoc = kc.argmin(1)  # thành phần i+1 -> chỉ số dòng
    # xa mọi dòng hơn 1 khoảng cách dòng (chữ nhỏ đầu/cuối trang, vết bẩn): bỏ. Ngưỡng chọn trên val (0,6 làm mất dấu).
    thuoc[kc.min(1) > 1.0 * np.median(np.diff(pk))] = -1
    muc = np.bincount(thuoc[thuoc >= 0], weights=st[1:, cv2.CC_STAT_AREA][thuoc >= 0], minlength=len(pk))  # mực mỗi dòng
    dong = []
    for k in range(len(pk)):
        ids = np.where(thuoc == k)[0] + 1
        if not len(ids) or muc[k] < 0.15 * np.median(muc):  # quá ít mực so với dòng khác: chữ nhỏ đầu trang, mảnh vụn
            continue
        x0, y0 = st[ids, 0].min(), st[ids, 1].min()
        x1, y1 = (st[ids, 0] + st[ids, 2]).max(), (st[ids, 1] + st[ids, 3]).max()
        mask = cv2.dilate(np.isin(nhan[y0:y1, x0:x1], ids).astype(np.uint8), np.ones((3, 3), np.uint8))  # giữ cả viền nét
        im = np.where(mask > 0, gray[y0:y1, x0:x1], 255).astype(np.uint8)
        dong.append(cv2.copyMakeBorder(im, 8, 8, 10, 10, cv2.BORDER_CONSTANT, value=255))
    return dong


@torch.no_grad()
def nhan_dang(m, bkt, h, gray, doan=False):
    """Ảnh xám (một dòng, hoặc đoạn/trang nếu doan=True) -> (văn bản, [(từ, độ tin cậy)])."""
    dev = next(m.parameters()).device
    gray = can_bang_nen(gray)
    txt, tu = [], []
    for im in (tach_dong(gray) if doan else [gray]):
        im = keo_tuong_phan(im)
        x = torch.from_numpy(chuan_hoa_anh(im, h)).float().div(255).sub(0.5).div(0.5)[None, None].to(dev)
        t, tc = giai_ma(m(x), torch.tensor([x.shape[-1]]), bkt)[0]
        if t and (not doan or any(c.isalnum() for c in t)):  # dòng chỉ toàn dấu câu = mảnh nhiễu (chữ nhỏ đầu trang, vết bẩn)
            txt.append(t); tu += tc
    return " ".join(txt), tu


# ---------------- giải mã + độ đo ----------------
def giai_ma(logits, widths, bkt):
    """Greedy CTC. Trả về [(văn bản, [(từ, độ tin cậy = min xác suất ký tự)])]."""
    prob, idx = logits.float().softmax(-1).max(-1)
    out = []
    for b in range(idx.shape[0]):
        T = int(widths[b]) // 4
        chars, confs, prev = [], [], 0
        for t in range(T):
            k = int(idx[b, t])
            if k and k != prev:
                chars.append(bkt.chars[k - 1])
                confs.append(float(prob[b, t]))
            prev = k
        txt = "".join(chars)
        tu, cur, cc = [], "", []
        for ch, p in zip(chars + [" "], confs + [1.0]):
            if ch == " ":
                if cur:
                    tu.append((cur, min(cc)))
                cur, cc = "", []
            else:
                cur += ch
                cc.append(p)
        out.append((txt, tu))
    return out


def do_do(du_doan, nhan):
    """CER, WER, CER sau khi bỏ dấu (lỗi chữ cái), và phần chênh = lỗi chỉ do dấu."""
    e = n = we = wn = e0 = 0
    for p, g in zip(du_doan, nhan):
        e += editdistance.eval(p, g); n += len(g)
        we += editdistance.eval(p.split(), g.split()); wn += len(g.split())
        e0 += editdistance.eval(bo_dau(p), bo_dau(g))
    cer, cer0 = e / max(n, 1), e0 / max(n, 1)
    return {"CER": cer, "WER": we / max(wn, 1), "CER_khong_dau": cer0, "loi_dau": cer - cer0}
