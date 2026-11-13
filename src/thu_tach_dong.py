"""Thử các cách tách dòng trên 108 đoạn val (cấu hình a = chỉ CRNN). Chỉ dùng val để chọn, không dùng test.
  python thu_tach_dong.py --run ../runs/crnn_line [--che_do cc,cat,seam,seam_gan] [--test]

Chế độ:
  cc       bản hiện tại: đỉnh chiếu ngang + gán nguyên thành phần liên thông về dòng gần tâm nhất
  cat      như cc, nhưng thành phần vắt qua ranh giới hai dòng (nét chạm nhau) bị cắt theo hàng trũng thẳng
  seam     như cat, nhưng cắt theo đường uốn (quy hoạch động) đi qua chỗ ít mực giữa hai dòng
  seam_gan mọi thành phần gán theo phía của đường uốn chứa phần lớn điểm ảnh (không dùng tâm)
Mức trần: nhận dạng ảnh dòng cắt sẵn của UIT-HWDB (cùng nội dung) rồi nối lại = tách dòng hoàn hảo.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import editdistance
import numpy as np
import torch
from scipy.signal import find_peaks

import htr

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--che_do", default="cc,theo150,theo100,theo75,theo100|w0.25,theo100|w0.5,theo100|t0.15,theo100|b,theo100|c")
ap.add_argument("--test", action="store_true", help="chạy trên test (chỉ sau khi đã chốt trên val)")
ap.add_argument("--ra", default="")
a = ap.parse_args()


def nhi_phan(gray):
    """Giống htr.tach_dong: Otsu, bỏ đốm, xoá dòng kẻ, bỏ thành phần chạm mép/quá lớn."""
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    H, W = bw.shape
    ngang = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, max(25, W // 4)), np.uint8))
    if cv2.connectedComponents(ngang)[0] - 1 >= 3:
        ke = ngang | cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((max(25, H // 3), 1), np.uint8))
        bw = bw & (1 - ke)
    n0, nhan0, st0, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    x, y, w, h = st0[:, 0], st0[:, 1], st0[:, 2], st0[:, 3]
    bo = (x == 0) | (y == 0) | (x + w >= W) | (y + h >= H) | (w > W / 2) | (h > H / 2)
    bo[0] = True
    return (~bo[nhan0]).astype(np.uint8)


def duong_uon(bw, r0, r1, thang, phat=0.3):
    """Đường từ trái sang phải trong dải hàng [r0, r1), mỗi cột lệch tối đa 1 hàng, chi phí = mực + phạt lệch khỏi hàng
    trũng `thang`. Trả về chỉ số hàng theo từng cột."""
    R = max(r1 - r0, 1)
    rows = np.arange(r0, r0 + R)
    cost = bw[r0:r0 + R].astype(np.float32) * 50 + phat * np.abs(rows - thang)[:, None]
    W = bw.shape[1]
    acc = cost[:, 0].copy()
    lui = np.zeros((R, W), np.int8)
    for x in range(1, W):
        tren = np.r_[np.inf, acc[:-1]]
        duoi = np.r_[acc[1:], np.inf]
        st = np.stack([tren + 1, acc, duoi + 1])  # đi chéo tốn thêm 1
        k = st.argmin(0)
        acc = st[k, np.arange(R)] + cost[:, x]
        lui[:, x] = k - 1  # -1: từ hàng trên, 0: ngang, +1: từ hàng dưới
    r = int(acc.argmin())
    out = np.empty(W, np.int32)
    for x in range(W - 1, -1, -1):
        out[x] = r
        r = min(max(r + int(lui[r, x]), 0), R - 1)
    return out + r0


def sua_dinh(ps, pk, che_do):
    """Sửa đỉnh dòng. che_do = "<gán>|g0.6|c|b": g = gộp đỉnh cách < g × khoảng dòng (giữ đỉnh cao hơn),
    c = chèn đỉnh vào khoảng trống > 1,6 × khoảng dòng, b = thêm đỉnh thấp ở trước đỉnh đầu / sau đỉnh cuối (dòng ngắn)."""
    tk = che_do.split("|")[1:]
    if not tk or not len(pk):
        return pk
    kd = float(np.median(np.diff(pk))) if len(pk) > 2 else float(htr.KHOANG_DONG_CHUAN)
    pk = list(pk)
    thap, _ = find_peaks(ps, distance=15, prominence=0.03 * ps.max())  # ứng viên đỉnh thấp
    thap = [int(x) for x in thap if ps[x] >= 0.08 * ps.max()]
    if "c" in tk:
        doi = True
        while doi:
            doi = False
            for a_, b_ in zip(pk[:-1], pk[1:]):
                if b_ - a_ > 1.6 * kd:
                    ung = [x for x in thap if a_ + 0.6 * kd <= x <= b_ - 0.6 * kd]
                    if ung:
                        pk = sorted(pk + [max(ung, key=lambda x: ps[x])]); doi = True; break
    if "b" in tk:
        cuoi = [x for x in thap if x >= pk[-1] + 0.6 * kd]
        while cuoi:
            x = min(cuoi); pk.append(x); cuoi = [y for y in cuoi if y >= x + 0.6 * kd]
        dau = [x for x in thap if x <= pk[0] - 0.6 * kd]
        while dau:
            x = max(dau); pk.insert(0, x); dau = [y for y in dau if y <= x - 0.6 * kd]
    g = [float(t[1:]) for t in tk if t.startswith("g")]
    if g:
        doi = True
        while doi and len(pk) > 1:
            doi = False
            for i in range(len(pk) - 1):
                if pk[i + 1] - pk[i] < g[0] * kd:
                    del pk[i if ps[pk[i]] < ps[pk[i + 1]] else i + 1]; doi = True; break
    return np.array(sorted(pk))


def tach(gray, che_do, _da_co_gian=False):
    bw = nhi_phan(gray)
    H, W = bw.shape
    ps = np.convolve(bw.sum(1).astype(float), np.ones(9) / 9, "same")
    if ps.max() == 0:
        return []
    pk, _ = find_peaks(ps, distance=25, prominence=0.10 * ps.max())
    if len(pk) > 2 and not _da_co_gian:
        k = float(np.median(np.diff(pk)))
        if not 55 <= k <= 110:
            f = min(4.0, max(0.25, htr.KHOANG_DONG_CHUAN / k))
            return tach(cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA), che_do, True)
    pk = sua_dinh(ps, pk, che_do)
    if len(pk) <= 1:
        return [gray]
    kd = float(np.median(np.diff(pk)))
    tk = che_do.split("|")[1:]
    cua = next((float(t[1:]) for t in tk if t.startswith("w")), 0.35)  # nửa cửa sổ tìm tâm dòng (× khoảng dòng)
    nguong_d = next((float(t[1:]) for t in tk if t.startswith("t")), 0.3)  # dải ít mực hơn ngưỡng này -> giữ tâm dải kề
    che_do = che_do.split("|")[0]
    n, nhan, st, tam = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if che_do.startswith("theo"):
        # bám tâm dòng theo dải dọc: dòng nghiêng/cong -> tâm dòng y_k(x) thay vì một hàng cố định
        rong = {"theo": 250, "theo2": 150}.get(che_do) or int(che_do[4:])
        S = max(4, W // rong)
        bx = np.linspace(0, W, S + 1).astype(int)
        tx = (bx[:-1] + bx[1:]) / 2
        Y = np.tile(pk.astype(float), (S, 1))
        giua = S // 2
        for s_ in list(range(giua, S)) + list(range(giua - 1, -1, -1)):
            ref = pk.astype(float) if s_ == giua else (Y[s_ - 1] if s_ > giua else Y[s_ + 1])
            p_s = np.convolve(bw[:, bx[s_]:bx[s_ + 1]].sum(1).astype(float), np.ones(9) / 9, "same")
            for k in range(len(pk)):
                lo, hi = int(max(0, ref[k] - cua * kd)), int(min(H, ref[k] + cua * kd))
                seg = p_s[lo:hi]
                Y[s_, k] = lo + int(seg.argmax()) if seg.size and seg.max() >= nguong_d * max(p_s.max(), 1) else ref[k]
        Yc = np.stack([np.interp(tam[1:, 0], tx, Y[:, k]) for k in range(len(pk))], 1)  # (thành phần, dòng)
        kc = np.abs(tam[1:, 1:2] - Yc)
        che_do = "cc"
    else:
        kc = np.abs(tam[1:, 1:2] - pk[None, :])
    thuoc = kc.argmin(1)
    thuoc[kc.min(1) > 1.0 * kd] = -1

    if che_do != "cc":
        # ranh giới giữa hai dòng kề: hàng trũng nhất (thẳng) hoặc đường uốn qua chỗ ít mực
        bien = [int(pk[k] + np.argmin(ps[pk[k]:pk[k + 1]])) for k in range(len(pk) - 1)]
        if che_do == "cat":
            duong = [np.full(W, b, np.int32) for b in bien]
        else:
            duong = [duong_uon(bw, int(pk[k]) + 3, int(pk[k + 1]) - 2, bien[k]) for k in range(len(pk) - 1)]
        D = np.stack(duong)  # (số ranh giới, W)
    lut = np.r_[-1, thuoc].astype(np.int32)
    lab = lut[nhan]  # dòng của từng điểm ảnh (theo thành phần)
    if che_do != "cc":
        for i in range(1, n):
            x0, y0, w, h = st[i, :4]
            if che_do != "seam_gan" and h < 0.5 * kd:
                continue
            ys, xs = np.nonzero(nhan[y0:y0 + h, x0:x0 + w] == i)
            ys += y0; xs += x0
            li = (ys[None, :] > D[:, xs]).sum(0)  # dòng theo ranh giới, từng điểm ảnh
            dem = np.bincount(li, minlength=len(pk))
            if che_do == "seam_gan":
                if thuoc[i - 1] >= 0:
                    lab[ys, xs] = int(dem.argmax())
                continue
            # cắt khi ≥ 2 dòng mỗi dòng có ≥ 20 % điểm ảnh và ≥ 30 điểm: nét chạm nhau giữa hai dòng
            lon = np.nonzero((dem >= 0.2 * dem.sum()) & (dem >= 30))[0]
            if len(lon) >= 2:
                lab[ys, xs] = li
    muc = np.bincount(lab[lab >= 0], minlength=len(pk))
    dong = []
    for k in range(len(pk)):
        mk = lab == k
        if not muc[k] or muc[k] < 0.15 * np.median(muc):
            continue
        yy, xx = np.nonzero(mk)
        y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        mask = cv2.dilate(mk[y0:y1, x0:x1].astype(np.uint8), np.ones((3, 3), np.uint8))
        im = np.where(mask > 0, gray[y0:y1, x0:x1], 255).astype(np.uint8)
        dong.append(cv2.copyMakeBorder(im, 8, 8, 10, 10, cv2.BORDER_CONSTANT, value=255))
    return dong


run = Path(a.run)
ch = json.load(open(run / "cau_hinh.json", encoding="utf-8"))
bkt = htr.BoKyTu(ch["chars"])
m = htr.CRNN(len(bkt), ch["h"]).cuda().eval()
m.load_state_dict(torch.load(run / "best.pt", map_location="cuda"))


@torch.no_grad()
def doc_dong(im):
    im = htr.keo_tuong_phan(im)
    x = torch.from_numpy(htr.chuan_hoa_anh(im, ch["h"])).float().div(255).sub(0.5).div(0.5)[None, None].cuda()
    return htr.giai_ma(m(x), torch.tensor([x.shape[-1]]), bkt)[0][0]


def doc_doan(gray, che_do):
    txt, ds = [], tach(htr.can_bang_nen(gray), che_do)
    for im in ds:
        t = doc_dong(im)
        if t and any(c.isalnum() for c in t):
            txt.append(t)
    return " ".join(txt), len(ds)


split = "test_data" if a.test else "train_data"
mau = htr.doc_mau("../data/UIT_HWDB_paragraph", split)
if not a.test:
    mau = [x for x in mau if x[2] in set(ch["val_writers"])]
mau.sort(key=lambda x: (int(x[2]), int(Path(x[0]).stem)))

# ánh xạ dòng UIT_HWDB_line -> đoạn (nhãn dòng là các dòng liên tiếp của đoạn, cùng người viết)
dong_gt = {}
for w in sorted({x[2] for x in mau}, key=int):
    lj = json.loads(Path(f"../data/UIT_HWDB_line/{split}/{w}/label.json").read_text(encoding="utf-8"))
    items = sorted(lj.items(), key=lambda kv: int(kv[0].split(".")[0]))
    j = 0
    for p, g, ww in [x for x in mau if x[2] == w]:
        acc = []
        while j < len(items) and len(" ".join(t for _, t in acc)) < len(g) - 3:
            acc.append(items[j]); j += 1
        s = htr.nfc(" ".join(t.strip() for _, t in acc))
        paths = [Path(f"../data/UIT_HWDB_line/{split}/{w}/{f}") for f, _ in acc]
        if acc and editdistance.eval(s, g) <= 0.05 * len(g) and all(q.exists() for q in paths):
            dong_gt[p] = paths

kq = {}
chi_tiet = {p: {"w": w, "gt": g, "so_dong_gt": len(dong_gt.get(p, []))} for p, g, w in mau}
if dong_gt:
    for p, g, w in mau:
        if p in dong_gt:
            chi_tiet[p]["tran"] = " ".join(t for t in (doc_dong(cv2.imread(str(q), cv2.IMREAD_GRAYSCALE)) for q in dong_gt[p]) if t)
for cd in a.che_do.split(","):
    for p, g, w in mau:
        t, sd = doc_doan(cv2.imread(p, cv2.IMREAD_GRAYSCALE), cd)
        chi_tiet[p][cd] = t
        chi_tiet[p]["so_dong_" + cd] = sd


def cer(key, chi=None):
    e = n = 0
    for p, d in chi_tiet.items():
        if key in d and (chi is None or p in chi):
            e += editdistance.eval(d[key], d["gt"]); n += len(d["gt"])
    return round(100 * e / max(n, 1), 2)


co_gt = set(dong_gt)
print(f"{len(mau)} đoạn, {len(co_gt)} đoạn ánh xạ được sang ảnh dòng chuẩn")
if co_gt:
    print("mức trần (ảnh dòng chuẩn):", cer("tran", co_gt), "% trên các đoạn ánh xạ được")
for cd in a.che_do.split(","):
    lech = [d["so_dong_" + cd] - d["so_dong_gt"] for p, d in chi_tiet.items() if p in co_gt]
    print(f"{cd:9s} CER {cer(cd)} % (đoạn ánh xạ được: {cer(cd, co_gt)} %) | số dòng: đúng {sum(x == 0 for x in lech)}, "
          f"thiếu {sum(x < 0 for x in lech)}, thừa {sum(x > 0 for x in lech)}")
    kq[cd] = cer(cd)
# so với cc theo người viết
if "cc" in a.che_do:
    for cd in a.che_do.split(","):
        if cd == "cc":
            continue
        E = defaultdict(lambda: [0, 0, 0])
        for p, d in chi_tiet.items():
            E[d["w"]][0] += editdistance.eval(d["cc"], d["gt"]); E[d["w"]][1] += editdistance.eval(d[cd], d["gt"]); E[d["w"]][2] += len(d["gt"])
        tot = sum(v[1] < v[0] for v in E.values()); xau = sum(v[1] > v[0] for v in E.values())
        print(f"{cd} so với cc: người viết tốt lên {tot}, xấu đi {xau}, như cũ {len(E) - tot - xau}")
if a.ra:
    json.dump({"ket_qua": kq, "chi_tiet": chi_tiet}, open(a.ra, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
