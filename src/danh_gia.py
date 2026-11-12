"""Đánh giá các cấu hình: a = CRNN thuần, b = + mô hình sửa lỗi (protonx-legal-tc) dùng nguyên đề xuất,
c = + sửa lỗi có kiểm soát (chỉ tại từ kém tin cậy), t = c + chặn bằng từ điển âm tiết của tập train.

Chọn ngưỡng trên toàn bộ val (647 dòng / 108 đoạn):
  python danh_gia.py --run ../runs/crnn_line --split val --cau_hinh a,b,c,t --nguong 0.3,0.5,0.7,0.9,1.01
  python danh_gia.py --run ../runs/crnn_line --goc ../data/UIT_HWDB_paragraph --split val --cau_hinh a,b,c,t --nguong 0.3,0.5,0.7,0.9,1.01
Chạy test với hai ngưỡng đã chọn trên val (mức dòng; thêm --goc ../data/UIT_HWDB_paragraph cho mức đoạn):
  python danh_gia.py --run ../runs/crnn_line --split test --cau_hinh a,b,c,t --nguong 0.9,1.01
Đối chứng LLM Qwen2.5-7B qua Ollama (Phụ lục §9; cần Ollama + qwen2.5:7b; tệp kết quả cũ không có hậu tố):
  python danh_gia.py --run ../runs/crnn_line --split test --cau_hinh a,b,c,t --nguong 0.3,0.5 --bo_sua qwen --hau_to qwen
Thống kê (KTC theo người viết, CER bỏ định dạng): python thong_ke.py ../runs/crnn_line/danh_gia_*.json
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import editdistance
import torch

import htr
import llm_sua

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--goc", default="../data/UIT_HWDB_line")
ap.add_argument("--split", choices=["val", "test"], default="test")
ap.add_argument("--cau_hinh", default="a,b,c")
ap.add_argument("--nguong", default="0.9")
ap.add_argument("--gioi_han", type=int, default=0)
ap.add_argument("--bo_sua", choices=["protonx", "qwen"], default="protonx", help="bộ sửa lỗi hậu kỳ (qwen = đối chứng, cần Ollama)")
ap.add_argument("--hau_to", default="tc", help="hậu tố tên tệp kết quả (tc = protonx-legal-tc)")
ap.add_argument("--so_beam", type=int, default=llm_sua.SO_BEAM, help="số nhánh beam search của mô hình sửa lỗi")
a = ap.parse_args()
llm_sua.SO_BEAM = a.so_beam

run = Path(a.run)
ch = json.load(open(run / "cau_hinh.json", encoding="utf-8"))
bkt = htr.BoKyTu(ch["chars"])
if a.split == "test":
    mau = htr.doc_mau(a.goc, "test_data")
else:
    mau = [m for m in htr.doc_mau(a.goc, "train_data") if m[2] in set(ch["val_writers"])]
if a.gioi_han:
    mau = mau[: a.gioi_han]

m = htr.CRNN(len(bkt), ch["h"]).cuda().eval()
m.load_state_dict(torch.load(run / "best.pt", map_location="cuda"))

# --- a: CRNN (mức đoạn: tách dòng rồi nhận dạng từng dòng)
doan = "paragraph" in a.goc
t0 = time.time()
crnn = [htr.nhan_dang(m, bkt, ch["h"], cv2.imread(p, cv2.IMREAD_GRAYSCALE), doan) for p, _, _ in mau]
t_crnn = (time.time() - t0) / len(mau)
gt = [g for _, g, _ in mau]
pa = [t for t, _ in crnn]


def so_sanh(pd):
    """Ngoài CER/WER: số dòng bước hậu kỳ làm tốt lên / xấu đi so với CRNN."""
    r = htr.do_do(pd, gt)
    tot = xau = 0
    for p0, p1, g in zip(pa, pd, gt):
        d = editdistance.eval(p1, g) - editdistance.eval(p0, g)
        tot += d < 0; xau += d > 0
    r.update(dong_tot_len=tot, dong_xau_di=xau, n=len(gt))
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}


kq, chi_tiet = {}, [{"anh": p, "gt": g, "a": t} for (p, g, _), t in zip(mau, pa)]
cfg = a.cau_hinh.split(",")
if "a" in cfg:
    kq["a"] = {**so_sanh(pa), "giay_moi_dong": round(t_crnn, 3)}

qwen = a.bo_sua == "qwen"
if qwen:  # ĐỐI CHỨNG: LLM Qwen qua Ollama, gọi từng mẫu (xem llm_sua_qwen.py)
    import llm_sua_qwen
    if "b" in cfg:
        t0 = time.time()
        pb = [llm_sua_qwen.sua_toan_bo(t) for t in pa]
        kq["b"] = {**so_sanh(pb), "giay_moi_dong": round(t_crnn + (time.time() - t0) / len(mau), 3)}
        for d, t in zip(chi_tiet, pb):
            d["b"] = t
else:
    # Mô hình sửa lỗi chạy MỘT lần trên toàn bộ đầu ra CRNN (theo lô); b dùng nguyên đề xuất,
    # c/t chỉ áp đề xuất tại từ kém tin cậy -> quét nhiều ngưỡng không phải chạy lại mô hình.
    dau_vao = [" ".join(w for w, _ in tc) for _, tc in crnn]
    if {"b", "c", "t"} & set(cfg):
        llm_sua.nap()
        t0 = time.time()
        goi_y = llm_sua.de_xuat(dau_vao)
        t_mh = (time.time() - t0) / len(mau)  # giây mô hình / mẫu (chạy lô 32, beam 5)
    if "b" in cfg:
        kq["b"] = {**so_sanh(goi_y), "giay_moi_dong": round(t_crnn + t_mh, 3)}
        for d, t in zip(chi_tiet, goi_y):
            d["b"] = t
# từ điển âm tiết: nhãn dòng của tập train, trừ người viết val (không dùng nhãn val/test)
tep_td = run / "tu_dien.txt"
if not tep_td.exists():
    val_ws = set(ch["val_writers"])
    nhan = [g for _, g, w in htr.doc_mau("../data/UIT_HWDB_line", "train_data") if w not in val_ws]
    tep_td.write_text("\n".join(sorted(llm_sua.tao_tu_dien(nhan))), encoding="utf-8")
tu_dien = set(tep_td.read_text(encoding="utf-8").split("\n"))

for ten_c, td in (("c", None), ("t", tu_dien)):  # c: tin cậy + khoảng cách sửa; t: c + từ điển
    if ten_c not in cfg:
        continue
    for ng in map(float, a.nguong.split(",")):
        pc, gui, thay, can_mh = [], 0, 0, 0
        if qwen:
            t0 = time.time()
            for _, tc in crnn:
                s, g_, t_ = llm_sua_qwen.sua_co_kiem_soat(tc, ng, tu_dien=td)
                pc.append(s); gui += g_; thay += t_
            kq[f"{ten_c}@{ng}"] = {**so_sanh(pc), "tu_gui_llm": gui, "tu_duoc_thay": thay,
                                   "giay_moi_dong": round(t_crnn + (time.time() - t0) / len(mau), 3)}
            for d, t in zip(chi_tiet, pc):
                d[f"{ten_c}@{ng}"] = t
            continue
        for (_, tc), gy in zip(crnn, goi_y):
            s, g_, t_ = llm_sua.kiem_soat(tc, gy, ng, td)
            pc.append(s); gui += g_; thay += t_; can_mh += g_ > 0
        # thời gian: chỉ mẫu có từ kém tin cậy mới cần gọi mô hình (ước tính theo thời gian TB/mẫu)
        kq[f"{ten_c}@{ng}"] = {**so_sanh(pc), "tu_gui_llm": gui, "tu_duoc_thay": thay, "mau_goi_mo_hinh": can_mh,
                               "giay_moi_dong": round(t_crnn + t_mh * can_mh / len(mau), 3)}
        for d, t in zip(chi_tiet, pc):
            d[f"{ten_c}@{ng}"] = t

ten = run / f"danh_gia_{'doan' if doan else 'dong'}_{a.split}_{a.cau_hinh.replace(',', '')}_{a.hau_to}.json"
json.dump({"ket_qua": kq, "chi_tiet": chi_tiet}, open(ten, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in kq.items():
    print(k, v)
