"""Thống kê kết quả từ các tệp danh_gia_*.json (đầu ra danh_gia.py).

- CER/WER thô và CER "bỏ qua định dạng" (chuẩn hoá khoảng trắng quanh dấu câu, kiểu dấu ngoặc, chữ hoa/thường):
  tách lỗi nội dung khỏi lỗi định dạng mà LLM hay gây ra (viết hoa đầu dòng, bỏ khoảng trắng trong " cốp ").
- Khoảng tin cậy 95 % bằng bootstrap theo CỤM NGƯỜI VIẾT (lấy mẫu lại người viết, giữ nguyên các dòng của họ),
  vì các dòng của cùng một người không độc lập; kèm bootstrap theo dòng để so sánh.
- Phân loại dòng bị cấu hình X làm xấu đi so với a: chỉ do định dạng / có lỗi nội dung.

python thong_ke.py ../runs/crnn_line/danh_gia_dong_test_abct.json [...]
"""
import json
import re
import sys
import unicodedata

import editdistance
import numpy as np

DAU_NGOAC = str.maketrans({"“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "‘": '"', "’": '"', "'": '"'})  # mọi kiểu dấu nháy -> "


def bo_dinh_dang(s):
    s = unicodedata.normalize("NFC", s).translate(DAU_NGOAC).lower()
    s = re.sub(r"\s*([.,;:!?()\"\-/])\s*", r"\1", s)  # bỏ khoảng trắng quanh dấu câu
    return re.sub(r"\s+", " ", s).strip()


def nguoi_viet(duong_dan):
    return re.search(r"/(\d+)/[^/]+$", duong_dan.replace("\\", "/")).group(1)


def bootstrap(loi, dai, nhom, B=5000, seed=0, muc=0.95):
    """loi, dai: mảng theo dòng; nhom: mã cụm (người viết) hoặc None = theo dòng. Trả về [2,5 %; 97,5 %]."""
    rng = np.random.default_rng(seed)
    if nhom is None:
        idx = rng.integers(0, len(loi), (B, len(loi)))
        v = loi[idx].sum(1) / dai[idx].sum(1)
    else:
        cum = sorted(set(nhom))
        tl = np.array([loi[nhom == c].sum() for c in cum]); td = np.array([dai[nhom == c].sum() for c in cum])
        idx = rng.integers(0, len(cum), (B, len(cum)))
        v = tl[idx].sum(1) / td[idx].sum(1)
    return np.percentile(v, [(1 - muc) / 2 * 100, (1 + muc) / 2 * 100]) * 100


def phan_tich(tep):
    d = json.load(open(tep, encoding="utf-8"))["chi_tiet"]
    gt = [x["gt"] for x in d]
    nhom = np.array([nguoi_viet(x["anh"]) for x in d])
    dai = np.array([len(g) for g in gt], float)
    dai_dd = np.array([len(bo_dinh_dang(g)) for g in gt], float)
    cfgs = [k for k in d[0] if k not in ("anh", "gt")]
    loi = {k: np.array([editdistance.eval(x[k], x["gt"]) for x in d], float) for k in cfgs}
    loi_dd = {k: np.array([editdistance.eval(bo_dinh_dang(x[k]), bo_dinh_dang(x["gt"])) for x in d], float) for k in cfgs}
    print(f"\n=== {tep}  ({len(d)} mẫu, {len(set(nhom))} người viết)")
    print(f"{'cấu hình':10s} {'CER':>6s} {'KTC95 người viết':>18s} {'KTC95 dòng':>16s} | {'CER bỏ đ.dạng':>13s} {'KTC95 người viết':>18s} | "
          f"{'Δ CER vs a (KTC người viết)':>30s} {'Δ bỏ đ.dạng vs a':>26s} | xấu đi: chỉ định dạng / nội dung")
    for k in cfgs:
        c = loi[k].sum() / dai.sum() * 100
        cd = loi_dd[k].sum() / dai_dd.sum() * 100
        s = (f"{k:10s} {c:6.2f} {str(np.round(bootstrap(loi[k], dai, nhom), 2)):>18s} {str(np.round(bootstrap(loi[k], dai, None), 2)):>16s} | "
             f"{cd:13.2f} {str(np.round(bootstrap(loi_dd[k], dai_dd, nhom), 2)):>18s} | ")
        if k != "a":
            dl, dd = loi[k] - loi["a"], loi_dd[k] - loi_dd["a"]
            s += (f"{dl.sum()/dai.sum()*100:+6.2f} {str(np.round(bootstrap(dl, dai, nhom), 2)):>22s} "
                  f"{dd.sum()/dai_dd.sum()*100:+6.2f} {str(np.round(bootstrap(dd, dai_dd, nhom), 2)):>18s} | ")
            xau = dl > 0
            s += f"{int((xau & (dd <= 0)).sum())} / {int((xau & (dd > 0)).sum())}"
            # theo người viết: bao nhiêu người CER xấu đi / tốt lên / như cũ (kiểm định dấu, ít giả định hơn bootstrap 6 cụm)
            cum = sorted(set(nhom)); ds = [dl[nhom == c].sum() for c in cum]
            s += f" | người viết xấu/tốt/như cũ: {sum(x > 0 for x in ds)}/{sum(x < 0 for x in ds)}/{sum(x == 0 for x in ds)}"
        print(s)


if __name__ == "__main__":
    assert bo_dinh_dang('Lượng tiền " cốp " tăng') == bo_dinh_dang('lượng tiền "cốp" tăng')
    assert bo_dinh_dang("hồ - hiệp") == bo_dinh_dang("hồ-hiệp") and bo_dinh_dang("Đảng") != bo_dinh_dang("đồng")
    assert nguoi_viet("../data/UIT_HWDB_line/test_data/250/12.jpg") == "250"
    # tuỳ chọn --muc=0.975 (Bonferroni ×2) / 0.9875 (×4) cho mọi KTC
    MUC = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--muc=")), 0.95))
    _bs = bootstrap
    bootstrap = lambda loi, dai, nhom, **k: _bs(loi, dai, nhom, muc=MUC, **k)  # noqa: E731
    print(f"Mức tin cậy của KTC: {MUC:.2%}")
    for t in [a for a in sys.argv[1:] if not a.startswith("--")]:
        phan_tich(t)
