"""Hiệu chỉnh hậu kỳ đầu ra CRNN bằng mô hình sửa lỗi văn bản tiếng Việt protonx-legal-tc
(ViT5-base tinh chỉnh, seq2seq, 226 triệu tham số; giấy phép ProtonX v1.3-NC – phi thương mại),
chạy cục bộ bằng Hugging Face transformers. "Built with ProtonX Text Correction Model."

- de_xuat(ds): mô hình sửa lỗi từng văn bản (cắt khúc <= 30 từ vì giới hạn ~128–160 token), chạy theo lô.
- Cấu hình b: dùng nguyên văn bản mô hình đề xuất.
- Cấu hình c: căn chỉnh từ giữa đầu vào và đề xuất, CHỈ nhận thay thế 1–1 tại từ CRNN kém tin cậy (< ngưỡng)
  và khác không quá 50 % độ dài; t = c + bản sửa phải có trong từ điển âm tiết của tập train.
"""
import difflib
import os
import re

import editdistance

MO_HINH = os.environ.get("MO_HINH_SUA", "/mnt/data/hs-htr/hf/protonx-legal-tc")
KHUC_TU = 30      # số từ tối đa mỗi khúc gửi mô hình
SO_BEAM = 10  # chọn trên val (beam 1/5/10: CER đoạn t@1.01 = 10,39/10,36/10,35 %)
_mo_hinh = _tok = None


def nap():
    """Nạp mô hình một lần. Tokenizer đọc thẳng tokenizer.json (AutoTokenizer của transformers 5.x lỗi với tệp này)."""
    global _mo_hinh, _tok
    if _mo_hinh is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
        _tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(MO_HINH, "tokenizer.json"),
                                       eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
        _mo_hinh = AutoModelForSeq2SeqLM.from_pretrained(MO_HINH).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return _mo_hinh, _tok


def _sinh(ds, lo=32):
    import torch
    m, tok = nap()
    ra = []
    for i in range(0, len(ds), lo):
        enc = tok(ds[i:i + lo], return_tensors="pt", padding=True, truncation=True, max_length=160).to(m.device)
        with torch.no_grad():
            out = m.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                             num_beams=SO_BEAM, max_new_tokens=160)
        ra += tok.batch_decode(out, skip_special_tokens=True)
    return ra


def de_xuat(ds):
    """[văn bản] -> [văn bản mô hình đề xuất]. Văn bản dài được cắt khúc theo từ rồi ghép lại."""
    khuc, chi_so = [], []
    for i, s in enumerate(ds):
        tu = s.split()
        for j in range(0, len(tu), KHUC_TU):
            khuc.append(" ".join(tu[j:j + KHUC_TU])); chi_so.append(i)
    ra = [[] for _ in ds]
    for i, s in zip(chi_so, _sinh(khuc) if khuc else []):
        ra[i].append(s.strip())
    return [" ".join(x) for x in ra]


def sua_toan_bo(txt):
    return de_xuat([txt])[0] if txt.strip() else txt


def tach(tok):
    """'(mồi,' -> ('(', 'mồi', ','): chỉ phần chữ được so khớp/thay, dấu câu giữ nguyên."""
    return re.match(r"^(\W*)(.*?)(\W*)$", tok).groups()


def chap_nhan(cu, moi, toi_da=0.5):
    """Chỉ nhận bản sửa là một từ, không khác quá toi_da độ dài từ cũ (chặn thay hẳn bằng từ khác)."""
    return bool(moi) and " " not in moi and editdistance.eval(cu, moi) <= max(1, int(len(cu) * toi_da))


def tao_tu_dien(nhan):
    """Tập âm tiết (chữ thường, bỏ dấu câu) từ nhãn huấn luyện -> lớp chặn: chỉ nhận bản sửa là âm tiết có thật."""
    return {tach(t)[1].lower() for s in nhan for t in s.split()} - {""}


def kiem_soat(tu_conf, goi_y, nguong, tu_dien=None):
    """Áp đề xuất `goi_y` của mô hình lên dãy từ CRNN có kiểm soát.
    Trả về (văn bản, số từ kém tin cậy đủ điều kiện, số từ được thay)."""
    tu = [t for t, _ in tu_conf]
    phan = [tach(t) for t in tu]
    nghi = {i for i, (_, c) in enumerate(tu_conf) if c < nguong and any(ch.isalpha() for ch in phan[i][1])}
    if not nghi:
        return " ".join(tu), 0, 0
    a = [p[1] for p in phan]
    b = [tach(t)[1] for t in goi_y.split()]
    thay = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "replace" or i2 - i1 != j2 - j1:  # chỉ thay 1–1; bỏ qua thêm/bớt/gộp/tách từ
            continue
        for i, moi in zip(range(i1, i2), b[j1:j2]):
            if i in nghi and moi != a[i] and chap_nhan(a[i], moi) and (tu_dien is None or moi.lower() in tu_dien):
                dau, _, duoi = phan[i]
                tu[i] = dau + moi + duoi
                thay += 1
    return " ".join(tu), len(nghi), thay


def sua_co_kiem_soat(tu_conf, nguong=0.5, tu_dien=None):
    tu = [t for t, _ in tu_conf]
    if not any(c < nguong for _, c in tu_conf):
        return " ".join(tu), 0, 0          # không có từ kém tin cậy -> không gọi mô hình
    return kiem_soat(tu_conf, de_xuat([" ".join(tu)])[0], nguong, tu_dien)


if __name__ == "__main__":
    # tự kiểm tra cơ chế kiểm soát (không cần mô hình)
    assert chap_nhan("tiên", "tiền") and not chap_nhan("cấp", "cốp tăng") and not chap_nhan("lộ", "phần trăm")
    assert tach('(mồi,') == ("(", "mồi", ",") and tach('"') == ('"', "", "")
    assert tao_tu_dien(['Cả sới, "gà"']) == {"cả", "sới", "gà"}
    vd = [("Lượng", .99), ("tiên", .2), ('"', .99), ("cấp", .3), ('"', .99), ("tăng", .95), ("chóng", .4)]
    # tiên->tiền (kém tin cậy, nhận); cấp->cốp (nhận); tăng->tang (tin cậy cao, KHÔNG nhận); chóng: không đổi
    s, n, k = kiem_soat(vd, 'Lượng tiền " cốp " tang chóng mặt', 0.5)
    assert s == 'Lượng tiền " cốp " tăng chóng' and (n, k) == (3, 2), (s, n, k)
    s, _, k = kiem_soat(vd, 'Lượng tiền " cốp " tăng chóng', 0.5, tu_dien={"tiền", "lượng"})  # cốp không có trong từ điển
    assert s == 'Lượng tiền " cấp " tăng chóng' and k == 1, s
    s, _, k = kiem_soat([("ắt", .2), ("phút,", .9)], "bắt đầu phút,", 0.5)  # chèn từ -> bỏ qua
    assert k == 0 and s == "ắt phút,", s
    print("tu kiem OK")
    if os.path.isdir(MO_HINH):
        print(de_xuat(["Chúng tôi táp trung nghiên cứu nhận dạng chữ viết tay tiếng Việt."]))  # câu tự đặt, không lấy từ dữ liệu
