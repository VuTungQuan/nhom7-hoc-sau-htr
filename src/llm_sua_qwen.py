"""ĐỐI CHỨNG (không triển khai): hiệu chỉnh hậu kỳ bằng LLM tổng quát Qwen2.5-7B-Instruct qua Ollama.
Giữ lại chỉ để tái lập kết quả ở Phụ lục §9 của ket-qua-thuc-nghiem.md. Web app KHÔNG dùng module này.

Cần Ollama chạy tại localhost:11434 với mô hình qwen2.5:7b. Chạy lại:
  python danh_gia.py --run ../runs/crnn_line --split test --cau_hinh a,b,c,t --nguong 0.3 --bo_sua qwen --hau_to qwen
"""
import json
import re
import urllib.request

from llm_sua import chap_nhan, tach  # dùng chung ràng buộc với bộ sửa lỗi chính

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"


def hoi(prompt, model=MODEL):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 1024}}).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"].strip()


P_TOAN_BO = """Dưới đây là văn bản tiếng Việt do máy nhận dạng từ chữ viết tay, có thể sai chính tả hoặc sai dấu.
Hãy sửa lỗi nhận dạng. Giữ nguyên nội dung, thứ tự từ, số và dấu câu; không thêm, không bớt, không diễn đạt lại.
Chỉ trả về văn bản đã sửa trên một dòng, không giải thích.

Văn bản: {dong}
Đã sửa:"""

P_KIEM_SOAT = """Dưới đây là văn bản tiếng Việt do máy nhận dạng từ chữ viết tay. Các từ trong ngoặc vuông [ ] là từ máy đọc không chắc chắn.
Với mỗi từ trong ngoặc, hãy đưa ra từ tiếng Việt đúng nhất theo ngữ cảnh, thường chỉ khác một vài chữ cái hoặc dấu. Nếu từ đã đúng thì giữ nguyên.
Trả về JSON dạng {{"1": "từ", "2": "từ", ...}} theo số thứ tự các từ trong ngoặc, không giải thích.

Văn bản: {dong}
JSON:"""


def sua_toan_bo(txt, model=MODEL):
    out = hoi(P_TOAN_BO.format(dong=txt), model).splitlines()
    # bỏ tiền tố LLM hay lặp lại từ prompt, để không phạt cấu hình b vì lỗi định dạng
    return re.sub(r"^(Văn bản|Dòng|Đã sửa)\s*:\s*", "", out[0].strip()) if out else txt


def sua_co_kiem_soat(tu_conf, nguong=0.5, model=MODEL, tu_dien=None):
    """Chỉ hỏi LLM các từ có độ tin cậy < nguong (đánh dấu [ ]), LLM trả JSON; nhận bản sửa nếu qua chap_nhan
    (+ từ điển nếu có). Trả về (văn bản, số từ gửi LLM, số từ được thay)."""
    tu = [t for t, _ in tu_conf]
    phan = [tach(t) for t in tu]
    nghi = [i for i, (_, c) in enumerate(tu_conf) if c < nguong and any(ch.isalpha() for ch in phan[i][1])]
    if not nghi:
        return " ".join(tu), 0, 0
    danh_dau = list(tu)
    for i in nghi:
        dau, loi, duoi = phan[i]
        danh_dau[i] = f"{dau}[{loi}]{duoi}"
    try:
        m = re.search(r"\{.*\}", hoi(P_KIEM_SOAT.format(dong=" ".join(danh_dau)), model), re.S)
        de_xuat = json.loads(m.group(0)) if m else {}
    except (ValueError, OSError):
        de_xuat = {}
    thay = 0
    for k, i in enumerate(nghi, 1):
        dau, loi, duoi = phan[i]
        moi = tach(str(de_xuat.get(str(k), "")).strip())[1]
        if moi != loi and chap_nhan(loi, moi) and (tu_dien is None or moi.lower() in tu_dien):
            tu[i] = dau + moi + duoi
            thay += 1
    return " ".join(tu), len(nghi), thay
