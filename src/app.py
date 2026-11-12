"""Web app nhận dạng chữ viết tay tiếng Việt: tải PDF/ảnh hoặc cắt vùng ảnh -> văn bản.
CRNN + hiệu chỉnh hậu kỳ bằng mô hình sửa lỗi protonx-legal-tc (transformers, chạy cục bộ; không dùng Ollama).

RUN=../runs/crnn_line uvicorn app:app --host 127.0.0.1 --port 8765
Công khai qua Cloudflare Tunnel (bin/chay_tunnel.sh) tại nhom7hocsau.cat.io.vn; hoặc truy cập riêng qua
đường hầm SSH: ssh -L 8765:localhost:8765 <tài khoản>@<máy chủ>
"""
import base64
import json
import os
import time
from pathlib import Path

import asyncio

import cv2
import fitz  # PyMuPDF
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

import htr
import llm_sua
import tai_khoan

RUN = Path(os.environ.get("RUN", "../runs/crnn_line"))
NGUONG = float(os.environ.get("NGUONG", "1.01"))  # ngưỡng sửa lỗi chọn trên toàn bộ val (t@1.01: tốt nhất mức đoạn)
TO_MAU = 0.5  # chỉ để tô màu từ kém tin cậy trên giao diện
TEP_TK = Path(os.environ.get("TAI_KHOAN", "../tai_khoan.json"))  # tạo bằng: python tai_khoan.py <tệp> <tên>
DS_TK = json.loads(TEP_TK.read_text())
PHIEN = tai_khoan.Phien()
ch = json.load(open(RUN / "cau_hinh.json", encoding="utf-8"))
bkt = htr.BoKyTu(ch["chars"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = htr.CRNN(len(bkt), ch["h"]).to(dev).eval()
m.load_state_dict(torch.load(RUN / "best.pt", map_location=dev))
llm_sua.nap()  # nạp sẵn mô hình sửa lỗi protonx-legal-tc (transformers) lúc khởi động
TU_DIEN = set((RUN / "tu_dien.txt").read_text(encoding="utf-8").split("\n")) if (RUN / "tu_dien.txt").exists() else None

app = FastAPI(title="Nhận dạng chữ viết tay tiếng Việt", docs_url=None, redoc_url=None, openapi_url=None)
TOI_DA_MB, TOI_DA_TRANG = 20, 20
MO = {"/dang-nhap"}  # đường dẫn không cần đăng nhập


@app.middleware("http")
async def bat_dang_nhap(request: Request, call_next):
    if request.url.path not in MO and not PHIEN.doc(request.cookies.get("phien")):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Chưa đăng nhập"}, 401)
        return RedirectResponse("/dang-nhap", 303)
    return await call_next(request)


@app.get("/dang-nhap")
def trang_dang_nhap():
    return FileResponse(Path(__file__).with_name("dang-nhap.html"))


@app.post("/dang-nhap")
async def dang_nhap(ten: str = Form(...), mat_khau: str = Form(...)):
    if not tai_khoan.kiem_tra(DS_TK, ten, mat_khau):
        await asyncio.sleep(1)  # làm chậm dò mật khẩu
        return RedirectResponse("/dang-nhap?loi=1", 303)
    r = RedirectResponse("/", 303)
    r.set_cookie("phien", PHIEN.tao(ten), max_age=PHIEN.giay, httponly=True, samesite="strict")
    return r


@app.post("/dang-xuat")
def dang_xuat():
    r = RedirectResponse("/dang-nhap", 303)
    r.delete_cookie("phien")
    return r


def anh_tu_tep(data, ten):
    if ten.lower().endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        if len(doc) > TOI_DA_TRANG:
            raise HTTPException(400, f"PDF tối đa {TOI_DA_TRANG} trang")
        for pg in doc:
            pix = pg.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
            yield np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    else:
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise HTTPException(400, "Không đọc được ảnh")
        yield im


@app.post("/api/pdf-trang")
async def pdf_trang(tep: UploadFile = File(...)):
    """PDF -> ảnh từng trang (200 dpi, xám, JPEG) để trình duyệt hiển thị và cắt vùng như ảnh thường."""
    data = await tep.read()
    if len(data) > TOI_DA_MB * 2**20:
        raise HTTPException(413, f"Tệp tối đa {TOI_DA_MB} MB")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise HTTPException(400, "Không đọc được PDF")
    if len(doc) > TOI_DA_TRANG:
        raise HTTPException(400, f"PDF tối đa {TOI_DA_TRANG} trang")
    return {"trang": ["data:image/jpeg;base64," + base64.b64encode(
        pg.get_pixmap(dpi=200, colorspace=fitz.csGRAY).tobytes("jpg", jpg_quality=90)).decode() for pg in doc]}


@app.post("/api/nhan-dang")
async def nhan_dang(tep: UploadFile = File(...), che_do: str = Form("t"), mot_dong: bool = Form(False)):
    data = await tep.read()
    if len(data) > TOI_DA_MB * 2**20:
        raise HTTPException(413, f"Tệp tối đa {TOI_DA_MB} MB")
    if che_do not in ("a", "b", "c", "t"):  # a: CRNN; b: mô hình sửa lỗi, dùng nguyên đề xuất; c: sửa lỗi có kiểm soát; t: c + từ điển
        raise HTTPException(400, "che_do phải là a, b, c hoặc t")
    t0, trang = time.time(), []
    for gray in anh_tu_tep(data, tep.filename or ""):
        txt, tu = htr.nhan_dang(m, bkt, ch["h"], gray, doan=not mot_dong)
        kq = txt
        if che_do == "b" and txt:
            kq = llm_sua.sua_toan_bo(txt)
        elif che_do in ("c", "t") and tu:
            kq = llm_sua.sua_co_kiem_soat(tu, NGUONG, tu_dien=TU_DIEN if che_do == "t" else None)[0]
        trang.append({"crnn": txt, "ket_qua": kq,
                      "tu": [{"tu": w, "tin_cay": round(c, 3)} for w, c in tu]})
    return {"trang": trang, "nguong": TO_MAU, "giay": round(time.time() - t0, 2)}


@app.get("/")
def trang_chu():
    return FileResponse(Path(__file__).with_name("index.html"))
