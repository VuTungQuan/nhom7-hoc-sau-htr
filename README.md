# Nhận dạng chữ viết tay tiếng Việt: CRNN + sửa lỗi có kiểm soát

Mã nguồn tiểu luận môn Học sâu (Nhóm 7, lớp 20261IT7332003): *Ứng dụng CNN và mô hình ngôn ngữ lớn trong nhận diện văn bản viết tay tiếng Việt*.

- **Nhận dạng:** CRNN (7 tầng tích chập kiểu VGG + BiLSTM 2×256 + CTC, 13,24 triệu tham số), ảnh dòng xám cao 64 px.
- **Tách dòng:** đỉnh hồ sơ chiếu ngang + gán thành phần liên thông, kèm tiền xử lý ảnh chụp.
- **Hậu kỳ:** [`protonx-models/protonx-legal-tc`](https://huggingface.co/protonx-models/protonx-legal-tc) qua Hugging Face transformers, chỉ nhận thay thế 1–1, khoảng cách sửa ≤ 50%, từ điển âm tiết. Qwen2.5-7B (Ollama) chỉ là đối chứng (`llm_sua_qwen.py`).
- **Web:** FastAPI, đăng nhập scrypt + phiên HMAC, ảnh/PDF/cắt vùng/camera điện thoại.

Kết quả chi tiết: [`ket-qua-thuc-nghiem.md`](ket-qua-thuc-nghiem.md). Trên 6 người viết test của UIT-HWDB, CER mức dòng 5,50% → 4,90%; mức đoạn 16,84% → 10,20% (tách dòng bám tâm theo dải + sửa lỗi, xem §6c).

## Cấu trúc

| Tệp | Nội dung |
|---|---|
| `src/htr.py` | Dữ liệu, mô hình CRNN, tiền xử lý, tách dòng, giải mã CTC, độ đo |
| `src/train.py` | Huấn luyện |
| `src/llm_sua.py` | Sửa lỗi bằng ProtonX + lớp kiểm soát |
| `src/danh_gia.py`, `src/thong_ke.py` | Đánh giá các cấu hình a/b/c/t, bootstrap theo người viết |
| `src/thu_tach_dong.py`, `src/so_tach_dong.py`, `src/chan_doan_dinh.py` | Thử nghiệm, so sánh và chẩn đoán các cách tách dòng trên val |
| `src/app.py`, `src/index.html`, `src/dang-nhap.html`, `src/tai_khoan.py` | Ứng dụng web |
| `src/ve_hinh*.py` | Vẽ hình cho báo cáo |
| `ket-qua/` | Log huấn luyện, đầu ra từng dòng/đoạn, thống kê |

## Chạy

Dữ liệu [UIT-HWDB](https://doi.org/10.1109/RIVF55975.2022.10013898) và trọng số **không** có trong repo.

```bash
cd src
python train.py --goc <thư mục UIT_HWDB_line> --ra ../runs/crnn_line   # 80 epoch, batch 16
python tai_khoan.py ../tai_khoan.json admin   # nhập mật khẩu qua stdin
RUN=../runs/crnn_line uvicorn app:app --host 127.0.0.1 --port 8765
```

Mô hình ProtonX dùng theo giấy phép **ProtonX v1.3-NC (phi thương mại)**.
