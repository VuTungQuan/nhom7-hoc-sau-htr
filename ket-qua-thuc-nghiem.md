# Kết quả thực nghiệm (cập nhật 12/11/2026 – thay Qwen2.5-7B/Ollama bằng ProtonX legal-tc/transformers; beam chọn lại trên val)

Nguyên liệu cho Chương 4. Số liệu gốc trong `ket-qua/`: log huấn luyện, đầu ra từng dòng/đoạn của mọi cấu hình (`*_tc.json` = mô hình sửa lỗi ProtonX), thống kê `thong_ke_tc.txt` (ProtonX) và `thong_ke.txt` (Qwen, phụ lục) – đầu ra của `src/thong_ke.py`.
Nguyên tắc: tham số chọn trên **tập kiểm định (val)** = 25 người viết tách ngẫu nhiên (seed 0) khỏi train; **tập kiểm thử (test)** = 6 người viết 250–255 của UIT-HWDB.

> **Dữ liệu:** UIT-HWDB là **bộ ảnh chữ viết tay tổng hợp** tạo bằng phương pháp "Transferring" (Nguyen et al., 2022 – nguon/43), không phải ảnh quét giấy viết thật. "Người viết" = mã người viết trong bộ dữ liệu.

> **Mô hình hiệu chỉnh hậu kỳ:** `protonx-models/protonx-legal-tc` – mô hình **sửa lỗi văn bản tiếng Việt kiểu seq2seq** (T5ForConditionalGeneration, tinh chỉnh từ **ViT5-base**, **226 triệu tham số**), huấn luyện trên ~800.000 cặp câu sai–đúng (chủ yếu văn bản pháp lý và đầu ra OCR), tối ưu cho hậu xử lý OCR (khôi phục dấu, sửa tách từ), cam kết "sửa chứ không viết lại". **Đây không phải mô hình ngôn ngữ lớn (LLM) theo nghĩa thông thường** (hàng tỉ tham số, sinh văn bản theo chỉ dẫn) mà là mô hình ngôn ngữ tiền huấn luyện cỡ vừa tinh chỉnh cho tác vụ sửa lỗi. Giấy phép ProtonX v1.3-NC: chỉ dùng phi thương mại.

## 1. Môi trường
- Máy chủ: GPU NVIDIA RTX 3090 24 GB, 10 nhân CPU, RAM 23 GB, Ubuntu 22.04.
- PyTorch 2.5 (CUDA 12.4), OpenCV, albumentations 1.4.2, **Hugging Face transformers 5.17** (tokenizers 0.23, safetensors 0.8).
- Mô hình sửa lỗi: `protonx-models/protonx-legal-tc` v1.3.1, tải từ Hugging Face (chỉ `model.safetensors` 904 MB + tokenizer/config; không tải `training_args.bin` dạng pickle), SHA-256 `ed4fee22…3e47` khớp Hugging Face. Lưu tại `/mnt/data/hs-htr/hf/protonx-legal-tc`. Chạy cục bộ trên GPU, không gửi dữ liệu ra ngoài. **Không dùng Ollama.**
  - Tokenizer nạp bằng `PreTrainedTokenizerFast(tokenizer_file=…)` vì `AutoTokenizer` của transformers 5.x lỗi khi dựng Unigram từ `tokenizer.json` của mô hình này.
  - Giải mã: beam search **10 nhánh (chọn trên val, §3.3)**, tối đa 160 token; văn bản cắt khúc ≤ 30 từ (giới hạn 128–160 token của mô hình), chạy theo lô 32.
- Web: FastAPI + uvicorn (127.0.0.1), công khai qua Cloudflare Tunnel (`nhom7hocsau.cat.io.vn`), đăng nhập băm scrypt + phiên ký HMAC.

## 2. Mô hình CRNN và huấn luyện
- Kiến trúc: 7 tầng tích chập kiểu VGG (BatchNorm, ReLU, MaxPool giảm chiều cao 16 lần, chiều rộng 4 lần) → BiLSTM 2 lớp × 256 → tuyến tính → CTC. 13,24 triệu tham số.
- Bộ ký tự: 161 ký tự + blank, lập từ nhãn **toàn bộ tập train gốc (gồm cả 25 người viết val)**, đã loại ký tự nhiễu `ƒ`. Chỉ dùng tập ký tự; test không có ký tự nào ngoài bộ này.
- Đầu vào: ảnh dòng xám, cao 64 px, giữ tỉ lệ. Tăng cường: affine (nghiêng ±8°, co giãn 0,9–1,05, xoay ±2°), làm mờ, độ sáng/tương phản.
- Huấn luyện: AdamW (lr 1e-3, weight decay 1e-4), OneCycle, batch 16, 80 epoch (~31.900 bước), AMP fp16, cắt gradient 5; ~37 giây/epoch, ~50 phút. Chọn checkpoint có CER val nhỏ nhất.
- Dữ liệu mức dòng: 6.381 dòng train (224 người viết) / 647 dòng val (25 người viết) / 201 dòng test (6 người viết); bỏ 44 ảnh thiếu nhãn. Dữ liệu mức từ **chưa dùng**.

| Epoch | Loss | CER val | WER val | Lỗi chỉ do dấu (val) |
|---|---|---|---|---|
| 1 | 4,31 | 100 % | 100 % | – |
| 5 | 1,00 | 15,15 % | 45,22 % | 3,00 % |
| 10 | 0,63 | 9,36 % | 30,08 % | 1,69 % |
| 20 | 0,49 | 5,24 % | 16,90 % | 1,00 % |
| 40 | 0,36 | 3,63 % | 11,57 % | 0,71 % |
| 60 | 0,30 | 3,12 % | 9,89 % | 0,59 % |
| **78 (tốt nhất)** | 0,30 | **3,02 %** | **9,41 %** | **0,57 %** |
| 80 | 0,29 | 3,04 % | 9,46 % | 0,58 % |

(Giải mã trong vòng huấn luyện, chưa qua tiền xử lý ảnh chụp. Với quy trình suy luận cuối, CER val 647 dòng = 3,10 %.)
*Lỗi chỉ do dấu* = CER − CER sau khi bỏ dấu ở cả dự đoán và nhãn.

## 3. Lựa chọn trên tập kiểm định

### 3.1 Tách dòng (mức đoạn, 108 đoạn val, cấu hình a)
| Cách tách | CER | Số đoạn hỏng (CER > 50 %) |
|---|---|---|
| Cắt ngang tại chỗ trũng, lề 0 / 0,1 / 0,2 / 0,3 × khoảng dòng | 19,9 / 17,6 / 18,9 / 22,2 % | – |
| Ước lượng khoảng dòng bằng tự tương quan + gán thành phần liên thông | 16,6–18,0 % | 12 |
| **Đỉnh nổi bật (prominence 10 %) + gán thành phần liên thông** | **12,0 %** | 1 |
| + cân bằng nền, kéo tương phản, tự co giãn, xoá dòng kẻ, bỏ thành phần xa (bản cuối) | **11,87 %** | – |

Ngưỡng bỏ thành phần xa dòng: 0,6 / 0,8 / **1,0** × khoảng dòng → CER 12,62 / 11,94 / **11,87 %**.

### 3.2 Hiệu chỉnh hậu kỳ bằng ProtonX: cấu hình và ngưỡng (TOÀN BỘ val: 647 dòng, 108 đoạn, 25 người viết)
Các cấu hình (cùng một đầu ra CRNN; mô hình sửa lỗi chạy **một lần** trên mỗi dòng/đoạn, các ngưỡng chỉ khác ở bước áp đề xuất):
- **a** – chỉ CRNN.
- **b** – dùng nguyên văn bản mô hình đề xuất.
- **c** – căn chỉnh từ (difflib) giữa đầu vào và đề xuất; **chỉ nhận thay thế 1–1** (bỏ qua thêm/bớt/gộp/tách từ), tại từ CRNN có độ tin cậy < ngưỡng, khác không quá 50 % độ dài từ cũ; giữ nguyên dấu câu bao quanh.
- **t** – c + bản sửa phải có trong **từ điển âm tiết** lập từ nhãn tập train (trừ người viết val), 2.829 âm tiết.
- Ngưỡng **1,01** (> 1) = mọi từ đều đủ điều kiện → **bỏ lọc theo độ tin cậy**, chỉ còn ràng buộc thay 1–1 + khoảng cách sửa (+ từ điển với t).

Mức dòng (647 dòng):
| Cấu hình | CER | WER | Δ CER so với a [KTC 95 % theo người viết] | Người viết xấu/tốt/như cũ | Từ đủ điều kiện / được thay |
|---|---|---|---|---|---|
| a | 3,10 % [2,15; 4,35] | 9,75 % | – | – | – |
| b | 3,23 % | 8,29 % | +0,13 [−0,04; +0,32] | 12/12/1 | – |
| c @0,3 | 3,10 % | 9,73 % | −0,00 [−0,01; 0,00] | 1/2/22 | 34 / 13 |
| c @0,5 | 3,05 % | 9,39 % | −0,05 [−0,09; −0,01] | 4/13/8 | 265 / 91 |
| c @0,7 | 3,00 % | 8,91 % | −0,10 [−0,16; −0,04] | 5/14/6 | 1.295 / 211 |
| **c @0,9** | **2,93 %** | 8,33 % | **−0,17** [−0,25; −0,09] | 5/19/1 | 2.755 / 328 |
| c @1,01 | 2,96 % | 8,21 % | −0,14 [−0,25; −0,03] | 6/18/1 | 9.850 / 450 |
| t @0,5 | 3,06 % | 9,39 % | −0,05 [−0,08; −0,01] | 4/13/8 | 265 / 86 |
| t @0,7 | 3,01 % | 8,91 % | −0,10 [−0,16; −0,04] | 5/14/6 | 1.295 / 204 |
| t @0,9 | 2,94 % | 8,35 % | −0,17 [−0,25; −0,09] | 5/19/1 | 2.755 / 317 |
| t @1,01 | 2,95 % | **8,19 %** (thấp nhất) | −0,15 [−0,26; −0,04] | 6/18/1 | 9.850 / 433 |

Mức đoạn (108 đoạn, đầu cuối gồm tách dòng):
| Cấu hình | CER | WER | Δ CER so với a [KTC 95 % theo người viết] | Người viết xấu/tốt/như cũ |
|---|---|---|---|---|
| a | 11,87 % [7,99; 16,75] | 27,74 % | – | – |
| b | 10,53 % | 19,54 % | −1,35 [−1,68; −1,01] | 2/23/0 |
| c / t @0,3 | 11,86 % | 27,66 % | −0,01 [−0,04; +0,01] | 7/10/8 – 6/9/10 |
| c @0,5 / t @0,5 | 11,74 % | 26,91 % | −0,13 [−0,20; −0,06] | 3/17/5 |
| c @0,7 / t @0,7 | 11,41 / 11,38 % | 24,90 % | −0,47 / −0,49 | 0/25/0 |
| c @0,9 / t @0,9 | 10,98 / 10,96 % | 22,76 / 22,82 % | −0,90 / −0,92 [−1,09; −0,75] | 0/25/0 |
| c @1,01 | 10,39 % | 19,98 % | −1,48 [−1,81; −1,18] | 0/25/0 |
| **t @1,01** | **10,36 %** | 20,01 % | **−1,52** [−1,82; −1,23] | **0/25/0** |

**Quyết định (chỉ dựa trên val):** cấu hình chính = **t @1,01** – tốt nhất ở mức đoạn (trường hợp dùng chính của web app: trang/đoạn) và chỉ kém cấu hình tốt nhất mức dòng (c @0,9) 0,02 điểm. Báo cáo kèm **t @0,9** (tốt nhất mức dòng). Nhận xét: lọc theo độ tin cậy không giúp ở mức đoạn – ràng buộc hữu ích là "chỉ thay 1–1, khác ít, có trong từ điển"; ngưỡng tin cậy càng thấp thì mô hình càng ít được can thiệp và lợi ích càng nhỏ. Ở mức dòng, WER thấp nhất thuộc t @1,01 (8,19 %) và c @1,01 (8,21 %), không phải b (8,29 %); b có CER tăng nhẹ (không có ý nghĩa thống kê) vì còn thêm/bớt/viết lại từ. (Bảng §3.2 chạy với beam 5; §3.3 cho thấy beam 10 cho số gần như trùng.)

### 3.3 Số nhánh beam search (toàn bộ val, chọn SAU khi đã thử trên 2 dòng test – xem §3.4)
| Beam | t @1,01 mức dòng: CER / WER | t @1,01 mức đoạn: CER / WER | c @1,01 mức đoạn | b mức đoạn |
|---|---|---|---|---|
| 1 (tham lam) | 2,94 % / 8,17 % | 10,39 % / 20,13 % | 10,42 % | 11,21 % |
| 5 | 2,95 % / 8,19 % | 10,36 % / 20,01 % | 10,39 % | 10,53 % |
| **10** (khuyến nghị trong model card) | 2,95 % / 8,19 % | **10,35 %** / 19,98 % | 10,38 % | 10,51 % |

Chênh giữa các beam ≤ 0,04 điểm CER ở cấu hình t → thực chất không khác biệt. Theo cùng quy tắc chọn ngưỡng (CER mức đoạn val thấp nhất), chọn **beam 10**; tập test chạy lại **một lần** với beam 10 (kết quả chính §4). Log: `ket-qua/chon_beam_val.log`.

### 3.4 Khai báo về quy trình (trung thực phương pháp)
- **Thứ tự thực nghiệm:** (1) thử Qwen2.5-7B (đối chứng LLM, §9) – không cải thiện; (2) **sau khi thấy kết quả Qwen**, nhóm chọn thêm mô hình sửa lỗi ProtonX. Việc chọn bộ sửa lỗi thứ hai là quyết định hậu nghiệm; so sánh Qwen–ProtonX là **khám phá**.
- **Đã nhìn 2 dòng test khi thử ProtonX lần đầu:** khi chạy thử mô hình, nhóm dùng 6 câu mẫu, trong đó **2 dòng thuộc tập test** (người viết 251 dòng 25 "Táp trung…", người viết 252 dòng 3 "dựng chỉnh đón Đảng…"; còn lại là 3 dòng val và 1 câu ví dụ của model card), và thấy beam 5 sửa đúng "chỉnh đốn" còn tham lam ra "chính đơn" → chọn beam 5 ban đầu. Lựa chọn này đã được **thay bằng lựa chọn trên val** (§3.3: beam 10). Ngưỡng và cấu hình (t @1,01) chọn hoàn toàn trên val. Hàm tự kiểm tra trong `llm_sua.py` đã đổi sang câu tự đặt.
- **Đa so sánh:** tập test đã dùng cho hai bộ sửa lỗi (Qwen, ProtonX) → báo cáo thêm KTC hiệu chỉnh Bonferroni (97,5 %) và p kiểm định dấu ×2 (§4).

## 4. Kết quả trên tập kiểm thử

**Phương pháp thống kê** (`src/thong_ke.py`):
- KTC 95 % bằng **bootstrap theo cụm người viết** (5.000 lần, seed 0, lấy mẫu lại người viết, giữ toàn bộ dòng/đoạn của họ; chênh lệch tính ghép cặp trên cùng mẫu). Với 6 cụm, bootstrap có thể đánh giá thấp độ bất định → kèm **đếm số người viết xấu đi/tốt lên/như cũ** (kiểm định dấu: 6/6 cùng hướng ⇔ p hai phía ≈ 0,031; 5/5 ⇔ p ≈ 0,063).
- **CER bỏ qua định dạng**: chuẩn hoá Unicode NFC, thống nhất mọi kiểu dấu nháy, chữ thường, bỏ khoảng trắng quanh dấu câu.
- Tập test cho ProtonX: chạy lần đầu với beam 5 (`*_tc_beam5.json`, `thong_ke_tc_beam5.txt`); sau khi chọn lại beam trên val, chạy **một lần** với beam 10 (kết quả chính dưới đây: `*_tc_beam10.json`, `thong_ke_tc.txt`). Hai lượt gần như trùng (mức dòng trùng hoàn toàn; mức đoạn t @1,01 WER 29,47 → 29,54 %).
- **Hiệu chỉnh đa so sánh** (2 bộ sửa lỗi thử trên test): KTC 97,5 % theo người viết cho t @1,01 – mức dòng [−0,84; −0,30], mức đoạn [−2,35; −0,73] – vẫn không chứa 0 (`ket-qua/thong_ke_tc_bonferroni2.txt`). Tập test thực tế đã chạy 4 lượt (Qwen @0,5 và @0,3; ProtonX beam 5 và beam 10); hiệu chỉnh ×4 (KTC 98,75 %, `thong_ke_tc_bonferroni4.txt`): mức dòng [−0,86; −0,25], mức đoạn [−2,44; −0,70] – vẫn không chứa 0. Kiểm định dấu 6/6: p = 0,031; ×2 = 0,062 (với 6 người viết, p nhỏ nhất có thể là 0,031, nên kiểm định dấu không thể đạt ngưỡng sau hiệu chỉnh) → bằng chứng chính là KTC bootstrap và sự nhất quán với val (25/25 người viết tốt lên ở mức đoạn).

### 4.1 Mức dòng (201 dòng, 6 người viết)
| Cấu hình | CER [KTC người viết] | WER | Lỗi chỉ do dấu | CER bỏ định dạng | Δ CER vs a [KTC] | Δ bỏ định dạng [KTC] | Người viết xấu/tốt/như cũ | Dòng xấu đi: chỉ định dạng / nội dung | Từ đủ ĐK / thay | Giây/dòng* |
|---|---|---|---|---|---|---|---|---|---|---|
| a | 5,50 % [1,77; 11,39] | 15,57 % | 0,83 % | 5,42 % | – | – | – | – | – | 0,04 |
| b | 5,32 % [1,25; 11,69] | **11,88 %** | 0,67 % | 5,23 % | −0,18 [−0,57; +0,28] | −0,18 [−0,59; +0,31] | 1/4/1 | 0 / 34 | – | 0,13 |
| c @0,9 | 5,06 % [1,45; 10,79] | 12,74 % | 0,73 % | 4,96 % | −0,44 [−0,63; −0,24] | −0,46 [−0,65; −0,25] | 0/6/0 | 0 / 18 | 935 / 184 | 0,12 |
| c @1,01 | 4,94 % [1,31; 10,68] | 11,94 % | 0,72 % | 4,83 % | −0,56 [−0,76; −0,34] | −0,58 [−0,78; −0,35] | 0/6/0 | 0 / 20 | 2.856 / 230 | 0,13 |
| t @0,9 | 5,03 % [1,40; 10,75] | 12,71 % | 0,71 % | 4,93 % | −0,47 [−0,68; −0,25] | −0,49 [−0,69; −0,26] | 0/6/0 | 0 / 15 | 935 / 175 | 0,12 |
| **t @1,01** | **4,90 %** [1,26; 10,62] | 11,91 % | 0,69 % | **4,79 %** | **−0,60** [−0,82; −0,35] | −0,62 [−0,84; −0,36] | **0/6/0** | 0 / 17 | 2.856 / 220 | 0,13 |

### 4.2 Mức đoạn – đầu cuối, gồm tách dòng (31 đoạn, 6 người viết)
| Cấu hình | CER [KTC người viết] | WER | Lỗi chỉ do dấu | CER bỏ định dạng | Δ CER vs a [KTC] | Δ bỏ định dạng [KTC] | Người viết xấu/tốt/như cũ | Đoạn xấu đi | Từ đủ ĐK / thay | Giây/đoạn* |
|---|---|---|---|---|---|---|---|---|---|---|
| a | 16,84 % [8,34; 26,03] | 37,86 % | 3,19 % | 17,04 % | – | – | – | – | – | 0,37 |
| b | 15,24 % [6,85; 25,18] | **28,51 %** | 1,72 % | 15,39 % | −1,60 [−2,86; −0,47] | −1,65 [−2,92; −0,50] | 1/5/0 | 6 | – | 0,56 |
| c @0,9 / t @0,9 | 15,84 / 15,85 % [7,44; 25,45] | 32,17 % | 2,37 % | 16,05 / 16,04 % | −1,00 / −0,99 [−1,56; −0,51] | −0,99 / −1,00 | 0/6/0 | 5 | 1.206 / 396 – 374 | 0,56 |
| c @1,01 | 15,36 % [7,06; 24,96] | 29,54 % | 1,87 % | 15,56 % | −1,48 [−2,27; −0,76] | −1,48 [−2,28; −0,76] | 0/6/0 | 5 | 2.844 / 544 | 0,56 |
| **t @1,01** | **15,36 %** [7,06; 24,95] | 29,54 % | 1,86 % | **15,55 %** | **−1,48** [−2,26; −0,78] | −1,49 [−2,27; −0,79] | **0/6/0** | 5 | 2.844 / 519 | 0,56 |

\* Đo bằng `danh_gia.py` ngoại tuyến trên máy chủ (mô hình sửa lỗi chạy lô 32, beam 10); không qua web. Qua web (đo một lần, đoạn 250/1, 10 dòng): chế độ a 1,09 s, chế độ t 1,82 s – gồm tải tệp, tiền xử lý, tách dòng.

### 4.3 Nhận xét chính
1. **CRNN**: CER 5,50 % / WER 15,6 % trên 6 người viết test (val: 3,10 %); KTC theo người viết rất rộng [1,8; 11,4] → độ chính xác thay đổi mạnh theo người viết.
2. **Hiệu chỉnh hậu kỳ bằng ProtonX có kiểm soát (t @1,01) cải thiện có ý nghĩa thống kê ở cả hai mức**: mức dòng CER −0,60 điểm [−0,82; −0,35] (5,50 → 4,90 %, giảm ~11 % tương đối), WER 15,6 → 11,9 %; mức đoạn CER −1,48 điểm [−2,26; −0,78] (16,84 → 15,36 %), WER 37,9 → 29,5 %. **Cả 6/6 người viết đều tốt lên** ở cả hai mức (kiểm định dấu p ≈ 0,031; 0,062 sau Bonferroni ×2); KTC 97,5 % (Bonferroni ×2) và 98,75 % (×4) vẫn không chứa 0; trên val (nơi chọn cấu hình, không độc lập) mọi ngưỡng ≥ 0,7 đều cho 25/25 người viết tốt lên ở mức đoạn. Lưu ý: ProtonX được chọn sau khi thấy kết quả Qwen (§3.4) → đây là bằng chứng cho cấu hình đã chọn trên val, không phải kiểm định xác nhận một giả thuyết đặt trước. Cải thiện **không đến từ định dạng** (CER bỏ định dạng giảm tương đương).
3. **Phần lớn cải thiện nằm ở WER** (sửa đúng cả từ: "phả→phải", "lình→lĩnh", "danh→doanh"), CER giảm ít hơn vì mỗi lần sửa thường chỉ đổi 1–2 ký tự; lỗi chỉ do dấu giảm (mức đoạn 3,19 → 1,86 %).
4. **Dùng nguyên đề xuất (b)**: tốt ở mức đoạn (−1,60) nhưng ở mức dòng không có ý nghĩa thống kê (−0,18 [−0,57; +0,28]) và có trường hợp mô hình **viết lại/thêm từ** (ví dụ "dỷ lụng phền phác" → "nhiêu vất vả phiền toái", thêm "Trang") – trái với cam kết "sửa chứ không viết lại" khi đầu vào nhiễu nặng. Ràng buộc thay 1–1 + khoảng cách sửa + từ điển của t loại bỏ các trường hợp này (mức dòng: số dòng xấu đi 34 → 17).
5. **Lọc theo độ tin cậy CTC không cần thiết** với mô hình này: ngưỡng thấp làm giảm lợi ích; ràng buộc cấu trúc (1–1, khác ít, từ điển) mới là phần "kiểm soát" có tác dụng.
6. **Tách dòng vẫn là nút thắt**: CER đầu cuối mức đoạn (15,4 %) gấp ~3 lần mức dòng (4,9 %). → Đã cải tiến ở §6c: mức đoạn t @1,01 còn 10,20 %.
7. **Thời gian**: thêm ~0,09 s/dòng, ~0,19 s/đoạn (ngoại tuyến, chạy lô, beam 10) – chấp nhận được cho web.

## 5. Ví dụ định tính (test, mức dòng – t @1,01 trừ khi ghi khác)
**Sửa đúng:**
- GT *… không phải là người trong lĩnh vực chuyên môn …* — CRNN *không **phả** là **ngàn** trong **lình** vực chuyên **mân*** → t *không **phải** là **ngành** trong **lĩnh** vực chuyên **môn*** (Δ −4).
- GT *thực hiện việc kê khai … kinh doanh của cán bộ* — CRNN ***cặc** hiện việc **vê** khai … kinh **danh** của cán **lộ*** → t ***các** hiện việc **kê** khai … kinh **doanh** của cán **bộ*** (Δ −3).
- GT *… thất nghiệp … như thế …* — CRNN ***hrất** nghiệp … **nhu thê*** → t ***thất** nghiệp … **như thế*** (Δ −4).
- Thử ban đầu (**2 dòng test**, xem §3.4): *Táp trung* → *Tập trung*; *chỉnh **đón** … (**iần** 2) … **táp tụo*** → *chỉnh **đốn** … (**lần** 2) … **tập trung***. Thử ban đầu (val): *phân **tắng*** → *phân **thắng***. Qua web (đoạn 250/1): *có chế → cơ chế*, *đâu tư → đầu tư*, *hoạc → hoặc*.

**Sửa sai (từ có thật nhưng sai ngữ cảnh – lớp từ điển không chặn được):**
- *phản **ứu*** (GT *ứng*) → *phản **cứu***; *dây chuyền **khá*** → ***khác***.
- *chũng cản* (GT *dũng cảm*) → *chung cần*; *kiệp **rà ây*** (GT *hiệp sĩ ấy*) → *hiệp **trà cây***.
- Thử ban đầu (val): *ắt phút* (GT *ít*) → *bắt phút*; *dân **đã** gà* (GT *đá gà*) → *dân đã **già***.

**b – dùng nguyên đề xuất, lỗi nặng nhất (viết lại/thêm từ):**
- CRNN *… bao **dỷ lụng phền phác*** (GT *hệ lụy phiền phức*) → b *… bao **nhiêu vất vả phiền toái*** (Δ +11).
- CRNN *… đến **qửi** tặng anh **Truang*** (GT *gửi … Trương*) → b *… đến **viếng** tặng anh **Trương Trang*** (Δ +7).

## 6. Ảnh ngoài miền dữ liệu (qua web app, định tính, không có nhãn chuẩn; đo ở mức CRNN, trước khi đổi mô hình sửa lỗi)
| Ảnh | Vấn đề ban đầu | Sau tiền xử lý |
|---|---|---|
| Ảnh chụp điện thoại lời giải toán (nền bàn cam, bóng đổ, bút bi xanh; ảnh thử cao 416 px cắt từ ảnh chụp màn hình) | Cả trang tách thành 1 dòng → "Phọc4cuợ4." | Tách được các dòng, đọc được phần lớn tiếng Việt: "Lời giải :… Để chứng minh … ta lấy … tùy ý … Theo định nghĩa … tồn tại … (ĐPCM)". Ký hiệu ∈, ⊂, ƒ không có trong bộ ký tự → đọc thành C, 8, 1 |
| Vở kẻ dòng, **phông chữ mô phỏng chữ viết tay** | CER 21,2 % (tự đối chiếu 66 ký tự) – 11/14 lỗi là "...." sinh từ dòng kẻ và chữ đầu trang | **CER 12,1 %**; còn lỗi kiểu chữ: "châu"/"cháu", "Pháu"/"Cháu", "đạnyg"/"dạy" |

Tiền xử lý cho ảnh thực tế (không làm tăng CER val mức đoạn: 11,88 → 11,87 %): cân bằng nền bằng phép giãn; bỏ thành phần chạm mép/dài quá nửa ảnh; xoá dòng kẻ (chỉ khi có ≥ 3 đường ngang dài ≥ W/4); tự co giãn để khoảng dòng ≈ 77 px; kéo giãn tương phản từng dòng; bỏ thành phần cách mọi dòng > 1 khoảng dòng; bỏ dòng ít mực/chỉ có dấu câu.

## 6b. Thí nghiệm phụ: huấn luyện dài hơn (20.000 bước, batch 64)
Cùng kiến trúc, lr tối đa 1e-3 (OneCycle), **batch 64 × 202 epoch ≈ 20.000 bước** (lượng dữ liệu đi qua mô hình gấp ~2,5 lần lượt chính). ~29–31 s/epoch, ~1 giờ 40 phút, VRAM 13,3 GB. Log: `ket-qua/log_20k.csv`.

| Epoch (lượt 20k) | 10 | 20 | 40 | 80 | 120 | 150 | 180 | **199 (tốt nhất)** | 202 |
|---|---|---|---|---|---|---|---|---|---|
| CER val (trong vòng huấn luyện) | 13,9 % | 6,3 % | 4,7 % | 3,5 % | 3,2 % | 3,0 % | 2,8 % | **2,75 %** | 2,76 % |

So sánh trên toàn bộ val bằng quy trình suy luận cuối (cấu hình a), ghép cặp, KTC theo người viết:
| | Mô hình chính | Mô hình 20k bước | Mới − cũ [KTC 95 %] | Người viết mới tốt hơn / kém hơn / như nhau |
|---|---|---|---|---|
| Mức dòng (647 dòng) | 3,10 % | 2,94 % | −0,16 [−0,33; +0,03] | 15 / 9 / 1 |
| Mức đoạn (108 đoạn) | **11,87 %** | 12,16 % | +0,28 [−0,07; +0,79] | 10 / 12 / 3 |

→ Không có ý nghĩa thống kê, ngược chiều → mô hình gần bão hoà; **giữ mô hình chính**. Hiệu năng: nút thắt là tính toán GPU (batch 16: 87 ms/bước GPU, 21 ms nạp dữ liệu, 2,2 GB VRAM; batch 64: 235 / 58 ms, 8 GB) do ảnh dòng rất rộng (900–2.400 px), phần đệm trong lô và LSTM tuần tự.

## 6c. Cải tiến tách dòng sau phân tích lỗi (13/11/2026)
**Khai báo thứ tự:** ý định cải tiến tách dòng xuất phát từ phân tích lỗi **trên tập test** (CER mức dòng vs mức đoạn theo người viết: bỏ người viết 253, CER dòng 2,09 % nhưng CER đoạn 11,68 %). Mọi lựa chọn phương pháp và tham số dưới đây làm **chỉ trên val** (108 đoạn, cấu hình a); test chạy **một lần** sau khi chốt (lượt chạy test thứ 5 của đề tài). Script: `src/thu_tach_dong.py`, log `ket-qua/thu_tach_dong_val*.log`, so sánh `src/so_tach_dong.py`.

**Chẩn đoán trên val:**
- Mức trần (nhận dạng ảnh dòng cắt sẵn của UIT-HWDB rồi nối, = tách dòng hoàn hảo): CER **2,99 %**, so với 11,87 % của tách dòng cũ → ~9 điểm CER do tách dòng.
- Tách cũ đúng số dòng ở 81/108 đoạn (thiếu 15, thừa 12); ngay cả đoạn đúng số dòng CER vẫn 8,8 % (trần 2,7 %) → lỗi chính là **dòng nghiêng/cong**: tâm dòng tính trên cả chiều ngang lệch ở hai đầu dòng, chữ cuối dòng bị gán sang dòng kề.
- Ảnh dòng và ảnh đoạn của UIT-HWDB là hai bản sinh riêng (khớp mẫu thất bại) → không có vị trí dòng thật, không huấn luyện được bộ phát hiện dòng có giám sát từ dữ liệu này.

| Cách tách (val, 108 đoạn, a) | CER | Đúng số dòng | Người viết tốt lên/xấu đi so với cũ |
|---|---|---|---|
| Cũ: đỉnh chiếu ngang + gán thành phần theo tâm | 11,87 % | 81 | – |
| Cắt thành phần vắt qua hai dòng (thẳng / đường uốn) | 12,17 / 11,70 % | 81 | 9/15 · 13/11 |
| Gộp đỉnh gần (< 0,6 khoảng dòng) | 11,54 % | 84 | 5/1 |
| Bám tâm dòng theo dải dọc 250 / 150 / **100** / 75 px | 8,76 / 8,27 / **7,92** / 9,05 % | 87 / 87 / 87 / 83 | 21/4 · 22/3 · 22/3 · 19/5 |
| Dải 100 px, cửa sổ ±0,25 / ±0,5 khoảng dòng (mặc định ±0,35) | 8,39 / 10,59 % | – | – |
| **Dải 100 px + thêm đỉnh thấp ở biên (dòng ngắn đầu/cuối)** | **7,58 %** | 87 | **22/2** |
| + chèn đỉnh vào khoảng trống > 1,6 khoảng dòng | 7,53 % | 85 | 22/2 (không giữ: lợi 0,05 điểm, thêm quy tắc) |

**Chốt:** dải 100 px + đỉnh biên (đưa vào `htr.tach_dong`; bản cũ lưu ở máy chủ `src/htr_tach_cu.py.bak`). Kiểm lại bằng `danh_gia.py` trên val: a 7,58 %, **t @1,01 6,17 %** (cũ 10,36 %).

**Test (31 đoạn, 6 người viết, chạy một lần; `danh_gia_doan_test_abct_tc_tachmoi.json`, `thong_ke_tc_tachmoi.txt`):**
| Cấu hình | CER tách cũ | CER tách mới | Mới − cũ [KTC 95 % người viết] | Người viết tốt lên/xấu đi | WER mới |
|---|---|---|---|---|---|
| a | 16,84 % | **11,84 %** | **−5,00** [−8,70; −2,18] | 5/1 | 30,37 % |
| t @1,01 | 15,36 % | **10,20 %** | **−5,16** [−8,47; −2,58] | 5/1 | 22,02 % |

- Theo người viết (a): 250 6,3→3,9; 251 21,1→8,2; 252 9,9→6,3; **253 34,4→31,6** (lỗi nhận dạng, CER dòng 17,1 %); 254 9,5→10,0 (xấu đi nhẹ); 255 10,4→3,1.
- Với tách mới, t @1,01 so với a: −1,64 [−2,49; −0,92], 6/6 người viết tốt lên (cũ: −1,48) → kết luận về bước hậu kỳ không đổi.
- Khoảng cách với mức trần vẫn còn (val 7,58 % vs 2,99 %); người viết 253 cần cải thiện mô hình nhận dạng, không phải tách dòng.

## 7. Ghi chú cài đặt (đối chiếu mã)
- `src/llm_sua.py`: `nap()` (nạp mô hình một lần), `de_xuat()` (cắt khúc 30 từ, lô 32, beam 10), `kiem_soat()` (căn chỉnh difflib, thay 1–1, khoảng cách ≤ 50 %, từ điển), có tự kiểm tra cơ chế kiểm soát không cần mô hình.
- `src/danh_gia.py`: chạy mô hình sửa lỗi **một lần** trên toàn bộ đầu ra CRNN rồi áp các ngưỡng; tệp kết quả hậu tố `_tc`.
- Web: mặc định **"CRNN + sửa lỗi có kiểm soát"** = t @1,01; "Chỉ CRNN" = a; "Sửa lỗi toàn bộ" = b. Tô màu từ kém tin cậy dùng ngưỡng riêng 0,5. Nạp sẵn mô hình lúc khởi động. Chân trang có ghi chú quyền riêng tư và ghi nguồn "Built with ProtonX Text Correction Model" (theo mục 5.2c và 7.3 của giấy phép).
- Tệp tải lên đọc vào bộ nhớ; Starlette có thể ghi tạm ra đĩa tệp > ~1 MB khi nhận (tệp tạm của thư viện). Ứng dụng không lưu ảnh/kết quả.
- Tài khoản đọc từ tệp khi khởi động → thêm/đổi mật khẩu cần khởi động lại. Trang tải phông Be Vietnam Pro từ Google Fonts.

## 8. Giới hạn
- Dữ liệu huấn luyện/đánh giá là **ảnh tổng hợp** (UIT-HWDB), văn bản phổ thông, 161 ký tự (không ký hiệu toán); giấy viết thật mới thử định tính 2 ảnh.
- Tập test chỉ 6 người viết (31 đoạn) → KTC cho CER tuyệt đối rất rộng; bootstrap 6 cụm có thể đánh giá thấp bất định.
- Mô hình sửa lỗi huấn luyện chủ yếu trên **văn bản pháp lý**, lệch miền với văn bản phổ thông của UIT-HWDB (vẫn có lợi); là **ViT5-base 226 triệu tham số, không phải LLM**; giấy phép **phi thương mại**.
- Lớp kiểm soát không chặn được thay bằng **từ có thật nhưng sai ngữ cảnh** ("ứu→cứu", "khá→khác").
- Tách dòng dựa trên chiếu ngang: giả định 1 cột, dòng gần nằm ngang.

## 9. Phụ lục – thí nghiệm trước với Qwen2.5-7B-Instruct qua Ollama (đã thay thế, không còn triển khai)
Cấu hình: LLM sinh văn bản theo chỉ dẫn (prompt tiếng Việt), Ollama 0.15.6, nhiệt độ 0; b = LLM viết lại cả dòng; c/t = chỉ hỏi LLM các từ kém tin cậy (trả JSON), cùng ràng buộc khoảng cách sửa (+ từ điển). Thống kê đầy đủ: `ket-qua/thong_ke.txt`.
- Val (toàn bộ): không cấu hình nào tốt hơn a; ngưỡng ít hại nhất 0,3 (mức dòng +0,00–0,01; mức đoạn +0,00).
- Test mức dòng: b **+8,51** điểm CER [6,59; 10,39], xấu đi ở 6/6 người viết (bỏ định dạng vẫn +7,64; 127/149 dòng xấu đi do nội dung); t @0,3 −0,02 [−0,08; +0,03] (gần như không can thiệp: 7 từ/201 dòng); t @0,5 +0,10 [+0,03; +0,17].
- Test mức đoạn: b +5,21 [1,87; 8,33]; c/t @0,3 −0,02 (không thay từ nào).
- Lỗi điển hình của b: **dịch sang tiếng Trung** ("H得了。从小时候他就梦想成为足球运动员…"), **chép lại chỉ dẫn trong prompt** ("Hãy sửa lỗi nhận dạng. Giữ nguyên nội dung…"), đổi "Đảng"→"đồng", "sách"→"sắc".
- **So sánh**: cùng đầu ra CRNN và cùng ý tưởng kiểm soát, LLM tổng quát 7B **gây hại hoặc vô ích**, còn mô hình chuyên sửa lỗi tiếng Việt 226M **cải thiện có ý nghĩa thống kê** → với hậu xử lý OCR tiếng Việt, mô hình nhỏ chuyên biệt phù hợp hơn LLM tổng quát chạy cục bộ.
