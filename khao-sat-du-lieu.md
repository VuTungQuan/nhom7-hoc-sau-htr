# Khảo sát dữ liệu UIT-HWDB (12/11/2026)

Nguồn: 3 file zip trong `code/` — `UIT_HWDB_word.zip`, `UIT_HWDB_line.zip`, `UIT_HWDB_paragraph.zip` (tổng 1,9 GB). Bộ dữ liệu công khai UIT-HWDB (Nguyen và cộng sự, 2022 — `nguon/43`).
Script: `scripts/khao_sat_du_lieu.py` (đọc thẳng zip, không giải nén).

## Cấu trúc
`UIT_HWDB_<mức>/{train_data,test_data}/<mã người viết>/<n>.jpg` + `label.json` (`{"1.jpg": "nhãn", ...}`), nhãn Unicode NFC.

## Số liệu

| Mức | Tập | Người viết | Ảnh | Ký tự TB/nhãn | Từ TB/nhãn | Kích thước ảnh (trung vị, w×h) |
|---|---|---|---|---|---|---|
| Từ | train | 249 | 107.864 (257 ảnh thiếu nhãn) | 3,3 | 1 | 128×106 |
| Từ | test | 6 | 2.881 | 3,4 | 1 | 128×85 |
| Dòng | train | 249 | 7.072 (44 ảnh thiếu nhãn) | 67,8 (max 158) | 15,7 | 2048×145 |
| Dòng | test | 6 | 201 | 65,2 | 15,0 | 2048×181 |
| Đoạn | train | 249 | 1.113 | 436,9 (max 715) | 100,1 | 2048×619 |
| Đoạn | test | 6 | 31 | 428,1 | 97,0 | 2048×616 |

- Bộ ký tự: 162 ký tự (chữ có dấu, số, dấu câu); test không có ký tự nào ngoài train. Có ký tự lạ `ƒ` (nhiễu gán nhãn) → loại khi huấn luyện.
- 26,7 % chữ cái mang dấu (dấu thanh hoặc dấu mũ/móc) — đồng đều giữa các mức và tập.
- Chia theo người viết: test = người viết 250–255 ở cả 3 mức, không trùng train → đánh giá độc lập người viết.
- Ảnh dòng là ảnh cắt từ ảnh đoạn của cùng người viết (nhãn nối dòng = nhãn đoạn).

## Hệ quả cho thiết kế
1. **Tập test rất nhỏ** (6 người viết, 201 dòng, 31 đoạn) → tách thêm tập validation theo người viết từ train (vd 25 người ≈ 10 %), báo cáo kết quả kèm khoảng tin cậy.
2. **Ảnh dòng rộng 2048 px** → resize theo chiều cao (vd h = 64–128) giữ tỉ lệ, đệm chiều rộng; nhãn dài tới 158 ký tự → số bước thời gian sau CNN phải > 158.
3. **Mức đoạn có nhãn văn bản nhưng không có toạ độ dòng** → tách dòng trên ảnh đoạn chỉ đánh giá gián tiếp qua CER đầu-cuối.
4. **Mức từ** (108 nghìn ảnh) dùng tiền huấn luyện / tăng cường, hoặc ghép từ thành dòng giả.
5. **Miền dữ liệu:** nội dung UIT-HWDB là văn bản phổ thông (bài luận, tin tức) → đề tài đã bỏ phạm vi "hành chính" (12/11/2026).
6. Bỏ 44 ảnh dòng và 257 ảnh từ thiếu nhãn.
