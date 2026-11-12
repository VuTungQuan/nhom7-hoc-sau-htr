"""Vẽ các sơ đồ cho báo cáo (Hình 1.1, 2.1, 2.2, 3.1, 3.2) và đường học (Hình 4.1).
Chạy: python ve_hinh.py <thư mục ra> <log.csv>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
LOG = Path(sys.argv[2]) if len(sys.argv) > 2 else None
plt.rcParams["font.family"] = "DejaVu Sans"
XANH, CAM, XAM, LUC = "#2b5c8a", "#c4622d", "#555555", "#2e7d4f"


def hop(ax, x, y, w, h, chu, mau=XANH, nen="#eef4fb", co=9, dam=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03", fc=nen, ec=mau, lw=1.4))
    ax.text(x, y, chu, ha="center", va="center", fontsize=co, weight="bold" if dam else "normal")


def mui(ax, a, b, chu=None, net="-", mau="#333", lech=0.18):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.2, linestyle=net, color=mau))
    if chu:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + lech, chu, ha="center", fontsize=7.5, style="italic", color=XAM)


def luu(fig, ten):
    fig.savefig(OUT / ten, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("ok", ten)


def h1_1_crnn():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 22); ax.set_ylim(0, 6.6); ax.axis("off")
    # ảnh dòng
    ax.add_patch(Rectangle((0.2, 2.6), 3.2, 1.0, fc="#f7f7f7", ec="#888"))
    ax.text(1.8, 3.1, "Cả sới gà náo động…", ha="center", va="center", fontsize=8.5, style="italic", family="DejaVu Serif")
    ax.text(1.8, 2.25, "Ảnh dòng xám\ncao 64 px × rộng W", ha="center", va="top", fontsize=7.5, color=XAM)
    # CNN
    for i, (h, c) in enumerate([(3.0, 64), (2.5, 128), (2.1, 256), (1.7, 256), (1.4, 512), (1.1, 512), (0.9, 512)]):
        x = 4.3 + i * 0.62
        ax.add_patch(Rectangle((x, 3.1 - h / 2), 0.42, h, fc="#d6e4f2", ec=XANH, lw=1))
        ax.text(x + 0.21, 3.1 - h / 2 - 0.15, str(c), ha="center", va="top", fontsize=6, color=XAM)
    ax.text(6.4, 5.35, "Tầng tích chập (7 lớp Conv3×3 + BN + ReLU,\nMaxPool: cao ÷16, rộng ÷4)", ha="center", fontsize=8, color=XANH, weight="bold")
    mui(ax, (3.45, 3.1), (4.25, 3.1))
    # chuỗi đặc trưng
    for i in range(7):
        ax.add_patch(Rectangle((9.3 + i * 0.32, 2.6), 0.26, 1.0, fc="#fde8d8", ec=CAM, lw=0.9))
    ax.text(10.4, 2.25, "Chuỗi đặc trưng\nT = W/4 bước", ha="center", va="top", fontsize=7.5, color=XAM)
    mui(ax, (8.75, 3.1), (9.25, 3.1))
    # BiLSTM
    hop(ax, 13.6, 3.1, 2.6, 2.0, "BiLSTM\n2 lớp × 256\n(hai chiều)", mau=CAM, nen="#fdf1e8", co=8.5, dam=True)
    ax.text(13.6, 5.35, "Tầng hồi quy", ha="center", fontsize=8, color=CAM, weight="bold")
    mui(ax, (11.6, 3.1), (12.25, 3.1))
    # Linear + softmax
    hop(ax, 16.5, 3.1, 2.0, 1.6, "Tuyến tính\n+ softmax\n162 lớp", mau=LUC, nen="#eaf5ee", co=8)
    mui(ax, (14.95, 3.1), (15.45, 3.1))
    # CTC
    hop(ax, 19.75, 3.1, 3.1, 1.6, "Giải mã CTC\n(gộp lặp, bỏ blank)\n+ độ tin cậy từ", mau=LUC, nen="#eaf5ee", co=8)
    ax.text(18.1, 5.35, "Tầng phiên mã", ha="center", fontsize=8, color=LUC, weight="bold")
    mui(ax, (17.55, 3.1), (18.15, 3.1))
    ax.text(19.75, 1.7, "\"Cả sới gà náo động…\"", ha="center", fontsize=8.5, style="italic")
    mui(ax, (19.75, 2.25), (19.75, 1.95))
    ax.text(11, 0.45, "Huấn luyện: hàm mất mát CTC trên nhãn cả dòng, không cần căn chỉnh từng ký tự · 13,24 triệu tham số",
            ha="center", fontsize=8, color=XAM)
    luu(fig, "h1-1-crnn.png")


def h2_1_quy_trinh():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 22); ax.set_ylim(0, 7); ax.axis("off")
    # (tên, tâm x, rộng)
    buoc = [("Bước 1–3\nKhảo sát,\nchuẩn bị", 1.4, 2.4), ("Bước 4\nQuét / chụp\ntài liệu", 4.4, 2.4),
            ("Bước 5\nKiểm tra lỗi\nảnh quét,\nbiên mục", 11.0, 2.6), ("Bước 6\nKý số", 13.9, 2.2),
            ("Bước 7\nĐóng gói\ncùng dữ\nliệu chủ", 16.6, 2.2), ("Bước 8\nNghiệm thu", 19.6, 2.6)]
    for chu, x, w in buoc:
        hop(ax, x, 4.3, w, 1.9, chu, co=8.5)
    hop(ax, 7.7, 4.3, 3.4, 2.2, "HỆ THỐNG\nNHẬN DẠNG\n(CRNN + sửa lỗi)", mau=CAM, nen="#fde8d8", co=8.5, dam=True)
    khung = [(1.4, 2.4), (4.4, 2.4), (7.7, 3.4), (11.0, 2.6), (13.9, 2.2), (16.6, 2.2), (19.6, 2.6)]
    for (a, wa), (b, wb) in zip(khung[:-1], khung[1:]):
        mui(ax, (a + wa / 2 + 0.05, 4.3), (b - wb / 2 - 0.05, 4.3))
    ax.text(7.7, 2.75, "lớp văn bản cho PDF/A hai lớp,\nhỗ trợ biên mục, tra cứu", ha="center", va="top", fontsize=8, color=CAM)
    ax.text(11, 6.5, "Quy trình số hóa tài liệu lưu trữ giấy (Điều 9, Thông tư 05/2025/TT-BNV) và vị trí của hệ thống",
            ha="center", fontsize=9.5, weight="bold")
    ax.text(11, 0.5, "Ghi chú: tên các bước 4–7 theo bản toàn văn đã tham khảo; tên bước 1–3 và 8 là khái quát – cần đối chiếu Công báo trước khi nộp.",
            ha="center", fontsize=7.5, style="italic", color="#a33")
    luu(fig, "h2-1-quy-trinh-so-hoa.png")


def tac_nhan(ax, x, y, ten):
    ax.add_patch(plt.Circle((x, y + 0.55), 0.18, fill=False, lw=1.5))
    ax.plot([x, x], [y + 0.37, y - 0.2], "k", lw=1.5)
    ax.plot([x - 0.3, x + 0.3], [y + 0.15, y + 0.15], "k", lw=1.5)
    ax.plot([x, x - 0.25], [y - 0.2, y - 0.6], "k", lw=1.5)
    ax.plot([x, x + 0.25], [y - 0.2, y - 0.6], "k", lw=1.5)
    ax.text(x, y - 0.95, ten, ha="center", va="top", fontsize=9.5)


def h2_2_use_case():
    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 9); ax.axis("off")
    ax.add_patch(FancyBboxPatch((2.5, 0.3), 8.0, 8.3, boxstyle="round,pad=0.02", fill=False, lw=1.2))
    ax.text(6.5, 8.25, "Hệ thống nhận dạng chữ viết tay tiếng Việt (web)", ha="center", fontsize=11, weight="bold")
    uc = {"01": (4.6, 7.2, "UC01 Đăng nhập"), "07": (8.5, 7.2, "UC07 Đăng xuất"),
          "02": (4.6, 5.4, "UC02 Đưa tệp vào\n(chọn, kéo thả, dán)"), "03": (8.5, 5.6, "UC03 Chọn vùng ảnh"),
          "04": (4.6, 3.6, "UC04 Nhận dạng\n(3 chế độ)"), "05": (8.5, 3.7, "UC05 Xem kết quả\nkèm độ tin cậy"),
          "06": (4.6, 1.8, "UC06 Sửa tay, sao chép,\ntải .txt"), "08": (8.5, 1.5, "UC08 Quản lý tài khoản\n(dòng lệnh)")}
    for x, y, t in uc.values():
        ax.add_patch(Ellipse((x, y), 3.1, 1.15, fc="#eef4fb", ec=XANH, lw=1.3))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.5)
    tac_nhan(ax, 1.1, 5.0, "Người dùng\nnghiệp vụ\n(cán bộ số hóa)")
    tac_nhan(ax, 11.9, 2.0, "Quản trị viên")
    for k in ["01", "02", "04", "05", "06", "07"]:
        x, y, _ = uc[k]
        ax.plot([1.45, x - 1.55 if x < 6 else x - 1.5], [5.2, y], color="#444", lw=0.9)
    ax.plot([11.55, uc["08"][0] + 1.55], [2.2, uc["08"][1]], color="#444", lw=0.9)
    mui(ax, (4.6, 4.2), (4.6, 4.8), net="--")
    ax.text(4.75, 4.45, "«include»", fontsize=7.5, style="italic", color=XAM)
    ax.add_patch(FancyArrowPatch((6.95, 5.6), (6.15, 5.45), arrowstyle="-|>", mutation_scale=13, lw=1.2, linestyle="--", color="#333"))
    ax.text(6.55, 5.85, "«extend»", ha="center", fontsize=7.5, style="italic", color=XAM)
    luu(fig, "h2-2-use-case.png")


def h3_1_kien_truc():
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 23); ax.set_ylim(0, 12.8); ax.axis("off")
    hop(ax, 2.2, 10.6, 3.6, 1.6, "Trình duyệt\n(chọn/kéo thả/dán,\ncắt vùng)", co=8.5)
    hop(ax, 7.6, 10.6, 3.8, 1.6, "Cloudflare Tunnel\nnhom7hocsau.cat.io.vn\n(HTTPS, kết nối ra ngoài)", mau=XAM, nen="#f2f2f2", co=8)
    mui(ax, (4.05, 10.6), (5.65, 10.6), "HTTPS")
    ax.add_patch(FancyBboxPatch((10.3, 0.4), 12.4, 11.6, boxstyle="round,pad=0.02", fill=False, ec="#999", lw=1.2, linestyle="--"))
    ax.text(16.5, 11.6, "Máy chủ của nhóm (RTX 3090) – không gọi API bên thứ ba", ha="center", fontsize=9, weight="bold", color=XAM)
    hop(ax, 13.2, 10.6, 5.2, 1.5, "FastAPI (app.py)\nđăng nhập scrypt\n+ phiên HMAC · /api/nhan-dang", mau=LUC, nen="#eaf5ee", co=8)
    mui(ax, (9.55, 10.6), (10.55, 10.6))
    ax.text(10.05, 9.6, "cloudflared", ha="center", fontsize=7.5, style="italic", color=XAM)
    hop(ax, 13.2, 8.3, 4.6, 1.4, "PDF → ảnh (PyMuPDF, 200 dpi)\nảnh → xám", co=8)
    hop(ax, 13.2, 6.1, 4.6, 1.7, "Tiền xử lý (htr.py)\ncân bằng nền · xóa dòng kẻ ·\ntự co giãn · kéo tương phản", co=8)
    hop(ax, 13.2, 3.8, 4.6, 1.7, "Tách dòng\nchiếu ngang + đỉnh nổi bật\n+ gán thành phần liên thông", co=8)
    hop(ax, 19.4, 3.8, 4.7, 1.7, "CRNN (best.pt)\nCNN + BiLSTM + CTC\n→ văn bản + độ tin cậy từ", mau=CAM, nen="#fde8d8", co=8, dam=True)
    hop(ax, 19.4, 6.6, 4.7, 2.1, "Sửa lỗi ProtonX\nprotonx-legal-tc (transformers)\nbeam 10, khúc ≤ 30 từ", mau=CAM, nen="#fde8d8", co=8, dam=True)
    hop(ax, 19.4, 9.3, 4.7, 1.7, "Lớp kiểm soát (llm_sua.py)\nthay 1–1 · khoảng cách ≤ 50 %\n· từ điển âm tiết", mau=CAM, nen="#fdf1e8", co=8)
    for a, b in [((13.2, 9.85), (13.2, 9.0)), ((13.2, 7.6), (13.2, 6.95)), ((13.2, 5.25), (13.2, 4.65)),
                 ((15.5, 3.8), (17.0, 3.8)), ((19.4, 4.65), (19.4, 5.55)), ((19.4, 7.65), (19.4, 8.45))]:
        mui(ax, a, b)
    ax.add_patch(FancyArrowPatch((17.0, 9.9), (15.85, 10.45), arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#333"))
    ax.text(15.6, 8.85, "JSON:\nvăn bản,\ntừ + độ\ntin cậy", fontsize=7, style="italic", color=XAM)
    ax.text(16.5, 1.3, "Chế độ a (chỉ CRNN) bỏ qua hai khối cam;\nchế độ b dùng nguyên đề xuất, không qua lớp kiểm soát.", ha="center", fontsize=7.5, color=XAM)
    luu(fig, "h3-1-kien-truc.png")


def h3_2_giao_dien():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 22); ax.set_ylim(0, 12.4); ax.axis("off")
    ax.add_patch(Rectangle((0.2, 0.2), 21.6, 12.0, fc="#f4f5f7", ec="#888"))
    ax.add_patch(Rectangle((0.2, 11.0), 21.6, 1.2, fc="white", ec="#ccc"))
    ax.add_patch(FancyBboxPatch((0.6, 11.25), 0.7, 0.7, boxstyle="round,pad=0.02", fc=XANH, ec=XANH))
    ax.text(1.6, 11.6, "Nhận dạng chữ viết tay tiếng Việt", fontsize=10, weight="bold", va="center")
    hop(ax, 20.4, 11.6, 2.2, 0.7, "Đăng xuất", co=8)
    # thẻ trái
    ax.add_patch(FancyBboxPatch((0.6, 2.0), 10.4, 8.6, boxstyle="round,pad=0.05", fc="white", ec="#ccc"))
    ax.text(1.0, 10.1, "① ẢNH ĐẦU VÀO", fontsize=8.5, weight="bold", color=XAM)
    ax.add_patch(FancyBboxPatch((1.0, 5.4), 9.6, 4.2, boxstyle="round,pad=0.05", fc="#f0f2f6", ec="#999", linestyle="--"))
    ax.text(5.8, 7.9, "Kéo thả ảnh hoặc PDF vào đây", ha="center", fontsize=9.5, weight="bold")
    ax.text(5.8, 7.0, "hoặc dán ảnh bằng Ctrl+V, hoặc bấm để chọn tệp\n(JPG, PNG, PDF · tối đa 20 MB, 20 trang)", ha="center", fontsize=8, color=XAM)
    ax.text(5.8, 5.8, "(ảnh hoặc từng trang PDF: kéo chuột để cắt vùng · PDF có nút chuyển trang)", ha="center", fontsize=7.5, style="italic", color=XAM)
    for i, (t, mac) in enumerate([("CRNN + sửa lỗi có kiểm soát", True), ("Chỉ CRNN", False), ("Sửa lỗi toàn bộ", False)]):
        x = [1.05, 5.75, 7.85][i]; w = [4.55, 1.95, 2.75][i]
        ax.add_patch(FancyBboxPatch((x, 4.2), w, 0.8, boxstyle="round,pad=0.02", fc="white" if mac else "#f0f2f6", ec=XANH if mac else "#bbb", lw=1.5 if mac else 1))
        ax.text(x + w / 2, 4.6, t, ha="center", va="center", fontsize=7.5, weight="bold" if mac else "normal")
    ax.text(1.2, 3.3, "☐ Ảnh là một dòng", fontsize=8)
    hop(ax, 7.0, 3.3, 2.0, 0.7, "Bỏ vùng chọn", co=7.5)
    ax.add_patch(FancyBboxPatch((8.4, 2.95), 2.3, 0.7, boxstyle="round,pad=0.02", fc=XANH, ec=XANH))
    ax.text(9.55, 3.3, "Nhận dạng", ha="center", va="center", fontsize=8, color="white", weight="bold")
    ax.text(1.2, 2.45, "Ctrl+Enter để chạy", fontsize=7, color=XAM)
    # thẻ phải
    ax.add_patch(FancyBboxPatch((11.4, 2.0), 10.0, 8.6, boxstyle="round,pad=0.05", fc="white", ec="#ccc"))
    ax.text(11.8, 10.1, "② KẾT QUẢ", fontsize=8.5, weight="bold", color=XAM)
    ax.text(11.8, 9.3, "Thứ trưởng Bộ Tài nguyên và môi trường Đặng Hùng", fontsize=8)
    ax.add_patch(Rectangle((11.75, 8.45), 1.3, 0.45, fc="#ffe8a3", ec="none"))
    ax.text(11.8, 8.6, "Võ tâm đắc với có chế khuyến khích …", fontsize=8)
    ax.text(11.8, 7.8, "■ nền vàng = từ CRNN đọc kém tin cậy (< 0,5)", fontsize=7, color="#a07800")
    ax.add_patch(Rectangle((11.8, 3.6), 9.2, 3.7, fc="#f0f2f6", ec="#bbb"))
    ax.text(12.0, 7.0, "Văn bản cuối – sửa tay được", fontsize=7.5, weight="bold")
    ax.text(12.1, 6.4, "Thứ trưởng Bộ Tài nguyên … tâm đắc với cơ chế …", fontsize=8)
    hop(ax, 12.8, 2.9, 1.8, 0.6, "Sao chép", co=7.5)
    hop(ax, 14.9, 2.9, 1.8, 0.6, "Tải .txt", co=7.5)
    ax.text(21.0, 2.9, "Xong trong 1,8 giây · 13 từ kém tin cậy", ha="right", fontsize=7, color=LUC)
    ax.text(0.6, 1.0, "Ảnh chỉ dùng để nhận dạng, xử lý trên máy chủ của nhóm, không lưu lại · Built with ProtonX Text Correction Model", fontsize=7, color=XAM)
    luu(fig, "h3-2-giao-dien.png")


def h4_1_duong_hoc():
    r = list(csv.DictReader(open(LOG, encoding="utf-8")))
    ep = [int(x["epoch"]) for x in r]
    loss = [float(x["loss"]) for x in r]
    cer = [float(x["val_CER"]) * 100 for x in r]
    wer = [float(x["val_WER"]) * 100 for x in r]
    dau = [float(x["val_loi_dau"]) * 100 for x in r]
    b = min(range(len(r)), key=lambda i: cer[i])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(ep, loss, color=XANH, lw=1.6)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss CTC (tập huấn luyện, thang log)"); a1.set_yscale("log"); a1.grid(alpha=0.3)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    a1.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 4])); a1.yaxis.set_minor_locator(NullLocator())
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    a1.set_title("Hàm mất mát", fontsize=10)
    a2.plot(ep, wer, color=CAM, lw=1.4, label="WER")
    a2.plot(ep, cer, color=XANH, lw=1.6, label="CER")
    a2.plot(ep, dau, color=LUC, lw=1.2, ls="--", label="Lỗi chỉ do dấu")
    a2.axvline(ep[b], color=XAM, ls=":", lw=1)
    a2.annotate(f"epoch {ep[b]}: CER {cer[b]:.2f} %".replace(".", ","), (ep[b], cer[b]), (ep[b] - 38, 22),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color=XAM))
    a2.set_ylim(0, 50); a2.set_xlabel("Epoch"); a2.set_ylabel("% trên tập kiểm định (25 người viết)")
    a2.set_title("Tỉ lệ lỗi trên tập kiểm định", fontsize=10); a2.grid(alpha=0.3); a2.legend(fontsize=8.5)
    fig.tight_layout()
    luu(fig, "h4-1-duong-hoc.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    h1_1_crnn(); h2_1_quy_trinh(); h2_2_use_case(); h3_1_kien_truc(); h3_2_giao_dien()
    if LOG:
        h4_1_duong_hoc()
