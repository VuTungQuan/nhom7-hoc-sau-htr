"""Tài khoản đăng nhập web app: mật khẩu băm scrypt, phiên ký HMAC (chỉ thư viện chuẩn).

Tạo/đổi mật khẩu (mật khẩu đọc từ stdin, không lộ trên dòng lệnh):
    python tai_khoan.py <tệp tài khoản> <tên đăng nhập>
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from pathlib import Path

N, R, P = 2**14, 8, 1


def bam(mk, salt):
    return hashlib.scrypt(mk.encode(), salt=salt, n=N, r=R, p=P, dklen=32)


def dat_mat_khau(tep, ten, mk):
    tep = Path(tep)
    ds = json.loads(tep.read_text()) if tep.exists() else {}
    salt = secrets.token_bytes(16)
    ds[ten] = {"salt": salt.hex(), "bam": bam(mk, salt).hex()}
    tep.write_text(json.dumps(ds, indent=1))
    os.chmod(tep, 0o600)


def kiem_tra(ds, ten, mk):
    tk = ds.get(ten)
    if not tk:
        bam(mk, b"\0" * 16)  # tốn thời gian như nhau, không lộ tên đăng nhập có tồn tại hay không
        return False
    return hmac.compare_digest(bam(mk, bytes.fromhex(tk["salt"])).hex(), tk["bam"])


class Phien:
    """Cookie = ten|het_han|chu_ky. Khoá ký sinh ngẫu nhiên mỗi lần khởi động: khởi động lại = đăng xuất hết."""

    def __init__(self, giay=8 * 3600):
        self.khoa, self.giay = secrets.token_bytes(32), giay

    def _ky(self, s):
        return base64.urlsafe_b64encode(hmac.new(self.khoa, s.encode(), "sha256").digest()).decode().rstrip("=")

    def tao(self, ten):
        s = f"{ten}|{int(time.time()) + self.giay}"
        return f"{s}|{self._ky(s)}"

    def doc(self, cookie):
        try:
            ten, het, ky = (cookie or "").rsplit("|", 2)
        except ValueError:
            return None
        if hmac.compare_digest(ky, self._ky(f"{ten}|{het}")) and int(het) > time.time():
            return ten
        return None


if __name__ == "__main__":
    if sys.argv[1:2] == ["--tu-kiem"]:
        import tempfile
        t = Path(tempfile.mkdtemp()) / "tk.json"
        dat_mat_khau(t, "u", "mk-dung")
        ds = json.loads(t.read_text())
        assert kiem_tra(ds, "u", "mk-dung") and not kiem_tra(ds, "u", "mk-sai") and not kiem_tra(ds, "x", "mk-dung")
        ph = Phien(); c = ph.tao("u")
        assert ph.doc(c) == "u" and ph.doc(c[:-1] + "A") is None and ph.doc("u|9999999999|x") is None and ph.doc(None) is None
        assert Phien().doc(c) is None  # khoá khác -> không nhận
        print("tu kiem OK")
    else:
        dat_mat_khau(sys.argv[1], sys.argv[2], sys.stdin.readline().rstrip("\n"))
        print("đã lưu tài khoản", sys.argv[2], "vào", sys.argv[1])
