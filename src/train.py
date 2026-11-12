"""Huấn luyện CRNN trên UIT-HWDB mức dòng.

python train.py --goc /mnt/data/hs-htr/data/UIT_HWDB_line --ra /mnt/data/hs-htr/runs/crnn_line
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import htr

ap = argparse.ArgumentParser()
ap.add_argument("--goc", required=True)
ap.add_argument("--ra", required=True)
ap.add_argument("--h", type=int, default=64)
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--gioi_han", type=int, default=0, help="chỉ dùng N mẫu (chạy thử)")
a = ap.parse_args()

torch.manual_seed(0)
ra = Path(a.ra); ra.mkdir(parents=True, exist_ok=True)
dev = "cuda"

train_all = htr.doc_mau(a.goc, "train_data")
test = htr.doc_mau(a.goc, "test_data")
train, val, val_ws = htr.chia_val(train_all)
if a.gioi_han:
    train, val = train[: a.gioi_han], val[: max(a.gioi_han // 10, 16)]
bkt = htr.BoKyTu("".join(t for _, t, _ in train_all))
json.dump({"chars": bkt.chars, "h": a.h, "val_writers": val_ws, "args": vars(a)},
          open(ra / "cau_hinh.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"train {len(train)}  val {len(val)}  test {len(test)}  ký tự {len(bkt.chars)}", flush=True)

gop = htr.gop_lo(bkt)
dl_tr = torch.utils.data.DataLoader(htr.TapDong(train, bkt, a.h, htr.tang_cuong()), a.bs, shuffle=True,
                                    num_workers=a.workers, collate_fn=gop, drop_last=True, persistent_workers=True)
dl_va = torch.utils.data.DataLoader(htr.TapDong(val, bkt, a.h), a.bs, num_workers=a.workers, collate_fn=gop)

m = htr.CRNN(len(bkt), a.h).to(dev)
opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl_tr), pct_start=0.1)
scaler = torch.amp.GradScaler()
print(f"tham số: {sum(p.numel() for p in m.parameters())/1e6:.2f} M", flush=True)


@torch.no_grad()
def danh_gia(dl):
    m.eval(); pd, gt = [], []
    for x, w, _, _, txt in dl:
        with torch.autocast("cuda", torch.float16):
            lg = m(x.to(dev))
        pd += [t for t, _ in htr.giai_ma(lg, w, bkt)]; gt += txt
    m.train()
    return htr.do_do(pd, gt), pd[:3], gt[:3]


log = csv.writer(open(ra / "log.csv", "a", newline="", encoding="utf-8", buffering=1))
log.writerow(["epoch", "loss", "val_CER", "val_WER", "val_loi_dau", "lr", "giay"])
best = 9.0
for ep in range(1, a.epochs + 1):
    t0, tong, n = time.time(), 0.0, 0
    for x, w, tg, tl, _ in dl_tr:
        with torch.autocast("cuda", torch.float16):
            lg = m(x.to(dev))
        lp = lg.float().log_softmax(-1).permute(1, 0, 2)  # T,B,C
        loss = F.ctc_loss(lp, tg, (w // 4).clamp(max=lp.shape[0]), tl, zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
        scaler.step(opt); scaler.update(); sch.step()
        tong += loss.item(); n += 1
    r, pd, gt = danh_gia(dl_va)
    log.writerow([ep, round(tong / n, 4), round(r["CER"], 4), round(r["WER"], 4), round(r["loi_dau"], 4),
                  f"{sch.get_last_lr()[0]:.2e}", round(time.time() - t0)])
    print(f"ep {ep} loss {tong/n:.3f} val CER {r['CER']:.4f} WER {r['WER']:.4f} dấu {r['loi_dau']:.4f} "
          f"({time.time()-t0:.0f}s)\n  pd: {pd[0][:80]}\n  gt: {gt[0][:80]}", flush=True)
    torch.save(m.state_dict(), ra / "last.pt")
    if r["CER"] < best:
        best = r["CER"]; torch.save(m.state_dict(), ra / "best.pt")
print("best val CER", best)
