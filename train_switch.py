#!/usr/bin/env python
"""Train a switch-state classifier (left / right [/ unknown]) on crops made by
tools/prepare_switch_crops.py.

    python train_switch.py                       # EfficientNet-B0, ImageNet-pretrained
    python train_switch.py --arch resnet18 --epochs 40

Augmentation: a mirrored left switch is a right switch, so horizontal flips swap the labels.
Reports accuracy per class and per box size (a proxy for distance to the switch).
Writes runs/switch_cls/best.pt (+ metrics.json).
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(img_bgr, size):
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = (img[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class SwitchCrops(Dataset):
    def __init__(self, root, split, classes, size, train, meta):
        self.items = [(Path(root) / m["file"], classes.index(m["class"]), m["box"][2] - m["box"][0])
                      for m in meta if m["file"].startswith(split + "/") and m["class"] in classes]
        self.classes, self.size, self.train = classes, size, train
        self.swap = {classes.index("left"): classes.index("right"), classes.index("right"): classes.index("left")}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y, box_w = self.items[i]
        img = cv2.imread(str(path))
        if self.train:
            if random.random() < 0.5:                       # mirror: left <-> right
                img = img[:, ::-1]
                y = self.swap.get(y, y)
            h, w = img.shape[:2]
            s = random.uniform(0.85, 1.0)                    # small random crop / shift
            ch, cw = int(h * s), int(w * s)
            y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
            img = img[y0:y0 + ch, x0:x0 + cw]
            if random.random() < 0.3:                        # far-away switches are blurry
                f = random.uniform(0.25, 0.6)
                img = cv2.resize(cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA),
                                 (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-20, 20)
            img = np.clip(img, 0, 255).astype(np.uint8)
        return to_tensor(np.ascontiguousarray(img), self.size), y, box_w


def build(arch, n_classes, pretrained=True):
    import torchvision.models as tvm
    if arch == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m


@torch.no_grad()
def evaluate(model, loader, device, n):
    model.eval()
    conf = np.zeros((n, n), int)
    by_size = {"<40px": [0, 0], "40-100px": [0, 0], ">=100px": [0, 0]}
    for x, y, bw in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for t, p, w in zip(y.tolist(), pred.tolist(), bw.tolist()):
            conf[t, p] += 1
            k = "<40px" if w < 40 else ("40-100px" if w < 100 else ">=100px")
            by_size[k][0] += int(t == p)
            by_size[k][1] += 1
    acc = conf.trace() / max(conf.sum(), 1)
    recall = conf.diagonal() / np.maximum(conf.sum(1), 1)
    return acc, recall, conf, {k: (round(a / b, 3) if b else None, b) for k, (a, b) in by_size.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/switch_crops")
    ap.add_argument("--out", default="runs/switch_cls")
    ap.add_argument("--arch", choices=["efficientnet_b0", "resnet18"], default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    meta = json.loads((Path(args.data) / "crops.json").read_text())
    classes = [c for c in ("left", "right", "unknown") if any(m["class"] == c for m in meta)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = SwitchCrops(args.data, "train", classes, args.size, True, meta)
    va = SwitchCrops(args.data, "val", classes, args.size, False, meta)
    print(f"Classes {classes} | train {len(tr)} | val {len(va)} | {args.arch} on {device}")
    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(va, args.batch, shuffle=False, num_workers=args.workers)

    counts = np.bincount([y for _, y, _ in tr.items], minlength=len(classes)).astype(np.float32)
    # left/right are balanced by the mirror augmentation; weight only the rest
    w = counts.sum() / np.maximum(counts, 1)
    if "unknown" not in classes:
        w[:] = 1.0
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), device=device), label_smoothing=0.05)
    model = build(args.arch, len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tl))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, _ in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        acc, rec, conf, by_size = evaluate(model, vl, device, len(classes))
        history.append({"epoch": ep + 1, "loss": tot / max(n, 1), "val_acc": acc,
                        "recall": dict(zip(classes, rec.round(3).tolist())), "by_box_size": by_size})
        print(f"epoch {ep + 1:2d}/{args.epochs} loss {tot / max(n, 1):.3f}  val acc {100 * acc:.1f}%  "
              + "  ".join(f"{c} {100 * r:.0f}%" for c, r in zip(classes, rec))
              + f"  | by size {by_size}  ({time.time() - t0:.0f}s)")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "arch": args.arch, "classes": classes, "size": args.size,
                        "crop_context": [2.0, 3.0], "val_acc": acc}, out / "best.pt")
            (out / "metrics.json").write_text(json.dumps({"best": history[-1], "confusion": conf.tolist(),
                                                          "classes": classes}, indent=1))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"\nBest val accuracy {100 * best:.1f}% -> {out / 'best.pt'}")
    m = json.loads((out / "metrics.json").read_text())
    print("Confusion (rows = true, cols = predicted):", m["classes"])
    for c, row in zip(m["classes"], m["confusion"]):
        print(f"  {c:8s} {row}")


if __name__ == "__main__":
    main()
