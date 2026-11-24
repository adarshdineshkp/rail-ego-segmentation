#!/usr/bin/env python
"""Evaluate a trained checkpoint on the validation split.

    python evaluate.py --ckpt runs/rs19_pidnet_s_rails/best.pt [--flip] [--save-vis out_dir]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from railseg.checkpoint import load_trained
from railseg.config import apply_overrides, load_config
from railseg.data.railsem19 import RailSem19
from railseg.data.classes import build_label_mapping, load_rs19_classes
from railseg.engine import evaluate, predict_logits
from railseg.metrics import mean_iou_of
from railseg.utils import colorize, format_iou_table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="override the config stored in the checkpoint")
    ap.add_argument("--flip", action="store_true", help="horizontal-flip test-time augmentation")
    ap.add_argument("--save-vis", default=None, help="folder for side-by-side prediction images")
    ap.add_argument("--max-vis", type=int, default=50)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, classes = load_trained(args.ckpt, device)
    cfg = load_config(args.config, args.overrides) if args.config else apply_overrides(cfg, args.overrides)
    src_names, src_colors = load_rs19_classes(cfg["data"]["splits_dir"])
    lut, names, _ = build_label_mapping(cfg["data"], src_names, src_colors)
    if names != classes["names"]:
        raise SystemExit(f"Label set in config {names} does not match checkpoint {classes['names']}")
    ignore = int(cfg["data"].get("ignore_label", 255))

    ds = RailSem19(cfg["data"], cfg["loss"], "val", lut, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    m = evaluate(model, loader, len(names), ignore, device, amp=True, flip=args.flip, log=print)
    key = [names.index(k) for k in (cfg["train"].get("key_classes") or []) if k in names]
    print(f"\nmIoU {100 * m['miou']:.2f}   pixel acc {100 * m['acc']:.2f}"
          + (f"   key-class mIoU {100 * mean_iou_of(m['iou'], key):.2f}" if key else ""))
    print(format_iou_table(names, m["iou"], key))

    if args.save_vis:
        out = Path(args.save_vis)
        out.mkdir(parents=True, exist_ok=True)
        colors = classes["colors"]
        for i, (img, lab, stem) in enumerate(loader):
            if i >= args.max_vis:
                break
            pred = predict_logits(model, img.to(device), lab.shape[-2:], args.flip).argmax(1)[0].cpu().numpy()
            raw = cv2.imread(str(ds.pairs[i][0]))
            raw = cv2.resize(raw, (pred.shape[1], pred.shape[0]))
            gt = lab[0].numpy().copy()
            gt[gt == ignore] = 0
            gt_c = colorize(gt, colors)[:, :, ::-1]
            pr_c = colorize(pred, colors)[:, :, ::-1]
            overlay = cv2.addWeighted(raw, 0.5, pr_c, 0.5, 0)
            panel = np.concatenate([overlay, gt_c, pr_c], axis=1)
            scale = 1920 / panel.shape[1]
            panel = cv2.resize(panel, None, fx=scale, fy=scale)
            cv2.imwrite(str(out / f"{stem[0]}.jpg"), panel)
        print(f"Saved visualisations (overlay | ground truth | prediction) to {out}/")


if __name__ == "__main__":
    main()
