#!/usr/bin/env python
"""Cut RailSem19 switch boxes into a classification dataset (switch state: left / right).

    python tools/prepare_switch_crops.py --root data/railsem19 --splits data/splits

RailSem19 annotates switches as boxes labelled switch-left, switch-right, switch-static (no moving
blades) and switch-unknown (state not visible). Each left/right box is cropped with surrounding
context (the blades are tiny, the tracks around them show which way they lead) and saved to
data/switch_crops/<train|val>/<left|right>/. --with-unknown also keeps switch-unknown as a class.
The train/val split follows the segmentation split, so no frame is in both.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

LABELS = {"switch-left": "left", "switch-right": "right", "switch-unknown": "unknown"}


def crop_with_context(img, box, ctx_w=2.0, ctx_h=3.0, min_side=48):
    """Square-ish crop around a box: ctx_w x its width, ctx_h x its height, at least min_side px."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    side = max(bw * ctx_w, bh * ctx_h, min_side)
    a0, b0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    a1, b1 = a0 + int(round(side)), b0 + int(round(side))
    pad = max(0, -a0, -b0, a1 - w, b1 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        a0, a1, b0, b1 = a0 + pad, a1 + pad, b0 + pad, b1 + pad
    return img[b0:b1, a0:a1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/railsem19")
    ap.add_argument("--splits", default="data/splits", help="folder with train.txt / val.txt (frame split)")
    ap.add_argument("--out", default="data/switch_crops")
    ap.add_argument("--with-unknown", action="store_true")
    ap.add_argument("--min-width", type=int, default=12, help="skip boxes narrower than this (px)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    images = {p.stem: p for p in root.rglob("*.jpg")}
    val_frames = set()
    vf = Path(args.splits) / "val.txt"
    if vf.exists():
        val_frames = {Path(line.split()[0]).stem for line in vf.read_text().splitlines() if line.strip()}
    else:
        print(f"[warn] {vf} not found: using every 17th frame for validation")
    keep = {k: v for k, v in LABELS.items() if v != "unknown" or args.with_unknown}

    counts, widths, skipped = Counter(), {"train": [], "val": []}, Counter()
    meta = []
    jsons = sorted(p for p in root.rglob("*.json")
                   if p.name not in ("json-classes.json", "rs19-config.json", "rs19_egopath.json"))
    for n, jp in enumerate(jsons):
        try:
            ann = json.loads(jp.read_text())
        except Exception:
            continue
        objs = [o for o in ann.get("objects", []) if o.get("label") in keep and "boundingbox" in o]
        if not objs or jp.stem not in images:
            continue
        img = cv2.imread(str(images[jp.stem]))
        if img is None:
            continue
        split = "val" if (jp.stem in val_frames if val_frames else n % 17 == 0) else "train"
        for k, o in enumerate(objs):
            x0, y0, x1, y1 = o["boundingbox"]
            if x1 - x0 < args.min_width:
                skipped["too small"] += 1
                continue
            cls = keep[o["label"]]
            crop = crop_with_context(img, (x0, y0, x1, y1))
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            name = f"{jp.stem}_{k}.jpg"
            cv2.imwrite(str(d / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[(split, cls)] += 1
            widths[split].append(x1 - x0)
            meta.append({"file": f"{split}/{cls}/{name}", "frame": jp.stem, "box": [x0, y0, x1, y1], "class": cls})
        if (n + 1) % 1000 == 0:
            print(f"  {n + 1}/{len(jsons)} annotation files")

    if not meta:
        sys.exit("No switch boxes found - are the RailSem19 jsons under --root?")
    (out / "crops.json").write_text(json.dumps(meta))
    print(f"\nSwitch crops written to {out}/")
    for split in ("train", "val"):
        row = ", ".join(f"{c}: {counts[(split, c)]}" for c in sorted({c for _, c in counts}))
        w = np.array(widths[split] or [0])
        print(f"  {split:5s} {row}   | box width px: median {np.median(w):.0f}, "
              f"<40: {int((w < 40).sum())}, 40-100: {int(((w >= 40) & (w < 100)).sum())}, >=100: {int((w >= 100).sum())}")
    if skipped:
        print(f"  skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
