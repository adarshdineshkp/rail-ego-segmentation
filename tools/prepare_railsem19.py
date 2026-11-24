#!/usr/bin/env python
"""Check the RailSem19 download and write train/val split lists.

RailSem19 ships as a single set of 8,500 frames (rs19_val), so we make our own split.
Neighbouring frame ids often come from the same drive, so validation is drawn in
contiguous blocks of ids rather than single random frames to reduce near-duplicate leakage.

Usage:
    python tools/prepare_railsem19.py --root data/railsem19
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from railseg.data.classes import DEFAULT_RS19_NAMES, default_palette  # noqa: E402


def find_pairs(root):
    images = {p.stem: p for p in root.rglob("*.jpg")}
    labels = {p.stem: p for p in root.rglob("*.png") if "uint8" in p.parts}
    stems = sorted(set(images) & set(labels))
    return images, labels, stems


def read_rs19_config(root):
    cfgs = list(root.rglob("rs19-config.json"))
    if not cfgs:
        print("  [warn] rs19-config.json not found, using built-in class names")
        return list(DEFAULT_RS19_NAMES), default_palette(len(DEFAULT_RS19_NAMES))
    d = json.loads(cfgs[0].read_text())
    labels = d.get("labels", d) if isinstance(d, dict) else d
    names, colors = [], []
    for item in labels:
        if isinstance(item, dict) and "name" in item:
            names.append(item["name"])
            colors.append(list(item.get("color", [0, 0, 0]))[:3])
    if len(names) < 19:
        print(f"  [warn] could not parse classes from {cfgs[0]}, using built-in names")
        return list(DEFAULT_RS19_NAMES), default_palette(len(DEFAULT_RS19_NAMES))
    if names != DEFAULT_RS19_NAMES:
        print("  [info] class list from rs19-config.json differs from built-in default; using the file's")
    return names, colors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/railsem19", help="folder where rs19_val.zip was extracted")
    ap.add_argument("--out", default="data/splits", help="where to write train.txt / val.txt / classes.json")
    ap.add_argument("--val-count", type=int, default=500)
    ap.add_argument("--block", type=int, default=50, help="size of contiguous id blocks used for val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", type=int, default=300, help="labels sampled for class-frequency stats (0 = skip)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    if not root.exists():
        sys.exit(f"Dataset root {root} does not exist. Download rs19_val.zip from "
                 f"https://www.wilddash.cc/railsem19 and extract it there.")
    print(f"Scanning {root} ...")
    images, labels, stems = find_pairs(root)
    print(f"  images: {len(images)}   uint8 label maps: {len(labels)}   matched pairs: {len(stems)}")
    if not stems:
        sys.exit("No image/label pairs found. Expected .jpg images and .png labels inside a 'uint8' "
                 "folder (as in the official rs19_val.zip).")
    missing = len(images) - len(stems)
    if missing:
        print(f"  [warn] {missing} images have no matching label and are skipped")

    names, colors = read_rs19_config(root)

    # sanity-check label values
    rng = random.Random(args.seed)
    sample = rng.sample(stems, min(len(stems), max(args.stats, 20)))
    counts = np.zeros(256, np.int64)
    for s in sample[: max(args.stats, 20)]:
        lab = cv2.imread(str(labels[s]), cv2.IMREAD_UNCHANGED)
        if lab is None:
            sys.exit(f"Could not read label {labels[s]}")
        if lab.ndim == 3:
            lab = lab[..., 0]
        counts += np.bincount(lab.ravel(), minlength=256)
    bad = [v for v in np.nonzero(counts)[0] if v >= len(names) and v != 255]
    if bad:
        print(f"  [warn] unexpected label values {bad}; they will be treated as ignore")

    # block split
    n_blocks = (len(stems) + args.block - 1) // args.block
    blocks = [stems[i * args.block:(i + 1) * args.block] for i in range(n_blocks)]
    order = list(range(n_blocks))
    rng.shuffle(order)
    val = []
    for b in order:
        if len(val) >= args.val_count:
            break
        val.extend(blocks[b])
    val_set = set(val)
    train = [s for s in stems if s not in val_set]
    if not train:
        sys.exit("Validation split took every frame; lower --val-count")

    out.mkdir(parents=True, exist_ok=True)

    def line(s):
        return f"{images[s].relative_to(root).as_posix()} {labels[s].relative_to(root).as_posix()}"

    (out / "train.txt").write_text("\n".join(line(s) for s in train) + "\n")
    (out / "val.txt").write_text("\n".join(line(s) for s in sorted(val)) + "\n")
    (out / "classes.json").write_text(json.dumps({"names": names, "colors": colors}, indent=1))
    print(f"  wrote {len(train)} train / {len(val)} val to {out}/")

    if args.stats:
        total = counts[: len(names)].sum()
        print(f"\nPixel share over {len(sample)} sampled labels:")
        for i, n in enumerate(names):
            print(f"  {i:2d} {n:<16} {100 * counts[i] / max(total, 1):6.2f}%")
        print(f"     {'(ignore=255)':<16} {100 * counts[255] / max(counts.sum(), 1):6.2f}% of all pixels")


if __name__ == "__main__":
    main()
