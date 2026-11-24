#!/usr/bin/env python
"""Build pixel masks from RailSem19's JSON annotations, for copies of the dataset that ship
without the dense `uint8` label maps (e.g. the Mendeley mirror: only jpgs/ + jsons/).

    python tools/rasterize_rs19_json.py --root data/railsem19

Writes <root>/uint8/<frame>.png and <root>/json-classes.json. Classes:
  0 background  1 rail  2 track (area between a rail pair)  3 human  4 on-rails (train cars)
  5 pole  6 fence  7 platform          255 = ignore (rail occluders, guard rails)

Rails come from 'polyline-pair' / 'polyline' annotations. They are drawn with a width that
scales with the local track gauge in the image, so they get thinner towards the horizon.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CLASSES = [
    ("background", [0, 0, 0]),
    ("rail", [255, 40, 40]),
    ("track", [255, 200, 0]),
    ("human", [220, 20, 220]),
    ("on-rails", [0, 120, 255]),
    ("pole", [153, 153, 153]),
    ("fence", [190, 153, 153]),
    ("platform", [70, 130, 70]),
]
ID = {n: i for i, (n, _) in enumerate(CLASSES)}
IGNORE = 255
RAIL_WIDTH_FRAC = 0.06   # rail head width relative to the gauge (~70 mm / 1435 mm, plus margin)


def _pts(v):
    a = np.asarray(v, dtype=np.float32).reshape(-1, 2)
    return a


def _poly(mask, pts, value):
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], int(value))


def _line(mask, pts, value, widths):
    p = np.round(pts).astype(np.int32)
    for i in range(len(p) - 1):
        cv2.line(mask, tuple(map(int, p[i])), tuple(map(int, p[i + 1])), int(value),
                 int(max(1, widths[i])), lineType=cv2.LINE_8)


def _gauge_widths(a, b):
    """Per-point rail width for polyline `a`, from the distance to the paired rail `b`."""
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)
    return np.clip(np.round(d * RAIL_WIDTH_FRAC), 2, 30)


def _fallback_widths(pts, h):
    """Rail width for unpaired polylines: grows linearly from the horizon to the image bottom."""
    y = np.clip(pts[:, 1] / h, 0, 1)
    return np.clip(np.round(2 + 14 * np.clip((y - 0.45) / 0.55, 0, 1)), 2, 16)


def rasterize(ann):
    h, w = int(ann["imgHeight"]), int(ann["imgWidth"])
    m = np.zeros((h, w), np.uint8)
    objs = ann.get("objects", [])

    def polys(*labels):
        return [_pts(o["polygon"]) for o in objs if o.get("label") in labels and "polygon" in o]

    for p in polys("platform"):
        _poly(m, p, ID["platform"])
    for p in polys("fence"):
        _poly(m, p, ID["fence"])
    for p in polys("pole"):
        _poly(m, p, ID["pole"])

    pairs, singles = [], []
    for o in objs:
        if o.get("label") != "rail":
            continue
        if "polyline-pair" in o and len(o["polyline-pair"]) == 2:
            a, b = _pts(o["polyline-pair"][0]), _pts(o["polyline-pair"][1])
            if len(a) >= 2 and len(b) >= 2:
                pairs.append((a, b))
        elif "polyline" in o:
            p = _pts(o["polyline"])
            if len(p) >= 2:
                singles.append(p)

    for a, b in pairs:  # track area first, rails on top
        if np.linalg.norm(a[0] - b[0]) > np.linalg.norm(a[0] - b[-1]):
            b = b[::-1]
        _poly(m, np.concatenate([a, b[::-1]]), ID["track"])
    for a, b in pairs:
        _line(m, a, ID["rail"], _gauge_widths(a, b))
        _line(m, b, ID["rail"], _gauge_widths(b, a))
    for p in singles:
        _line(m, p, ID["rail"], _fallback_widths(p, h))

    for o in objs:
        if o.get("label") == "guard-rail" and "polyline" in o:
            p = _pts(o["polyline"])
            _line(m, p, IGNORE, _fallback_widths(p, h))
    for p in polys("train-car"):
        _poly(m, p, ID["on-rails"])
    for p in polys("person", "person-group"):
        _poly(m, p, ID["human"])
    for p in polys("rail-occluder"):
        _poly(m, p, IGNORE)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/railsem19")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", type=int, default=6, help="save N overlay previews to <root>/../mask_preview")
    args = ap.parse_args()

    root = Path(args.root)
    jsons = sorted(p for p in root.rglob("*.json")
                   if p.name not in ("json-classes.json", "rs19-config.json"))
    images = {p.stem: p for p in root.rglob("*.jpg")}
    if not jsons:
        sys.exit(f"No annotation .json files found under {root}")
    out_dir = root / "uint8"
    out_dir.mkdir(exist_ok=True)
    prev_dir = root.parent / "mask_preview"  # outside root so previews are never taken as training images
    n_done = n_skip = n_bad = 0
    for i, jp in enumerate(jsons):
        dst = out_dir / f"{jp.stem}.png"
        if dst.exists() and not args.overwrite:
            n_skip += 1
            continue
        try:
            ann = json.loads(jp.read_text())
            if "objects" not in ann:
                continue
            m = rasterize(ann)
        except Exception as e:  # keep going on a malformed file
            print(f"  [warn] {jp.name}: {e}")
            n_bad += 1
            continue
        cv2.imwrite(str(dst), m)
        n_done += 1
        if args.preview and n_done <= args.preview and jp.stem in images:
            prev_dir.mkdir(exist_ok=True)
            img = cv2.imread(str(images[jp.stem]))
            lut = np.zeros((256, 3), np.uint8)
            for k, (_, c) in enumerate(CLASSES):
                lut[k] = c[::-1]
            lut[IGNORE] = (255, 255, 255)
            col = lut[m]
            sel = m > 0
            img[sel] = (0.45 * img[sel] + 0.55 * col[sel]).astype(np.uint8)
            cv2.imwrite(str(prev_dir / f"{jp.stem}.jpg"), img)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(jsons)}")

    (root / "json-classes.json").write_text(json.dumps(
        {"labels": [{"name": n, "color": c} for n, c in CLASSES]}, indent=1))
    print(f"Masks: {n_done} written, {n_skip} already present, {n_bad} failed -> {out_dir}")
    if args.preview:
        print(f"Previews: {prev_dir}")


if __name__ == "__main__":
    main()
