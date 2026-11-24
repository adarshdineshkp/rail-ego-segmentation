#!/usr/bin/env python
"""Run a trained model on your own images or video (e.g. locomotive camera footage).

    python infer.py --ckpt runs/rs19_pidnet_s_rails/best.pt --input frame.jpg --out preds/
    python infer.py --ckpt ... --input frames_dir/ --out preds/
    python infer.py --ckpt ... --input drive.mp4 --out preds/          # writes preds/drive_seg.mp4
    (Ctrl+C stops early and still saves a playable video of the frames done so far)

For each image it saves <name>_overlay.jpg and <name>_label.png (class ids, for your switch logic).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from railseg.checkpoint import load_trained
from railseg.data.transforms import normalize
from railseg.engine import predict_logits
from railseg.utils import colorize

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def segment(model, frame_bgr, device, resize=None, flip=False):
    h, w = frame_bgr.shape[:2]
    inp = frame_bgr if not resize else cv2.resize(frame_bgr, (resize[1], resize[0]))
    x = torch.from_numpy(normalize(inp))[None].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = predict_logits(model, x, (h, w), flip)
    return logits.argmax(1)[0].byte().cpu().numpy()


def overlay(frame_bgr, label, colors, alpha=0.5, skip_id=None):
    col = colorize(label, colors)[:, :, ::-1]
    mask = label != skip_id if skip_id is not None else np.ones(label.shape, bool)
    out = frame_bgr.copy()
    out[mask] = cv2.addWeighted(frame_bgr, 1 - alpha, col, alpha, 0)[mask]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="preds")
    ap.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=None,
                    help="resize network input (default: native resolution)")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0, help="stop each video after N frames (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, classes = load_trained(args.ckpt, device)
    colors = classes["colors"]
    skip = classes["names"].index("background") if "background" in classes["names"] else None
    print("Classes:", ", ".join(f"{i}={n}" for i, n in enumerate(classes["names"])))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.input)

    if src.suffix.lower() in VID_EXT:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dst = out / f"{src.stem}_seg.mp4"
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        n, t0 = 0, time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok or (args.max_frames and n >= args.max_frames):
                    break
                lab = segment(model, frame, device, args.resize, args.flip)
                writer.write(overlay(frame, lab, colors, args.alpha, skip))
                n += 1
                if n % 100 == 0:
                    print(f"  {n}/{total or '?'} frames, {n / (time.time() - t0):.1f} fps")
        except KeyboardInterrupt:
            print("  stopped with Ctrl+C - saving what was processed")
        finally:
            cap.release()
            writer.release()  # writes the MP4 index, so a partial video still plays
        print(f"Wrote {dst} ({n} frames)")
        return

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    for p in files:
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"  skip unreadable {p}")
            continue
        lab = segment(model, frame, device, args.resize, args.flip)
        cv2.imwrite(str(out / f"{p.stem}_overlay.jpg"), overlay(frame, lab, colors, args.alpha, skip))
        cv2.imwrite(str(out / f"{p.stem}_label.png"), lab.astype(np.uint8))
    print(f"Wrote predictions for {len(files)} image(s) to {out}/")


if __name__ == "__main__":
    main()
