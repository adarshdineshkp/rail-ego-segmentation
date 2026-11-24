#!/usr/bin/env python
"""Export a trained checkpoint to ONNX (e.g. for TensorRT on a Jetson).

    python tools/export_onnx.py --ckpt runs/rs19_pidnet_s_rails/best.pt --size 1088 1920
    trtexec --onnx=runs/rs19_pidnet_s_rails/best.onnx --fp16 --saveEngine=pidnet_s.engine

The graph outputs logits at 1/8 resolution (N x C x H/8 x W/8); upsample + argmax afterwards.
Use an input size divisible by 32 (1080 -> 1088 with padding, or resize).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from railseg.checkpoint import load_trained  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=[1088, 1920])
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model, _, classes = load_trained(args.ckpt, torch.device("cpu"), augment=False)
    dummy = torch.randn(1, 3, *args.size)
    out = Path(args.out or Path(args.ckpt).with_suffix(".onnx"))
    torch.onnx.export(model, dummy, str(out), input_names=["image"], output_names=["logits"],
                      opset_version=args.opset, do_constant_folding=True)
    print(f"Exported {out}  input 1x3x{args.size[0]}x{args.size[1]}  classes: {classes['names']}")


if __name__ == "__main__":
    main()
