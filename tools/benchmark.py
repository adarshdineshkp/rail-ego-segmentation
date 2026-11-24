#!/usr/bin/env python
"""Measure pure model inference time on the GPU (no video decoding, drawing or saving).

    python tools/benchmark.py --ckpt runs/rs19ego_pidnet_s_b4/best.pt
    python tools/benchmark.py --ckpt ... --sizes 1080x1920 720x1280 --iters 300

Reports mean / median / p95 latency per frame and FPS, in FP16 (default) and FP32,
for the model forward pass, plus forward + upsample + argmax (what infer.py needs).
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from railseg.checkpoint import load_trained  # noqa: E402


def run(model, x, out_size, iters, fp16, full):
    times = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        for i in range(iters + 20):
            torch.cuda.synchronize()
            t = time.perf_counter()
            y = model(x)
            if full:
                y = F.interpolate(y.float(), size=out_size, mode="bilinear", align_corners=False).argmax(1)
            torch.cuda.synchronize()
            if i >= 20:  # first iterations are warm-up
                times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return statistics.mean(times), times[len(times) // 2], times[int(0.95 * len(times)) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sizes", nargs="+", default=["1080x1920", "720x1280", "544x960"])
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found.")
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    model, cfg, _ = load_trained(args.ckpt, dev)
    print(f"GPU: {torch.cuda.get_device_name(0)} | model: {cfg['model']['name']} | {args.iters} timed runs each\n")
    print(f"{'input':>10} {'precision':>9} {'what':>22} {'mean ms':>8} {'median':>7} {'p95':>6} {'FPS':>6}")
    for s in args.sizes:
        h, w = (int(v) for v in s.lower().split("x"))
        x = torch.randn(1, 3, h, w, device=dev)
        for fp16 in (True, False):
            for full in (False, True):
                mean, med, p95 = run(model, x, (h, w), args.iters, fp16, full)
                what = "forward + argmax" if full else "forward only"
                print(f"{s:>10} {'FP16' if fp16 else 'FP32':>9} {what:>22} {mean:8.2f} {med:7.2f} {p95:6.2f} {1000 / mean:6.1f}")
    print("\nThese are batch-1 latencies in PyTorch. TensorRT (tools/export_onnx.py + trtexec) is usually faster.")


if __name__ == "__main__":
    main()
