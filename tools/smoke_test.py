#!/usr/bin/env python
"""End-to-end check that the environment works, in a minute or two, without RailSem19.

    python tools/smoke_test.py                       # model + tiny train/eval/infer run
    python tools/smoke_test.py --weights pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt

1. builds PIDNet-S/M/L and checks output shapes
2. optionally checks that pretrained weights load (how many tensors match)
3. writes a tiny fake dataset in RailSem19 layout, then runs prepare -> train -> evaluate -> infer
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from railseg.models import VARIANTS, build_model, load_pretrained  # noqa: E402


def check_models():
    x = torch.randn(2, 3, 256, 384)
    for name in VARIANTS:
        m = build_model(name, 6, augment=True).train()
        p, main, d = m(x)
        assert p.shape == (2, 6, 32, 48) and main.shape == (2, 6, 32, 48) and d.shape == (2, 1, 32, 48), \
            (p.shape, main.shape, d.shape)
        (p.mean() + main.mean() + d.mean()).backward()
        m.eval().augment = False
        with torch.no_grad():
            out = m(torch.randn(1, 3, 270, 480))  # odd sizes like 1080p / 4
        assert out.shape[:2] == (1, 6), out.shape
        n = sum(t.numel() for t in m.parameters()) / 1e6
        print(f"  [ok] {name}: {n:.2f} M params, train+eval forward/backward fine")


def make_fake_rs19(root, n=12, h=216, w=384):
    img_dir = root / "rs19_val" / "jpgs" / "rs19_val"
    lab_dir = root / "rs19_val" / "uint8" / "rs19_val"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        lab = np.full((h, w), 10, np.uint8)                 # sky
        lab[h // 2:] = 9                                    # terrain
        lab[h // 2:, w // 3: 2 * w // 3] = 15               # trackbed
        lab[h // 2:, w // 3 + 20: w // 3 + 26] = 17         # rail-raised
        lab[h // 2:, 2 * w // 3 - 26: 2 * w // 3 - 20] = 17
        lab[:20, :20] = 255
        img = (rng.rand(h, w, 3) * 60).astype(np.uint8) + (lab[..., None] * 9).astype(np.uint8)
        cv2.imwrite(str(img_dir / f"rs{i:05d}.jpg"), img)
        cv2.imwrite(str(lab_dir / f"rs{i:05d}.png"), lab)


def run(cmd):
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="pretrained file to test-load into PIDNet-S")
    ap.add_argument("--model", default="pidnet_s")
    args = ap.parse_args()
    py = sys.executable

    print("1) model shapes")
    check_models()

    if args.weights:
        print("2) pretrained weights")
        m = build_model(args.model, 19)
        n, total = load_pretrained(m, args.weights, skip_head=False)
        if n < 0.95 * total:
            sys.exit(f"Only {n}/{total} tensors matched - wrong file or model variant?")
        print("  [ok] weights match the architecture")

    print("3) tiny end-to-end run on fake data (CPU or GPU)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data, splits, runs = tmp / "rs19", tmp / "splits", tmp / "runs"
        make_fake_rs19(data)
        run([py, "tools/prepare_railsem19.py", "--root", str(data), "--out", str(splits),
             "--val-count", "3", "--block", "3", "--stats", "0"])
        common = [f"data.root={data}", f"data.splits_dir={splits}", f"output_dir={runs}",
                  "data.num_workers=0", "data.train.crop_size=[128,192]", "train.batch_size=2",
                  "train.epochs=2", "train.max_iters=6", "train.eval_every=1", "train.log_every=2",
                  "train.warmup_iters=2", "loss.ohem_min_kept=1000", "model.pretrained=null",
                  f"model.name={args.model}"]
        run([py, "train.py", "--config", "configs/railsem19_pidnet_s_rails.yaml"] + common)
        ckpt = runs / "rs19_pidnet_s_rails" / "best.pt"
        assert ckpt.is_file(), "best.pt was not written"
        run([py, "train.py", "--config", "configs/railsem19_pidnet_s_rails.yaml", "--resume", "auto"]
            + common + ["train.epochs=3", "train.max_iters=9"])
        run([py, "evaluate.py", "--ckpt", str(ckpt), "--save-vis", str(tmp / "vis"), "--max-vis", "2",
             f"data.root={data}", f"data.splits_dir={splits}"])
        run([py, "infer.py", "--ckpt", str(ckpt), "--input", str(data / "rs19_val" / "jpgs" / "rs19_val"),
             "--out", str(tmp / "preds")])
        assert any((tmp / "preds").glob("*_label.png")), "infer wrote no labels"
    print("\nSMOKE TEST PASSED - the environment is ready. Next: scripts/train.sh")


if __name__ == "__main__":
    main()
