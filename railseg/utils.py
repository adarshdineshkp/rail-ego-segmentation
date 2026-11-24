import os
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.distributed as dist
except ImportError:  # data-only tools (prepare script) work without torch
    torch = None
    dist = None


# --------------------------------------------------------------------------- distributed
def init_distributed():
    """Initialise DDP when launched with torchrun. Returns (distributed, rank, world, local_rank)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        # long timeout: rank 0 runs validation while the other ranks wait at a barrier
        from datetime import timedelta
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        return True, rank, world, local_rank
    return False, 0, 1, 0


def is_main():
    return dist is None or not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# --------------------------------------------------------------------------- misc
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path):
    """torch.load that prefers weights_only=True and falls back for older checkpoint formats."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        print(f"[info] {path}: weights-only load failed, retrying with full unpickling "
              f"(only do this for files from a source you trust)", file=sys.stderr)
        return torch.load(path, map_location="cpu", weights_only=False)


class Logger:
    def __init__(self, path=None, enabled=True):
        self.enabled = enabled
        self.f = open(path, "a", buffering=1) if (enabled and path) else None

    def __call__(self, msg):
        if not self.enabled:
            return
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")


def format_iou_table(names, iou, highlight=()):
    width = max(len(n) for n in names) + 2
    rows = []
    for i, (n, v) in enumerate(zip(names, iou)):
        mark = " *" if i in highlight else ""
        val = "   n/a" if v != v else f"{100 * v:6.2f}"  # v != v -> NaN (class absent in val set)
        rows.append(f"  {n:<{width}}{val}{mark}")
    return "\n".join(rows)


def colorize(label, colors):
    """label HxW int array -> HxWx3 uint8 RGB using `colors` (list of [r,g,b])."""
    lut = np.zeros((256, 3), np.uint8)
    lut[: len(colors)] = np.asarray(colors, np.uint8)
    return lut[label.astype(np.uint8)]


def project_root():
    return Path(__file__).resolve().parent.parent
