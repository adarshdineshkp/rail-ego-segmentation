#!/usr/bin/env python
"""Train PIDNet on RailSem19.

    python train.py --config configs/railsem19_pidnet_s_rails.yaml
    python train.py --config ... train.batch_size=4 train.epochs=50      # overrides
    python train.py --config ... --resume auto                           # continue last run
    torchrun --nproc_per_node=2 train.py --config ...                    # multi-GPU
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from railseg.config import load_config, save_config
from railseg.data import build_datasets
from railseg.engine import evaluate
from railseg.losses import PIDNetLoss
from railseg.metrics import mean_iou_of
from railseg.models import build_model, load_pretrained
from railseg.utils import Logger, format_iou_table, init_distributed, is_main, safe_torch_load, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="path to a last.pt, or 'auto' for <run dir>/last.pt")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. train.batch_size=4")
    return p.parse_args()


def lr_at(it, total, cfg):
    base, warm = cfg["lr"], cfg.get("warmup_iters", 0) or 0
    if it < warm:
        return base * (0.1 + 0.9 * it / warm)
    return base * max(0.0, 1 - it / max(1, total)) ** cfg["poly_power"]


def init_weights(model, mcfg, log):
    paths = mcfg.get("pretrained")
    if not paths:
        log("WARNING: no pretrained weights configured; training from scratch converges slowly.")
        return
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        if Path(p).is_file():
            load_pretrained(model, p, skip_head=mcfg.get("skip_head_weights", True), log=log)
            return
    log("WARNING: none of the pretrained files exist: " + ", ".join(paths) +
        "\n         Run `python tools/download_weights.py` first. Training from scratch.")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    distributed, rank, world, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    out_dir = Path(cfg["output_dir"]) / cfg["experiment"]
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    log = Logger(out_dir / "train.log", enabled=is_main())
    tcfg, ecfg = cfg["train"], cfg.get("eval", {})
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    if device.type == "cpu":
        log("WARNING: no CUDA GPU found - running on CPU (fine for the smoke test, far too slow to train).")
    set_seed(int(tcfg.get("seed", 0)) + rank)

    # ------------------------------------------------------------------ data
    train_ds, val_ds, class_info = build_datasets(cfg)
    names = class_info["names"]
    num_classes = len(names)
    ignore = int(cfg["data"].get("ignore_label", 255))
    key_names = tcfg.get("key_classes") or []
    unknown = [k for k in key_names if k not in names]
    if unknown:
        raise SystemExit(f"train.key_classes {unknown} not in classes {names}")
    key_idx = [names.index(k) for k in key_names]

    if is_main():
        save_config({**cfg, "classes": class_info}, out_dir / "config.yaml")
    log(f"Run dir: {out_dir}")
    log(f"Classes ({num_classes}): {', '.join(names)}")
    log(f"Train {len(train_ds)} / val {len(val_ds)} images | device {device} | GPUs {world} | AMP {amp}")

    workers = int(cfg["data"].get("num_workers", 4))
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if distributed else None
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=workers, pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=ecfg.get("batch_size", 1), shuffle=False,
        num_workers=min(workers, 4), pin_memory=device.type == "cuda")
    if len(train_loader) == 0:
        raise SystemExit("Training set is smaller than one batch; lower train.batch_size.")

    # ------------------------------------------------------------------ model
    resume_path = None
    if args.resume == "auto":
        if (out_dir / "last.pt").is_file():
            resume_path = out_dir / "last.pt"
        else:
            log("--resume auto: no last.pt yet, starting a fresh run")
    elif args.resume:
        resume_path = Path(args.resume)

    model = build_model(cfg["model"]["name"], num_classes, augment=True)
    if resume_path is None:
        init_weights(model, cfg["model"], log)
    if distributed and tcfg.get("sync_bn", True):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    raw_model = model
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    log(f"Model {cfg['model']['name']}: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.2f} M params")

    criterion = PIDNetLoss(cfg["loss"], ignore)
    optimizer = torch.optim.SGD(raw_model.parameters(), lr=tcfg["lr"], momentum=tcfg["momentum"],
                                weight_decay=tcfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    epochs = int(tcfg["epochs"])
    iters_per_epoch = len(train_loader)
    total_iters = epochs * iters_per_epoch
    if tcfg.get("max_iters"):
        total_iters = min(total_iters, int(tcfg["max_iters"]))
    start_epoch, global_it, best = 0, 0, -1.0

    if resume_path is not None:
        path = resume_path
        ck = safe_torch_load(path)
        raw_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch, global_it, best = ck["epoch"] + 1, ck["iter"], ck.get("best_score", -1.0)
        log(f"Resumed from {path} at epoch {start_epoch}, iter {global_it}, best {best:.4f}")

    tb = None
    if is_main():
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(out_dir / "tb"))
        except Exception:
            log("(tensorboard not installed - logging to train.log / metrics.csv only)")

    def save(name, epoch):
        torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "iter": global_it,
                    "best_score": best, "cfg": cfg, "classes": class_info}, out_dir / name)

    metrics_csv = out_dir / "metrics.csv"
    log(f"Training {epochs} epochs x {iters_per_epoch} iters = {total_iters} iterations")
    t_start, t_last = time.time(), time.time()

    for epoch in range(start_epoch, epochs):
        if global_it >= total_iters:
            break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for images, labels, edges in train_loader:
            if global_it >= total_iters:
                break
            lr = lr_at(global_it, total_iters, tcfg)
            for g in optimizer.param_groups:
                g["lr"] = lr
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            edges = edges.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model(images)
                loss, parts = criterion(outputs, labels, edges)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_it += 1

            if global_it % tcfg.get("log_every", 50) == 0 or global_it == 1:
                now = time.time()
                n = 1 if global_it == 1 else tcfg.get("log_every", 50)
                sec_it = (now - t_last) / n
                t_last = now
                eta_h = sec_it * (total_iters - global_it) / 3600
                p = {k: v.item() for k, v in parts.items()}
                log(f"ep {epoch + 1}/{epochs} it {global_it}/{total_iters} lr {lr:.5f} "
                    f"loss {loss.item():.4f} (aux {p['aux']:.3f} main {p['main']:.3f} "
                    f"bd {p['bd']:.3f} sb {p['sb']:.3f}) {sec_it:.2f}s/it ETA {eta_h:.1f}h")
                if tb:
                    tb.add_scalar("train/loss", loss.item(), global_it)
                    tb.add_scalar("train/lr", lr, global_it)
                    for k, v in p.items():
                        tb.add_scalar(f"train/{k}", v, global_it)

        finished = epoch == epochs - 1 or global_it >= total_iters
        if (epoch + 1) % tcfg.get("eval_every", 5) == 0 or finished:
            if is_main():
                log(f"Evaluating on {len(val_ds)} val images ...")
                m = evaluate(raw_model, val_loader, num_classes, ignore, device, amp=amp,
                             flip=ecfg.get("flip_test", False), log=log)
                score = mean_iou_of(m["iou"], key_idx) if key_idx else m["miou"]
                log(f"epoch {epoch + 1}: mIoU {100 * m['miou']:.2f}  pixel acc {100 * m['acc']:.2f}"
                    + (f"  key-class mIoU ({', '.join(key_names)}) {100 * score:.2f}" if key_idx else ""))
                log("per-class IoU:\n" + format_iou_table(names, m["iou"], key_idx))
                new_file = not metrics_csv.exists()
                with open(metrics_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if new_file:
                        w.writerow(["epoch", "iter", "miou", "acc", "score"] + names)
                    w.writerow([epoch + 1, global_it, f"{m['miou']:.4f}", f"{m['acc']:.4f}", f"{score:.4f}"]
                               + [f"{v:.4f}" for v in m["iou"]])
                if tb:
                    tb.add_scalar("val/miou", m["miou"], global_it)
                    tb.add_scalar("val/score", score, global_it)
                    for n_, v in zip(names, m["iou"]):
                        if v == v:
                            tb.add_scalar(f"val_iou/{n_}", v, global_it)
                if score > best:
                    best = score
                    save("best.pt", epoch)
                    (out_dir / "best.json").write_text(json.dumps(
                        {"epoch": epoch + 1, "score": score, "miou": m["miou"],
                         "iou": dict(zip(names, m["iou"]))}, indent=1))
                    log(f"New best ({100 * best:.2f}) -> {out_dir / 'best.pt'}")
            if distributed:
                dist.barrier()
        if is_main():
            save("last.pt", epoch)

    log(f"Done in {(time.time() - t_start) / 3600:.2f} h. Best score {100 * max(best, 0):.2f}. "
        f"Checkpoints in {out_dir}")
    if tb:
        tb.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
