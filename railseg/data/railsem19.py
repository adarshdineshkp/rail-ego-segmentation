from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .classes import build_label_mapping, load_rs19_classes
from .transforms import train_transform, val_transform


def read_split(list_file, root):
    list_file, root = Path(list_file), Path(root)
    if not list_file.exists():
        raise FileNotFoundError(
            f"{list_file} not found. Run: python tools/prepare_railsem19.py --root {root}")
    pairs = []
    for line in list_file.read_text().splitlines():
        line = line.strip()
        if line:
            img_rel, lab_rel = line.split()
            pairs.append((root / img_rel, root / lab_rel))
    return pairs


class RailSem19(Dataset):
    def __init__(self, data_cfg, loss_cfg, split, lut, train):
        self.root = Path(data_cfg["root"])
        self.pairs = read_split(Path(data_cfg["splits_dir"]) / f"{split}.txt", self.root)
        self.lut = lut
        self.train = train
        self.ignore = int(data_cfg.get("ignore_label", 255))
        self.tcfg = data_cfg.get("train", {})
        self.vcfg = data_cfg.get("val", {})
        self.edge_pad = loss_cfg.get("edge_pad", 6)
        self.edge_dilate = loss_cfg.get("edge_dilate", 4)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lab_path = self.pairs[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        lab = cv2.imread(str(lab_path), cv2.IMREAD_UNCHANGED)
        if img is None or lab is None:
            raise IOError(f"Could not read {img_path} or {lab_path}")
        if lab.ndim == 3:
            lab = lab[..., 0]
        lab = self.lut[lab]
        if self.train:
            img, lab, edge = train_transform(img, lab, self.tcfg, self.ignore,
                                             self.edge_pad, self.edge_dilate)
            return torch.from_numpy(img), torch.from_numpy(lab.astype(np.int64)), torch.from_numpy(edge)
        img, lab = val_transform(img, lab, self.vcfg)
        return torch.from_numpy(img), torch.from_numpy(lab.astype(np.int64)), img_path.stem


def class_info_from_cfg(cfg):
    data_cfg = cfg["data"]
    src_names, src_colors = load_rs19_classes(data_cfg["splits_dir"])
    lut, names, colors = build_label_mapping(data_cfg, src_names, src_colors)
    return lut, {"names": names, "colors": colors}


def build_datasets(cfg):
    lut, info = class_info_from_cfg(cfg)
    train_ds = RailSem19(cfg["data"], cfg["loss"], "train", lut, train=True)
    val_ds = RailSem19(cfg["data"], cfg["loss"], "val", lut, train=False)
    return train_ds, val_ds, info
