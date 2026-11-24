"""Augmentations and target generation (pure NumPy / OpenCV)."""
import random

import cv2
import numpy as np

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers

MEAN_RGB = np.array([0.485, 0.456, 0.406], np.float32)
STD_RGB = np.array([0.229, 0.224, 0.225], np.float32)
PAD_BGR = tuple(float(v) for v in (MEAN_RGB[::-1] * 255))


def random_scale(img, lab, lo, hi):
    s = random.uniform(lo, hi)
    h, w = lab.shape
    nh, nw = max(1, int(h * s + 0.5)), max(1, int(w * s + 0.5))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lab = cv2.resize(lab, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img, lab


def pad_and_random_crop(img, lab, crop_h, crop_w, ignore):
    h, w = lab.shape
    ph, pw = max(crop_h - h, 0), max(crop_w - w, 0)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=PAD_BGR)
        lab = cv2.copyMakeBorder(lab, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=int(ignore))
    h, w = lab.shape
    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)
    return img[y:y + crop_h, x:x + crop_w], lab[y:y + crop_h, x:x + crop_w]


def color_jitter(img, strength):
    """Random brightness / contrast / saturation on a BGR image."""
    if strength <= 0:
        return img
    img = img.astype(np.float32)
    img *= 1.0 + random.uniform(-strength, strength)
    mean = img.mean()
    img = (img - mean) * (1.0 + random.uniform(-strength, strength)) + mean
    gray = img.mean(axis=2, keepdims=True)
    img = (img - gray) * (1.0 + random.uniform(-strength, strength)) + gray
    return np.clip(img, 0, 255)


def make_edge(lab, pad=6, dilate=4):
    """Boundary target for PIDNet's D branch (same recipe as the official code)."""
    edge = cv2.Canny(np.ascontiguousarray(lab), 0.1, 0.2)
    if pad > 0:
        edge[:pad], edge[-pad:], edge[:, :pad], edge[:, -pad:] = 0, 0, 0, 0
    kernel = np.ones((dilate, dilate), np.uint8)
    return (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.float32)


def normalize(img_bgr):
    """BGR uint8/float HxWx3 -> normalised RGB float32 3xHxW."""
    img = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - MEAN_RGB) / STD_RGB
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def train_transform(img, lab, tcfg, ignore, edge_pad=6, edge_dilate=4):
    lo, hi = tcfg.get("scale_range", [0.75, 1.5])
    img, lab = random_scale(img, lab, lo, hi)
    ch, cw = tcfg.get("crop_size", [768, 1024])
    img, lab = pad_and_random_crop(img, lab, ch, cw, ignore)
    if tcfg.get("hflip", True) and random.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    img = color_jitter(img, tcfg.get("color_jitter", 0.0))
    lab = np.ascontiguousarray(lab)
    edge = make_edge(lab, edge_pad, edge_dilate)
    return normalize(img), lab, edge


def val_transform(img, lab, vcfg):
    size = vcfg.get("resize")  # [h, w] or None
    if size:
        h, w = size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
    return normalize(img), np.ascontiguousarray(lab)
