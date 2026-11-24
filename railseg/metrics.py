import math

import torch


class ConfusionMatrix:
    def __init__(self, num_classes, ignore_label=255, device="cpu"):
        self.n = num_classes
        self.ignore = ignore_label
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred, target):
        mask = (target != self.ignore) & (target >= 0) & (target < self.n)
        idx = target[mask].long() * self.n + pred[mask].long()
        self.mat += torch.bincount(idx, minlength=self.n * self.n).view(self.n, self.n)

    def compute(self):
        m = self.mat.double()
        tp = m.diag()
        denom = m.sum(0) + m.sum(1) - tp
        iou = torch.where(denom > 0, tp / denom.clamp(min=1), torch.full_like(tp, float("nan")))
        valid = iou[~torch.isnan(iou)]
        miou = valid.mean().item() if valid.numel() else 0.0
        acc = (tp.sum() / m.sum().clamp(min=1)).item()
        return {"miou": miou, "acc": acc, "iou": iou.cpu().tolist()}


def mean_iou_of(iou, indices):
    vals = [iou[i] for i in indices if not math.isnan(iou[i])]
    return sum(vals) / len(vals) if vals else 0.0
