"""PIDNet training loss, following the official recipe:
  0.4 * CE(P-branch) + 1.0 * OHEM-CE(main) + 20 * weighted-BCE(boundary)
  + 1.0 * OHEM-CE(main, only where the boundary head is confident)  ("SB" loss)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.9, min_kept=131072):
        super().__init__()
        self.ignore = ignore_label
        self.thresh = thresh
        self.min_kept = max(1, int(min_kept))

    def forward(self, logits, target):
        logits = logits.float()
        pixel_losses = F.cross_entropy(logits, target, ignore_index=self.ignore,
                                       reduction="none").view(-1)
        mask = target.view(-1) != self.ignore
        if not mask.any():
            return logits.sum() * 0.0
        with torch.no_grad():
            prob = F.softmax(logits, dim=1)
            tmp = target.clone()
            tmp[tmp == self.ignore] = 0
            prob = prob.gather(1, tmp.unsqueeze(1)).view(-1)[mask]
            sorted_prob, _ = prob.sort()
            k = min(self.min_kept, sorted_prob.numel() - 1)
            threshold = max(sorted_prob[k].item(), self.thresh)
            keep = prob < threshold
        valid = pixel_losses[mask]
        if not keep.any():
            return valid.mean()
        return valid[keep].mean()


def weighted_boundary_bce(logit, target):
    """Class-balanced BCE for the boundary (D) head. logit: N1HW, target: NHW in {0,1}."""
    logit = logit[:, 0].float()
    target = target.float()
    pos = target == 1
    neg = target == 0
    n_pos = pos.sum().float()
    n_neg = neg.sum().float()
    total = (n_pos + n_neg).clamp(min=1.0)
    weight = torch.zeros_like(target)
    weight[pos] = n_neg / total
    weight[neg] = n_pos / total
    return F.binary_cross_entropy_with_logits(logit, target, weight, reduction="mean")


class PIDNetLoss(nn.Module):
    def __init__(self, cfg, ignore_label=255, align_corners=False):
        super().__init__()
        self.ignore = ignore_label
        self.align_corners = align_corners
        self.aux_w = cfg["aux_weight"]
        self.main_w = cfg["main_weight"]
        self.bd_w = cfg["boundary_weight"]
        self.sb_w = cfg["sb_weight"]
        self.sb_thresh = cfg["sb_thresh"]
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_label)
        if cfg.get("ohem", True):
            self.main_loss = OhemCrossEntropy(ignore_label, cfg["ohem_thresh"], cfg["ohem_min_kept"])
        else:
            self.main_loss = lambda x, y: self.ce(x.float(), y)

    def forward(self, outputs, labels, edges):
        p, main, d = outputs
        size = labels.shape[-2:]

        def up(t):
            return F.interpolate(t, size=size, mode="bilinear", align_corners=self.align_corners)

        p, main, d = up(p), up(main), up(d)
        l_aux = self.ce(p.float(), labels)
        l_main = self.main_loss(main, labels)
        l_bd = weighted_boundary_bce(d, edges)
        with torch.no_grad():
            filler = torch.full_like(labels, self.ignore)
            sb_labels = torch.where(torch.sigmoid(d[:, 0].float()) > self.sb_thresh, labels, filler)
        l_sb = self.main_loss(main, sb_labels)
        total = self.aux_w * l_aux + self.main_w * l_main + self.bd_w * l_bd + self.sb_w * l_sb
        parts = {"aux": l_aux.detach(), "main": l_main.detach(), "bd": l_bd.detach(), "sb": l_sb.detach()}
        return total, parts
