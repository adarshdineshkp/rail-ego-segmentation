"""Switch-state reading for the path lock: where the switch is comes from the map (RTK position +
track map), which way it is set comes from the camera (classifier trained by train_switch.py).

    voter = SwitchVoter("runs/switch_cls/best.pt", device)
    box = switch_box(meta, d_ahead)                # image box of the switch blades, from the map
    voter.add(img, box)                            # one frame's vote
    voter.decision                                 # None until confident, then "left" / "right"

Label meaning (RailSem19 convention as used here): the switch is set for the LEFT or RIGHT of its
two branches. For a right-hand turnout, "left" = straight on, "right" = diverging.

Votes are summed as log-odds across frames (each frame clipped, so one overconfident frame cannot
decide alone); the decision is taken once the sum passes `threshold` with at least `min_votes`
frames. Only boxes at least min_width px wide vote (validation: below ~40 px the blade gap is
too small to read and the model falls back on a bias); bigger boxes count for more. After the
decision the model keeps classifying (logged, not voted) so its accuracy can be checked.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from prepare_switch_crops import crop_with_context  # noqa: E402  (same crop as in training)
from train_switch import build, to_tensor  # noqa: E402

GAUGE, RAIL_HALF = 1.435, 0.79       # rail centre line from track centre, m
BLADE_LEN = 9.0                      # toe -> heel, m


def project(meta, fwd, lat, up):
    """Pinhole projection of a point fwd m ahead, lat m to the LEFT, up m above rail top."""
    f, cx, cy = meta["f_px"], meta["cx"], meta["cy"]
    h = meta.get("cam_height_above_rail_m", 3.0) - up
    p = math.radians(meta.get("pitch_deg", 0.0))
    depth = fwd * math.cos(p) + h * math.sin(p)
    down = h * math.cos(p) - fwd * math.sin(p)
    if depth <= 0.5:
        return None
    return cx - f * lat / depth, cy + f * down / depth


def switch_box(meta, d_ahead, margin=0.35):
    """Image box (x0, y0, x1, y1) around the switch blades, toe d_ahead m in front of the camera,
    on a straight approach. None if out of view or too close for the blades to be seen."""
    if d_ahead is None or d_ahead < 6.0:
        return None
    pts = []
    for fwd in (d_ahead - 0.5, d_ahead + BLADE_LEN):
        for lat in (-(RAIL_HALF + margin), RAIL_HALF + margin):
            for up in (0.0, 0.25):
                q = project(meta, fwd, lat, up)
                if q is None:
                    return None
                pts.append(q)
    pts = np.array(pts)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    if x1 < 0 or y1 < 0 or x0 > meta["width"] or y0 > meta["height"]:
        return None
    return float(x0), float(y0), float(min(x1, meta["width"] - 1)), float(min(y1, meta["height"] - 1))


class SwitchVoter:
    def __init__(self, ckpt, device, threshold=4.6, min_votes=5, clip=2.5, min_width=60):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.classes, self.size = ck["classes"], ck["size"]
        self.ctx = ck.get("crop_context", [2.0, 3.0])
        self.model = build(ck["arch"], len(self.classes), pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.device = device
        self.il, self.ir = self.classes.index("left"), self.classes.index("right")
        self.threshold, self.min_votes, self.clip, self.min_width = threshold, min_votes, clip, min_width
        self.reset()

    def reset(self):
        self.score, self.votes, self.decision, self.decided_at, self.history = 0.0, 0, None, None, []

    @torch.no_grad()
    def add(self, img, box, frame=None, d_ahead=None):
        """Classify one frame's crop; returns p(right) for this frame (or None if not usable)."""
        if box is None or box[2] - box[0] < self.min_width:
            return None
        crop = crop_with_context(img, box, *self.ctx)
        x = to_tensor(np.ascontiguousarray(crop), self.size)[None].to(self.device)
        voting = self.decision is None
        prob = torch.softmax(self.model(x).float(), 1)[0].cpu().numpy()
        pl, pr = float(prob[self.il]), float(prob[self.ir])
        lo = float(np.clip(math.log((pr + 1e-6) / (pl + 1e-6)), -self.clip, self.clip))
        wgt = float(np.clip((box[2] - box[0]) / 150.0, 0.4, 1.0))   # far = small = less weight
        if voting:
            self.score += wgt * lo
            self.votes += 1
        self.history.append({"frame": frame, "d_ahead_m": None if d_ahead is None else round(d_ahead, 1),
                             "box_w": round(box[2] - box[0]), "p_right": round(pr / (pl + pr), 3),
                             "score": round(self.score, 2), "voted": voting})
        if self.decision is None and self.votes >= self.min_votes and abs(self.score) >= self.threshold:
            self.decision = "right" if self.score > 0 else "left"
            self.decided_at = {"frame": frame, "d_ahead_m": None if d_ahead is None else round(d_ahead, 1),
                               "votes": self.votes}
        return pr / (pl + pr)

    def single_frame_accuracy(self, truth):
        h = [v for v in self.history if v["p_right"] is not None]
        if not h:
            return None
        ok = [(v["p_right"] > 0.5) == (truth == "right") for v in h]
        return round(sum(ok) / len(ok), 3)

    def accuracy_by_width(self, truth):
        out = {}
        for name, lo, hi in (("60-100px", 60, 100), ("100-200px", 100, 200), (">=200px", 200, 1e9)):
            h = [v for v in self.history if lo <= v["box_w"] < hi]
            if h:
                ok = sum((v["p_right"] > 0.5) == (truth == "right") for v in h)
                out[name] = {"accuracy": round(ok / len(h), 3), "frames": len(h)}
        return out

