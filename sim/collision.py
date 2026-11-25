"""Camera-only collision warning from a segmentation map (NumPy/OpenCV, no torch).

Per frame:
  1. ego corridor   - the ego-track mask, widened row by row to the vehicle's clearance
                      envelope (a train is wider than the gauge)
  2. in-path test   - rail vehicles / people (connected components) whose bottom edge falls
                      inside the corridor, or sits just above where the visible ego-track ends
  3. distance       - two independent estimates:
                        gauge:  Z = f * gauge / (ego-track width in px at that row)
                        ground: Z = h / tan(pitch + atan((y - cy) / f))   (flat track)
  4. alert          - persistence filter (k of last n frames) and braking-distance thresholds:
                        stop distance = v * reaction_time + v^2 / (2 * deceleration)
                        BRAKE if d <= brake_factor * stop, WARNING if d <= warn_factor * stop
"""
import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

LEVELS = ("CLEAR", "TRACKING", "WARNING", "BRAKE")


@dataclass
class Params:
    f_px: float
    cy: float
    cam_height_m: float = 3.0
    pitch_deg: float = 2.0
    gauge_m: float = 1.5           # width the ego-track mask spans (rail to rail)
    envelope_m: float = 3.4        # vehicle clearance width
    min_area_px: int = 60
    gap_rows: int = 40             # object may sit this far above the last ego-track row
    reaction_s: float = 2.0
    decel_mps2: float = 0.7        # service braking
    warn_factor: float = 2.0
    brake_factor: float = 1.2
    persist_k: int = 3
    persist_n: int = 5
    distance_source: str = "gauge"  # "gauge" or "ground"


@dataclass
class Detection:
    cls: str
    bbox: tuple                     # x0, y0, x1, y1
    bottom_y: int
    in_path: bool
    dist_gauge: float = float("nan")
    dist_ground: float = float("nan")


@dataclass
class State:
    hist: deque = field(default_factory=lambda: deque(maxlen=5))
    dists: deque = field(default_factory=lambda: deque(maxlen=5))


def ego_rows(ego):
    """Per image row: (left, right) column of the ego-track mask, or None."""
    rows = {}
    ys, xs = np.nonzero(ego)
    if len(ys) == 0:
        return rows
    order = np.argsort(ys, kind="stable")
    ys, xs = ys[order], xs[order]
    bounds = np.flatnonzero(np.diff(ys)) + 1
    for seg_y, seg_x in zip(np.split(ys, bounds), np.split(xs, bounds)):
        rows[int(seg_y[0])] = (int(seg_x.min()), int(seg_x.max()))
    return rows


def corridor_at(rows, y, p, top_y):
    """Clearance corridor (x0, x1) at row y; above the visible end of the ego-track, the
    topmost rows are extended (the object itself hides the track behind it)."""
    if y in rows:
        l, r = rows[y]
    elif top_y is not None and top_y - p.gap_rows <= y < top_y:
        l, r = rows[top_y]
    else:
        return None
    w = max(r - l, 1)
    extra = w * (p.envelope_m / p.gauge_m - 1) / 2
    return l - extra, r + extra, w


def ground_distance(y, p):
    ang = math.radians(p.pitch_deg) + math.atan((y - p.cy) / p.f_px)
    return p.cam_height_m / math.tan(ang) if ang > 1e-4 else float("inf")


def analyse(label, class_ids, p):
    """label: HxW class map. class_ids: dict name -> id (needs 'ego-track', optional
    'on-rails', 'human', 'obstacle'). Returns (detections, ego_rows)."""
    ego = label == class_ids["ego-track"]
    rows = ego_rows(ego)
    top_y = min(rows) if rows else None
    dets = []
    for cname in ("on-rails", "human", "obstacle"):
        if cname not in class_ids:
            continue
        mask = (label == class_ids[cname]).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < p.min_area_px:
                continue
            yb = int(y + h - 1)
            # the column span of the component's bottom rows decides whether it stands in the corridor
            band = cc[max(yb - 3, y): yb + 1, x: x + w] == i
            cols = np.flatnonzero(band.any(axis=0)) + x
            corr = corridor_at(rows, yb, p, top_y)
            in_path = corr is not None and cols.size > 0 and cols.max() >= corr[0] and cols.min() <= corr[1]
            d = Detection(cname, (int(x), int(y), int(x + w), int(y + h)), yb, bool(in_path))
            if corr is not None:
                d.dist_gauge = p.f_px * p.gauge_m / corr[2]
            d.dist_ground = ground_distance(yb, p)
            dets.append(d)
    return dets, rows


def stop_distance(v, p):
    return v * p.reaction_s + v * v / (2 * p.decel_mps2)


def decide(dets, speed_mps, state, p):
    """Returns (level, nearest in-path detection or None, smoothed distance, ttc, stop distance)."""
    in_path = [d for d in dets if d.in_path]
    key = "dist_gauge" if p.distance_source == "gauge" else "dist_ground"
    near = min(in_path, key=lambda d: (getattr(d, key) if not math.isnan(getattr(d, key)) else 1e9),
               default=None)
    state.hist.append(near is not None)
    if state.hist.maxlen != p.persist_n:
        state.hist = deque(state.hist, maxlen=p.persist_n)
    confirmed = sum(state.hist) >= p.persist_k
    dist = float("nan")
    if near is not None:
        dist = getattr(near, key)
        if math.isnan(dist):
            dist = near.dist_ground
        state.dists.append(dist)
        dist = float(np.median(state.dists))
    stop = stop_distance(speed_mps, p)
    ttc = dist / speed_mps if speed_mps > 0 and not math.isnan(dist) else float("inf")
    if not confirmed or near is None:
        level = "CLEAR"
    elif dist <= p.brake_factor * stop:
        level = "BRAKE"
    elif dist <= p.warn_factor * stop:
        level = "WARNING"
    else:
        level = "TRACKING"
    return level, near, dist, ttc, stop


COLORS = {"CLEAR": (60, 170, 60), "TRACKING": (200, 150, 40), "WARNING": (0, 165, 255), "BRAKE": (0, 0, 230)}


def draw_hud(img, level, near, dist, ttc, stop, speed, true_dist=None, dets=()):
    """Draw alert banner, boxes and numbers onto a BGR image (in place)."""
    h, w = img.shape[:2]
    for d in dets:
        col = (0, 0, 255) if d.in_path else (200, 200, 200)
        cv2.rectangle(img, d.bbox[:2], d.bbox[2:], col, 3 if d.in_path else 1)
    cv2.rectangle(img, (0, 0), (w, 110), COLORS[level], -1)
    cv2.putText(img, level, (30, 80), cv2.FONT_HERSHEY_DUPLEX, 2.4, (255, 255, 255), 4)
    info = f"speed {speed * 3.6:.0f} km/h   stop dist {stop:.0f} m"
    if near is not None:
        info = f"{near.cls} in path: {dist:.0f} m   TTC {ttc:.1f} s   " + info
    cv2.putText(img, info, (430, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    if true_dist is not None:
        err = f"   (estimate error {dist - true_dist:+.1f} m)" if near is not None else ""
        cv2.putText(img, f"true distance {true_dist:.0f} m{err}", (430, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img
