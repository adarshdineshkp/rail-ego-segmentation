"""Camera-only collision warning from a segmentation map (NumPy/OpenCV, no torch).

Per frame:
  1. ego corridor   - the ego-track mask, widened row by row to the vehicle's clearance
                      envelope (a train is wider than the gauge)
  2. track width    - a straight-line fit of ego-track width against image row (on flat track
                      the width falls linearly to zero at the horizon); robust far away, where
                      a single row is only a few pixels wide
  3. in-path test   - objects (segmented rail vehicles / people, or detector boxes) whose bottom
                      edge falls inside the corridor, or just above where the ego-track ends
  4. path cut-off   - class-agnostic: if the visible end of the ego-track approaches at about
                      the train's own speed, something static is standing on the track, even
                      if no detector recognises it. (A visibility limit moves with the train.)
  5. distance       - gauge:  Z = f * gauge / fitted_width(row)
                      ground: Z = h / tan(pitch + atan((y - cy) / f))   (flat track, known pose)
  6. alert          - k-of-n persistence and braking-distance thresholds:
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
    cutoff_max_m: float = 180.0     # path cut-off cue only within this range
    cutoff_window: int = 20         # frames used to measure how fast the path end approaches
    cutoff_min_ratio: float = 0.6   # closing speed must be >= this fraction of train speed
    cutoff_min_rows: float = 5.0    # resolution guard, see path_cutoff


@dataclass
class Detection:
    cls: str
    bbox: tuple                     # x0, y0, x1, y1
    bottom_y: int
    in_path: bool
    dist_gauge: float = float("nan")
    dist_ground: float = float("nan")
    source: str = "seg"


@dataclass
class State:
    hist: deque = field(default_factory=lambda: deque(maxlen=5))
    dists: deque = field(default_factory=lambda: deque(maxlen=5))
    path_end: deque = field(default_factory=lambda: deque(maxlen=30))   # (t, distance of ego-track end)


def ego_rows(ego):
    """Per image row: (left, right) column of the ego-track mask."""
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


def fit_width(rows, min_w=4):
    """Fit width = a*y + b over the ego-track rows (two passes, dropping outliers).
    Returns (a, b) or None."""
    if len(rows) < 20:
        return None
    y = np.array(sorted(rows), dtype=np.float64)
    w = np.array([rows[int(v)][1] - rows[int(v)][0] + 1 for v in y], dtype=np.float64)
    keep = w >= min_w
    y, w = y[keep], w[keep]
    if len(y) < 20:
        return None
    a, b = np.polyfit(y, w, 1)
    res = np.abs(w - (a * y + b)) / np.maximum(a * y + b, 1)
    inl = res < 0.25
    if inl.sum() >= 20:
        a, b = np.polyfit(y[inl], w[inl], 1)
    return (float(a), float(b)) if a > 0 else None


def width_at(fit, y):
    return fit[0] * y + fit[1] if fit else float("nan")


def gauge_distance(fit, y, p):
    w = width_at(fit, y)
    return p.f_px * p.gauge_m / w if fit and w > 1.0 else float("nan")


def ground_distance(y, p):
    ang = math.radians(p.pitch_deg) + math.atan((y - p.cy) / p.f_px)
    return p.cam_height_m / math.tan(ang) if ang > 1e-4 else float("inf")


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
    return l - extra, r + extra


def _in_corridor(rows, top_y, x0, x1, yb, p):
    corr = corridor_at(rows, yb, p, top_y)
    return corr is not None and x1 >= corr[0] and x0 <= corr[1]


def analyse(label, class_ids, p):
    """label: HxW class map. class_ids: name -> id (needs 'ego-track'; uses 'on-rails', 'human',
    'obstacle' when present). Returns (detections, ego rows, width fit)."""
    ego = label == class_ids["ego-track"]
    rows = ego_rows(ego)
    fit = fit_width(rows)
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
            band = cc[max(yb - 3, y): yb + 1, x: x + w] == i   # the object's footprint
            cols = np.flatnonzero(band.any(axis=0)) + x
            in_path = cols.size > 0 and _in_corridor(rows, top_y, cols.min(), cols.max(), yb, p)
            dets.append(Detection(cname, (int(x), int(y), int(x + w), int(y + h)), yb, bool(in_path),
                                  gauge_distance(fit, yb, p), ground_distance(yb, p), "seg"))
    return dets, rows, fit


def detections_from_boxes(boxes, rows, fit, p, source="yolo"):
    """boxes: [(x0, y0, x1, y1, class_name, score)] from any detector, e.g. your YOLO models."""
    top_y = min(rows) if rows else None
    out = []
    for x0, y0, x1, y1, name, score in boxes:
        yb = int(y1)
        inset = 0.25 * (x1 - x0)  # use the middle of the box bottom as the footprint
        in_path = _in_corridor(rows, top_y, x0 + inset, x1 - inset, yb, p)
        out.append(Detection(f"{name} {score:.2f}", (int(x0), int(y0), int(x1), int(y1)), yb, bool(in_path),
                             gauge_distance(fit, yb, p), ground_distance(yb, p), source))
    return out


def path_cutoff(rows, fit, state, t, speed_mps, p):
    """Class-agnostic blocked-path cue. Returns a Detection or None."""
    if not rows or not fit:
        state.path_end.clear()
        return None
    top_y = min(rows)
    z = gauge_distance(fit, top_y, p)
    if not math.isfinite(z):
        return None
    state.path_end.append((t, z))
    pts = list(state.path_end)[-p.cutoff_window:]
    if len(pts) < p.cutoff_window or speed_mps <= 0:
        return None
    # Only trust the cue where the path end, approaching at train speed, moves by at least
    # cutoff_min_rows image rows over the window (one row spans ~Z^2/(h*f) metres at range Z).
    span_s = pts[-1][0] - pts[0][0]
    z_res = math.sqrt(speed_mps * span_s * p.cam_height_m * p.f_px / p.cutoff_min_rows)
    if z > min(p.cutoff_max_m, z_res):
        return None
    ts, zs = np.array(pts).T
    # Theil-Sen slope over pairs at least half a window apart: robust to the one-row jumps of
    # the path end far away (one pixel row is several metres at 150 m+)
    n = len(ts)
    slopes = [(zs[j] - zs[i]) / (ts[j] - ts[i]) for i in range(n) for j in range(i + n // 2, n)
              if ts[j] > ts[i]]
    slope = float(np.median(slopes))            # m/s; a static blockage gives about -speed
    if -slope < p.cutoff_min_ratio * speed_mps:
        return None
    l, r = rows[top_y]
    pad = max(r - l, 4)
    return Detection("path-blocked", (int(l - pad), int(top_y - 3 * pad), int(r + pad), int(top_y)), top_y, True,
                     z, ground_distance(top_y, p), "cutoff")


def stop_distance(v, p):
    return v * p.reaction_s + v * v / (2 * p.decel_mps2)


def decide(dets, speed_mps, state, p):
    """Returns (level, nearest in-path detection or None, smoothed distance, ttc, stop distance)."""
    def dist_of(d):
        v = d.dist_gauge if p.distance_source == "gauge" else d.dist_ground
        if not math.isfinite(v):
            v = d.dist_ground if p.distance_source == "gauge" else d.dist_gauge
        return v if math.isfinite(v) else 1e9

    in_path = [d for d in dets if d.in_path]
    near = min(in_path, key=dist_of, default=None)
    if state.hist.maxlen != p.persist_n:
        state.hist = deque(state.hist, maxlen=p.persist_n)
    state.hist.append(near is not None)
    confirmed = sum(state.hist) >= p.persist_k
    dist = float("nan")
    if near is not None:
        state.dists.append(dist_of(near))
        dist = float(np.median(state.dists))
    else:
        state.dists.clear()
    stop = stop_distance(speed_mps, p)
    ttc = dist / speed_mps if speed_mps > 0 and math.isfinite(dist) else float("inf")
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
SRC_COLORS = {"seg": (0, 0, 255), "yolo": (255, 0, 255), "cutoff": (0, 255, 255)}


def draw_hud(img, level, near, dist, ttc, stop, speed, true_dist=None, dets=()):
    """Draw alert banner, boxes and numbers onto a BGR image (in place)."""
    h, w = img.shape[:2]
    for d in dets:
        col = SRC_COLORS.get(d.source, (0, 0, 255)) if d.in_path else (200, 200, 200)
        cv2.rectangle(img, d.bbox[:2], d.bbox[2:], col, 3 if d.in_path else 1)
        if d.in_path:
            cv2.putText(img, d.cls, (d.bbox[0], max(d.bbox[1] - 8, 130)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.rectangle(img, (0, 0), (w, 110), COLORS[level], -1)
    cv2.putText(img, level, (30, 80), cv2.FONT_HERSHEY_DUPLEX, 2.4, (255, 255, 255), 4)
    info = f"speed {speed * 3.6:.0f} km/h   stop dist {stop:.0f} m"
    if near is not None:
        info = f"{near.cls} in path: {dist:.0f} m   TTC {ttc:.1f} s   " + info
    cv2.putText(img, info, (430, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    if true_dist is not None:
        err = f"   (estimate error {dist - true_dist:+.1f} m)" if near is not None and math.isfinite(dist) else ""
        cv2.putText(img, f"true distance {true_dist:.0f} m{err}", (430, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img
