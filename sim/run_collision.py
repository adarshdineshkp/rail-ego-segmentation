#!/usr/bin/env python
"""Run segmentation + collision warning on a rendered (or recorded) frame sequence.

    python sim/run_collision.py --run sim_out/run1 --ckpt runs/rs19ego_pidnet_s_b4/best.pt --watch --show
    python sim/run_collision.py ... --yolo yolo/rail_equipment.pt yolo/human.pt     # add your detectors

--watch  processes frames as Blender writes them (stops when done.txt appears)
--show   live window (q / Esc to stop)
--yolo   one or more Ultralytics YOLO weight files (pip install ultralytics); their boxes go
         through the same in-path test as the segmentation

Writes into the run folder: collision.mp4 (frames with overlay + alert banner),
collision_log.csv (per frame), summary.json (alerts vs ground truth, verdict).
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
from collision import (Params, State, analyse, decide, detections_from_boxes, draw_hud,  # noqa: E402
                       path_cutoff, stop_distance)
from infer import overlay, segment  # noqa: E402
from railseg.checkpoint import load_trained  # noqa: E402


def read_gt(run):
    gt = {}
    p = run / "gt.csv"
    if p.exists():
        for row in csv.DictReader(open(p)):
            gt[int(row["frame"])] = row
    return gt


def load_yolo(paths):
    if not paths:
        return []
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("--yolo needs the ultralytics package:  pip install ultralytics")
    models = []
    for p in paths:
        m = YOLO(p)
        print(f"YOLO {p}: classes {list(m.names.values())[:12]}{' ...' if len(m.names) > 12 else ''}")
        models.append(m)
    return models


def yolo_boxes(models, img, device, conf):
    out = []
    for m in models:
        r = m.predict(img, conf=conf, verbose=False, device=0 if device.type == "cuda" else "cpu")[0]
        for (x0, y0, x1, y1), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            out.append((x0, y0, x1, y1, m.names[int(c)], s))
    return out


def fmt(v):
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="folder with frames/, meta.json (and gt.csv)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--yolo", nargs="*", default=[])
    ap.add_argument("--yolo-conf", type=float, default=0.35)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-cutoff", action="store_true", help="disable the class-agnostic path cut-off cue")
    ap.add_argument("--distance", choices=["gauge", "ground"], default="gauge")
    ap.add_argument("--speed", type=float, default=None, help="m/s, if meta.json has none")
    args = ap.parse_args()

    run = Path(args.run)
    frames_dir = run / "frames"
    t_wait = time.time()
    while not (run / "meta.json").exists():
        if not args.watch or time.time() - t_wait > 600:
            sys.exit(f"{run / 'meta.json'} not found")
        time.sleep(1)
    time.sleep(0.5)
    meta = json.loads((run / "meta.json").read_text())
    speed = args.speed or meta.get("speed_mps", 0.0)
    fps = meta.get("fps", 10)
    p = Params(f_px=meta["f_px"], cy=meta["cy"], cam_height_m=meta.get("cam_height_above_rail_m", 3.0),
               pitch_deg=meta.get("pitch_deg", 0.0), distance_source=args.distance)
    in_route = meta.get("obstacle", {}).get("in_route", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, classes = load_trained(args.ckpt, device)
    names = classes["names"]
    if "ego-track" not in names:
        sys.exit(f"Checkpoint has no ego-track class: {names}")
    ids = {n: i for i, n in enumerate(names)}
    colors = classes["colors"]
    skip = ids.get("background")
    yolos = load_yolo(args.yolo)
    print(f"Scenario: {meta.get('obstacle', {}).get('scenario', '?')} | expected: {meta.get('expected', '?')}")
    print(f"Device: {device} | speed {speed} m/s | distance from {args.distance} | "
          f"cut-off cue {'off' if args.no_cutoff else 'on'} | YOLO models: {len(yolos)}")

    if args.show:
        cv2.namedWindow("collision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("collision", 1280, 720)
    writer = None
    state = State()
    log = open(run / "collision_log.csv", "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["frame", "level", "in_path", "source", "est_distance_m", "dist_gauge_m", "dist_ground_m",
                 "true_distance_m", "ttc_s", "stop_distance_m", "model_ms"])
    first, levels = {}, []
    errs = {"gauge": [], "ground": []}
    gt = read_gt(run)
    i, t_last_new = 0, time.time()
    try:
        while True:
            fp = frames_dir / f"{i:06d}.jpg"
            if not fp.exists():
                if (run / "done.txt").exists() or not args.watch or time.time() - t_last_new > 300:
                    break
                time.sleep(0.2)
                if args.show:
                    cv2.waitKey(1)
                continue
            t_last_new = time.time()
            img = cv2.imread(str(fp))
            if img is None:
                time.sleep(0.2)
                continue
            if i not in gt:
                gt = read_gt(run)
            td = gt.get(i, {}).get("true_distance_m", "")
            true_d = float(td) if td not in ("", None) else None

            t0 = time.perf_counter()
            label = segment(model, img, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_ms = (time.perf_counter() - t0) * 1000
            dets, rows, fit = analyse(label, ids, p)
            if yolos:
                dets += detections_from_boxes(yolo_boxes(yolos, img, device, args.yolo_conf), rows, fit, p)
            if not args.no_cutoff:
                cut = path_cutoff(rows, fit, state, i / fps, speed, p)
                if cut is not None:
                    dets.append(cut)
            level, near, dist, ttc, stop = decide(dets, speed, state, p)
            levels.append(level)

            vis = overlay(img, label, colors, 0.45, skip)
            draw_hud(vis, level, near, dist, ttc, stop, speed, true_d if in_route else None, dets)
            if not in_route and true_d is not None:
                cv2.putText(vis, f"vehicle {true_d:.0f} m away, NOT on the train's route",
                            (430, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(vis, f"model {model_ms:.0f} ms", (30, vis.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(run / "collision.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(vis)
            if args.show:
                cv2.imshow("collision", vis)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("stopped from the window")
                    break

            if level not in first and level in ("TRACKING", "WARNING", "BRAKE"):
                first[level] = {"frame": i, "time_s": round(i / fps, 1), "est_distance_m": round(dist, 1),
                                "true_distance_m": true_d, "source": near.source, "what": near.cls}
                print(f"  frame {i}: first {level} ({near.cls}, {near.source}) at est. {dist:.0f} m"
                      + (f" (true {true_d:.0f} m)" if true_d is not None else ""))
            if near is not None and true_d is not None and in_route:
                for k, v in (("gauge", near.dist_gauge), ("ground", near.dist_ground)):
                    if math.isfinite(v):
                        errs[k].append(v - true_d)
            lw.writerow([i, level, near.cls if near else "", near.source if near else "",
                         fmt(dist if near else None), fmt(near.dist_gauge if near else None),
                         fmt(near.dist_ground if near else None), fmt(true_d), fmt(ttc if near else None),
                         f"{stop:.1f}", f"{model_ms:.1f}"])
            if i % 20 == 0:
                print(f"  frame {i}: {level}" + (f", {near.cls} at {dist:.0f} m" if near else "")
                      + f", model {model_ms:.0f} ms")
            i += 1
    except KeyboardInterrupt:
        print("stopped with Ctrl+C")
    finally:
        log.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    summary = {"scenario": meta.get("obstacle", {}).get("scenario"), "expected": meta.get("expected"),
               "frames": i, "speed_mps": speed, "stop_distance_m": round(stop_distance(speed, p), 1),
               "first_alerts": first,
               "frames_per_level": {lv: levels.count(lv) for lv in ("CLEAR", "TRACKING", "WARNING", "BRAKE")}}
    for k, e in errs.items():
        if e:
            a = np.abs(np.array(e))
            summary[f"{k}_distance_error_m"] = {"mean_abs": round(float(a.mean()), 1),
                                                "median_abs": round(float(np.median(a)), 1),
                                                "mean_signed": round(float(np.mean(e)), 1)}
    if in_route:
        if "BRAKE" in first and first["BRAKE"]["true_distance_m"] is not None:
            margin = first["BRAKE"]["true_distance_m"] - summary["stop_distance_m"]
            summary["brake_margin_m"] = round(margin, 1)
            summary["verdict"] = ("PASS: stops in time" if margin >= 0 else "FAIL: too late") + \
                f" (braking needs {summary['stop_distance_m']} m, BRAKE at {first['BRAKE']['true_distance_m']} m)"
        else:
            summary["verdict"] = "FAIL: no BRAKE alert was raised"
    else:
        bad = summary["frames_per_level"]["WARNING"] + summary["frames_per_level"]["BRAKE"]
        summary["verdict"] = ("PASS: no false alarm" if bad == 0 else
                              f"FAIL: false alarm on {bad} frames (vehicle is not on the train's route)")
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nSummary:\n" + json.dumps(summary, indent=1))
    print(f"\nVideo: {run / 'collision.mp4'}")


if __name__ == "__main__":
    main()
