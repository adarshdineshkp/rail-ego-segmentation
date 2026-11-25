#!/usr/bin/env python
"""Run segmentation + collision warning on a rendered (or recorded) frame sequence.

    python sim/run_collision.py --run sim_out/run1 --ckpt runs/rs19ego_pidnet_s_b4/best.pt --watch --show

--watch  processes frames as Blender writes them (stops when done.txt appears)
--show   live window (q / Esc to stop)

Writes into the run folder: collision.mp4 (frames with overlay + alert banner),
collision_log.csv (per frame), summary.json (when alerts fired, distance error vs ground truth).
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
from collision import Params, State, analyse, decide, draw_hud, stop_distance  # noqa: E402
from infer import overlay, segment  # noqa: E402
from railseg.checkpoint import load_trained  # noqa: E402


def read_gt(run):
    gt = {}
    p = run / "gt.csv"
    if p.exists():
        for row in csv.DictReader(open(p)):
            gt[int(row["frame"])] = row
    return gt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="folder with frames/, meta.json (and gt.csv)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--show", action="store_true")
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
    meta = json.loads((run / "meta.json").read_text())
    speed = args.speed or meta.get("speed_mps", 0.0)
    p = Params(f_px=meta["f_px"], cy=meta["cy"], cam_height_m=meta.get("cam_height_above_rail_m", 3.0),
               pitch_deg=meta.get("pitch_deg", 0.0), distance_source=args.distance)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, classes = load_trained(args.ckpt, device)
    names = classes["names"]
    if "ego-track" not in names:
        sys.exit(f"Checkpoint has no ego-track class: {names}")
    ids = {n: i for i, n in enumerate(names)}
    colors = classes["colors"]
    skip = ids.get("background")
    print(f"Model classes: {names}\nDevice: {device} | speed {speed} m/s | distance from {args.distance}")

    if args.show:
        cv2.namedWindow("collision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("collision", 1280, 720)
    writer = None
    state = State()
    log = open(run / "collision_log.csv", "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["frame", "level", "in_path_class", "est_distance_m", "dist_gauge_m", "dist_ground_m",
                 "true_distance_m", "ttc_s", "stop_distance_m", "infer_ms"])
    first = {}
    errs = {"gauge": [], "ground": []}
    gt = read_gt(run)
    i, t_last_new = 0, time.time()
    try:
        while True:
            fp = frames_dir / f"{i:06d}.jpg"
            if not fp.exists():
                done = (run / "done.txt").exists()
                if not args.watch or (done and not fp.exists()) or time.time() - t_last_new > 300:
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
            gt = gt if i in gt else read_gt(run)
            true_d = float(gt[i]["true_distance_m"]) if i in gt else None

            t0 = time.perf_counter()
            label = segment(model, img, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000
            dets, _ = analyse(label, ids, p)
            level, near, dist, ttc, stop = decide(dets, speed, state, p)

            vis = overlay(img, label, colors, 0.45, skip)
            draw_hud(vis, level, near, dist, ttc, stop, speed, true_d, dets)
            cv2.putText(vis, f"model {infer_ms:.0f} ms", (30, vis.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(run / "collision.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         meta.get("fps", 10), (w, h))
            writer.write(vis)
            if args.show:
                cv2.imshow("collision", vis)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("stopped from the window")
                    break

            if level not in first and level in ("TRACKING", "WARNING", "BRAKE"):
                first[level] = {"frame": i, "time_s": i / meta.get("fps", 10), "est_distance_m": round(dist, 1),
                                "true_distance_m": true_d}
                print(f"  frame {i}: first {level} at est. {dist:.0f} m"
                      + (f" (true {true_d:.0f} m)" if true_d is not None else ""))
            if near is not None and true_d is not None:
                for k, v in (("gauge", near.dist_gauge), ("ground", near.dist_ground)):
                    if not math.isnan(v) and math.isfinite(v):
                        errs[k].append(v - true_d)
            fmt = lambda v: "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.1f}"  # noqa: E731
            lw.writerow([i, level, near.cls if near else "", fmt(dist if near else None),
                         fmt(near.dist_gauge if near else None), fmt(near.dist_ground if near else None),
                         fmt(true_d), fmt(ttc if near else None), f"{stop:.1f}", f"{infer_ms:.1f}"])
            if i % 20 == 0:
                print(f"  frame {i}: {level}" + (f", {near.cls} at {dist:.0f} m" if near else "")
                      + f", model {infer_ms:.0f} ms")
            i += 1
    except KeyboardInterrupt:
        print("stopped with Ctrl+C")
    finally:
        log.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    summary = {"frames": i, "speed_mps": speed, "stop_distance_m": round(stop_distance(speed, p), 1),
               "first_alerts": first}
    for k, e in errs.items():
        if e:
            a = np.abs(np.array(e))
            summary[f"{k}_distance_error_m"] = {"mean_abs": round(float(a.mean()), 1),
                                                "median_abs": round(float(np.median(a)), 1),
                                                "mean_signed": round(float(np.mean(e)), 1)}
    if "BRAKE" in first and first["BRAKE"]["true_distance_m"] is not None:
        margin = first["BRAKE"]["true_distance_m"] - summary["stop_distance_m"]
        summary["brake_margin_m"] = round(margin, 1)
        summary["verdict"] = ("stops in time" if margin >= 0 else "TOO LATE") + \
            f" (braking needs {summary['stop_distance_m']} m, alert at {first['BRAKE']['true_distance_m']} m)"
    elif "BRAKE" not in first:
        summary["verdict"] = "no BRAKE alert was raised"
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nSummary:\n" + json.dumps(summary, indent=1))
    print(f"\nVideo: {run / 'collision.mp4'}")


if __name__ == "__main__":
    main()
