#!/usr/bin/env python
"""Print a scoreboard of all simulated runs: python sim/summarize.py [sim_out]"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "sim_out")
rows = []
for f in sorted(root.glob("*/summary.json")):
    s = json.loads(f.read_text())
    fa = s.get("first_alerts", {})

    def at(level):
        a = fa.get(level)
        if not a:
            return "-"
        d = a.get("true_distance_m")
        return f"{d:.0f} m ({a.get('source', '?')})" if d is not None else f"est {a['est_distance_m']:.0f} m"

    rows.append((f.parent.name, f"{s.get('speed_mps', 0) * 3.6:.0f} km/h", f"{s.get('stop_distance_m', 0):.0f} m",
                 at("WARNING"), at("BRAKE"), s.get("verdict", "?")))
if not rows:
    sys.exit(f"No summary.json under {root}")
head = ("run", "speed", "needs", "first WARNING", "first BRAKE", "verdict")
widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(5)]
print("\n" + "  ".join(h.ljust(w) for h, w in zip(head, widths)) + "  verdict")
for r in rows:
    print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)) + "  " + r[5])
print("\n(distances are the TRUE distance to the obstacle when the alert first fired; "
      "the source says which cue fired: seg = segmentation, cutoff = path cut-off, yolo = detector)")
