"""RailSem19 class definitions and label remapping (pure NumPy, no torch needed)."""
import json
from pathlib import Path

import numpy as np

# Fallback only. tools/prepare_railsem19.py reads the real names/colours from the
# dataset's rs19-config.json and writes them to <splits_dir>/classes.json.
DEFAULT_RS19_NAMES = [
    "road", "sidewalk", "construction", "tram-track", "fence", "pole", "traffic-light",
    "traffic-sign", "vegetation", "terrain", "sky", "human", "rail-track", "car", "truck",
    "trackbed", "on-rails", "rail-raised", "rail-embedded",
]


def default_palette(n):
    rng = np.random.RandomState(7)
    cols = rng.randint(40, 255, size=(n, 3))
    return cols.tolist()


def load_rs19_classes(splits_dir):
    """Returns (names, colors) of the 19 source classes, as written by the prepare script."""
    p = Path(splits_dir) / "classes.json"
    if p.exists():
        d = json.loads(p.read_text())
        names = d["names"]
        colors = d.get("colors") or default_palette(len(names))
        return names, colors
    return list(DEFAULT_RS19_NAMES), default_palette(len(DEFAULT_RS19_NAMES))


def build_label_mapping(data_cfg, src_names, src_colors):
    """Build a 256-entry lookup table that maps RailSem19 ids to training ids.

    label_set: full   -> the 19 RailSem19 classes as they are.
    label_set: groups -> data.groups is an ordered {new_name: [rs19 names]} dict. A group
                         with an empty list collects every class not named elsewhere; with
                         no such group, unlisted classes become ignore.
    Returns (lut uint8[256], names, colors).
    """
    ignore = int(data_cfg.get("ignore_label", 255))
    lut = np.full(256, ignore, dtype=np.uint8)
    mode = data_cfg.get("label_set", "full")

    if mode == "full":
        for i in range(len(src_names)):
            lut[i] = i
        return lut, list(src_names), [list(c) for c in src_colors]

    if mode != "groups":
        raise ValueError(f"data.label_set must be 'full' or 'groups', got {mode!r}")
    groups = data_cfg.get("groups") or {}
    if not groups:
        raise ValueError("data.label_set is 'groups' but data.groups is empty")
    catch_all = [g for g, srcs in groups.items() if not srcs]
    if len(catch_all) > 1:
        raise ValueError(f"Only one group may have an empty list (catch-all); got {catch_all}")

    name_to_id = {n: i for i, n in enumerate(src_names)}
    names, colors, used = [], [], set()
    for new_id, (gname, srcs) in enumerate(groups.items()):
        names.append(gname)
        srcs = srcs or []
        for s in srcs:
            if s not in name_to_id:
                raise ValueError(f"Group {gname!r}: unknown RailSem19 class {s!r}. "
                                 f"Known classes: {src_names}")
            if s in used:
                raise ValueError(f"RailSem19 class {s!r} is listed in more than one group")
            used.add(s)
            lut[name_to_id[s]] = new_id
        colors.append(list(src_colors[name_to_id[srcs[0]]]) if srcs else [0, 0, 0])
    if catch_all:
        cid = list(groups).index(catch_all[0])
        for n, i in name_to_id.items():
            if n not in used:
                lut[i] = cid
    lut[ignore] = ignore
    return lut, names, colors
