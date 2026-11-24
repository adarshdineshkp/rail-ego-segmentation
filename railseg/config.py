"""YAML configs with `_base_` inheritance and `key.sub=value` command-line overrides."""
import copy
import os

import yaml


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        base_cfg = _load(base_path)
        # a child that redefines `groups` replaces it entirely instead of merging
        if isinstance(cfg.get("data"), dict) and "groups" in cfg["data"]:
            base_cfg.setdefault("data", {})["groups"] = None
        cfg = _merge(base_cfg, cfg)
    return cfg


def load_config(path, overrides=()):
    return apply_overrides(_load(path), overrides)


def apply_overrides(cfg, overrides=()):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Overrides must look like key.sub=value, got {item!r}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
