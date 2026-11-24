"""Import the torch-based dataset lazily so the NumPy-only helpers work without torch."""


def build_datasets(cfg):
    from .railsem19 import build_datasets as _build
    return _build(cfg)


def class_info_from_cfg(cfg):
    from .railsem19 import class_info_from_cfg as _info
    return _info(cfg)
