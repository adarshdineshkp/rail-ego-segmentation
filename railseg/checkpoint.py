from .models import build_model
from .utils import safe_torch_load


def load_trained(path, device, augment=False):
    """Load a checkpoint written by train.py. Returns (model, cfg, classes)."""
    ck = safe_torch_load(path)
    cfg, classes = ck["cfg"], ck["classes"]
    model = build_model(cfg["model"]["name"], len(classes["names"]), augment=augment)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg, classes
