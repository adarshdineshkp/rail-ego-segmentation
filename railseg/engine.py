import torch
import torch.nn.functional as F

from .metrics import ConfusionMatrix


def _main_output(out):
    if isinstance(out, (list, tuple)):
        return out[1] if len(out) == 3 else out[0]
    return out


@torch.no_grad()
def predict_logits(model, images, out_size, flip=False, align_corners=False):
    logits = F.interpolate(_main_output(model(images)).float(), size=out_size,
                           mode="bilinear", align_corners=align_corners)
    if flip:
        flipped = _main_output(model(torch.flip(images, dims=[3]))).float()
        flipped = F.interpolate(flipped, size=out_size, mode="bilinear", align_corners=align_corners)
        logits = logits + torch.flip(flipped, dims=[3])
    return logits


@torch.no_grad()
def evaluate(model, loader, num_classes, ignore_label, device, amp=False, flip=False, log_every=100,
             log=None):
    was_training = model.training
    model.eval()
    cm = ConfusionMatrix(num_classes, ignore_label, device)
    use_amp = amp and device.type == "cuda"
    for i, batch in enumerate(loader):
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = predict_logits(model, images, labels.shape[-2:], flip)
        cm.update(logits.argmax(1), labels)
        if log and log_every and (i + 1) % log_every == 0:
            log(f"  eval {i + 1}/{len(loader)}")
    if was_training:
        model.train()
    return cm.compute()
