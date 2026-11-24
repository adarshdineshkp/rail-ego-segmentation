#!/usr/bin/env python
"""Get Cityscapes-pretrained PIDNet weights from MMSegmentation and convert them to this repo's
(official-PIDNet) parameter names. Used because the original Google Drive links are dead.

    python tools/convert_mmseg_pidnet.py            # PIDNet-S
    python tools/convert_mmseg_pidnet.py --model m

Writes pretrained_models/cityscapes/PIDNet_<S|M|L>_Cityscapes_val.pt and reports how many
tensors of the backbone were filled. Segmentation heads are re-initialised during fine-tuning
anyway, so only the backbone is converted.
"""
import argparse
import re
import sys
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from railseg.models import build_model  # noqa: E402

BASE = "https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/"
URLS = {
    "s": BASE + "pidnet-s_2xb6-120k_1024x1024-cityscapes/"
                "pidnet-s_2xb6-120k_1024x1024-cityscapes_20230302_191700-bb8e3bcc.pth",
    "m": BASE + "pidnet-m_2xb6-120k_1024x1024-cityscapes/"
                "pidnet-m_2xb6-120k_1024x1024-cityscapes_20230301_143452-f9bcdbf3.pth",
}

# (pattern, replacement) applied in order to backbone keys (after removing "backbone.")
RULES = [
    # stem = conv, conv, layer1, relu, layer2, relu
    (r"^stem\.0\.conv\.", "conv1.0."), (r"^stem\.0\.bn\.", "conv1.1."),
    (r"^stem\.1\.conv\.", "conv1.3."), (r"^stem\.1\.bn\.", "conv1.4."),
    (r"^stem\.2\.", "layer1."), (r"^stem\.4\.", "layer2."),
    # branches
    (r"^i_branch_layers\.0\.", "layer3."), (r"^i_branch_layers\.1\.", "layer4."),
    (r"^i_branch_layers\.2\.", "layer5."),
    (r"^p_branch_layers\.0\.", "layer3_."), (r"^p_branch_layers\.1\.", "layer4_."),
    (r"^p_branch_layers\.2\.", "layer5_."),
    (r"^d_branch_layers\.0\.", "layer3_d."), (r"^d_branch_layers\.1\.", "layer4_d."),
    (r"^d_branch_layers\.2\.", "layer5_d."),
    (r"^compression_1\.", "compression3."), (r"^compression_2\.", "compression4."),
    (r"^diff_1\.", "diff3."), (r"^diff_2\.", "diff4."),
    (r"^pag_1\.", "pag3."), (r"^pag_2\.", "pag4."),
    # residual blocks: ConvModule(conv, bn) -> convN / bnN
    (r"\.conv(\d)\.conv\.", r".conv\1."), (r"\.conv(\d)\.bn\.", r".bn\1."),
    (r"\.downsample\.conv\.", ".downsample.0."), (r"\.downsample\.bn\.", ".downsample.1."),
    # conv-bn pairs
    (r"^(compression[34]|diff[34])\.conv\.", r"\1.0."), (r"^(compression[34]|diff[34])\.bn\.", r"\1.1."),
    # PagFM: mmseg f_p (P branch) / f_i (I branch) -> official f_x / f_y
    (r"^(pag[34])\.f_p\.conv\.", r"\1.f_x.0."), (r"^(pag[34])\.f_p\.bn\.", r"\1.f_x.1."),
    (r"^(pag[34])\.f_i\.conv\.", r"\1.f_y.0."), (r"^(pag[34])\.f_i\.bn\.", r"\1.f_y.1."),
    (r"^(pag[34])\.up\.conv\.", r"\1.up.0."), (r"^(pag[34])\.up\.bn\.", r"\1.up.1."),
    # LightBag (S/M) and Bag (L)
    (r"^dfm\.f_p\.conv\.", "dfm.conv_p.0."), (r"^dfm\.f_p\.bn\.", "dfm.conv_p.1."),
    (r"^dfm\.f_i\.conv\.", "dfm.conv_i.0."), (r"^dfm\.f_i\.bn\.", "dfm.conv_i.1."),
    (r"^dfm\.conv\.bn\.", "dfm.conv.0."), (r"^dfm\.conv\.conv\.", "dfm.conv.2."),
    # PAPPM / DAPPM (ConvModules in norm-act-conv order)
    (r"^spp\.scales\.0\.bn\.", "spp.scale0.0."), (r"^spp\.scales\.0\.conv\.", "spp.scale0.2."),
    (r"^spp\.scales\.([1-4])\.1\.bn\.", r"spp.scale\1.1."),
    (r"^spp\.scales\.([1-4])\.1\.conv\.", r"spp.scale\1.3."),
    (r"^spp\.processes\.bn\.", "spp.scale_process.0."), (r"^spp\.processes\.conv\.", "spp.scale_process.2."),
    (r"^spp\.processes\.0\.bn\.", "spp.process1.0."), (r"^spp\.processes\.0\.conv\.", "spp.process1.2."),
    (r"^spp\.processes\.1\.bn\.", "spp.process2.0."), (r"^spp\.processes\.1\.conv\.", "spp.process2.2."),
    (r"^spp\.processes\.2\.bn\.", "spp.process3.0."), (r"^spp\.processes\.2\.conv\.", "spp.process3.2."),
    (r"^spp\.processes\.3\.bn\.", "spp.process4.0."), (r"^spp\.processes\.3\.conv\.", "spp.process4.2."),
    (r"^spp\.(compression|shortcut)\.bn\.", r"spp.\1.0."), (r"^spp\.(compression|shortcut)\.conv\.", r"spp.\1.2."),
]


def convert_key(k):
    for pat, rep in RULES:
        k = re.sub(pat, rep, k)
    return k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(URLS), default="s")
    ap.add_argument("--src", default=None, help="already-downloaded mmseg .pth (skips download)")
    args = ap.parse_args()
    letter = args.model.upper()

    src = Path(args.src) if args.src else ROOT / "pretrained_models" / "mmseg" / Path(URLS[args.model]).name
    if not src.is_file():
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URLS[args.model]}")
        urllib.request.urlretrieve(URLS[args.model], src)
    print(f"Source: {src} ({src.stat().st_size / 1e6:.1f} MB)")

    ck = torch.load(src, map_location="cpu", weights_only=False)  # openmmlab.com checkpoint
    sd = ck.get("state_dict", ck)

    model = build_model(f"pidnet_{args.model}", 19)
    own = model.state_dict()
    out, unused, bad_shape = {}, [], []
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        nk = convert_key(k[len("backbone."):])
        if nk not in own:
            unused.append(f"{k} -> {nk}")
        elif own[nk].shape != v.shape:
            bad_shape.append(f"{k} -> {nk}: {tuple(v.shape)} vs {tuple(own[nk].shape)}")
        else:
            out[nk] = v

    # mmseg's stem convs have no bias (BN follows); the official ones do - zero them
    for b in ("conv1.0.bias", "conv1.3.bias"):
        if b in own and b not in out:
            out[b] = torch.zeros_like(own[b])

    head = ("final_layer.", "seghead_p.", "seghead_d.")
    backbone_keys = [k for k in own if not k.startswith(head)]
    missing = [k for k in backbone_keys if k not in out]
    n_real = lambda ks: sum(1 for k in ks if not k.endswith("num_batches_tracked"))  # noqa: E731

    print(f"Backbone tensors filled: {n_real(out)}/{n_real(backbone_keys)}")
    for title, items in (("Unmapped source keys", unused), ("Shape mismatches", bad_shape),
                         ("Backbone keys still missing", missing)):
        if items:
            print(f"{title} ({len(items)}):")
            for s in items[:40]:
                print("   ", s)

    if n_real(missing) > 0 or bad_shape:
        sys.exit("CONVERSION INCOMPLETE - send this output to Claude. Nothing was saved.")

    dst = ROOT / "pretrained_models" / "cityscapes" / f"PIDNet_{letter}_Cityscapes_val.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out}, dst)
    print(f"OK - saved {dst}")


if __name__ == "__main__":
    main()
