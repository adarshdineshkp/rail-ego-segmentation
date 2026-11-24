#!/usr/bin/env python
"""Download the official PIDNet pretrained weights from Google Drive (needs `pip install gdown`).

    python tools/download_weights.py            # PIDNet-S: ImageNet + Cityscapes
    python tools/download_weights.py --model m  # PIDNet-M

The author notes that some Drive links have stopped working. If a download fails, get the
file manually from the shared folder below and save it under the path printed for it.
"""
import argparse
import sys
from pathlib import Path

FOLDER = ("https://drive.google.com/drive/folders/0BySIOtxxULinfjlGdGFiT3NQVUdLVDBxWnhhTjB4VXNBRkFOa281WHl"
          "kektYY2VBcWVZb1k?resourcekey=0-w0JIXUekD-FCW-Rm1Z-HfQ&usp=sharing")

# file ids from https://github.com/XuJiacong/PIDNet README
FILES = {
    "s": {
        "pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar": "1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-",
        "pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt": "1JakgBam_GrzyUMp-NbEVVBPEIXLSCssH",
    },
    "m": {
        "pretrained_models/imagenet/PIDNet_M_ImageNet.pth.tar": "1gB9RxYVbdwi9eO5lbT073q-vRoncpYT1",
        "pretrained_models/cityscapes/PIDNet_M_Cityscapes_val.pt": "1q0i4fVWmO7tpBKq_eOyIXe-mRf_hIS7q",
    },
    "l": {
        "pretrained_models/imagenet/PIDNet_L_ImageNet.pth.tar": "1Eg6BwEsnu3AkKLO8lrKsoZ8AOEb2KZHY",
        "pretrained_models/cityscapes/PIDNet_L_Cityscapes_val.pt": "1AR8LHC3613EKwG23JdApfTGsyOAcH0_L",
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(FILES), default="s")
    args = ap.parse_args()
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is not installed: pip install gdown")

    root = Path(__file__).resolve().parent.parent
    ok = 0
    for rel, fid in FILES[args.model].items():
        dst = root / rel
        if dst.is_file() and dst.stat().st_size > 1_000_000:
            print(f"already present: {rel}")
            ok += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {rel} ...")
        try:
            res = gdown.download(id=fid, output=str(dst), quiet=False)
        except Exception as e:  # gdown raises on quota / missing files
            res = None
            print(f"  failed: {e}")
        if res and dst.is_file() and dst.stat().st_size > 1_000_000:
            ok += 1
        else:
            if dst.exists():
                dst.unlink()
            print(f"  could not download {rel}. Get it manually from\n  {FOLDER}\n  and save it as {dst}")
    print(f"{ok}/{len(FILES[args.model])} weight files ready.")
    if not ok:
        print("The Google Drive links are dead. Use the MMSegmentation mirror instead:\n"
              f"  python tools/convert_mmseg_pidnet.py --model {args.model}   (Windows: get_weights.bat)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
