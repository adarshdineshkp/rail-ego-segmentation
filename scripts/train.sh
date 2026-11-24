#!/usr/bin/env bash
# One command to train: checks the dataset, writes splits, fetches weights, starts training.
#
#   bash scripts/train.sh                                      # rail-focused PIDNet-S, all GPUs
#   bash scripts/train.sh configs/railsem19_pidnet_s.yaml      # all 19 classes
#   DATA_ROOT=/data/rs19 bash scripts/train.sh                 # dataset somewhere else
#   bash scripts/train.sh configs/... train.batch_size=4       # extra config overrides
#
# Re-running the same command resumes from runs/<experiment>/last.pt.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/railsem19_pidnet_s_rails.yaml}"
[ $# -gt 0 ] && shift
DATA_ROOT="${DATA_ROOT:-data/railsem19}"
SPLITS="${SPLITS:-data/splits}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "RailSem19 not found at $DATA_ROOT."
  echo "Download rs19_val.zip from https://www.wilddash.cc/railsem19 (free account) and run:"
  echo "  mkdir -p $DATA_ROOT && unzip rs19_val.zip -d $DATA_ROOT"
  exit 1
fi

if [ ! -f "$SPLITS/train.txt" ]; then
  python tools/prepare_railsem19.py --root "$DATA_ROOT" --out "$SPLITS"
fi

MODEL_LETTER=$(python - "$CONFIG" <<'EOF'
import sys; sys.path.insert(0, ".")
from railseg.config import load_config
print(load_config(sys.argv[1])["model"]["name"][-1])
EOF
)
UPPER=$(echo "$MODEL_LETTER" | tr '[:lower:]' '[:upper:]')
if [ ! -f "pretrained_models/cityscapes/PIDNet_${UPPER}_Cityscapes_val.pt" ] && \
   [ ! -f "pretrained_models/imagenet/PIDNet_${UPPER}_ImageNet.pth.tar" ]; then
  python tools/download_weights.py --model "$MODEL_LETTER" || \
    echo "WARNING: pretrained weights unavailable - training will start from scratch."
fi

NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
ARGS=(--config "$CONFIG" --resume auto "data.root=$DATA_ROOT" "data.splits_dir=$SPLITS" "$@")
if [ "$NGPU" -gt 1 ]; then
  echo "Launching on $NGPU GPUs"
  exec torchrun --standalone --nproc_per_node="$NGPU" train.py "${ARGS[@]}"
else
  exec python train.py "${ARGS[@]}"
fi
