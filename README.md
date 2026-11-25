# rail-ego-segmentation

A ready-to-run setup for training **PIDNet** (real-time semantic segmentation) on **RailSem19**. It is built for a locomotive-mounted front camera and outputs rail and track masks for downstream switch-state logic.

## Quick start

```bash
# 1. environment (install the PyTorch build for your CUDA version first)
pip install -r requirements.txt

# 2. check that everything works (about 1-2 min, no dataset needed)
python tools/smoke_test.py

# 3. dataset: download rs19_val.zip from https://www.wilddash.cc/railsem19 (free account)
mkdir -p data/railsem19 && unzip rs19_val.zip -d data/railsem19

# 4. train
bash scripts/train.sh
```

`scripts/train.sh` checks the dataset, writes the train/val split, downloads the pretrained weights, uses every GPU it finds and starts training. If training stops, run the same command again to resume from `runs/<experiment>/last.pt`.

### Windows (e.g. a laptop RTX 3050, 6 GB)

1. Double-click `setup_windows.bat`. It creates `.venv`, installs PyTorch with CUDA 12.6 and the requirements, runs the smoke test and downloads the weights.
2. Extract `rs19_val.zip` into `data\railsem19`.
3. Double-click `train_rtx3050.bat` (batch size 2, 640×1024 crops, 40 epochs). Run it again to resume.

## What gets trained

| Config | Classes | Use |
|---|---|---|
| `configs/railsem19_pidnet_s_rails.yaml` (default) | background, rail, track, trackbed, on-rails, human | rail-focused; best.pt is chosen by rail + track IoU |
| `configs/railsem19_pidnet_s.yaml` | all 19 RailSem19 classes | full scene understanding |
| `configs/railsem19_json_pidnet_s.yaml` | background, rail, track, human, on-rails, pole, fence, platform | for copies of RailSem19 without `uint8/` masks; masks drawn from the JSON annotations by `tools/rasterize_rs19_json.py` (`train_rtx3050.bat` does this automatically) |
| `configs/railsem19_json_ego_pidnet_s.yaml` | the above + ego-track (the train’s own path) | used automatically when `data/railsem19/rs19_egopath.json` is present ([ego-path annotations](https://github.com/irtrailenium/train-ego-path-detection), CC BY-NC-SA 4.0) |
| `configs/railsem19_pidnet_m_rails.yaml` | rail-focused, PIDNet-M | more accurate, about 2x slower |

To change the label set, edit `data.groups` in the rails config. Each group lists RailSem19 class names, and the group with an empty list collects every class not listed elsewhere.

Recipe (from the official PIDNet code): SGD with lr 0.01 and poly decay, OHEM cross-entropy, a boundary loss weighted ×20, and a boundary-aware "SB" loss. Training uses AMP and starts from Cityscapes-pretrained weights, with the classifier re-initialised for the new classes. Crops are 768×1024, and random scales stay between 0.75 and 1.5 so the rails remain a few pixels wide.

Override any setting from the command line:

```bash
bash scripts/train.sh configs/railsem19_pidnet_s_rails.yaml train.batch_size=4 train.epochs=60
DATA_ROOT=/mnt/datasets/rs19 NGPU=1 bash scripts/train.sh
```

**Split:** RailSem19 comes as a single set of 8,500 frames. `tools/prepare_railsem19.py` holds out 500 frames for validation in contiguous blocks of 50 IDs, which keeps near-duplicate neighbouring frames out of the validation set.

## Outputs

`runs/<experiment>/` contains:
- `best.pt`, `last.pt`: checkpoints
- `best.json`: per-class IoU of the best epoch
- `train.log`, `metrics.csv`
- `tb/`: training curves (`tensorboard --logdir runs`)
- `config.yaml`: the exact config used

## After training

```bash
# per-class IoU on the val split, plus side-by-side images
python evaluate.py --ckpt runs/rs19_pidnet_s_rails/best.pt --save-vis vis/

# your own footage: images, a folder, or a video
python infer.py --ckpt runs/rs19_pidnet_s_rails/best.pt --input loco_drive.mp4 --out preds/

# ONNX for TensorRT / Jetson
python tools/export_onnx.py --ckpt runs/rs19_pidnet_s_rails/best.pt --size 1088 1920
```

`infer.py` writes `*_label.png` files containing class IDs. Feed these to your switch-state logic in place of the RandomForest pixel classifier.

## Layout

```
train.py  evaluate.py  infer.py      entry points
scripts/train.sh                     one-command training
configs/                             experiment configs (_base_ inheritance)
railseg/models/pidnet.py             PIDNet-S/M/L, parameter names match the official repo
railseg/losses.py                    OHEM CE + boundary BCE + SB loss
railseg/data/                        RailSem19 dataset, label grouping, augmentations
tools/                               prepare split, download weights, smoke test, ONNX export
```

## Notes

- **Pretrained weights** are hosted on the PIDNet author's Google Drive, and the author says some links no longer work. If `tools/download_weights.py` fails, download the files manually from the folder link it prints. Then run `python tools/smoke_test.py --weights <file>` to confirm they load (it reports how many tensors matched).
- **Memory:** PIDNet-S with batch size 8 at 768×1024 needs about 16 GB with AMP. On a smaller GPU, use `train.batch_size=4` or a smaller `data.train.crop_size`.
- **License:** RailSem19 is for research use under its own terms. PIDNet is MIT-licensed.

## Simulated drive (Blender)

`run_sim.bat` (or `blender -b --factory-startup --python sim/build_scene.py -- --out sim_out/run1`
followed by `python sim/run_collision.py --run sim_out/run1 --ckpt <best.pt> --show`) renders a
procedural railway scene from a cab camera, with a vehicle or a person standing on the track in a
curve, and runs the collision warning on every frame as it is rendered. `sim/collision.py` holds the
logic: ego corridor at clearance width, in-path test, distance from the track gauge and from the
ground plane, k-of-n persistence, and braking-distance alert levels. Each run writes
`collision.mp4`, `collision_log.csv` and `summary.json` (when each alert fired vs the true distance,
and whether braking would have been in time).
