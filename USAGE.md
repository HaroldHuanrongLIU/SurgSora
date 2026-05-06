# SurgWMBench Usage

This guide documents the Python command path for the SurgWMBench 20-anchor
adaptation. It trains one SurgSora checkpoint with 5 context anchors and 15
future anchors. The model input is the first 5 anchor frames plus their 5
observed trajectory points, and the output is the future 15 anchor frames plus
the future 15 trajectory points. Evaluation reports horizons 5, 10, and 15
against original-size frames and original-resolution trajectory pixels.

Use `--prediction-task image-only` to train or evaluate a strict image-only
variant: the model uses only the first 5 anchor frames as input, predicts future
frames 6-20, and does not build, save, or load `trajectory_head.pt`.

## Environment

Create the Python 3.11 environment from the locked uv project:

```bash
uv sync
source .venv/bin/activate
```

For LPIPS evaluation, include the optional metrics extra:

```bash
uv sync --extra metrics
source .venv/bin/activate
```

Check the core runtime:

```bash
uv run --frozen python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Paths

The default dataset root is:

```text
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

Override it with `--dataset-root`; keep manifest paths relative to that root,
for example `--train-manifest manifests/train.jsonl`. Do not edit the official
manifests or create random train/val/test splits.

The default pretrained checkpoint path is:

```text
./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1
```

## Joint Image + Trajectory Workflow

Use this mode when the input is the first 5 anchor frames plus their first 5
trajectory points, and the model should predict both future frames and future
trajectory points for anchors 6-20. This is the default `--prediction-task
joint` mode.

Joint training applies robustness augmentation to the observed input trajectory
points by default: Gaussian noise in normalized coordinate space and random
per-point masking. Future trajectory labels remain clean. Set
`--trajectory-input-noise-std 0.0 --trajectory-input-mask-prob 0.0` to disable
this augmentation.

To keep random masking but disable Gaussian noise, use:

```bash
--trajectory-input-noise-std 0.0
```

To disable both Gaussian noise and random masking, use:

```bash
--trajectory-input-noise-std 0.0 \
--trajectory-input-mask-prob 0.0
```

Joint training writes:

```text
unet_context/
controlnet/
trajectory_head.pt
training_args.json
```

### Single-GPU Joint Training

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python Training/train_surgwmbench_20anchor.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --pretrained-model-name-or-path ./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1 \
  --output-dir ./Training/logs/surgwmbench_20anchor \
  --per-gpu-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --mixed-precision fp16 \
  --trajectory-loss-weight 10.0 \
  --trajectory-velocity-loss-weight 1.0 \
  --trajectory-input-noise-std 0.01 \
  --trajectory-input-mask-prob 0.2 \
  --trajectory-input-mask-value -1.0 \
  --trajectory-hidden-dim 512 \
  --trajectory-num-layers 2 \
  --trajectory-num-heads 8
```

### Multi-GPU Joint Training

Run DDP through Accelerate. Set `--num_processes` to the number of visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --frozen python -m accelerate.commands.launch --num_processes 4 \
  Training/train_surgwmbench_20anchor.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --pretrained-model-name-or-path ./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1 \
  --output-dir ./Training/logs/surgwmbench_20anchor \
  --per-gpu-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --mixed-precision fp16 \
  --trajectory-loss-weight 10.0 \
  --trajectory-velocity-loss-weight 1.0 \
  --trajectory-input-noise-std 0.01 \
  --trajectory-input-mask-prob 0.2 \
  --trajectory-input-mask-value -1.0 \
  --trajectory-hidden-dim 512 \
  --trajectory-num-layers 2 \
  --trajectory-num-heads 8
```

### Joint Evaluation

Evaluate original-resolution image metrics plus trajectory ADE/FDE for horizons
5, 10, and 15:

```bash
uv run --frozen python Training/eval_surgwmbench_20anchor.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/val.jsonl \
  --pretrained-model-name-or-path ./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1 \
  --checkpoint-dir ./Training/logs/surgwmbench_20anchor \
  --output-dir ./Training/eval/surgwmbench_20anchor \
  --prediction-task joint \
  --batch-size 1
```

## Image-Only Workflow

Use this mode when the input is only the first 5 anchor frames and the model
should predict future frames 6-20. It does not use trajectory points as inputs
or labels, and does not build, save, or load `trajectory_head.pt`.

Image-only training writes:

```text
unet_context/
controlnet/
training_args.json
```

### Single-GPU Image-Only Training

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python Training/train_surgwmbench_20anchor.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --pretrained-model-name-or-path ./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1 \
  --output-dir ./Training/logs/surgwmbench_20anchor_image_only \
  --per-gpu-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --mixed-precision fp16 \
  --prediction-task image-only
```

### Multi-GPU Image-Only Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --frozen python -m accelerate.commands.launch --num_processes 4 \
  Training/train_surgwmbench_20anchor.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --pretrained-model-name-or-path ./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1 \
  --output-dir ./Training/logs/surgwmbench_20anchor_image_only \
  --per-gpu-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --mixed-precision fp16 \
  --prediction-task image-only
```

### Image-Only Evaluation

Evaluate original-resolution image metrics only. Evaluation can auto-detect the
mode from `training_args.json`, but passing `--prediction-task image-only` keeps
the command explicit:

```bash
uv run --frozen python Training/eval_surgwmbench_20anchor.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/val.jsonl \
  --pretrained-model-name-or-path ./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1 \
  --checkpoint-dir ./Training/logs/surgwmbench_20anchor_image_only \
  --output-dir ./Training/eval/surgwmbench_20anchor_image_only \
  --prediction-task image-only \
  --batch-size 1
```

## Training Notes

For a small real-data smoke run, add these flags to either training command:

```bash
--max-clips 1 --max-train-batches 1 --num-train-epochs 1
```

Effective batch size for either workflow is:

```text
per-gpu-batch-size * num_processes * gradient-accumulation-steps
```

Add `--compute-lpips` only after syncing with `uv sync --extra metrics`.
