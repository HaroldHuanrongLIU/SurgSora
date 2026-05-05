# SurgWMBench Usage

This guide documents the Python command path for the SurgWMBench 20-anchor
adaptation. It trains one SurgSora checkpoint with 5 context anchors and 15
future anchors, then evaluates horizons 5, 10, and 15 against original-size
frames.

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

## Single-GPU Training

Run one GPU with the direct Python entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python Training/train_surgwmbench_20anchor.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --pretrained-model-name-or-path ./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1 \
  --output-dir ./Training/logs/surgwmbench_20anchor \
  --per-gpu-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --mixed-precision fp16
```

For a small real-data smoke run, add:

```bash
--max-clips 1 --max-train-batches 1 --num-train-epochs 1
```

## Multi-GPU Training

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
  --mixed-precision fp16
```

Effective batch size is:

```text
per-gpu-batch-size * num_processes * gradient-accumulation-steps
```

## Evaluation

Evaluate one checkpoint and report original-resolution metrics for horizons
5, 10, and 15:

```bash
uv run --frozen python Training/eval_surgwmbench_20anchor.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/val.jsonl \
  --pretrained-model-name-or-path ./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1 \
  --checkpoint-dir ./Training/logs/surgwmbench_20anchor \
  --output-dir ./Training/eval/surgwmbench_20anchor \
  --batch-size 1
```

Add `--compute-lpips` only after syncing with `uv sync --extra metrics`.
