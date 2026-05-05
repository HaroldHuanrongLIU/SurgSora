#!/usr/bin/env python
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from diffusers import AutoencoderKLTemporalDecoder
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Training.surgwmbench_modeling import (
    build_5frame_latent_input,
    decode_latents_to_frames,
    encode_context_images,
    expand_conv_in_channels,
    get_add_time_ids,
    make_zero_control_tensors,
    resize_dual_control_fusion,
    tensor_to_vae_latent,
)
from Training.train_utils.surgwmbench_dataset import SurgWMBench20AnchorDataset, surgwmbench_collate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SurgSora SurgWMBench 20-anchor future prediction.")
    parser.add_argument("--dataset-root", default="/mnt/hdd1/neurips2026_dataset_track/SurgWMBench")
    parser.add_argument("--manifest", default="manifests/val.jsonl")
    parser.add_argument("--pretrained-model-name-or-path", default="./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="./Training/eval_outputs/surgwmbench_20anchor")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--target-frames", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--motion-bucket-id", type=int, default=127)
    parser.add_argument("--noise-aug-strength", type=float, default=0.02)
    parser.add_argument("--decode-chunk-size", type=int, default=8)
    parser.add_argument("--save-samples", type=int, default=4)
    parser.add_argument("--compute-lpips", action="store_true")
    return parser.parse_args()


def _load_models(args, device: torch.device, dtype: torch.dtype):
    from models.Control_Backbone import UNetControlNetModel
    from models.Control_Encoder import DualFlowControlNet
    from utils.scheduling_euler_discrete_karras_fix import EulerDiscreteScheduler

    checkpoint_dir = Path(args.checkpoint_dir)
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
        variant="fp16",
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        variant="fp16",
    )
    unet_dir = checkpoint_dir / "unet_context"
    controlnet_dir = checkpoint_dir / "controlnet"
    if unet_dir.exists():
        unet = UNetControlNetModel.from_pretrained(unet_dir)
    else:
        unet = UNetControlNetModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
            variant="fp16",
        )
        expand_conv_in_channels(unet, 4 + args.context_frames * 4, context_frames=args.context_frames)
    if not controlnet_dir.exists():
        raise FileNotFoundError(f"Missing trained controlnet directory: {controlnet_dir}")
    controlnet = DualFlowControlNet.from_pretrained(controlnet_dir)
    resize_dual_control_fusion(controlnet, args.target_frames - 1)

    scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    for module in (image_encoder, vae, unet, controlnet):
        module.requires_grad_(False)
        module.to(device=device, dtype=dtype)
        module.eval()
    return feature_extractor, image_encoder, vae, unet, controlnet, scheduler


def _make_loader(args):
    dataset = SurgWMBench20AnchorDataset(
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        image_size=(args.width, args.height),
        context_frames=args.context_frames,
        target_frames=args.target_frames,
        max_clips=args.max_clips,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=surgwmbench_collate,
    )


@torch.no_grad()
def _predict_batch(args, batch, models, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    feature_extractor, image_encoder, vae, unet, controlnet, scheduler = models
    context_frames = batch["context_frames"].to(device=device, dtype=dtype)
    batch_size = context_frames.shape[0]

    context_latents = tensor_to_vae_latent(context_frames, vae, sample=False)
    encoder_hidden_states = encode_context_images(context_frames, feature_extractor, image_encoder, dtype)
    latent_height, latent_width = context_latents.shape[-2:]

    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(
        batch_size,
        args.target_frames,
        unet.config.out_channels,
        latent_height,
        latent_width,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    controls = make_zero_control_tensors(context_frames, args.target_frames, dtype=dtype)
    added_time_ids = get_add_time_ids(
        batch_size,
        encoder_hidden_states.dtype,
        device,
        unet,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
    )

    for timestep in scheduler.timesteps:
        latent_model_input = scheduler.scale_model_input(latents, timestep)
        latent_model_input = build_5frame_latent_input(latent_model_input, context_latents, vae.config.scaling_factor)
        down_samples, mid_sample, _, _ = controlnet(
            latent_model_input,
            timestep,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
            controlnet_cond=controls["controlnet_cond"],
            controlnet_flow=controls["controlnet_flow"],
            controlnet_depth=controls["controlnet_depth"],
            controlnet_mask=controls["controlnet_mask"],
            return_dict=False,
        )
        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
            down_block_additional_residuals=[sample.to(dtype=dtype) for sample in down_samples],
            mid_block_additional_residual=mid_sample.to(dtype=dtype),
        ).sample
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return decode_latents_to_frames(latents, vae, decode_chunk_size=args.decode_chunk_size)


def _ssim(gt: np.ndarray, pred: np.ndarray) -> Optional[float]:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None
    try:
        return float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))
    except TypeError:
        return float(structural_similarity(gt, pred, multichannel=True, data_range=1.0))


def _load_gt(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _pred_to_original(pred: torch.Tensor, original_size) -> np.ndarray:
    image = pred.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    pil = Image.fromarray((image * 255.0).round().astype(np.uint8))
    pil = pil.resize(tuple(original_size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _save_gif(path: Path, frames: List[np.ndarray], fps: int = 7) -> None:
    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(round(1000 / fps))
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=duration_ms, loop=0)


def _frame_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Optional[float]]:
    error = pred - gt
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    psnr = float("inf") if mse == 0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": _ssim(gt, pred)}


def _load_lpips_model(enabled: bool, device: torch.device):
    if not enabled:
        return None
    try:
        import lpips
    except Exception as exc:
        raise ImportError("Install lpips or omit --compute-lpips") from exc
    return lpips.LPIPS(net="alex").to(device).eval()


@torch.no_grad()
def _lpips_value(model, gt: np.ndarray, pred: np.ndarray, device: torch.device) -> Optional[float]:
    if model is None:
        return None
    gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    gt_tensor = gt_tensor * 2.0 - 1.0
    pred_tensor = pred_tensor * 2.0 - 1.0
    return float(model(gt_tensor, pred_tensor).item())


def _mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def main():
    args = parse_args()
    if args.context_frames != 5 or args.target_frames != 15:
        raise ValueError("This task is fixed to 5 context anchors and 15 target anchors.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    models = _load_models(args, device, dtype)
    lpips_model = _load_lpips_model(args.compute_lpips, device)
    dataloader = _make_loader(args)

    horizons = {5: [], 10: [], 15: []}
    sample_artifacts = []
    clip_count = 0

    for batch in dataloader:
        predictions = _predict_batch(args, batch, models, device, dtype)
        batch_size = predictions.shape[0]
        for batch_idx in range(batch_size):
            original_size = batch["original_size"][batch_idx]
            frame_metrics = []
            pred_frames_for_sample = []
            for frame_idx in range(args.target_frames):
                pred = _pred_to_original(predictions[batch_idx, frame_idx], original_size)
                gt = _load_gt(batch["target_frame_paths"][batch_idx][frame_idx])
                metrics = _frame_metrics(gt, pred)
                metrics["lpips"] = _lpips_value(lpips_model, gt, pred, device)
                frame_metrics.append(metrics)
                if clip_count < args.save_samples:
                    pred_frames_for_sample.append((pred * 255.0).round().astype(np.uint8))

            for horizon in horizons:
                subset = frame_metrics[:horizon]
                horizons[horizon].append({metric: _mean([item[metric] for item in subset]) for metric in subset[0]})

            if clip_count < args.save_samples:
                name = f"{batch['patient_id'][batch_idx]}_{batch['trajectory_id'][batch_idx]}"
                gif_path = sample_dir / f"{clip_count:04d}_{name}.gif"
                _save_gif(gif_path, pred_frames_for_sample, fps=7)
                sample_artifacts.append(str(gif_path))
            clip_count += 1

    metrics = {}
    for horizon, rows in horizons.items():
        metrics[f"horizon_{horizon}"] = {
            key: _mean([row[key] for row in rows])
            for key in ("mse", "mae", "psnr", "ssim", "lpips")
        }

    report = {
        "dataset_name": "SurgWMBench",
        "task": "20_anchor_first5_predict_6_to_20",
        "manifest": args.manifest,
        "checkpoint_dir": args.checkpoint_dir,
        "context_frames": args.context_frames,
        "target_frames": args.target_frames,
        "metrics_original_resolution": metrics,
        "num_clips": clip_count,
        "sample_artifacts": sample_artifacts,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report["metrics_original_resolution"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
