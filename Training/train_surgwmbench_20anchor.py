#!/usr/bin/env python
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Training.surgwmbench_modeling import (
    augment_context_trajectory_coords,
    build_5frame_latent_input,
    encode_context_images,
    expand_conv_in_channels,
    get_add_time_ids,
    make_zero_control_tensors,
    rand_cosine_interpolated,
    resize_dual_control_fusion,
    tensor_to_vae_latent,
)
from Training.train_utils.surgwmbench_dataset import SurgWMBench20AnchorDataset, surgwmbench_collate
from Training.trajectory_head import TrajectoryPredictionHead, infer_image_embed_dim, save_trajectory_head


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SurgSora on SurgWMBench 20-anchor future prediction.")
    parser.add_argument("--dataset-root", default="/mnt/hdd1/neurips2026_dataset_track/SurgWMBench")
    parser.add_argument("--train-manifest", default="manifests/train.jsonl")
    parser.add_argument("--pretrained-model-name-or-path", default="./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1")
    parser.add_argument("--controlnet-model-name-or-path", default=None)
    parser.add_argument("--output-dir", default="./Training/logs/surgwmbench_20anchor")
    parser.add_argument("--train-height", type=int, default=256)
    parser.add_argument("--train-width", type=int, default=256)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--target-frames", type=int, default=15)
    parser.add_argument("--per-gpu-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--conditioning-dropout-prob", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--motion-bucket-id", type=int, default=127)
    parser.add_argument("--noise-aug-strength", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument(
        "--prediction-task",
        choices=["joint", "image-only"],
        default="joint",
        help="joint trains future images and trajectories; image-only trains future images without trajectory input or supervision.",
    )
    parser.add_argument("--trajectory-loss-weight", type=float, default=10.0)
    parser.add_argument("--trajectory-velocity-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--trajectory-input-noise-std",
        type=float,
        default=0.01,
        help="Gaussian noise std for observed normalized trajectory points in joint training.",
    )
    parser.add_argument(
        "--trajectory-input-mask-prob",
        type=float,
        default=0.2,
        help="Per-point probability of masking observed trajectory inputs in joint training.",
    )
    parser.add_argument(
        "--trajectory-input-mask-value",
        type=float,
        default=-1.0,
        help="Sentinel value used for masked observed trajectory inputs.",
    )
    parser.add_argument("--trajectory-hidden-dim", type=int, default=512)
    parser.add_argument("--trajectory-num-layers", type=int, default=2)
    parser.add_argument("--trajectory-num-heads", type=int, default=8)
    return parser.parse_args()


def _weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _prepare_models(args, accelerator: Accelerator, weight_dtype: torch.dtype):
    from models.Control_Backbone import UNetControlNetModel
    from models.Control_Encoder import DualFlowControlNet

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
    unet = UNetControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    input_channels = 4 + args.context_frames * 4
    expand_conv_in_channels(unet, input_channels, context_frames=args.context_frames)
    unet.register_to_config(num_frames=args.target_frames)

    if args.controlnet_model_name_or_path:
        controlnet = DualFlowControlNet.from_pretrained(args.controlnet_model_name_or_path)
        expand_conv_in_channels(controlnet, input_channels, context_frames=args.context_frames)
    else:
        controlnet = DualFlowControlNet.from_unet(unet)
    controlnet.register_to_config(num_frames=args.target_frames)
    resize_dual_control_fusion(controlnet, args.target_frames - 1)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.conv_in.requires_grad_(True)
    controlnet.requires_grad_(True)

    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.conv_in.to(dtype=torch.float32)
    controlnet.to(accelerator.device, dtype=torch.float32)
    trajectory_head = None
    if args.prediction_task == "joint":
        trajectory_head = TrajectoryPredictionHead(
            image_embed_dim=infer_image_embed_dim(image_encoder),
            hidden_dim=args.trajectory_hidden_dim,
            context_frames=args.context_frames,
            target_frames=args.target_frames,
            num_layers=args.trajectory_num_layers,
            num_heads=args.trajectory_num_heads,
        )
        trajectory_head.to(accelerator.device, dtype=torch.float32)
    return feature_extractor, image_encoder, vae, unet, controlnet, trajectory_head


def _make_dataloader(args):
    train_dataset = SurgWMBench20AnchorDataset(
        dataset_root=args.dataset_root,
        manifest=args.train_manifest,
        image_size=(args.train_width, args.train_height),
        context_frames=args.context_frames,
        target_frames=args.target_frames,
        max_clips=args.max_clips,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_gpu_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=surgwmbench_collate,
        pin_memory=True,
    )


def main():
    args = parse_args()
    if args.context_frames != 5 or args.target_frames != 15:
        raise ValueError("This task is fixed to 5 context anchors and 15 target anchors.")
    if args.trajectory_input_noise_std < 0:
        raise ValueError("--trajectory-input-noise-std must be non-negative.")
    if args.trajectory_input_mask_prob < 0 or args.trajectory_input_mask_prob > 1:
        raise ValueError("--trajectory-input-mask-prob must be in [0, 1].")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    project_config = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "runs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        project_config=project_config,
    )
    weight_dtype = _weight_dtype(accelerator)

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "training_args.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)

    feature_extractor, image_encoder, vae, unet, controlnet, trajectory_head = _prepare_models(args, accelerator, weight_dtype)
    train_dataloader = _make_dataloader(args)

    trainable_params = list(controlnet.parameters()) + list(unet.conv_in.parameters())
    if trajectory_head is not None:
        trainable_params += list(trajectory_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    if trajectory_head is not None:
        unet, controlnet, trajectory_head, optimizer, train_dataloader = accelerator.prepare(
            unet, controlnet, trajectory_head, optimizer, train_dataloader
        )
    else:
        unet, controlnet, optimizer, train_dataloader = accelerator.prepare(unet, controlnet, optimizer, train_dataloader)

    steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch

    global_step = 0
    for epoch in range(args.num_train_epochs):
        controlnet.train()
        unet.train()
        if trajectory_head is not None:
            trajectory_head.train()
        for batch_idx, batch in enumerate(train_dataloader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            if trajectory_head is None:
                accumulate_context = accelerator.accumulate(controlnet)
            else:
                accumulate_context = accelerator.accumulate(controlnet, trajectory_head)
            with accumulate_context:
                context_frames = batch["context_frames"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                target_frames = batch["target_frames"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)

                with torch.no_grad():
                    target_latents = tensor_to_vae_latent(target_frames, vae, sample=True)
                    context_latents = tensor_to_vae_latent(context_frames, vae, sample=False)
                    image_tokens = encode_context_images(
                        context_frames,
                        feature_extractor,
                        image_encoder,
                        weight_dtype,
                        return_frame_tokens=trajectory_head is not None,
                    )

                if trajectory_head is None:
                    encoder_hidden_states = image_tokens.to(dtype=weight_dtype)
                    pred_coords_norm = None
                    target_coords_norm = None
                else:
                    anchor_coords_norm = batch["anchor_coords_norm"].to(accelerator.device, dtype=torch.float32, non_blocking=True)
                    context_coords_norm = anchor_coords_norm[:, : args.context_frames]
                    target_coords_norm = anchor_coords_norm[:, args.context_frames : args.context_frames + args.target_frames]
                    context_coords_norm = augment_context_trajectory_coords(
                        context_coords_norm,
                        noise_std=args.trajectory_input_noise_std,
                        mask_prob=args.trajectory_input_mask_prob,
                        mask_value=args.trajectory_input_mask_value,
                    )
                    trajectory_outputs = trajectory_head(image_tokens, context_coords_norm)
                    encoder_hidden_states = trajectory_outputs["encoder_hidden_states"].to(dtype=weight_dtype)
                    pred_coords_norm = trajectory_outputs["pred_coords_norm"]

                noise = torch.randn_like(target_latents)
                batch_size = target_latents.shape[0]
                sigmas = rand_cosine_interpolated([batch_size], device=target_latents.device, dtype=target_latents.dtype)
                sigmas_reshaped = sigmas
                while sigmas_reshaped.ndim < target_latents.ndim:
                    sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)

                noisy_latents = target_latents + noise * sigmas_reshaped
                inp_noisy_latents = noisy_latents / ((sigmas_reshaped**2 + 1) ** 0.5)
                inp_noisy_latents = build_5frame_latent_input(inp_noisy_latents, context_latents, vae.config.scaling_factor)

                if args.conditioning_dropout_prob:
                    keep = (torch.rand(batch_size, device=target_latents.device) >= args.conditioning_dropout_prob).view(batch_size, 1, 1, 1, 1)
                    inp_noisy_latents[:, :, 4:] = inp_noisy_latents[:, :, 4:] * keep.to(inp_noisy_latents.dtype)

                timesteps = torch.tensor([0.25 * sigma.log() for sigma in sigmas], device=target_latents.device, dtype=target_latents.dtype)
                added_time_ids = get_add_time_ids(
                    batch_size,
                    encoder_hidden_states.dtype,
                    target_latents.device,
                    accelerator.unwrap_model(unet),
                    fps=args.fps,
                    motion_bucket_id=args.motion_bucket_id,
                    noise_aug_strength=args.noise_aug_strength,
                )
                controls = make_zero_control_tensors(context_frames, args.target_frames, dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample, _, _ = controlnet(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    controlnet_cond=controls["controlnet_cond"],
                    controlnet_flow=controls["controlnet_flow"],
                    controlnet_depth=controls["controlnet_depth"],
                    controlnet_mask=controls["controlnet_mask"],
                    return_dict=False,
                )
                model_pred = unet(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                c_out = -sigmas_reshaped / ((sigmas_reshaped**2 + 1) ** 0.5)
                c_skip = 1 / (sigmas_reshaped**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas_reshaped**2) * (sigmas_reshaped**-2.0)
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() - target_latents.float()) ** 2).reshape(batch_size, -1),
                    dim=1,
                ).mean()
                image_loss = loss
                trajectory_loss = None
                trajectory_velocity_loss = None
                if trajectory_head is not None:
                    trajectory_loss = F.smooth_l1_loss(pred_coords_norm, target_coords_norm)
                    trajectory_velocity_loss = F.smooth_l1_loss(
                        pred_coords_norm[:, 1:] - pred_coords_norm[:, :-1],
                        target_coords_norm[:, 1:] - target_coords_norm[:, :-1],
                    )
                    loss = (
                        image_loss
                        + args.trajectory_loss_weight * trajectory_loss
                        + args.trajectory_velocity_loss_weight * trajectory_velocity_loss
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    if trajectory_head is None:
                        logger.info(
                            "epoch=%s step=%s task=image-only loss=%.6f image_loss=%.6f",
                            epoch,
                            global_step,
                            loss.detach().item(),
                            image_loss.detach().item(),
                        )
                    else:
                        logger.info(
                            "epoch=%s step=%s task=joint loss=%.6f image_loss=%.6f traj_loss=%.6f traj_vel_loss=%.6f",
                            epoch,
                            global_step,
                            loss.detach().item(),
                            image_loss.detach().item(),
                            trajectory_loss.detach().item(),
                            trajectory_velocity_loss.detach().item(),
                        )
                if global_step % args.checkpointing_steps == 0:
                    accelerator.save_state(str(output_dir / f"checkpoint-{global_step}"))
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        unwrapped_unet.save_pretrained(output_dir / "unet_context")
        unwrapped_controlnet.save_pretrained(output_dir / "controlnet")
        if trajectory_head is not None:
            unwrapped_trajectory_head = accelerator.unwrap_model(trajectory_head)
            save_trajectory_head(unwrapped_trajectory_head, output_dir / "trajectory_head.pt")
        logger.info("Saved SurgWMBench 20-anchor checkpoint to %s", output_dir)


if __name__ == "__main__":
    main()
