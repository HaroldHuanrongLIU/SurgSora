import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def tensor_to_vae_latent(frames: torch.Tensor, vae, sample: bool = True) -> torch.Tensor:
    """Encode [B, F, C, H, W] RGB frames in [0, 1] to scaled VAE latents."""
    video_length = frames.shape[1]
    flat = rearrange(frames, "b f c h w -> (b f) c h w")
    distribution = vae.encode(flat).latent_dist
    latents = distribution.sample() if sample else distribution.mode()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    return latents * vae.config.scaling_factor


def decode_latents_to_frames(latents: torch.Tensor, vae, decode_chunk_size: int = 8) -> torch.Tensor:
    """Decode scaled VAE latents to [B, F, C, H, W] frames in [0, 1]."""
    batch_size, num_frames = latents.shape[:2]
    flat = rearrange(latents / vae.config.scaling_factor, "b f c h w -> (b f) c h w")
    decoded = []
    for start in range(0, flat.shape[0], decode_chunk_size):
        chunk = flat[start : start + decode_chunk_size]
        decoded.append(vae.decode(chunk, num_frames=chunk.shape[0]).sample)
    frames = torch.cat(decoded, dim=0)
    frames = rearrange(frames, "(b f) c h w -> b f c h w", b=batch_size, f=num_frames)
    return frames.float().add_(1.0).div_(2.0).clamp_(0.0, 1.0)


def encode_context_images(
    context_frames: torch.Tensor,
    feature_extractor,
    image_encoder,
    dtype: torch.dtype,
    return_frame_tokens: bool = False,
) -> torch.Tensor:
    """Encode context frames as CLIP tokens.

    By default this returns one mean-pooled token with shape [B, 1, D]. Set
    return_frame_tokens=True when downstream modules need one token per context
    frame, returning [B, F, D].
    """
    batch_size, context_count = context_frames.shape[:2]
    flat = rearrange(context_frames, "b f c h w -> (b f) c h w")
    flat = flat * 2.0 - 1.0
    flat = F.interpolate(flat, size=(224, 224), mode="bicubic", align_corners=True)
    flat = (flat + 1.0) / 2.0
    pixel_values = feature_extractor(
        images=flat,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values
    pixel_values = pixel_values.to(device=context_frames.device, dtype=dtype)
    embeds = image_encoder(pixel_values).image_embeds
    tokens = embeds.view(batch_size, context_count, -1)
    if return_frame_tokens:
        return tokens
    return tokens.mean(dim=1, keepdim=True)


def augment_context_trajectory_coords(
    context_coords_norm: torch.Tensor,
    noise_std: float = 0.0,
    mask_prob: float = 0.0,
    mask_value: float = -1.0,
) -> torch.Tensor:
    """Add training-time robustness noise to observed normalized trajectory points."""
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    if mask_prob < 0 or mask_prob > 1:
        raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")
    if noise_std == 0 and mask_prob == 0:
        return context_coords_norm

    augmented = context_coords_norm.clone()
    if noise_std > 0:
        augmented = (augmented + torch.randn_like(augmented) * noise_std).clamp(0.0, 1.0)
    if mask_prob > 0:
        mask = torch.rand(augmented.shape[:-1], device=augmented.device) < mask_prob
        augmented = augmented.masked_fill(mask.unsqueeze(-1), mask_value)
    return augmented


def build_5frame_latent_input(
    noisy_target_latents: torch.Tensor,
    context_latents: torch.Tensor,
    vae_scaling_factor: float,
) -> torch.Tensor:
    """Concatenate noisy target latents with five context latent blocks."""
    batch_size, target_count = noisy_target_latents.shape[:2]
    context_count = context_latents.shape[1]
    context_blocks = context_latents / vae_scaling_factor
    context_blocks = context_blocks.reshape(batch_size, 1, context_count * context_blocks.shape[2], *context_blocks.shape[-2:])
    context_blocks = context_blocks.repeat(1, target_count, 1, 1, 1)
    return torch.cat([noisy_target_latents, context_blocks], dim=2)


def expand_conv_in_channels(module: nn.Module, new_in_channels: int, context_frames: int = 5) -> None:
    """Expand a SurgSora/SVD input conv from 8 channels to 4 + context_frames * 4 channels."""
    old_conv = module.conv_in
    if old_conv.in_channels == new_in_channels:
        return
    if old_conv.in_channels != 8:
        raise ValueError(f"Expected old conv_in to have 8 channels, got {old_conv.in_channels}")
    expected_channels = 4 + context_frames * 4
    if new_in_channels != expected_channels:
        raise ValueError(f"Expected {expected_channels} input channels, got {new_in_channels}")

    new_conv = nn.Conv2d(
        new_in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :4].copy_(old_conv.weight[:, :4])
        condition_weight = old_conv.weight[:, 4:8] / float(context_frames)
        for idx in range(context_frames):
            start = 4 + idx * 4
            new_conv.weight[:, start : start + 4].copy_(condition_weight)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    module.conv_in = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
    if hasattr(module, "register_to_config"):
        module.register_to_config(in_channels=new_in_channels)


def make_zero_control_tensors(
    context_frames: torch.Tensor,
    target_frames: int,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Create non-leaking ControlNet conditions from the observed context only."""
    dtype = dtype or context_frames.dtype
    batch_size, _, _, height, width = context_frames.shape
    device = context_frames.device
    flow = torch.zeros(batch_size, target_frames - 1, 2, height, width, device=device, dtype=dtype)
    depth = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, 256, height // 4, width // 4, device=device, dtype=dtype)
    return {
        "controlnet_cond": context_frames[:, -1].to(dtype=dtype),
        "controlnet_flow": flow,
        "controlnet_depth": depth,
        "controlnet_mask": mask,
    }


def resize_dual_control_fusion(controlnet: nn.Module, flow_frames: int) -> None:
    """Resize SurgSora DualFlowControlNet fusion blocks from the original 20-flow setup."""
    blocks = getattr(controlnet, "control_fusion_block", None)
    if blocks is None:
        return
    for idx, block in enumerate(blocks):
        first = block[0]
        if first.in_channels == flow_frames and first.out_channels == flow_frames:
            continue
        new_block = nn.Sequential(
            nn.Conv3d(flow_frames, flow_frames, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(flow_frames, flow_frames, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)),
            nn.SiLU(),
        ).to(device=first.weight.device, dtype=first.weight.dtype)
        for old_conv, new_conv in zip((block[0], block[1]), (new_block[0], new_block[1])):
            with torch.no_grad():
                new_conv.weight.zero_()
                out_count = min(old_conv.weight.shape[0], new_conv.weight.shape[0])
                in_count = min(old_conv.weight.shape[1], new_conv.weight.shape[1])
                new_conv.weight[:out_count, :in_count].copy_(old_conv.weight[:out_count, :in_count])
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.zero_()
                    new_conv.bias[:out_count].copy_(old_conv.bias[:out_count])
        blocks[idx] = new_block


def get_add_time_ids(
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    unet,
    fps: int = 6,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
) -> torch.Tensor:
    add_time_ids = torch.tensor(
        [[fps, motion_bucket_id, noise_aug_strength]],
        dtype=dtype,
        device=device,
    ).repeat(batch_size, 1)
    expected = unet.add_embedding.linear_1.in_features
    actual = unet.config.addition_time_embed_dim * add_time_ids.shape[1]
    if expected != actual:
        raise ValueError(f"UNet expects added time dim {expected}, got {actual}")
    return add_time_ids


def rand_cosine_interpolated(
    shape,
    image_d: int = 64,
    noise_d_low: int = 32,
    noise_d_high: int = 64,
    sigma_data: float = 0.5,
    min_value: float = 0.002,
    max_value: float = 700,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(low, high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = torch.rand(shape, dtype=dtype, device=device)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data
