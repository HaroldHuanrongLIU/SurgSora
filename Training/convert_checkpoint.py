#!/usr/bin/env python
"""Convert an accelerator save_state checkpoint into the from_pretrained layout that
eval_surgwmbench_20anchor.py expects (unet_context/ + controlnet/ subdirs)."""
import argparse
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Training.surgwmbench_modeling import expand_conv_in_channels, resize_dual_control_fusion
from Training.trajectory_head import TrajectoryPredictionHead, save_trajectory_head
from models.Control_Backbone import UNetControlNetModel
from models.Control_Encoder import DualFlowControlNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-name-or-path", default="./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--target-frames", type=int, default=15)
    parser.add_argument("--image-embed-dim", type=int, default=1024)
    parser.add_argument("--trajectory-hidden-dim", type=int, default=512)
    parser.add_argument("--trajectory-num-layers", type=int, default=2)
    parser.add_argument("--trajectory-num-heads", type=int, default=8)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    unet = UNetControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=True, variant="fp16"
    )
    expand_conv_in_channels(unet, 4 + args.context_frames * 4, context_frames=args.context_frames)
    unet.register_to_config(num_frames=args.target_frames)

    controlnet = DualFlowControlNet.from_unet(unet)
    controlnet.register_to_config(num_frames=args.target_frames)
    resize_dual_control_fusion(controlnet, args.target_frames - 1)

    unet_state = load_file(ckpt / "model.safetensors")
    controlnet_state = load_file(ckpt / "model_1.safetensors")
    unet.load_state_dict(unet_state)
    controlnet.load_state_dict(controlnet_state)

    unet.save_pretrained(out / "unet_context")
    controlnet.save_pretrained(out / "controlnet")
    trajectory_state_path = ckpt / "model_2.safetensors"
    if trajectory_state_path.exists():
        trajectory_head = TrajectoryPredictionHead(
            image_embed_dim=args.image_embed_dim,
            hidden_dim=args.trajectory_hidden_dim,
            context_frames=args.context_frames,
            target_frames=args.target_frames,
            num_layers=args.trajectory_num_layers,
            num_heads=args.trajectory_num_heads,
        )
        trajectory_head.load_state_dict(load_file(trajectory_state_path))
        save_trajectory_head(trajectory_head, out / "trajectory_head.pt")
        print(f"Wrote {out / 'unet_context'}, {out / 'controlnet'}, and {out / 'trajectory_head.pt'}")
    else:
        print(f"Wrote {out / 'unet_context'} and {out / 'controlnet'}; no model_2.safetensors trajectory head found")


if __name__ == "__main__":
    main()
