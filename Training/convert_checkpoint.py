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
from models.Control_Backbone import UNetControlNetModel
from models.Control_Encoder import DualFlowControlNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-name-or-path", default="./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--target-frames", type=int, default=15)
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
    print(f"Wrote {out / 'unet_context'} and {out / 'controlnet'}")


if __name__ == "__main__":
    main()
