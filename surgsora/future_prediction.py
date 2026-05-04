from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from benchmark.surgwmbench import BaselineSpec, run_cli


def predict_native_surgsora_frames(
    dataset_root: Path,
    window: object,
    args: argparse.Namespace,
) -> List[np.ndarray]:
    request_path = write_native_request(dataset_root, window, args)
    checkpoint = Path(str(args.checkpoint))
    if checkpoint.suffix == ".json":
        raise RuntimeError(
            "SurgSora native prediction requires real SurgSora/controlnet checkpoints, "
            f"not the adapter metadata checkpoint: {checkpoint}. "
            f"Wrote the SurgWMBench-to-SurgSora native request to {request_path}."
        )

    missing_assets = missing_required_assets(args)
    if missing_assets:
        formatted = ", ".join(f"{name}={path}" for name, path in missing_assets.items())
        raise RuntimeError(
            "SurgSora native prediction requires the upstream Gradio pipeline assets "
            f"before generation can run. Missing: {formatted}. "
            f"Wrote the SurgWMBench-to-SurgSora native request to {request_path}. "
            "Set SURGSORA_CONTROLNET_CKPT, SURGSORA_DAV2_CKPT, SURGSORA_SAM_CKPT, "
            "and SURGSORA_CMP_CONFIG as needed."
        )

    raise RuntimeError(
        "SurgSora native request preparation is implemented, but non-interactive "
        "generation is intentionally not launched from the benchmark adapter yet because "
        "the upstream release exposes generation through gradio_demo_run.py, whose module "
        "initializes CUDA models and launches a Gradio app at import time. "
        f"Use the request JSON at {request_path} to run the SurgSora pipeline after "
        "extracting the Gradio runtime into a non-interactive command."
    )


def write_native_request(dataset_root: Path, window: object, args: argparse.Namespace) -> Path:
    first_frame_path = dataset_root / window.context_frame_paths[-1]
    with Image.open(first_frame_path) as image:
        width, height = image.size

    predicted_coords = predict_trajectory(
        window.context_coords,
        int(args.prediction_horizon),
        str(args.trajectory_predictor),
    )
    start = list(window.context_coords[-1])
    tracking_points_norm = [[start, *predicted_coords]]
    tracking_points_px = [
        [[float(point[0]) * width, float(point[1]) * height] for point in track]
        for track in tracking_points_norm
    ]
    request_dir = args.output.parent / "native_requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"{safe_id(window.clip_id)}_h{args.prediction_horizon}.json"
    payload: Dict[str, Any] = {
        "dataset_name": "SurgWMBench",
        "baseline": "surgsora",
        "model": "SurgSora",
        "clip_id": window.clip_id,
        "data_track": window.data_track,
        "prediction_task": args.prediction_task,
        "context_frames": args.context_frames,
        "prediction_horizon": args.prediction_horizon,
        "first_frame_path": str(first_frame_path),
        "context_indices": window.context_indices,
        "future_indices": window.future_indices,
        "context_coords_norm": window.context_coords,
        "predicted_future_coords_norm": predicted_coords,
        "surgsora_tracking_points_norm": tracking_points_norm,
        "surgsora_tracking_points_px": tracking_points_px,
        "image_width": width,
        "image_height": height,
        "motion_brush_mask": "default_all_zero",
        "bbox_points": [],
        "checkpoint_assets": required_assets(args),
        "native_entrypoint": "gradio_demo_run.py",
        "notes": (
            "SurgSora controls generation from a first frame and object trajectory. "
            "This request uses the last context anchor frame as the first frame and "
            "the adapter trajectory predictor as the future control path."
        ),
    }
    request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return request_path


def predict_trajectory(
    context_coords: Sequence[Sequence[float]],
    horizon: int,
    predictor: str,
) -> List[List[float]]:
    context = np.asarray(context_coords, dtype=np.float64)
    if predictor == "copy_last" or len(context) < 2:
        step = np.zeros(2, dtype=np.float64)
    else:
        step = context[-1] - context[-2]
    start = context[-1]
    predictions = [np.clip(start + step * (idx + 1), 0.0, 1.0).tolist() for idx in range(horizon)]
    return [[float(item[0]), float(item[1])] for item in predictions]


def required_assets(args: argparse.Namespace) -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "SURGSORA_CONTROLNET_CKPT": os.environ.get("SURGSORA_CONTROLNET_CKPT", str(args.checkpoint)),
        "SURGSORA_DAV2_CKPT": os.environ.get(
            "SURGSORA_DAV2_CKPT",
            str(repo_root / "models" / "DAV2" / "depth_anything_v2_vitb.pth"),
        ),
        "SURGSORA_SAM_CKPT": os.environ.get(
            "SURGSORA_SAM_CKPT",
            str(repo_root / "models" / "sam" / "sam_vit_h_4b8939.pth"),
        ),
        "SURGSORA_CMP_CONFIG": os.environ.get(
            "SURGSORA_CMP_CONFIG",
            str(
                repo_root
                / "models"
                / "cmp"
                / "experiments"
                / "semiauto_annot"
                / "resnet50_vip+mpii_liteflow"
                / "config.yaml"
            ),
        ),
    }


def missing_required_assets(args: argparse.Namespace) -> Dict[str, str]:
    missing: Dict[str, str] = {}
    for name, value in required_assets(args).items():
        if not value or not Path(value).exists():
            missing[name] = value
    return missing


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


SPEC = BaselineSpec(
    baseline="surgsora",
    model="SurgSora",
    native_entrypoint="gradio_demo_run.py",
    native_train_entrypoint="Training/train_stage2.py",
    native_frame_predictor=(
        "SurgWMBench-to-SurgSora native request preparation is implemented. "
        "Full generation still needs the upstream Gradio runtime extracted into a "
        "non-interactive command with DAV2/CMP/SAM/controlnet assets."
    ),
    native_frame_predictor_fn=predict_native_surgsora_frames,
    notes=(
        "SurgSora is object/trajectory-control oriented. The SurgWMBench adapter "
        "validates official manifests and trajectory targets, writes native request "
        "artifacts, and keeps dense pseudo-coordinate metrics separate from the primary "
        "sparse human-anchor track."
    ),
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli(SPEC, argv)


if __name__ == "__main__":
    main()
