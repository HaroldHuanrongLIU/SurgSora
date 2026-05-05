import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resolve_existing_path(dataset_root: Path, relative_path: str) -> Path:
    path = dataset_root / relative_path
    if path.exists():
        return path
    if "\uf021" in relative_path:
        alt = dataset_root / relative_path.replace("\uf021", "*")
        if alt.exists():
            return alt
    raise FileNotFoundError(path)


def _load_rgb_tensor(path: Path, size: Optional[Tuple[int, int]]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.BILINEAR)
    data = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
    return data.permute(2, 0, 1).float().div_(255.0)


class SurgWMBench20AnchorDataset(Dataset):
    """Manifest-backed SurgWMBench loader for 5-context to 15-target anchor prediction."""

    def __init__(
        self,
        dataset_root: str,
        manifest: str = "manifests/train.jsonl",
        image_size: Tuple[int, int] = (256, 256),
        context_frames: int = 5,
        target_frames: int = 15,
        max_clips: Optional[int] = None,
        load_pixels: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = self.dataset_root / manifest
        self.image_size = image_size
        self.context_frames = context_frames
        self.target_frames = target_frames
        self.load_pixels = load_pixels

        if context_frames != 5 or target_frames != 15:
            raise ValueError("SurgWMBench 20-anchor task requires context_frames=5 and target_frames=15.")

        rows: List[Dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if max_clips is not None and idx >= max_clips:
                    break
                rows.append(json.loads(line))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _load_annotation(self, row: Dict[str, Any]) -> Dict[str, Any]:
        annotation_path = _resolve_existing_path(self.dataset_root, row["annotation_path"])
        return json.loads(annotation_path.read_text(encoding="utf-8"))

    def _anchor_frame_records(self, annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
        sampled_indices = annotation["sampled_indices"]
        frames = annotation["frames"]
        if len(sampled_indices) != 20:
            raise ValueError(f"Expected 20 sampled_indices, got {len(sampled_indices)}")
        records = [frames[index] for index in sampled_indices]
        for anchor_idx, record in enumerate(records):
            if not record["is_human_labeled"] or record["anchor_idx"] != anchor_idx:
                raise ValueError("sampled_indices do not align with the 20 human anchors")
        return records

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        annotation = self._load_annotation(row)
        anchor_records = self._anchor_frame_records(annotation)

        original_size = (
            int(annotation["image_size"]["width"]),
            int(annotation["image_size"]["height"]),
        )
        context_records = anchor_records[: self.context_frames]
        target_records = anchor_records[self.context_frames : self.context_frames + self.target_frames]
        if len(target_records) != self.target_frames:
            raise ValueError("Clip does not contain enough target anchors")

        context_paths = [_resolve_existing_path(self.dataset_root, item["frame_path"]) for item in context_records]
        target_paths = [_resolve_existing_path(self.dataset_root, item["frame_path"]) for item in target_records]

        sample: Dict[str, Any] = {
            "dataset_version": row["dataset_version"],
            "patient_id": row["patient_id"],
            "source_video_id": row["source_video_id"],
            "trajectory_id": row["trajectory_id"],
            "difficulty": row.get("difficulty"),
            "num_frames": int(row["num_frames"]),
            "sampled_indices": list(row["sampled_indices"]),
            "original_size": original_size,
            "context_frame_paths": [str(path) for path in context_paths],
            "target_frame_paths": [str(path) for path in target_paths],
            "anchor_coords_px": torch.tensor(
                [anchor["coord_px"] for anchor in annotation["human_anchors"]],
                dtype=torch.float32,
            ),
            "anchor_coords_norm": torch.tensor(
                [anchor["coord_norm"] for anchor in annotation["human_anchors"]],
                dtype=torch.float32,
            ),
        }

        if self.load_pixels:
            context = torch.stack([_load_rgb_tensor(path, self.image_size) for path in context_paths], dim=0)
            target = torch.stack([_load_rgb_tensor(path, self.image_size) for path in target_paths], dim=0)
            sample["context_frames"] = context
            sample["target_frames"] = target

        return sample


def surgwmbench_collate(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    tensor_keys = {"context_frames", "target_frames", "anchor_coords_px", "anchor_coords_norm"}
    for key in samples[0].keys():
        values = [sample[key] for sample in samples]
        if key in tensor_keys:
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values
    return batch
