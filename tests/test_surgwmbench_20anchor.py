from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from Training.surgwmbench_modeling import (
    augment_context_trajectory_coords,
    encode_context_images,
    expand_conv_in_channels,
    resize_dual_control_fusion,
)
from Training.train_utils.surgwmbench_dataset import SurgWMBench20AnchorDataset
from Training.trajectory_head import (
    TrajectoryPredictionHead,
    load_trajectory_head,
    normalized_to_pixel_coords,
    save_trajectory_head,
    trajectory_ade_fde,
)


DATASET_ROOT = Path("/mnt/hdd1/neurips2026_dataset_track/SurgWMBench")


@pytest.mark.skipif(not DATASET_ROOT.exists(), reason="SurgWMBench dataset is not available")
def test_surgwmbench_20anchor_dataset_loads_one_clip():
    dataset = SurgWMBench20AnchorDataset(
        dataset_root=str(DATASET_ROOT),
        manifest="manifests/train.jsonl",
        image_size=(64, 64),
        max_clips=1,
    )
    sample = dataset[0]

    assert sample["context_frames"].shape == (5, 3, 64, 64)
    assert sample["target_frames"].shape == (15, 3, 64, 64)
    assert sample["anchor_coords_px"].shape == (20, 2)
    assert sample["anchor_coords_norm"].shape == (20, 2)
    assert len(sample["sampled_indices"]) == 20
    assert sample["sampled_indices"][0] == 0
    assert sample["sampled_indices"][-1] == sample["num_frames"] - 1
    assert all(path.endswith(".png") for path in sample["context_frame_paths"] + sample["target_frame_paths"])


class FakeConfigurableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.config = SimpleNamespace(in_channels=8)

    def register_to_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)


def test_expand_conv_in_channels_to_five_context_frames():
    module = FakeConfigurableModule()
    old_condition = module.conv_in.weight[:, 4:8].detach().clone()

    expand_conv_in_channels(module, 24, context_frames=5)

    assert module.conv_in.in_channels == 24
    assert module.config.in_channels == 24
    for idx in range(5):
        start = 4 + idx * 4
        assert torch.allclose(module.conv_in.weight[:, start : start + 4], old_condition / 5)


class FakeDualControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.control_fusion_block = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(20, 20, kernel_size=(1, 1, 1)),
                    nn.Conv3d(20, 20, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                    nn.SiLU(),
                )
                for _ in range(4)
            ]
        )


def test_resize_dual_control_fusion_to_15_target_frames():
    module = FakeDualControlNet()

    resize_dual_control_fusion(module, flow_frames=14)

    for block in module.control_fusion_block:
        assert block[0].in_channels == 14
        assert block[0].out_channels == 14
        assert block[1].in_channels == 14
        assert block[1].out_channels == 14


class FakeFeatureExtractor:
    def __call__(self, images, **kwargs):
        return SimpleNamespace(pixel_values=images)


class FakeImageEncoder(nn.Module):
    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        image_embeds = torch.arange(batch * 4, dtype=pixel_values.dtype, device=pixel_values.device).view(batch, 4)
        return SimpleNamespace(image_embeds=image_embeds)


def test_encode_context_images_can_pool_or_return_frame_tokens():
    context_frames = torch.rand(2, 5, 3, 8, 8)
    feature_extractor = FakeFeatureExtractor()
    image_encoder = FakeImageEncoder()

    frame_tokens = encode_context_images(
        context_frames,
        feature_extractor,
        image_encoder,
        torch.float32,
        return_frame_tokens=True,
    )
    pooled_tokens = encode_context_images(context_frames, feature_extractor, image_encoder, torch.float32)

    assert frame_tokens.shape == (2, 5, 4)
    assert pooled_tokens.shape == (2, 1, 4)
    assert torch.equal(pooled_tokens[:, 0], frame_tokens.mean(dim=1))


def test_augment_context_trajectory_coords_masks_inputs_without_changing_source():
    coords = torch.full((2, 5, 2), 0.5)

    augmented = augment_context_trajectory_coords(coords, mask_prob=1.0, mask_value=-1.0)

    assert torch.equal(coords, torch.full((2, 5, 2), 0.5))
    assert torch.equal(augmented, torch.full((2, 5, 2), -1.0))


def test_augment_context_trajectory_coords_adds_clamped_noise():
    torch.manual_seed(0)
    coords = torch.full((4, 5, 2), 0.5)

    augmented = augment_context_trajectory_coords(coords, noise_std=0.5)

    assert augmented.shape == coords.shape
    assert not torch.equal(augmented, coords)
    assert torch.all(augmented >= 0.0)
    assert torch.all(augmented <= 1.0)


def test_trajectory_prediction_head_shapes_and_range():
    head = TrajectoryPredictionHead(
        image_embed_dim=32,
        hidden_dim=16,
        context_frames=5,
        target_frames=15,
        num_layers=1,
        num_heads=4,
    )

    outputs = head(torch.randn(2, 5, 32), torch.rand(2, 5, 2))

    assert outputs["encoder_hidden_states"].shape == (2, 10, 32)
    assert outputs["pred_coords_norm"].shape == (2, 15, 2)
    assert torch.all(outputs["pred_coords_norm"] >= 0)
    assert torch.all(outputs["pred_coords_norm"] <= 1)


def test_trajectory_head_save_and_load_round_trip(tmp_path):
    head = TrajectoryPredictionHead(
        image_embed_dim=32,
        hidden_dim=16,
        context_frames=5,
        target_frames=15,
        num_layers=1,
        num_heads=4,
    )
    path = tmp_path / "trajectory_head.pt"

    save_trajectory_head(head, path)
    loaded = load_trajectory_head(path, map_location="cpu")

    assert loaded.config_dict() == head.config_dict()
    for key, value in head.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_trajectory_coordinate_helpers():
    coords_norm = torch.tensor([[[0.5, 0.25], [1.0, 0.0]]])
    coords_px = normalized_to_pixel_coords(coords_norm, (200, 100))
    expected_px = torch.tensor([[[100.0, 25.0], [200.0, 0.0]]])

    assert torch.allclose(coords_px, expected_px)

    pred = torch.tensor([[[3.0, 4.0], [0.0, 10.0]]])
    target = torch.zeros_like(pred)
    ade, fde = trajectory_ade_fde(pred, target)

    assert torch.isclose(ade, torch.tensor(7.5))
    assert torch.isclose(fde, torch.tensor(10.0))
