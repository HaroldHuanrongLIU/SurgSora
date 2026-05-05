from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from Training.surgwmbench_modeling import expand_conv_in_channels, resize_dual_control_fusion
from Training.train_utils.surgwmbench_dataset import SurgWMBench20AnchorDataset


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

