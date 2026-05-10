from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn


class TrajectoryPredictionHead(nn.Module):
    """Condition diffusion on observed trajectory points and predict future anchors."""

    def __init__(
        self,
        image_embed_dim: int,
        hidden_dim: int = 512,
        context_frames: int = 5,
        target_frames: int = 15,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.image_embed_dim = image_embed_dim
        self.hidden_dim = hidden_dim
        self.context_frames = context_frames
        self.target_frames = target_frames
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.image_proj = nn.Linear(image_embed_dim, hidden_dim)
        self.coord_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_time_embed = nn.Parameter(torch.zeros(context_frames, hidden_dim))
        self.token_type_embed = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.normal_(self.context_time_embed, std=0.02)
        nn.init.normal_(self.token_type_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.future_queries = nn.Parameter(torch.randn(target_frames, hidden_dim) * 0.02)
        self.future_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.condition_proj = nn.Linear(hidden_dim, image_embed_dim)
        self.coord_out = nn.Linear(hidden_dim, 2)

    def config_dict(self) -> Dict[str, int | float]:
        return {
            "image_embed_dim": self.image_embed_dim,
            "hidden_dim": self.hidden_dim,
            "context_frames": self.context_frames,
            "target_frames": self.target_frames,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }

    def forward(self, image_tokens: torch.Tensor, context_coords_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        if image_tokens.shape[1] != self.context_frames:
            raise ValueError(f"Expected {self.context_frames} image tokens, got {image_tokens.shape[1]}")
        if context_coords_norm.shape[1:] != (self.context_frames, 2):
            raise ValueError(f"Expected context coords shape [B, {self.context_frames}, 2], got {tuple(context_coords_norm.shape)}")

        batch_size = image_tokens.shape[0]
        image_hidden = self.image_proj(image_tokens.float())
        coord_hidden = self.coord_proj(context_coords_norm.float())
        time = self.context_time_embed.unsqueeze(0)
        image_hidden = image_hidden + time + self.token_type_embed[0].view(1, 1, -1)
        coord_hidden = coord_hidden + time + self.token_type_embed[1].view(1, 1, -1)

        memory = torch.cat([image_hidden, coord_hidden], dim=1)
        memory = self.context_encoder(memory)

        coord_condition_tokens = self.condition_proj(memory[:, self.context_frames :])
        all_tokens = torch.cat([image_tokens.float(), coord_condition_tokens], dim=1)
        encoder_hidden_states = all_tokens.mean(dim=1, keepdim=True)

        queries = self.future_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.future_decoder(queries, memory)
        pred_coords_norm = torch.sigmoid(self.coord_out(decoded))
        return {
            "encoder_hidden_states": encoder_hidden_states,
            "pred_coords_norm": pred_coords_norm,
        }


def infer_image_embed_dim(image_encoder) -> int:
    config = getattr(image_encoder, "config", None)
    projection_dim = getattr(config, "projection_dim", None)
    if isinstance(projection_dim, int):
        return projection_dim
    visual_projection = getattr(image_encoder, "visual_projection", None)
    out_features = getattr(visual_projection, "out_features", None)
    if isinstance(out_features, int):
        return out_features
    raise ValueError("Could not infer CLIP image embedding dimension from image_encoder")


def save_trajectory_head(model: TrajectoryPredictionHead, path: str | Path) -> None:
    torch.save({"config": model.config_dict(), "state_dict": model.state_dict()}, path)


def load_trajectory_head(path: str | Path, map_location=None) -> TrajectoryPredictionHead:
    payload = torch.load(path, map_location=map_location)
    model = TrajectoryPredictionHead(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model


def normalized_to_pixel_coords(coords_norm: torch.Tensor, original_size: Tuple[float, float]) -> torch.Tensor:
    scale = torch.tensor(original_size, device=coords_norm.device, dtype=coords_norm.dtype)
    return coords_norm * scale


def trajectory_ade_fde(pred_coords: torch.Tensor, target_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    distances = torch.linalg.vector_norm(pred_coords - target_coords, dim=-1)
    return distances.mean(), distances[..., -1].mean()
