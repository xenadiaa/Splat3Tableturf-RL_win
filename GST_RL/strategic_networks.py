from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class StrategicStateEncoder(nn.Module):
    def __init__(self, map_channels: int, scalar_dim: int, hidden_dim: int = 320) -> None:
        super().__init__()
        self.map_net = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4 + 96, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, map_obs: Tensor, scalar_obs: Tensor) -> Tensor:
        map_emb = self.map_net(map_obs)
        scalar_emb = self.scalar_net(scalar_obs)
        return self.fc(torch.cat([map_emb, scalar_emb], dim=-1))


class StrategicPolicyValueNet(nn.Module):
    """Separate PPO model for strategic shaping rewards and richer state statistics."""

    def __init__(
        self,
        map_channels: int,
        scalar_dim: int,
        action_feature_dim: int,
        hidden_dim: int = 320,
    ) -> None:
        super().__init__()
        self.state_encoder = StrategicStateEncoder(map_channels, scalar_dim, hidden_dim=hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + action_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward_single(self, map_obs: Tensor, scalar_obs: Tensor, action_features: Tensor) -> Tuple[Tensor, Tensor]:
        if map_obs.ndim != 3:
            raise ValueError("map_obs must have shape [C,H,W]")
        if scalar_obs.ndim != 1:
            raise ValueError("scalar_obs must have shape [S]")
        if action_features.ndim != 2:
            raise ValueError("action_features must have shape [A,F]")

        state_emb = self.state_encoder(map_obs.unsqueeze(0), scalar_obs.unsqueeze(0)).squeeze(0)
        num_actions = action_features.shape[0]
        expanded_state = state_emb.unsqueeze(0).expand(num_actions, -1)
        logits = self.action_head(torch.cat([expanded_state, action_features], dim=-1)).squeeze(-1)
        value = self.value_head(state_emb).squeeze(-1)
        return logits, value
