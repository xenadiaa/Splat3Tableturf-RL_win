from __future__ import annotations

import torch
from torch import Tensor, nn


class StateEncoder(nn.Module):
    def __init__(self, map_channels: int, scalar_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.map_net = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4 + 64, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, map_obs: Tensor, scalar_obs: Tensor) -> Tensor:
        map_emb = self.map_net(map_obs)
        scalar_emb = self.scalar_net(scalar_obs)
        return self.fc(torch.cat([map_emb, scalar_emb], dim=-1))


class DQNNet(nn.Module):
    """Q-network over dynamic legal action candidates."""

    def __init__(
        self,
        map_channels: int,
        scalar_dim: int,
        action_feature_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_encoder = StateEncoder(map_channels, scalar_dim, hidden_dim=hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + action_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_single(self, map_obs: Tensor, scalar_obs: Tensor, action_features: Tensor) -> Tensor:
        if map_obs.ndim != 3:
            raise ValueError("map_obs must have shape [C,H,W]")
        if scalar_obs.ndim != 1:
            raise ValueError("scalar_obs must have shape [S]")
        if action_features.ndim != 2:
            raise ValueError("action_features must have shape [A,F]")

        state_emb = self.state_encoder(map_obs.unsqueeze(0), scalar_obs.unsqueeze(0)).squeeze(0)
        num_actions = action_features.shape[0]
        expanded_state = state_emb.unsqueeze(0).expand(num_actions, -1)
        return self.q_head(torch.cat([expanded_state, action_features], dim=-1)).squeeze(-1)
