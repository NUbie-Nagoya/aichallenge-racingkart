"""Compact 1-D CNN + GRU maneuver policy and deployment wrapper."""

from __future__ import annotations

from typing import Optional, Tuple  # noqa: UP035 - TorchScript 1.8 parser contract

import numpy as np
import torch
from torch import Tensor, nn

from .normalization import Normalizer


class TemporalPolicy(nn.Module):
    def __init__(
        self, lidar_embedding: int = 32, aux_embedding: int = 16, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.lidar_embedding = lidar_embedding
        self.aux_embedding = aux_embedding
        self.hidden_size = hidden_size
        self.scan_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=3, padding=4),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, stride=3, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(16 * 8, lidar_embedding),
            nn.ReLU(),
        )
        self.aux_encoder = nn.Sequential(
            nn.Linear(11, 32), nn.ReLU(), nn.Linear(32, aux_embedding), nn.ReLU()
        )
        self.gru = nn.GRU(
            lidar_embedding + aux_embedding, hidden_size, batch_first=True
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 2), nn.Tanh()
        )

    def forward(
        self,
        lidar: Tensor,
        aux: Tensor,
        hidden: Optional[Tensor] = None,  # noqa: UP045 - TorchScript 1.8
    ) -> Tuple[Tensor, Tensor]:  # noqa: UP006 - TorchScript 1.8
        if (
            lidar.dim() != 3
            or aux.dim() != 3
            or lidar.size(2) != 360
            or aux.size(2) != 11
        ):
            raise ValueError("expected lidar [B,T,360] and aux [B,T,11]")
        if lidar.size(0) != aux.size(0) or lidar.size(1) != aux.size(1):
            raise ValueError("lidar and auxiliary batch/time shapes differ")
        batch, time = lidar.size(0), lidar.size(1)
        scan = self.scan_encoder(lidar.reshape(batch * time, 1, 360)).reshape(
            batch, time, self.lidar_embedding
        )
        state = self.aux_encoder(aux.reshape(batch * time, 11)).reshape(
            batch, time, self.aux_embedding
        )
        sequence, next_hidden = self.gru(torch.cat((scan, state), dim=2), hidden)
        return self.action_head(sequence[:, -1]), next_hidden


class ExportPolicy(nn.Module):
    """Owns raw-input normalization and normalized-to-physical action scaling."""

    def __init__(
        self,
        policy: TemporalPolicy,
        lidar_mean: Tensor,
        lidar_std: Tensor,
        aux_mean: Tensor,
        aux_std: Tensor,
        action_low: Tensor,
        action_high: Tensor,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.register_buffer("lidar_mean", lidar_mean)
        self.register_buffer("lidar_std", lidar_std)
        self.register_buffer("aux_mean", aux_mean)
        self.register_buffer("aux_std", aux_std)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

    @classmethod
    def from_normalizer(
        cls,
        policy: TemporalPolicy,
        normalizer: Normalizer,
        action_low: tuple[float, float],
        action_high: tuple[float, float],
    ) -> ExportPolicy:
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return cls(
            policy,
            tensor(normalizer.lidar_mean),
            tensor(normalizer.lidar_std),
            tensor(normalizer.aux_mean),
            tensor(normalizer.aux_std),
            tensor(action_low),
            tensor(action_high),
        )

    def forward(self, raw_lidar: Tensor, raw_aux: Tensor) -> Tensor:
        normalized_lidar = (raw_lidar - self.lidar_mean) / self.lidar_std
        normalized_aux = (raw_aux - self.aux_mean) / self.aux_std
        normalized_action, _ = self.policy(normalized_lidar, normalized_aux)
        return self.action_low + (normalized_action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )
