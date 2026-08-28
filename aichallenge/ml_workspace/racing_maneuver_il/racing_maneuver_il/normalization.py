"""Train-only raw-input normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Normalizer:
    lidar_mean: np.ndarray
    lidar_std: np.ndarray
    aux_mean: np.ndarray
    aux_std: np.ndarray

    @classmethod
    def fit(
        cls,
        lidar: np.ndarray,
        aux: np.ndarray,
        train_indices: np.ndarray,
        epsilon: float = 1e-6,
    ) -> Normalizer:
        if len(train_indices) == 0:
            raise ValueError("cannot fit normalizer without training frames")
        selected_lidar = np.asarray(lidar[train_indices], dtype=np.float64)
        selected_aux = np.asarray(aux[train_indices], dtype=np.float64)
        lmean, amean = selected_lidar.mean(axis=0), selected_aux.mean(axis=0)
        lstd, astd = selected_lidar.std(axis=0), selected_aux.std(axis=0)
        return cls(
            lmean.astype(np.float32),
            np.maximum(lstd, epsilon).astype(np.float32),
            amean.astype(np.float32),
            np.maximum(astd, epsilon).astype(np.float32),
        )

    def transform(
        self, lidar: np.ndarray, aux: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return ((np.asarray(lidar) - self.lidar_mean) / self.lidar_std).astype(
            np.float32
        ), ((np.asarray(aux) - self.aux_mean) / self.aux_std).astype(np.float32)

    def inverse_transform(
        self, lidar: np.ndarray, aux: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return ((np.asarray(lidar) * self.lidar_std) + self.lidar_mean).astype(
            np.float32
        ), ((np.asarray(aux) * self.aux_std) + self.aux_mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            name: getattr(self, name).tolist()
            for name in ("lidar_mean", "lidar_std", "aux_mean", "aux_std")
        }

    @classmethod
    def from_dict(cls, values: dict[str, list[float]]) -> Normalizer:
        return cls(
            *(
                np.asarray(values[name], dtype=np.float32)
                for name in ("lidar_mean", "lidar_std", "aux_mean", "aux_std")
            )
        )
