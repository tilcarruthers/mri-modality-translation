from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from mri_translation.data.transforms import to_float_tensor


def sample_intensities(dataset, key: str, sample_step: int = 10) -> np.ndarray:
    values = []
    for idx in range(0, len(dataset), max(1, sample_step)):
        tensor = to_float_tensor(dataset[idx][key])
        values.append(tensor.flatten().numpy())
    return np.concatenate(values, axis=0)


def compute_global_minmax(
    dataset, keys: Iterable[str], sample_step: int = 10
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for key in keys:
        mins, maxs = [], []
        for idx in range(0, len(dataset), max(1, sample_step)):
            tensor = to_float_tensor(dataset[idx][key])
            mins.append(float(torch.min(tensor).item()))
            maxs.append(float(torch.max(tensor).item()))
        stats[key] = {"min": min(mins), "max": max(maxs)}
    return stats


def compute_percentiles(
    dataset,
    keys: Iterable[str],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    sample_step: int = 10,
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for key in keys:
        intensities = sample_intensities(dataset, key=key, sample_step=sample_step)
        stats[key] = {
            "min": float(np.percentile(intensities, lower_percentile)),
            "max": float(np.percentile(intensities, upper_percentile)),
        }
    return stats
