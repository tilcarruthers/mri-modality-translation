from __future__ import annotations

from dataclasses import dataclass

import torch

from mri_translation.data.stats import compute_global_minmax, compute_percentiles
from mri_translation.data.transforms import clamp_unit_range


@dataclass
class RangeNormalizer:
    stats: dict[str, dict[str, float]]

    def normalize(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        min_val = self.stats[key]["min"]
        max_val = self.stats[key]["max"]
        denom = max(max_val - min_val, 1e-8)
        return clamp_unit_range((tensor - min_val) / denom)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return self.stats


def build_normalizer(dataset, data_config: dict) -> RangeNormalizer:
    normalization = data_config.get("normalization", {})
    method = normalization.get("method", "global_minmax")
    sample_step = normalization.get("sample_step", 10)
    keys = [data_config["input_key"], data_config["target_key"]]

    if method == "global_minmax":
        stats = compute_global_minmax(dataset, keys=keys, sample_step=sample_step)
    elif method == "percentile_minmax":
        stats = compute_percentiles(
            dataset,
            keys=keys,
            lower_percentile=normalization.get("lower_percentile", 1.0),
            upper_percentile=normalization.get("upper_percentile", 99.0),
            sample_step=sample_step,
        )
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return RangeNormalizer(stats=stats)
