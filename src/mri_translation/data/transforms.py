from __future__ import annotations

import torch


def to_float_tensor(image) -> torch.Tensor:
    tensor = image if isinstance(image, torch.Tensor) else torch.tensor(image)
    tensor = tensor.float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def clamp_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, 0.0, 1.0)
