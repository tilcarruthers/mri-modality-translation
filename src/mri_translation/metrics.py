from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity


def mse_per_pixel(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


def mae_per_pixel(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def rmse_per_pixel(pred: torch.Tensor, target: torch.Tensor) -> float:
    return math.sqrt(mse_per_pixel(pred, target))


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = mse_per_pixel(pred, target)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim_per_image(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    pred_np = pred.detach().cpu().squeeze().numpy()
    target_np = target.detach().cpu().squeeze().numpy()
    return float(structural_similarity(target_np, pred_np, data_range=data_range))


def accumulate_error_sums(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, int]:
    diff = pred - target
    sq_error_sum = float(torch.sum(diff ** 2).item())
    abs_error_sum = float(torch.sum(torch.abs(diff)).item())
    numel = diff.numel()
    return sq_error_sum, abs_error_sum, numel
