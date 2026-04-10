from __future__ import annotations

from pathlib import Path

import torch
from torch.amp import autocast

from mri_translation.engine.train import resolve_device
from mri_translation.metrics import (
    accumulate_error_sums,
    psnr_per_image,
    ssim_per_image,
)


def load_checkpoint(model, checkpoint_path: str | Path, device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    metric_names: list[str],
    checkpoint_path: str | Path | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    device = resolve_device(device) if isinstance(device, str) else device
    model = model.to(device)
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, device)

    model.eval()

    sq_error_sum = 0.0
    abs_error_sum = 0.0
    total_numel = 0
    psnr_values = []
    ssim_values = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        if device.type == "cuda":
            with autocast(device_type="cuda"):
                pred = model(x)
        else:
            pred = model(x)

        pred = torch.clamp(pred, 0.0, 1.0)

        batch_sq_error_sum, batch_abs_error_sum, batch_numel = accumulate_error_sums(pred, y)
        sq_error_sum += batch_sq_error_sum
        abs_error_sum += batch_abs_error_sum
        total_numel += batch_numel

        for i in range(pred.size(0)):
            psnr_values.append(psnr_per_image(pred[i], y[i]))
            ssim_values.append(ssim_per_image(pred[i], y[i]))

    mse = sq_error_sum / max(1, total_numel)
    mae = abs_error_sum / max(1, total_numel)
    rmse = mse**0.5

    outputs = {}
    if "mse" in metric_names:
        outputs["mse"] = mse
    if "mae" in metric_names:
        outputs["mae"] = mae
    if "rmse" in metric_names:
        outputs["rmse"] = rmse
    if "psnr" in metric_names:
        outputs["psnr"] = float(sum(psnr_values) / max(1, len(psnr_values)))
    if "ssim" in metric_names:
        outputs["ssim"] = float(sum(ssim_values) / max(1, len(ssim_values)))

    return outputs


def get_visual_batch(loader, num_samples: int = 8) -> dict[str, torch.Tensor]:
    batch = next(iter(loader))
    return {
        "input": batch["input"][:num_samples],
        "target": batch["target"][:num_samples],
    }
