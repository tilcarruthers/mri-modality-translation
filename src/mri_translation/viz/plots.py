from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mri_translation.engine.evaluate import load_checkpoint
from mri_translation.engine.train import resolve_device


def plot_training_history(
    history: dict, save_path: str | Path, title: str = "training history"
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="validation")
    if "lr" in history:
        ax2 = ax.twinx()
        ax2.plot(epochs, history["lr"], linestyle="--", alpha=0.6, label="lr")
        ax2.set_ylabel("Learning rate")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _robust_display_range(
    *arrays: np.ndarray, lower: float = 1.0, upper: float = 99.0
) -> tuple[float, float]:
    stacked = np.concatenate([arr.reshape(-1) for arr in arrays])
    vmin = float(np.percentile(stacked, lower))
    vmax = float(np.percentile(stacked, upper))

    if not np.isfinite(vmin):
        vmin = float(stacked.min())
    if not np.isfinite(vmax):
        vmax = float(stacked.max())

    if vmax <= vmin:
        vmax = vmin + 1e-6

    return vmin, vmax


@torch.no_grad()
def plot_prediction_grid(
    model,
    batch: dict[str, torch.Tensor],
    device,
    save_path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> None:
    device = resolve_device(device) if isinstance(device, str) else device
    model = model.to(device)

    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, device)

    model.eval()
    x = batch["input"].to(device)
    y = batch["target"].to(device)
    pred = model(x)

    # Clamp predictions to the target range for visualization
    pred = torch.clamp(pred, 0.0, 1.0)

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    pred = pred.cpu().numpy()

    num_samples = x.shape[0]
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(num_samples):
        inp = x[idx, 0]
        tgt = y[idx, 0]
        out = pred[idx, 0]

        # Use one robust display range per row across input/target/prediction
        vmin, vmax = _robust_display_range(inp, tgt, out, lower=1.0, upper=99.0)

        row = axes[idx]
        row[0].imshow(inp, cmap="gray", vmin=vmin, vmax=vmax)
        row[0].set_title("T1 input")
        row[1].imshow(tgt, cmap="gray", vmin=vmin, vmax=vmax)
        row[1].set_title("T2 target")
        row[2].imshow(out, cmap="gray", vmin=vmin, vmax=vmax)
        row[2].set_title("T2 prediction")

        for ax in row:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
