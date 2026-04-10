from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


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

    x = x.cpu()
    y = y.cpu()
    pred = pred.cpu()

    num_samples = x.size(0)
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for idx in range(num_samples):
        row = axes[idx]
        row[0].imshow(x[idx, 0], cmap="gray")
        row[0].set_title("T1 input")
        row[1].imshow(y[idx, 0], cmap="gray")
        row[1].set_title("T2 target")
        row[2].imshow(pred[idx, 0], cmap="gray")
        row[2].set_title("T2 prediction")
        for ax in row:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
