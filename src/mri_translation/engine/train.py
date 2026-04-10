from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm


def resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def build_loss(name: str):
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {name}")


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, use_amp: bool = False, scaler=None
):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="train", leave=False):
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None and device.type == "cuda":
            with autocast(device_type="cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1

    return running_loss / max(1, num_batches)


@torch.no_grad()
def validate_one_epoch(model, val_loader, criterion, device, use_amp: bool = False):
    model.eval()
    running_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="val", leave=False):
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with autocast(device_type="cuda"):
                pred = model(x)
                loss = criterion(pred, y)
        else:
            pred = model(x)
            loss = criterion(pred, y)

        running_loss += float(loss.item())
        num_batches += 1

    return running_loss / max(1, num_batches)


def save_checkpoint(
    path: Path, model, optimizer, epoch: int, history: dict, train_loss: float, val_loss: float
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def fit(model, train_loader, val_loader, training_config: dict, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(training_config["device"])
    model = model.to(device)

    if training_config.get("compile_model", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["lr"])
    criterion = build_loss(training_config["loss"])
    use_amp = bool(training_config.get("use_amp", False))
    scaler = GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    start_time = time.time()
    for epoch in range(1, training_config["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler
        )
        val_loss = validate_one_epoch(model, val_loader, criterion, device, use_amp)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        save_checkpoint(last_path, model, optimizer, epoch, history, train_loss, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, history, train_loss, val_loss)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.2f}s")
    return history, best_path, last_path, device
