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


def build_scheduler(optimizer, scheduler_config: dict | None):
    if not scheduler_config:
        return None

    name = scheduler_config.get("name")
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 4),
            min_lr=scheduler_config.get("min_lr", 1.0e-6),
        )

    raise ValueError(f"Unsupported scheduler: {name}")


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0, monitor: str = "val_loss") -> None:
        if monitor != "val_loss":
            raise ValueError(f"Unsupported early stopping monitor: {monitor}")
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_value: float | None = None

    def step(self, current_value: float) -> bool:
        if self.best_value is None or current_value < (self.best_value - self.min_delta):
            self.best_value = current_value
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def maybe_compile_model(model, training_config: dict):
    compile_requested = bool(training_config.get("compile_model", False))
    if not compile_requested or not hasattr(torch, "compile"):
        return model, False

    try:
        compiled_model = torch.compile(model)
        print("Model compilation enabled.")
        return compiled_model, True
    except Exception as exc:
        print(f"Model compilation failed during setup; continuing without compile. Reason: {exc}")
        return model, False


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    use_amp: bool = False,
    scaler=None,
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
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    history: dict,
    train_loss: float,
    val_loss: float,
):
    base_model = unwrap_model(model)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        },
        path,
    )


def fit(
    model,
    train_loader,
    val_loader,
    training_config: dict,
    run_dir: Path,
    scheduler_config: dict | None = None,
    early_stopping_config: dict | None = None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(training_config["device"])
    base_model = model.to(device)
    model, model_is_compiled = maybe_compile_model(base_model, training_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["lr"])
    scheduler = build_scheduler(optimizer, scheduler_config)
    criterion = build_loss(training_config["loss"])
    use_amp = bool(training_config.get("use_amp", False))
    scaler = GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    early_stopper = None
    if early_stopping_config and early_stopping_config.get("enabled", False):
        early_stopper = EarlyStopping(
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.0),
            monitor=early_stopping_config.get("monitor", "val_loss"),
        )

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    start_time = time.time()
    epoch = 0
    while epoch < training_config["epochs"]:
        epoch += 1
        current_lr = float(optimizer.param_groups[0]["lr"])

        try:
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, use_amp, scaler
            )
            val_loss = validate_one_epoch(model, val_loader, criterion, device, use_amp)
        except Exception as exc:
            if model_is_compiled:
                print(
                    "Compiled model failed during execution; falling back to eager mode. "
                    f"Reason: {exc}"
                )
                model = base_model
                model_is_compiled = False
                optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]["lr"])
                scheduler = build_scheduler(optimizer, scheduler_config)
                scaler = GradScaler("cuda") if (use_amp and device.type == "cuda") else None
                epoch -= 1
                continue
            raise

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        save_checkpoint(
            last_path, model, optimizer, scheduler, epoch, history, train_loss, val_loss
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, history, train_loss, val_loss
            )

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"lr={current_lr:.6g} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        if early_stopper is not None:
            stop = early_stopper.step(val_loss)
            if stop:
                print(f"Early stopping triggered at epoch {epoch:02d}.")
                break

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.2f}s")
    return history, best_path, last_path, device
