from __future__ import annotations

from typing import Any

REQUIRED_TOP_LEVEL_KEYS = {
    "experiment",
    "seed",
    "data",
    "loader",
    "model",
    "training",
    "evaluation",
}


def _require_keys(config: dict[str, Any], keys: set[str], scope: str) -> None:
    missing = sorted(keys - set(config))
    if missing:
        raise KeyError(f"Missing required keys in {scope}: {missing}")


def validate_train_config(config: dict[str, Any]) -> None:
    _require_keys(config, REQUIRED_TOP_LEVEL_KEYS, "config")
    _require_keys(
        config["data"],
        {"dataset_name", "train_split", "val_split", "input_key", "target_key"},
        "data",
    )
    _require_keys(config["loader"], {"batch_size", "num_workers", "pin_memory"}, "loader")
    _require_keys(config["model"], {"name"}, "model")
    _require_keys(
        config["training"],
        {"epochs", "lr", "loss", "device", "use_amp", "compile_model"},
        "training",
    )


def validate_eval_config(config: dict[str, Any]) -> None:
    _require_keys(config, REQUIRED_TOP_LEVEL_KEYS, "config")
    _require_keys(
        config["data"],
        {"dataset_name", "train_split", "val_split", "input_key", "target_key"},
        "data",
    )
    _require_keys(config["loader"], {"batch_size", "num_workers", "pin_memory"}, "loader")
    _require_keys(config["model"], {"name"}, "model")
    _require_keys(config["training"], {"device", "use_amp", "compile_model"}, "training")
