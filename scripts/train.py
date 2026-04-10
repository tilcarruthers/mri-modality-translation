from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mri_translation.config import validate_train_config
from mri_translation.data.datasets import build_dataloaders
from mri_translation.engine.evaluate import evaluate_model, get_visual_batch
from mri_translation.engine.train import fit
from mri_translation.models.factory import build_model
from mri_translation.utils.io import ensure_dir, save_json
from mri_translation.utils.seed import set_seed
from mri_translation.viz.plots import plot_prediction_grid, plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MRI modality translation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_train_config(config)
    set_seed(config["seed"])

    experiment_name = config["experiment"]["name"]
    run_dir = Path(config["experiment"]["output_dir"]) / experiment_name
    ensure_dir(run_dir)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    loaders, data_info = build_dataloaders(config["data"], config["loader"])
    model = build_model(config["model"]["name"])

    history, best_path, last_path, device = fit(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        training_config=config["training"],
        scheduler_config=config.get("scheduler"),
        early_stopping_config=config.get("early_stopping"),
        run_dir=run_dir,
    )

    metrics = evaluate_model(
        model=model,
        loader=loaders["val"],
        device=device,
        metric_names=config["evaluation"]["metrics"],
        checkpoint_path=best_path,
        max_batches=config["evaluation"].get("max_batches"),
    )

    visual_batch = get_visual_batch(
        loaders["val"], num_samples=config["evaluation"]["num_visual_samples"]
    )
    prediction_path = run_dir / "prediction_grid.png"
    plot_prediction_grid(
        model=model,
        batch=visual_batch,
        device=device,
        save_path=prediction_path,
        checkpoint_path=best_path,
    )

    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics.json"
    data_info_path = run_dir / "data_info.json"

    save_json(history, history_path)
    save_json(metrics, metrics_path)
    save_json(data_info, data_info_path)
    plot_training_history(history, run_dir / "training_history.png", title=experiment_name)

    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
