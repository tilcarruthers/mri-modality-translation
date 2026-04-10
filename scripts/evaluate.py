from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mri_translation.config import validate_eval_config
from mri_translation.data.datasets import build_dataloaders
from mri_translation.engine.evaluate import evaluate_model, get_visual_batch
from mri_translation.models.factory import build_model
from mri_translation.utils.io import ensure_dir, save_json
from mri_translation.utils.seed import set_seed
from mri_translation.viz.plots import plot_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MRI modality translation checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model checkpoint.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name in config, e.g. baseline_encoder_decoder or unet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_eval_config(config)
    if args.model_name:
        config["model"]["name"] = args.model_name

    set_seed(config["seed"])

    eval_dir = Path(args.checkpoint).parent
    ensure_dir(eval_dir)

    loaders, _ = build_dataloaders(config["data"], config["loader"])
    model = build_model(config["model"]["name"])

    metrics = evaluate_model(
        model=model,
        loader=loaders["val"],
        device=config["training"]["device"],
        metric_names=config["evaluation"]["metrics"],
        checkpoint_path=args.checkpoint,
        max_batches=config["evaluation"].get("max_batches"),
    )
    save_json(metrics, eval_dir / "metrics_eval.json")

    visual_batch = get_visual_batch(
        loaders["val"], num_samples=config["evaluation"]["num_visual_samples"]
    )
    plot_prediction_grid(
        model=model,
        batch=visual_batch,
        device=config["training"]["device"],
        save_path=eval_dir / "prediction_grid_eval.png",
        checkpoint_path=args.checkpoint,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
