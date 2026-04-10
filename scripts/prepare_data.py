from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mri_translation.config import validate_train_config
from mri_translation.data.download import download_and_prepare_local_dataset
from mri_translation.data.splits import (
    create_grouped_split_manifest,
    load_split_manifest,
    save_split_manifest,
)
from mri_translation.utils.io import ensure_dir, save_json
from mri_translation.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download, cache, and split the MRI dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload and rebuild the local dataset even if it already exists.",
    )
    parser.add_argument(
        "--regenerate-splits",
        action="store_true",
        help="Regenerate split manifests even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_train_config(config)
    set_seed(config["seed"])

    data_config = config["data"]
    split_config = data_config["split"]

    dataset = download_and_prepare_local_dataset(
        dataset_name=data_config["dataset_name"],
        source_train_split=data_config["source_train_split"],
        source_val_split=data_config["source_val_split"],
        cache_dir=data_config["cache_dir"],
        local_dataset_dir=data_config["local_dataset_dir"],
        force_download=args.force_download,
    )

    split_dir = Path(split_config["split_dir"])
    manifest_path = split_dir / "split_manifest.json"
    reuse_existing = bool(split_config.get("reuse_existing", True)) and not args.regenerate_splits

    if reuse_existing and manifest_path.exists():
        manifest = load_split_manifest(split_dir)
    else:
        manifest = create_grouped_split_manifest(
            dataset=dataset,
            group_key=split_config["group_key"],
            train_ratio=split_config["train_ratio"],
            val_ratio=split_config["val_ratio"],
            test_ratio=split_config["test_ratio"],
            seed=split_config["seed"],
        )
        save_split_manifest(manifest, split_dir)

    summary = {
        "dataset_name": data_config["dataset_name"],
        "local_dataset_dir": data_config["local_dataset_dir"],
        "cache_dir": data_config["cache_dir"],
        "split_dir": str(split_dir),
        "summary": manifest["summary"],
    }

    ensure_dir(split_dir)
    save_json(summary, split_dir / "prepare_data_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
