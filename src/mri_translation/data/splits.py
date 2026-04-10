from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from mri_translation.utils.io import ensure_dir, load_json, save_json


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")


def _partition_groups(groups: list[str], train_ratio: float, val_ratio: float):
    n_groups = len(groups)
    n_train = max(1, int(round(n_groups * train_ratio)))
    n_val = max(1, int(round(n_groups * val_ratio)))

    if n_train + n_val >= n_groups:
        n_val = max(1, n_groups - n_train - 1)

    train_groups = groups[:n_train]
    val_groups = groups[n_train : n_train + n_val]
    test_groups = groups[n_train + n_val :]

    if not test_groups:
        test_groups = val_groups[-1:]
        val_groups = val_groups[:-1]

    return train_groups, val_groups, test_groups


def create_grouped_split_manifest(
    dataset,
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    sample_id_key: str = "sample_id",
) -> dict:
    validate_ratios(train_ratio, val_ratio, test_ratio)

    group_to_sample_ids: dict[str, list[str]] = defaultdict(list)
    for row in dataset:
        group_value = str(row[group_key])
        group_to_sample_ids[group_value].append(str(row[sample_id_key]))

    groups = sorted(group_to_sample_ids)
    rng = random.Random(seed)
    rng.shuffle(groups)

    train_groups, val_groups, test_groups = _partition_groups(groups, train_ratio, val_ratio)

    split_to_groups = {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }
    split_to_sample_ids = {
        split_name: sorted(
            sample_id for group in split_groups for sample_id in group_to_sample_ids[group]
        )
        for split_name, split_groups in split_to_groups.items()
    }

    manifest = {
        "strategy": "patient_grouped",
        "group_key": group_key,
        "sample_id_key": sample_id_key,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "splits": split_to_sample_ids,
        "summary": {
            "num_total_samples": len(dataset),
            "num_total_groups": len(groups),
            "num_train_samples": len(split_to_sample_ids["train"]),
            "num_val_samples": len(split_to_sample_ids["val"]),
            "num_test_samples": len(split_to_sample_ids["test"]),
            "num_train_groups": len(train_groups),
            "num_val_groups": len(val_groups),
            "num_test_groups": len(test_groups),
        },
    }
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: dict) -> None:
    splits = manifest["splits"]
    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    if train_ids & val_ids:
        raise ValueError("Train and val splits overlap.")
    if train_ids & test_ids:
        raise ValueError("Train and test splits overlap.")
    if val_ids & test_ids:
        raise ValueError("Val and test splits overlap.")


def save_split_manifest(manifest: dict, split_dir: str | Path) -> None:
    split_dir = ensure_dir(split_dir)
    save_json(manifest, split_dir / "split_manifest.json")
    save_json(manifest["splits"]["train"], split_dir / "train_ids.json")
    save_json(manifest["splits"]["val"], split_dir / "val_ids.json")
    save_json(manifest["splits"]["test"], split_dir / "test_ids.json")
    save_json(manifest["summary"], split_dir / "split_summary.json")


def load_split_manifest(split_dir: str | Path) -> dict:
    split_dir = Path(split_dir)
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {manifest_path}. "
            "Run `python scripts/prepare_data.py --config <config-path>` first."
        )
    manifest = load_json(manifest_path)
    validate_split_manifest(manifest)
    return manifest
