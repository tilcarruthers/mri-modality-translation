from __future__ import annotations

from pathlib import Path

from datasets import concatenate_datasets, load_dataset, load_from_disk

from mri_translation.utils.io import ensure_dir


def _add_sample_ids(dataset, prefix: str):
    def with_ids(example, idx: int):
        example["sample_id"] = f"{prefix}_{idx:06d}"
        example["source_split"] = prefix
        return example

    return dataset.map(with_ids, with_indices=True)


def download_and_prepare_local_dataset(
    dataset_name: str,
    source_train_split: str,
    source_val_split: str,
    cache_dir: str | Path,
    local_dataset_dir: str | Path,
    force_download: bool = False,
):
    local_dataset_dir = Path(local_dataset_dir)
    cache_dir = ensure_dir(cache_dir)

    if local_dataset_dir.exists() and not force_download:
        return load_from_disk(str(local_dataset_dir))

    train_ds = load_dataset(dataset_name, split=source_train_split, cache_dir=str(cache_dir))
    val_ds = load_dataset(dataset_name, split=source_val_split, cache_dir=str(cache_dir))

    train_ds = _add_sample_ids(train_ds, "train")
    val_ds = _add_sample_ids(val_ds, "validation")

    merged = concatenate_datasets([train_ds, val_ds])
    ensure_dir(local_dataset_dir.parent)
    merged.save_to_disk(str(local_dataset_dir))
    return merged


def load_local_dataset(local_dataset_dir: str | Path):
    local_dataset_dir = Path(local_dataset_dir)
    if not local_dataset_dir.exists():
        raise FileNotFoundError(
            f"Local dataset not found at {local_dataset_dir}. "
            "Run `python scripts/prepare_data.py --config <config-path>` first."
        )
    return load_from_disk(str(local_dataset_dir))
