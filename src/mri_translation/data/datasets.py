from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset

from mri_translation.data.download import load_local_dataset
from mri_translation.data.normalization import RangeNormalizer, build_normalizer
from mri_translation.data.splits import load_split_manifest
from mri_translation.data.transforms import to_float_tensor


@dataclass
class DatasetBundle:
    full_raw: object
    train_raw: object
    val_raw: object
    test_raw: object
    train: Dataset
    val: Dataset
    test: Dataset
    normalizer: RangeNormalizer


class MRIPairedHFDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        input_key: str,
        target_key: str,
        normalizer: RangeNormalizer | None = None,
        metadata_keys: list[str] | None = None,
    ) -> None:
        self.dataset = hf_dataset
        self.input_key = input_key
        self.target_key = target_key
        self.normalizer = normalizer
        self.metadata_keys = metadata_keys or []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        x = to_float_tensor(sample[self.input_key])
        y = to_float_tensor(sample[self.target_key])

        if self.normalizer is not None:
            x = self.normalizer.normalize(x, self.input_key)
            y = self.normalizer.normalize(y, self.target_key)

        item = {"input": x, "target": y}
        for key in self.metadata_keys:
            if key in sample:
                item[key] = sample[key]
        return item


def _select_split(dataset, sample_ids: list[str], sample_id_key: str = "sample_id"):
    id_to_index = {str(dataset[idx][sample_id_key]): idx for idx in range(len(dataset))}
    indices = [id_to_index[sample_id] for sample_id in sample_ids]
    return dataset.select(indices)


def build_dataset_bundle(data_config: dict) -> DatasetBundle:
    full_raw = load_local_dataset(data_config["local_dataset_dir"])
    split_manifest = load_split_manifest(data_config["split"]["split_dir"])
    sample_id_key = split_manifest.get("sample_id_key", "sample_id")

    train_raw = _select_split(
        full_raw, split_manifest["splits"]["train"], sample_id_key=sample_id_key
    )
    val_raw = _select_split(full_raw, split_manifest["splits"]["val"], sample_id_key=sample_id_key)
    test_raw = _select_split(
        full_raw, split_manifest["splits"]["test"], sample_id_key=sample_id_key
    )

    normalizer = build_normalizer(train_raw, data_config)
    metadata_keys = data_config.get("metadata_keys", [])

    train_ds = MRIPairedHFDataset(
        train_raw,
        input_key=data_config["input_key"],
        target_key=data_config["target_key"],
        normalizer=normalizer,
        metadata_keys=metadata_keys,
    )
    val_ds = MRIPairedHFDataset(
        val_raw,
        input_key=data_config["input_key"],
        target_key=data_config["target_key"],
        normalizer=normalizer,
        metadata_keys=metadata_keys,
    )
    test_ds = MRIPairedHFDataset(
        test_raw,
        input_key=data_config["input_key"],
        target_key=data_config["target_key"],
        normalizer=normalizer,
        metadata_keys=metadata_keys,
    )

    return DatasetBundle(
        full_raw=full_raw,
        train_raw=train_raw,
        val_raw=val_raw,
        test_raw=test_raw,
        train=train_ds,
        val=val_ds,
        test=test_ds,
        normalizer=normalizer,
    )


def build_dataloaders(data_config: dict, loader_config: dict):
    bundle = build_dataset_bundle(data_config)

    train_loader = DataLoader(
        bundle.train,
        batch_size=loader_config["batch_size"],
        shuffle=True,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
    )
    val_loader = DataLoader(
        bundle.val,
        batch_size=loader_config["batch_size"],
        shuffle=False,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
    )
    test_loader = DataLoader(
        bundle.test,
        batch_size=loader_config["batch_size"],
        shuffle=False,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
    )

    info = {
        "num_train_samples": len(bundle.train),
        "num_val_samples": len(bundle.val),
        "num_test_samples": len(bundle.test),
        "normalization": bundle.normalizer.to_dict(),
        "dataset_name": data_config["dataset_name"],
        "local_dataset_dir": data_config["local_dataset_dir"],
        "split_dir": data_config["split"]["split_dir"],
    }
    return {"train": train_loader, "val": val_loader, "test": test_loader}, info
