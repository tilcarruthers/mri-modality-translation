from mri_translation.data.datasets import MRIPairedHFDataset
from mri_translation.data.normalization import RangeNormalizer


def test_dataset_wrapper_returns_expected_keys_and_shape():
    dummy = [
        {"t1": [[0.0, 1.0], [2.0, 3.0]], "t2": [[1.0, 2.0], [3.0, 4.0]], "view": "axial"},
    ]
    normalizer = RangeNormalizer(
        stats={"t1": {"min": 0.0, "max": 3.0}, "t2": {"min": 1.0, "max": 4.0}}
    )
    ds = MRIPairedHFDataset(
        dummy, input_key="t1", target_key="t2", normalizer=normalizer, metadata_keys=["view"]
    )
    sample = ds[0]

    assert sample["input"].shape == (1, 2, 2)
    assert sample["target"].shape == (1, 2, 2)
    assert sample["view"] == "axial"
    assert float(sample["input"].min()) >= 0.0
    assert float(sample["input"].max()) <= 1.0
