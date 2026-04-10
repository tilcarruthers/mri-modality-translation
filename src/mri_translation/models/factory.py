from __future__ import annotations

from mri_translation.models.baseline import SimpleEncoderDecoder
from mri_translation.models.resunet import ResUNet
from mri_translation.models.unet import UNet


def build_model(name: str):
    registry = {
        "baseline_encoder_decoder": SimpleEncoderDecoder,
        "unet": UNet,
        "resunet": ResUNet,
    }
    if name not in registry:
        raise ValueError(f"Unknown model name: {name}. Available: {sorted(registry)}")
    return registry[name]()
