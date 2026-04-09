import torch

from mri_translation.models.baseline import SimpleEncoderDecoder
from mri_translation.models.unet import UNet


def test_baseline_output_shape():
    model = SimpleEncoderDecoder()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_unet_output_shape():
    model = UNet()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
