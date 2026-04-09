import math

import torch

from mri_translation.metrics import mae_per_pixel, mse_per_pixel, rmse_per_pixel


def test_metrics_are_per_pixel():
    pred = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    target = torch.zeros_like(pred)

    assert math.isclose(mse_per_pixel(pred, target), 0.5, rel_tol=1e-6)
    assert math.isclose(mae_per_pixel(pred, target), 0.5, rel_tol=1e-6)
    assert math.isclose(rmse_per_pixel(pred, target), math.sqrt(0.5), rel_tol=1e-6)
