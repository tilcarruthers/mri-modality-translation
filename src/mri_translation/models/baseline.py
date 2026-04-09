from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoderDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dec1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = self.upsample(x)
        x = F.relu(self.dec1(x))
        x = self.upsample(x)
        return self.dec2(x)
