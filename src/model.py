"""A small convolutional classifier used as the export target.

The network is intentionally compact so it trains, exports, and runs on CPU
in a fraction of a second. It accepts single channel square images and
produces logits over a configurable number of classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """A two block CNN with a linear classification head.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of output logits.
        image_size: Height and width of the square input. The two pooling
            layers each halve the spatial dimensions, so ``image_size`` must
            be divisible by 4.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        reduced = image_size // 4
        self.flat_dim = 16 * reduced * reduced
        self.classifier = nn.Linear(self.flat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
