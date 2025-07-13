"""eca.py: Efficient Channel Attention enhancement for Faster R-CNN FPN backbone."""

from typing import Any

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

__author__ = "Vojtěch Sýkora"


class ECALayer(nn.Module):
    """Efficient Channel Attention"""

    # https://arxiv.org/pdf/1910.03151

    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        assert k_size > 0
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)

        return x * y.expand_as(x)


def attach_eca(model: FasterRCNN, kernel_size=3) -> None:
    """Wrap each FPN conv output with an ECALayer."""
    assert kernel_size > 0
    fpn = model.backbone.fpn
    print(f"--ECA: Using Kernel Size {kernel_size}")

    # Apply ECA to inner blocks (lateral convolutions)
    if hasattr(fpn, "inner_blocks"):
        for idx, conv in enumerate(fpn.inner_blocks):
            c_out = conv.out_channels
            fpn.inner_blocks[idx] = nn.Sequential(
                conv, ECALayer(c_out, k_size=kernel_size)
            )

    # Apply ECA to layer blocks (output convolutions)
    if hasattr(fpn, "layer_blocks"):
        for idx, conv in enumerate(fpn.layer_blocks):
            c_out = conv.out_channels
            fpn.layer_blocks[idx] = nn.Sequential(
                conv, ECALayer(c_out, k_size=kernel_size)
            )


if __name__ == "__main__":
    backbone = resnet_fpn_backbone("resnet152", pretrained=True, trainable_layers=3)
    model = FasterRCNN(backbone, num_classes=61)
    attach_eca(model)
