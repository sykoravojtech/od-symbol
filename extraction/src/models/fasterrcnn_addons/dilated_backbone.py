"""dilated_backbone.py: Dilated convolution enhancement for Faster R-CNN"""

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN

__author__ = "Vojtěch Sýkora"


class DilatedConvBlock(nn.Module):
    """Simple dilated convolution block for multi-scale feature extraction"""

    def __init__(self, channels, dilation_rates=[1, 2, 4]):
        super(DilatedConvBlock, self).__init__()

        num_branches = len(dilation_rates)
        base_channels = channels // num_branches
        remaining_channels = channels - (base_channels * (num_branches - 1))

        # make parallel dilated convolutions
        self.dilated_convs = nn.ModuleList()
        for i, rate in enumerate(dilation_rates):
            # Last branch gets remaining channels to ensure exact match
            out_channels = (
                remaining_channels if i == num_branches - 1 else base_channels
            )
            self.dilated_convs.append(
                nn.Conv2d(
                    channels,
                    out_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                    bias=False,
                )
            )

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply parallel dilated convs
        outputs = [conv(x) for conv in self.dilated_convs]
        out = torch.cat(outputs, dim=1)
        out = self.relu(self.bn(out))

        # Residual connection
        return out + x


def attach_dilated_convs(model: FasterRCNN, dilation_rates=[1, 2, 4]) -> None:
    """Attach dilated convolution blocks to FPN layers"""
    fpn = model.backbone.fpn
    print(f"--DILATED: Using dilation rates {dilation_rates}")

    # Apply to inner blocks (lateral convolutions)
    if hasattr(fpn, "inner_blocks"):
        for idx, conv in enumerate(fpn.inner_blocks):
            c_out = conv.out_channels
            fpn.inner_blocks[idx] = nn.Sequential(
                conv, DilatedConvBlock(c_out, dilation_rates=dilation_rates)
            )

    # Apply to layer blocks (output convolutions)
    if hasattr(fpn, "layer_blocks"):
        for idx, conv in enumerate(fpn.layer_blocks):
            c_out = conv.out_channels
            fpn.layer_blocks[idx] = nn.Sequential(
                conv, DilatedConvBlock(c_out, dilation_rates=dilation_rates)
            )
