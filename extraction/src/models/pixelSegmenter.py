"""pixelSegmenter.py: Classification of Image Patch Centers"""

# Third-Party Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project Imports
from extraction.src.models.cnn import CNN

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



class PixelSegmenter(nn.Module):
    """Classification of Image Patch Centers"""

    def __init__(self, conv_features, kernel_sizes, pooling, patch_size):
        super().__init__()

        self.cnn = CNN(3, conv_features, kernel_sizes, pooling)
        self.cnn_mag = CNN(3, [2], [patch_size], [1])
        self.fusion = nn.Conv2d(3, 1, 1)
        self.patch_size = patch_size

    def forward(self, x):
        """Standard Pytorch Forward"""

        if x.shape[2] == self.patch_size and x.shape[3] == self.patch_size:
            x_agg = self.cnn(x)
            x_mag = self.cnn_mag(x)

            x_fused = torch.cat((x_agg, x_mag), dim=1)
            x_fused = self.fusion(x_fused)

            return torch.nn.functional.relu(x_fused)

        else:
            result = torch.zeros([1, 1, 300, 300], requires_grad=False)

            for i in range(300):
                for j in range(300):
                    patch = x[:, :, i:i+self.patch_size, j:j+self.patch_size]

                    x_agg = self.cnn(patch)
                    x_mag = self.cnn_mag(patch)

                    x_fused = torch.cat((x_agg, x_mag), dim=1)
                    x_fused = self.fusion(x_fused)

                    result[0, 0, i, j] = torch.nn.functional.relu(x_fused)
                print(i)

            return result
