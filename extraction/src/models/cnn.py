"""cnn.py: Vanilla, Fully Parameterized Convolutional Neural Network"""

# System Imports
from math import floor
from typing import List

# Third-Party Imports
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



def conv_arith(i, k, s):
    """Size of a Convolution Operation given Input, Kernel size and Stride """

    return floor((i-k)/s)+1


class ConvBlock(nn.Module):
    """A Single Convolutional Block"""

    def __init__(self, features_in: int, features_out: int, kernel_size: int, pooling_size: int):
        super().__init__()

        self.conv = nn.Conv2d(features_in, features_out, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(features_out)
        self.do = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d((pooling_size, pooling_size))


    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.do(x)
        x = self.pool(x)

        return x


class CNN(nn.Module):
    """Vanilla, Fully Parameterized Convolutional Neural Network"""

    def __init__(self, in_features: int, conv_features: List, kernel_sizes: List, pooling: List):
        super().__init__()

        if len(conv_features) != len(kernel_sizes) or len(kernel_sizes) != len(pooling):
            print("Error: Lists of Conv Features, Kernel Size and Pooling Have to be same length ")

        self.conv_features = conv_features
        self.kernel_sizes = kernel_sizes
        self.pooling = pooling
        self.conv_blocks = nn.ModuleList()

        for block_nbr in range(len(conv_features)):
            self.conv_blocks.append(ConvBlock(conv_features[block_nbr-1] if block_nbr != 0 else in_features,
                                              conv_features[block_nbr],
                                              kernel_sizes[block_nbr],
                                              pooling[block_nbr]))


    def arith(self, i):
        """Calculates Output size based on Model Parameters and Input Size in one Dimension"""

        x = i

        for block_nbr in range(len(self.conv_blocks)):
            x = conv_arith(x, self.kernel_sizes[block_nbr], 1)
            x = conv_arith(x, self.pooling[block_nbr], self.pooling[block_nbr])

        return x


    def arith_2d(self, width, height):

        return self.arith(width)*self.arith(height)*self.conv_features[-1]


    def forward(self, x):
        """Standard Pytorch Forward"""

        for conv_block in self.conv_blocks:
            x = conv_block(x)

        return x
