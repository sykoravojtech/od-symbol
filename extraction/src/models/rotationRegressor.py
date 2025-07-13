
# System Imports
from typing import List

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



class RotationRegressor(nn.Module):
    """OCR by LSTM"""

    def __init__(self, conv_features: List, kernel_sizes: List, pooling: List, lin_features: List,
                 img_width: int, img_height: int, in_channels):
        super().__init__()

        self.cnn = CNN(in_channels, conv_features, kernel_sizes, pooling)

        # output
        self.lin_out1 = nn.Linear(self.cnn.arith_2d(img_width, img_height), lin_features[0])
        self.do_out1 = nn.Dropout(0.5)
        self.lin_out2 = nn.Linear(lin_features[0], lin_features[1])
        self.do_out2 = nn.Dropout(0.5)
        self.lin_out3 = nn.Linear(lin_features[1], lin_features[2])
        self.do_out3 = nn.Dropout(0.5)
        self.lin_out4 = nn.Linear(lin_features[2], 2)


    def forward(self, x):
        """Standard Pytorch Forward"""

        x = self.cnn(x)

        x = torch.flatten(x, start_dim=1)
        x = self.lin_out1(x)
        x = F.relu(x)
        x = self.do_out1(x)
        x = self.lin_out2(x)
        x = F.relu(x)
        #x = self.do_out2(x)
        x = self.lin_out3(x)
        x = F.relu(x)
        #x = self.do_out3(x)
        x = self.lin_out4(x)

        return torch.tanh(x)*2
