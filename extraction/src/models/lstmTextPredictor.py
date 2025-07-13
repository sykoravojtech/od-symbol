"""lstmTextPredictor.py: Handwriting Recognition by CNN/LSTM"""

# System Imports
from math import floor

# Project Imports
from extraction.src.models.cnn import CNN

# Third-Party Imports
import torch
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


class LSTMTextPredictor(nn.Module):
    """Handwriting Recognition by CNN and LSTM"""

    def __init__(self, conv_features, kernel_sizes, pooling, lstm_cells, lstm_layers,
                 max_text_len, class_count, image_width, image_height):
        super().__init__()

        self.max_text_len = max_text_len
        self.class_count = class_count

        # Input
        self.cnn = CNN(3, conv_features, kernel_sizes, pooling)

        # LSTM
        self.lstm = nn.LSTM(self.cnn.arith(image_height)*conv_features[-1],
                            lstm_cells, lstm_layers, bidirectional=True, batch_first=True)

        # Output
        self.conv_out_1 = nn.Conv2d(lstm_cells*2*lstm_layers, 512, 1)
        self.conv_out_2 = nn.Conv2d(512, class_count, 1)


    def forward(self, x):
        """Standard Pytorch Forward"""

        x = self.cnn(x)

        x = torch.transpose(x, 1, 3)
        x = torch.flatten(x, start_dim=2)

        x, _ = self.lstm(x)

        x = torch.transpose(x, 1, 2)
        x = torch.unsqueeze(x, 3)

        x = self.conv_out_1(x)
        x = F.relu(x)
        x = self.conv_out_2(x)


        x = torch.squeeze(x, dim=3)
        x = torch.transpose(x, 1, 2)

        return x[:, 0:self.max_text_len, :]
