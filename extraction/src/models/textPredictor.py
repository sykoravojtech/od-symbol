"""textPredictor.py: Simple Text Predictor"""

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



class TextPredictor(nn.Module):
    """Handwriting Recognition by CNN"""

    def __init__(self, conv_features, kernel_sizes, pooling, lin_features,
                 max_text_len, class_count, image_width, image_height):
        super().__init__()

        self.max_text_len = max_text_len
        self.class_count = class_count

        self.cnn = CNN(3, conv_features, kernel_sizes, pooling)

        # output
        self.lin_out1 = nn.Linear(self.cnn.arith_2d(image_width, image_height), lin_features[0])
        self.do_out1 = nn.Dropout(0.1)
        self.lin_out2 = nn.Linear(lin_features[0], lin_features[1])
        self.do_out2 = nn.Dropout(0.1)
        self.lin_out3 = nn.Linear(lin_features[1], class_count*max_text_len)


    def forward(self, x):
        """Standard Pytorch Forward"""

        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.lin_out1(x)
        x = self.do_out1(x)
        x = F.relu(x)
        x = self.lin_out2(x)
        x = F.relu(x)
        x = self.do_out2(x)
        x = self.lin_out3(x)
        x = F.relu(x)

        x = x.reshape((-1, self.class_count, self.max_text_len))
        x = torch.permute(x, (0, 2, 1))
        #x = F.softmax(x, dim=2)
        #print(x.size())
        return x
