"""binaryAccuracy.py: Simple Binary Classification Accuracy Function"""

# System Imports
from math import sin, cos, pi, atan2

# Third-Party Imports
import torch

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



def bin_acc(pred: torch.Tensor, target: torch.Tensor):

    return torch.sum(torch.where(torch.abs(pred-target) < 0.5, 1.0, 0.0)).tolist() / torch.numel(pred)
