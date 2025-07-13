"""rotationSnippetProcessor.py: Processor for Node Rotation Prediction"""

# System Imports
from typing import List, Optional

# Project Imports
from extraction.src.processor.nodeSnippetProcessor import NodeSnippetProcessor
from extraction.src.utils.rotation import encode_rotation, decode_rotation

# Third-Party Imports
import cv2
from numpy import ndarray
import torch
from torch import Tensor
from torchvision import transforms

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



class AdaptiveNormalize(object):
    """Scales all Images to the Same Height"""

    def __call__(self, img):
        return transforms.Normalize(img.mean((1, 2)), img.std((1, 2))*2)(img)


class GaussianNoise(object):

    def __call__(self, img):
        return img + torch.randn(img.shape)*0.15


class RotationSnippetProcessor(NodeSnippetProcessor):
    """Processor for Node Rotation Prediction"""

    def __init__(self, augment: bool, train: bool, debug: bool,
                 rand_crop: int, width: int, height: int, supported_classes: List[str], symmetric_classes: List[str]):
        super().__init__(augment, train, debug, rand_crop)
        self.supported_classes = supported_classes
        self.symmetric_classes = symmetric_classes
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((width, height)),
                                             AdaptiveNormalize()])
        self.transform_augment = transforms.Compose([transforms.ToTensor(),
                                                     GaussianNoise(),
                                                     transforms.RandomGrayscale(),
                                                     transforms.RandomInvert(),
                                                     transforms.Resize((width, height)),
                                                     AdaptiveNormalize()])


    def _node_filter(self, node_data: dict) -> bool:
        """Decides Whether a Node Can be Processed"""

        return node_data['type'] in self.supported_classes and (bool(node_data['position']["rotation"]) == self.train)


    def _make_input(self, snippet: ndarray, node_data: dict) -> List[Tensor]:
        """"Creates a List of Input Tensors for the Model"""

        if self.augment:
            return [self.transform(snippet),
                    self.transform(cv2.rotate(snippet, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                    self.transform(cv2.rotate(snippet, cv2.ROTATE_90_CLOCKWISE)),
                    self.transform(cv2.rotate(snippet, cv2.ROTATE_180)),
                    self.transform_augment(snippet),
                    self.transform_augment(cv2.rotate(snippet, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                    self.transform_augment(cv2.rotate(snippet, cv2.ROTATE_90_CLOCKWISE)),
                    self.transform_augment(cv2.rotate(snippet, cv2.ROTATE_180))]

        else:
            return [self.transform(snippet)]


    def _make_target(self, node_data: dict) -> List[Tensor]:
        """Creates a List of Target Models for the Model"""

        symmetric = node_data['type'] in self.symmetric_classes

        if self.augment:
            return [encode_rotation(node_data['position']["rotation"], symmetric),
                    encode_rotation(node_data['position']["rotation"]+90, symmetric),
                    encode_rotation(node_data['position']["rotation"]-90, symmetric),
                    encode_rotation(node_data['position']["rotation"]+180, symmetric),
                    encode_rotation(node_data['position']["rotation"], symmetric),
                    encode_rotation(node_data['position']["rotation"] + 90, symmetric),
                    encode_rotation(node_data['position']["rotation"] - 90, symmetric),
                    encode_rotation(node_data['position']["rotation"] + 180, symmetric)]

        else:
            return [encode_rotation(node_data['position']["rotation"], symmetric)]


    def _integrate_data(self, pred: Tensor, node_data: dict) -> None:
        """Integrates the Model's Predictions into the Node's Attributes"""

        node_data['position']['rotation'] = decode_rotation(pred)
