"""textSnippetProcessor.py: Processor for Text Prediction"""

# System Imports
import random
import importlib
from random import randint
from typing import Optional, List

# Project Imports
from extraction.src.processor.nodeSnippetProcessor import NodeSnippetProcessor
from extraction.src.utils.text import encode_text, decode_text, set_character_set

# Third-Party Imports
import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"


class TextSnippetProcessor(NodeSnippetProcessor):
    """Processor for Text Prediction"""

    def __init__(self, augment: bool, train: bool, debug: bool,
                 rand_crop: int, image_height: int, image_width: Optional[int] = None,
                 max_text_len: Optional[int] = None, char_set: List[str] = None):
        """Texts Snippets are Scaled to image_height, optionally padded to image_width."""
        super().__init__(augment, train, debug, rand_crop)
        
        self.image_width = image_width
        self.image_height = image_height
        self.max_text_len = max_text_len

        self.tse = TextSnippetEncoder(image_width, image_height)
        set_character_set(char_set)

        #statistics = importlib.import_module(f"{db_path}.consistency")
        # TODO: Still needed? -> self.char_set = statistics.unique_characters([text for _, text in self.snippets])
        #self.snippets_by_char = [[snippet for snippet in self.snippets if char in snippet[1]] for char in self.char_set]
        #statistics.character_distribution([text for _, text in self.snippets], self.char_set)
        #statistics.character_distribution([text for _, text in [self.draw() for _ in range(1000)]], self.char_set)


    def _node_filter(self, node_data: dict) -> bool:
        """Decides Whether a Node Can be Processed"""

        if not node_data['type'] == "text":
            return False

        if not bool(node_data.get('text', None)) == self.train:
            return False

        if self.max_text_len and len(node_data['text']) > self.max_text_len:
            return False

        width_after_rotation = node_data['position']['width']
        height_after_rotation = node_data['position']['height']

        if node_data['position']['rotation'] == 90 or node_data['position']['rotation'] == 270:
            width_after_rotation, height_after_rotation = height_after_rotation, width_after_rotation

        width_after_transform = width_after_rotation * (self.image_height/height_after_rotation)

        if self.image_width and width_after_transform > self.image_width:
            return False

        return True


    def _make_input(self, snippet: np.ndarray, node_data: dict) -> List[Tensor]:
        """"Creates a List of Input Tensors for the Model"""

        if self.augment:
            return [self.tse.encode_image(snippet, node_data),
                    self.tse.encode_image(snippet, node_data, True)]

        else:
            return [self.tse.encode_image(snippet, node_data)]


    def _make_target(self, node_data: dict) -> List[Tensor]:
        """Creates a List of Target Models for the Model"""

        if self.augment:
            return [encode_text(node_data['text'], self.max_text_len),
                    encode_text(node_data['text'], self.max_text_len)]

        else:
            return [encode_text(node_data['text'], self.max_text_len)]


    def _integrate_data(self, pred: Tensor, node_data: dict) -> None:
        """Integrates the Model's Predictions into the Node's Attributes"""

        node_data['text'] = decode_text(pred)


    #def draw(self):
    #    """Returns a (n uniformly) Randomly Selected Sample"""
    #
    #    char = randint(0, len(self.char_set) - 1)
    #    return self.snippets_by_char[char][randint(0, len(self.snippets_by_char[char]) - 1)]




class FixedHeightResize(object):
    """Scales all Images to the Same Height"""

    def __init__(self, image_height):
        self.image_height = image_height

    def __call__(self, img):
        return transforms.Resize((self.image_height, self._calc_new_width(img)))(img)

    def _calc_new_width(self, img):
        channels, old_height, old_width = img.size()
        aspect_ratio = old_width / old_height
        return round(self.image_height * aspect_ratio)


class TextSnippetEncoder:

    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             FixedHeightResize(image_height),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.transform_augment_1 = transforms.Compose([transforms.ToTensor(),
                                                       FixedHeightResize(image_height),
                                                       #transforms.RandomPosterize(2),
                                                       transforms.RandomAdjustSharpness(2),
                                                       transforms.RandomInvert(),
                                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.transform_augment_2 = transforms.Compose([transforms.ToTensor(),
                                                       FixedHeightResize(image_height),
                                                       transforms.RandomAutocontrast(),
                                                       #transforms.RandomEqualize(),
                                                       transforms.ColorJitter(),
                                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])



    def encode_image(self, snippet: np.ndarray, node_data: dict, augment: bool = False):

        rotation = node_data['position']["rotation"] if "rotation" in node_data['position'] else 0

        if rotation == 90:
            snippet = cv2.rotate(snippet, cv2.ROTATE_90_CLOCKWISE)

        if rotation == 270:
            snippet = cv2.rotate(snippet, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if rotation == 180:
            snippet = cv2.rotate(snippet, cv2.ROTATE_180)

        if not self.image_width:
            return self.transform(snippet)

        img = (self.transform_augment_1 if random.getrandbits(1) else self.transform_augment_2)(snippet) if augment else self.transform(snippet)
        img = torch.cat((img, 0.5 + torch.zeros((3, self.image_height, self.image_width - img.shape[2]))),
                        2)

        # TODO have this normalization everywhere
        img_max = torch.max(img)
        img_min = torch.min(img)
        img = (img - ((img_max + img_min) / 2)) / (img_max - img_min)

        return img
