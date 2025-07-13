"""SegmentationPixelProcessor.py: Processor for Classification of Image Patch Center Pixel"""

# System Imports
from random import randint
from typing import List, Tuple, Union

# Third-Party Imports
from numpy import ndarray
from torch import Tensor
from torchvision import transforms

# Project Imports
from converter.core.engineeringGraph import EngGraph
from extraction.src.core.processor import Processor

__author__ = "Johannes Bayer, Vojtěch Sýkora"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"


class SegmentationPixelProcessor(Processor):
    """Processor for Classification of Image Patch Center Pixel"""

    def __init__(
        self,
        augment: bool,
        train: bool,
        debug: bool,
        patch_size: int,
        samples_per_image: int,
    ):
        super().__init__(augment, train, debug)

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.patch_size = patch_size
        self.samples_per_image = samples_per_image

    def pre_process(
        self, img_raw: ndarray, img_map: ndarray
    ) -> List[Tuple[Tensor, Union[Tensor, any]]]:
        """Turns ONE Raw Image and ONE Graph Structure into a List of Tuples of Input and Target/Info Tensors"""

        samples = []

        if self.train:
            img_height, img_width, _ = img_raw.shape
            fg_count = 0
            bg_count = 0

            while (fg_count < self.samples_per_image / 2) or (
                bg_count < self.samples_per_image / 2
            ):
                offset_x = randint(0, img_width - self.patch_size)
                offset_y = randint(0, img_height - self.patch_size)
                center_map = img_map[
                    offset_y + self.patch_size // 2, offset_x + self.patch_size // 2
                ]
                patch_raw = img_raw[
                    offset_y : offset_y + self.patch_size,
                    offset_x : offset_x + self.patch_size,
                ]

                if center_map == 0 and fg_count < self.samples_per_image / 2:
                    samples += [(self.transform(patch_raw), Tensor([[[1]]]), None)]
                    fg_count += 1

                if center_map == 255 and bg_count < self.samples_per_image / 2:
                    samples += [(self.transform(patch_raw), Tensor([[[0]]]), None)]
                    bg_count += 1

        else:
            samples += [(self.transform(img_raw), None, None)]

        return samples

    def post_process(
        self, pred_list: List[Tuple[Tensor, any]], graph: EngGraph
    ) -> None:
        """Integrates a List of Prediction and Info Tensors into a Graph Structure"""

        seg_map = pred_list[0][0]
        import matplotlib.pyplot as plt

        plt.imshow(seg_map[0].detach().numpy())
        plt.show()
