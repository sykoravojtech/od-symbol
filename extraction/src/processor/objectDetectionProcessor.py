"""objectDetectionProcessor.py: Pre-/Post-Processing for Object Detection"""

# System Imports
from typing import Any, Dict, List, Tuple, Union

import torch
from converter.core.boundingbox import BoundingBox

# Project Imports
from converter.core.engineeringGraph import EngGraph
from extraction.src.core.processor import Processor

# Third-Party Imports
from numpy import ndarray
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import (
    ColorJitter,
    GaussianBlur,
    RandomAffine,
    RandomAutocontrast,
    RandomErasing,
    RandomGrayscale,
    RandomPerspective,
    RandomResizedCrop,
    RandomRotation,
)

__author__ = "Johannes Bayer, Vojtěch Sýkora"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"


class ObjectDetectionProcessor(Processor):
    """Pre-/Post-Processing for Object Detection"""

    def __init__(
        self,
        augment: bool,
        train: bool,
        debug: bool,
        classes: List[str],
        score_threshold: float,
    ):
        super().__init__(augment, train, debug)

        self.classes = classes
        if augment:
            print("-- Adding transforms --")
            self.transformer = self.get_cghd_transforms(augment)
        else:
            print("-- without transforms --")
            self.transformer = transforms.ToTensor()
        self.score_threshold = score_threshold

    def get_cghd_transforms(self, augmentations) -> transforms.Compose:
        """x% chance ofusing transforms, else just ToTensor."""
        print(f"Using transforms = {augmentations}")

        aug_block = []
        if "rotation" in augmentations:
            aug_block.append(RandomRotation(degrees=5))
        if "affine" in augmentations:
            aug_block.append(RandomAffine(degrees=0, scale=(0.9, 1.1)))
        if "perspective" in augmentations:
            aug_block.append(RandomPerspective(distortion_scale=0.05, p=1.0))
        if "erasing" in augmentations:
            aug_block.append(RandomErasing(p=1, scale=(0.005, 0.02), ratio=(0.3, 3)))
        if "color_jitter" in augmentations:
            aug_block.append(
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.01, hue=0.05)
            )
        if "autocontrast" in augmentations:
            aug_block.append(RandomAutocontrast(p=1))
        if "grayscale" in augmentations:
            aug_block.append(RandomGrayscale(p=1))
        if "gaussian_blur" in augmentations:
            aug_block.append(GaussianBlur(kernel_size=3))
        if "noise" in augmentations:
            aug_block.append(
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02)
            )

        # RandomApply([RandomSubset(aug_block)], p=0.5) would take a subset of the block
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomApply(
                    aug_block, p=0.5
                ),  # 50% data normal, 50% data augmented
            ]
        )

    def pre_process(
        self, image: ndarray, graph: EngGraph
    ) -> List[Tuple[Tensor, Union[Tensor, any]]]:
        """Turns ONE Raw Image and ONE Graph Structure into a List of Tuples of Input and Target Tensors"""

        if self.train:
            target = {"boxes": [], "labels": []}

            # Extract annotations from the graph
            node_data = graph.nodes(data=True)
            for id, ann in node_data:
                pos = ann["position"]
                xmin = pos["x"] - pos["width"] / 2
                ymin = pos["y"] - pos["height"] / 2
                xmax = pos["x"] + pos["width"] / 2
                ymax = pos["y"] + pos["height"] / 2

                target["boxes"].append([xmin, ymin, xmax, ymax])

                # Convert string labels to integer class indices using self.classes list
                class_id = self.get_class_id(ann["type"])
                target["labels"].append(class_id)

            # Convert lists to tensors
            target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32)
            target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)

            return [(self.transformer(image), target, None)]

        else:
            return [(self.transformer(image), None, None)]

    def post_process(self, pred_list: List[Tuple[Tensor, any]], graph: EngGraph):
        """Integrates a List of Prediction and Info Tensors into a Graph Structure"""

        boxes = pred_list[0][0]["boxes"].tolist()
        labels = pred_list[0][0]["labels"].tolist()
        scores = pred_list[0][0]["scores"].tolist()

        for box, label, score in zip(boxes, labels, scores):
            if score > self.score_threshold:
                graph.add_node(
                    type=self.classes[label],
                    position=BoundingBox(box[0], box[1], box[2], box[3]).position,
                )

    def filter_pred_dict_using_score(
        self, pred_dict: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Filter out detections with score below the threshhold"""
        scores = pred_dict["scores"]
        mask = scores > self.score_threshold
        return {key: tensor[mask] for key, tensor in pred_dict.items()}

    def get_class_id(self, ann_type: str) -> int:
        """Get the id of the object detection class"""
        class_id = (
            self.classes.index(ann_type)
            if ann_type in self.classes
            else self.classes.index("unknown")
        )
        return class_id

    def get_class_name_from_id(self, class_id: int) -> str:
        """Get the class name of the object detection class"""
        return self.classes[class_id]
