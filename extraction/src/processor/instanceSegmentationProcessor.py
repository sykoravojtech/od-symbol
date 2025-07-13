"""segmentationProcessor.py: Processor for Semantic Segmentation on Patches"""

# System Imports
from typing import Tuple, List, Union
from random import randint

# Project Imports
from converter.core.engineeringGraph import EngGraph
from converter.core.utils import pos_to_bbox
from converter.core.geometry import transform, Point
from extraction.src.core.processor import Processor

# Third-Party Imports
import cv2
import numpy as np
from torch import Tensor
from torchvision import transforms
import matplotlib.pyplot as plt

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2024-2025"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



class InstanceSegmentationProcessor(Processor):
    """Processor for Semantic Segmentation on Patches"""

    def __init__(self, augment: bool, train: bool, debug: bool, patch_size: int):
        super().__init__(augment, train, debug)

        self.patch_size = patch_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(size=(self.patch_size, self.patch_size))])


    def pre_process(self, images: Union[Tuple[np.ndarray, np.ndarray], np.ndarray], graph: EngGraph) \
            -> List[Tuple[Tensor, Tensor, any]]:
        """Turns ONE Raw Image and ONE Graph Structure into a List of Tuples of Input and Target/Info Tensors"""

        if self.train:
            img_raw, img_map = images
            return [self.make_training_sample(img_raw, img_map, graph, node_id) for node_id in graph.nodes]

        else:
            return [self.make_inference_sample(images, graph, node_id) for node_id in graph.nodes]


    def post_process(self, pred_list: List[Tuple[Tensor, any]], graph: EngGraph) -> None:
        """Integrates a List of Prediction and Info Tensors into a Graph Structure"""

        for seg_map_tensor, node_id in pred_list:
            seg_map = seg_map_tensor[0].detach().numpy()
            seg_map_thres = (255 * (seg_map < 0)).astype(np.uint8)

            if self.debug:
                plt.imshow(seg_map)
                plt.show()
                plt.imshow(seg_map_thres)
                plt.show()

            cntrs = cv2.findContours(seg_map_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
            good_contours = [contour for contour in cntrs if cv2.contourArea(contour) > 10]

            if good_contours:
                contours_combined = np.vstack(good_contours)
                hull = cv2.convexHull(contours_combined)
                poly = np.squeeze(hull).tolist()

                node_pos = {k: (0.0 if k == "rotation" else v)  # Don't use existing Rotation, In-Place Detection
                            for k, v in graph.nodes[node_id]["position"].items()}
                poly_global = [tuple(transform(Point(x/self.patch_size, 1-(y/self.patch_size)), node_pos))
                               for x, y in poly]

                graph.nodes[node_id]["shape"] = poly_global


    def make_training_sample(self, img_raw: np.ndarray, img_map: np.ndarray, graph: EngGraph, node_id: any) \
            -> Tuple[Tensor, Tensor, any]:

        left, right, top, bottom, _, _, _ = pos_to_bbox(graph.nodes[node_id]["position"])
        polygon = np.array([[x-left, y-top] for x, y in graph.nodes[node_id]["shape"]], dtype=np.int32)

        snippet_raw = img_raw[top:bottom, left:right]
        snippet_map = img_map[top:bottom, left:right]
        snippet_poly = cv2.fillPoly(np.zeros((bottom-top, right-left), dtype=np.uint8),
                                    pts=[polygon], color=255)
        snippet_instance = 255-np.bitwise_and(255-snippet_map, snippet_poly)

        if self.debug:
            plt.imshow(snippet_raw)
            plt.show()
            plt.imshow(snippet_map)
            plt.show()
            plt.imshow(snippet_poly)
            plt.show()
            plt.imshow(snippet_instance)
            plt.show()

        return self.transform(snippet_raw), self.transform(snippet_instance), node_id


    def make_inference_sample(self, img_raw: np.ndarray, graph: EngGraph, node_id: any) -> Tuple[Tensor, Tensor, any]:

        left, right, top, bottom, _, _, _ = pos_to_bbox(graph.nodes[node_id]["position"])
        snippet_raw = img_raw[top:bottom, left:right]

        return self.transform(snippet_raw), Tensor(), node_id
