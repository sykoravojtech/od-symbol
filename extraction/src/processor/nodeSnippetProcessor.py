"""nodeSnippetProcessor.py: Base Class for Pre-/Post-Processing of Node Snippets"""

# System Imports
from abc import abstractmethod
from typing import Tuple, List, Union, Dict
from random import randint

# Project Imports
from converter.core.engineeringGraph import EngGraph
from extraction.src.core.processor import Processor

# Third-Party Imports
from numpy import ndarray
from torch import Tensor

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



class NodeSnippetProcessor(Processor):
    """Base Class for Pre-/Post-Processing of Node Snippets"""

    def __init__(self, augment: bool, train: bool, debug: bool, rand_crop: int):
        super().__init__(augment, train, debug)
        self.rand_crop = rand_crop


    @abstractmethod
    def _node_filter(self, node_data: dict) -> bool:
        """Decides Whether a Node Can be Processed"""

        pass


    @abstractmethod
    def _make_input(self, snippet: ndarray, node_data: dict) -> List[Tensor]:
        """"Creates a List of Input Tensors for the Model"""

        pass


    @abstractmethod
    def _make_target(self, node_data: dict) -> List[Tensor]:
        """Creates a List of Target Models for the Model"""

        pass


    @abstractmethod
    def _integrate_data(self, pred: Tensor, node_data: dict) -> None:
        """Integrates the Model's Predictions into the Node's Attributes"""

        pass


    def crop(self, image: ndarray, bbox: Dict[str, int], margins: Dict[str, int] = None):
        """Crops a Snippet from an image while maintaining safety Limits"""

        top = int(bbox["y"] - bbox["height"] / 2)
        bottom = int(bbox["y"] + bbox["height"] / 2)
        left = int(bbox["x"] - bbox["width"] / 2)
        right = int(bbox["x"] + bbox["width"] / 2)

        if margins is not None:
            top = max(0, top + margins["top"])
            bottom = min(image.shape[0], bottom + margins["bottom"])
            left = max(0, left + margins["left"])
            right = min(image.shape[1], right + margins["right"])

        return image[top:bottom, left: right]


    def pre_process(self, image: ndarray, graph: EngGraph) -> List[Tuple[Tensor, Union[Tensor, any]]]:
        """Turns ONE Raw Image and ONE Graph Structure into a List of Tuples of Input and Target/Info Tensors"""

        samples = []

        for node_id, node_data in graph.nodes.items():
            if self._node_filter(node_data):
                snippet = self.crop(image, node_data['position'])

                if self.train:
                    inputs = self._make_input(snippet, node_data)
                    targets = self._make_target(node_data)
                    samples += zip(inputs, targets, [node_data for _ in inputs])

                    if self.augment:
                        snippet_augment = self.crop(image, node_data['position'],
                                                    {dim: randint(-self.rand_crop, self.rand_crop)
                                                     for dim in ["top", "bottom", "left", "right"]})

                        inputs = self._make_input(snippet_augment, node_data)
                        targets = self._make_target(node_data)
                        samples += zip(inputs, targets, [node_data for _ in inputs])

                else:
                    samples += [(snippet, None, node_id) for snippet in self._make_input(snippet, node_data)]

        return samples


    def post_process(self, pred_list: List[Tuple[Tensor, any]], graph: EngGraph) -> None:
        """Integrates a List of Prediction and Info Tensors into a Graph Structure"""

        for pred, node_id in pred_list:
            self._integrate_data(pred, graph.nodes[node_id])
