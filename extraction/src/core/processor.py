"""processor.py: Abstract Base Class for Complete Inference Process AND Training Preparation"""

# System Imports
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch

# Third-Party Imports
from numpy import ndarray
from torch import Tensor

# Project Imports
from converter.core.engineeringGraph import EngGraph

__author__ = "Johannes Bayer, Vojtěch Sýkora"
__copyright__ = "Copyright 2023-2024, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"


class Processor(ABC):
    """Abstract Base Class for Complete Inference Process AND Training Preparation"""

    cpu = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, augment: bool, train: bool, debug: bool):
        self.augment: bool = augment
        self.train: bool = train
        self.debug: bool = debug
        self.model: Optional[torch.device] = None

    @abstractmethod
    def pre_process(
        self, image: ndarray, graph: EngGraph
    ) -> List[Tuple[Tensor, Tensor, any]]:
        """Abstract Preprocessing Method, to be overridden in Subclasses
        Turns ONE Raw Image and ONE Graph Structure into a List of Samples
        where a Sample is a Tuple of Input Tensor, Target Tensor and Info Data"""

        return []

    @abstractmethod
    def post_process(
        self, pred_list: List[Tuple[Tensor, any]], graph: EngGraph
    ) -> None:
        """Integrates a List of Prediction and Info Tensors into a Graph Structure"""

        pass

    def set_model(self, model):
        """Sets a model, allowing the processor to inference. Deactivates Gradient Calculation."""

        print(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        torch.set_grad_enabled(False)
        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def inference(self, image: ndarray, graph: EngGraph) -> List[Tuple[Tensor, any]]:
        """Complete Inference Cycle"""

        if self.model is None:
            print("Error: No Model Provided!")
            return None

        print("Preprocessing...", end="", flush=True)
        data_in = self.pre_process(image, graph)
        data_out = []
        print("OK")

        print("Inferencing", end="")
        for tensor_in, _, data_info in data_in:
            try:
                # Model Application and Write Result
                pred = self.model(tensor_in[None, :].to(self.device))[0]
                data_out += [
                    (
                        pred.to(self.cpu) if type(pred) is torch.Tensor else pred,
                        data_info,
                    )
                ]
                print(".", end="", flush=True)

            except:
                print("X", end="", flush=True)

        print("OK")

        print("Postprocessing...", end="", flush=True)
        self.post_process(data_out, graph)
        print("OK")
        return data_out
