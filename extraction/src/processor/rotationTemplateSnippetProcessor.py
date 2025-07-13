"""rotationTemplateSnippetProcessor.py: Processor for Template-Assisted Rotation Prediction"""

# System Imports
from os.path import join
from typing import List

# Project Imports
from extraction.src.processor.rotationSnippetProcessor import RotationSnippetProcessor
from converter.core.engineeringGraph import EngGraph
from converter.converter.pngConverter import PngConverter

# Third-Party Imports
import cv2
import torch
import numpy as np
from torchvision import transforms

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



class RotationTemplateSnippetProcessor(RotationSnippetProcessor):
    """Processor for Template-Assisted Rotation Prediction"""


    def __init__(self, augment: bool, train: bool, debug: bool,
                 rand_crop: int, width: int, height: int, supported_classes: List[str], symmetric_classes: List[str],
                 template_margin: int):
        super().__init__(augment, train, debug, rand_crop, width, height, supported_classes, symmetric_classes)

        self.pngConverter = PngConverter()
        self.templates = {'text': np.zeros([width, height, 3], dtype=np.uint8)}

        for symbol in self.pngConverter.renderer.symbols:
            graph = EngGraph("", width, height)
            graph.add_node(type=symbol, position={'x': width//2, 'y': height//2,
                                                  'width': width-template_margin, 'height': height-template_margin})
            self.templates[symbol] = self.pngConverter.to_array(graph, mode="complex",
                                                                rectangles=False, drawNodeId=False, drawNodeType=False,
                                                                symbol=True, symbolColor=(0, 0, 0))

            if self.debug:
                for tmp_name, tmp_img in self.templates.items():
                    cv2.imwrite(join("debug", f"template-{tmp_name}.png"), tmp_img)


    def _make_input(self, snippet: np.ndarray, node_data: dict) -> List[torch.Tensor]:

        return [torch.cat(tensors=(input_tensor,
                                   transforms.ToTensor()(self.templates[node_data['type']])*2 -1),
                          dim=0)[0:4]
                for input_tensor in super()._make_input(snippet, node_data)]
