"""imageGraphDatasetLoader.py: Loads Raw Images and EngGraphs from the CGHD Dataset and Processes Them"""

# System Imports
from typing import List, Tuple, Type
from os import listdir
from os.path import join, split

# Project Imports
from extraction.src.core.processor import Processor
from extraction.src.core.dataloader import Dataloader
from converter.converter.pascalVocConverter import PascalVocConverter

# Third-Party Imports
import cv2

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



class ImageGraphLoader(Dataloader):
    """Loads Raw Images and EngGraphs from the CGHD Dataset and Processes Them"""

    def __init__(self, drafters: list, db_path: str, augment: bool, processor_cls: Type, processor_args: dict,
                 debug: bool, caching: bool = True):

        self.converter = PascalVocConverter()
        super().__init__(drafters, db_path, augment, processor_cls, processor_args, debug, caching)


    def _load_samples(self, db_path: str, drafter: int, processor: Processor) -> Tuple[List, int]:

        samples = []
        raw_file_count = 0

        for graph_path in listdir(join(db_path, f"drafter_{drafter}", "annotations")):
            graph = self.converter.load(join(db_path, f"drafter_{drafter}", "annotations", graph_path))
            img_path = join(db_path, f"drafter_{drafter}", "images", split(graph.graph['image'])[-1])
            img = cv2.imread(img_path)
            samples += processor.pre_process(img, graph)
            raw_file_count += 1

        return samples, raw_file_count
