"""imageSegmentationGraphLoader.py: Provider of Raw Image, Segmentation Map and Instance Graph Triples"""

# System Imports
from os.path import join
from os import listdir
from typing import Type, Tuple, List

# Project Imports
from extraction.src.core.processor import Processor
from extraction.src.core.dataloader import Dataloader
from converter.converter.labelmeConverter import LabelMeConverter

# Third-Party Imports
import cv2

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2024"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



class ImageSegmentationGraphLoader(Dataloader):
    """Provider of Raw Image, Segmentation Map and Instance Graph Triples"""

    def __init__(self, drafters: list, db_path: str, augment: bool, processor_cls: Type, processor_args: dict,
                 debug: bool, caching: bool = True):

        self.converter = LabelMeConverter()
        super().__init__(drafters, db_path, augment, processor_cls, processor_args, debug, caching)


    def _load_samples(self, db_path: str, drafter: int, processor: Processor) -> Tuple[List, int]:

        samples = []
        raw_file_count = 0

        for file_name in listdir(join(db_path, f"drafter_{drafter}", "segmentation")):
            img_raw = cv2.imread(join(db_path, f"drafter_{drafter}", "images", file_name))
            img_map = cv2.imread(join(db_path, f"drafter_{drafter}", "segmentation", file_name), cv2.IMREAD_GRAYSCALE)
            graph = self.converter.load(join(db_path, f"drafter_{drafter}", "instances", file_name.split(".")[0]+".json"))
            samples += processor.pre_process((img_raw, img_map), graph)
            raw_file_count += 1

        return samples, raw_file_count
