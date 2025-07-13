"""loader.py: Abstract Base Class for Cached Loader"""

# System Imports
from typing import List, Tuple, Type
from os.path import join, isfile
from hashlib import sha256
from abc import ABC, abstractmethod

# Project Imports
from extraction.src.core.processor import Processor

# Third-Party Imports
from torch import Tensor, save, load

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"



class Dataloader(ABC):
    """Abstract Base Class for Cached Loader"""

    def __init__(self, drafters: list, db_path: str, augment: bool, processor_cls: Type, processor_args: dict,
                 debug: bool, caching: bool = True):
        """Prepares the Training Data Utilizing either the Cache or Processor"""

        self.drafters = drafters
        self.db_path = db_path
        self.augment = augment
        self.processor_cls = processor_cls
        self.processor_args = processor_args
        self._samples = []
        self.debug = debug

        self.processor: Processor = self.processor_cls(augment=self.augment, train=True,
                                                       debug=debug, **self.processor_args)

        if caching:
            if self._cache_exists():
                self._load_cache()

            else:
                self._create_samples()
                self._create_cache()

        else:
            self._create_samples()


    @abstractmethod
    def _load_samples(self, db_path: str, drafter: int, processor: Processor) -> Tuple[List, int]:
        """Loads the Files Required by the Processor and Preprocesses them"""
        pass


    def _create_samples(self) -> None:

        raw_file_count = 0

        for drafter in self.drafters:
            if self.debug:
                print(f"Processing drafter {drafter}...", end="")

            drafter_samples, drafter_raw_file_count = self._load_samples(self.db_path, drafter, self.processor)
            self._samples += drafter_samples
            raw_file_count += drafter_raw_file_count

            if self.debug:
                print(f"OK. Loaded {len(drafter_samples)} Samples from {drafter_raw_file_count} Raw Files")

        print(f"Loaded {len(self._samples)} Samples from {raw_file_count} Raw Files")


    def _cache_exists(self):
        """Returns whether a Suitable Cache File already Exists"""

        return isfile(join("extraction", "loaders", self._create_signature()))


    def _create_signature(self) -> str:
        """Create a Cache File Name Bearing a Unique Signature of the Parameters"""

        hash_1 = f"{self.drafters}{self.db_path}{self.augment}{self.processor_cls.__name__}{self.__class__.__name__}"
        hash_2 = f"frozenset(self.processor_args.items())"  # TODO REAL Hash Here
        return sha256(f"{hash_1}{hash_2}".encode()).hexdigest() + ".pt"


    def _create_cache(self) -> None:

        print("Generating Cache...", end="")
        save(self._samples, join("extraction", "loaders", self._create_signature()))
        print("OK")


    def _load_cache(self) -> None:

        self._samples = load(join("extraction", "loaders", self._create_signature()))
        print(f"Loaded {len(self._samples)} Snippets from Cache")


    def __len__(self) -> int:
        """Amount of Samples"""

        return len(self._samples)


    def __getitem__(self, item) -> List[Tuple[Tensor, Tensor]]:
        """Returns a Sample"""

        return self._samples[item]
