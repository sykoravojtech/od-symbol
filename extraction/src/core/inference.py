"""inference.py: Performs Model Inference given an Image and optional Graph Structure"""

# System Imports
import json
import sys
from os.path import join
from typing import Callable, Dict, Optional, Type

# Third-Party Imports
import cv2
import torch

# Project Imports
from converter.converter.jsonConverter import JSONConverter
from extraction.src.core.package_loader import class_from_package
from extraction.src.core.processor import Processor

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023-2024, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"


def processor_from_config(config_file: str) -> Processor:
    """Ready-for-Inference Creates based on Configuration Script"""

    with open(config_file) as config_data:
        config = json.loads(config_data.read())

    model_cls = class_from_package(config["model_class"])
    model_args = config["model_parameter"]
    model_path = join(
        "extraction", "model", config["training_parameter"]["name"], "model_state.pt"
    )
    processor_cls = class_from_package(config["processor_class"])
    processor_args = config["processor_parameter"]

    model = model_cls(**model_args)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    processor: Processor = processor_cls(
        augment=False, train=False, debug=False, **processor_args
    )
    processor.set_model(model=model)

    return processor


def inference(
    model_path: str,
    model_cls: Type,
    model_args: Dict,
    path_image: str,
    path_graph: str,
    processor_cls: Type,
    processor_args: Dict,
    name: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Loads an Image, an EngGraph in Pascal VOC Format, and a Model,
    Performs Preprocessing, Applies the Model to all Tensors, Postprocesses them and Saves the Results
    """

    print(f"Performing {name}")

    print("Loading Graph...")
    graph = JSONConverter().load(path_graph)

    print("Loading Image...")
    image = cv2.imread(path_image)

    print("Loading Model...")
    model = model_cls(**model_args)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    print("Loading Processor...")
    processor = processor_cls(augment=False, train=False, debug=debug, **processor_args)
    processor.set_model(model=model)
    processor.inference(image=image, graph=graph)

    print("Saving Graph...")
    # JSONConverter().store(graph, path_graph)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Error: Must provide paths to files: Image (PNG), Graph (Pascal VOC), Config (JSON)"
        )

    else:
        path_image = sys.argv[2]
        path_graph = sys.argv[3]

        with open(sys.argv[1]) as json_file:
            config = json.loads(json_file.read())

        inference(
            model_path=join(
                "extraction",
                "model",
                config["training_parameter"]["name"],
                "model_state.pt",
            ),
            model_cls=class_from_package(config["model_class"]),
            model_args=config["model_parameter"],
            path_image=path_image,
            path_graph=path_graph,
            processor_cls=class_from_package(config["processor_class"]),
            processor_args=config["processor_parameter"],
            name=config["training_parameter"]["name"],
            debug=config["training_parameter"]["debug"],
        )
