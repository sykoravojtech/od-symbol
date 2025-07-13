"""package_loader.py: Utilities for Training/Inference"""

# System Imports
import json
import os
import sys
from importlib import import_module
from inspect import getmembers
from typing import Tuple, Type

import torch
from extraction.src.core.processor import Processor

__author__ = "Johannes Bayer, Vojtěch Sýkora"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"


def class_from_package(descriptor: str) -> Type:
    """Returns a Class given a Package/Class Descriptor String"""

    try:
        module_name, cls_name = descriptor.split("/")
        module = import_module(module_name)
        cls = [value for key, value in getmembers(module) if key == cls_name][0]
        return cls

    except (IndexError, ValueError, TypeError, AttributeError, ModuleNotFoundError):
        sys.exit(f"Error: Improper Package Descriptor: '{descriptor}'")


def build_processor_model(
    config: str, device: str = "cpu"
) -> Tuple[Processor, torch.nn.Module]:
    """Ready-for-Inference Creates based on Configuration Script"""
    model_cls = class_from_package(config["model_class"])
    model_args = config["model_parameter"]
    model = model_cls(**model_args)

    # add ECA layers
    if config.get("use_eca", False):
        from extraction.src.models.fasterrcnn import FasterRCNN
        from extraction.src.models.fasterrcnn_addons.eca import attach_eca

        if isinstance(model, FasterRCNN):
            k_size = 5  # Default kernel size used in our training
            attach_eca(model, kernel_size=k_size)
            print(f"ECA successfully attached to model with kernel size {k_size}")

    # add dilated convolution layers
    if config.get("use_dilated", False):
        from extraction.src.models.fasterrcnn import FasterRCNN
        from extraction.src.models.fasterrcnn_addons.dilated_backbone import (
            attach_dilated_convs,
        )

        if isinstance(model, FasterRCNN):
            dilation_rates = [1, 2, 4]  # default dilation rates in our training
            attach_dilated_convs(model, dilation_rates=dilation_rates)
            print(
                f"Dilated convolutions successfully attached to model with dilation rates {dilation_rates}"
            )

    model_path = config.get("model_path", None)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    processor_cls = class_from_package(config["processor_class"])
    processor_args = config["processor_parameter"]

    processor: Processor = processor_cls(
        augment=False, train=False, debug=False, **processor_args
    )
    processor.set_model(model=model)
    name = config["training_parameter"]["name"]
    print(f"Processor and model built ({name})")

    return processor, model


def clsstr(cls: type) -> str:
    """ "Returns a String Describing a Class"""

    return f"{cls.__module__}/{cls.__name__}"
