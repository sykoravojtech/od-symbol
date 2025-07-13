import json
from typing import Dict

__author__ = "Vojtěch Sýkora"


def save_config(config: Dict, save_path: str) -> None:
    with open(save_path, "w") as file:
        json.dump(config, file, indent=2)


def load_config(load_path: str) -> Dict:
    with open(load_path) as json_file:
        config = json.loads(json_file.read())
    return config
