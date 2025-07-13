"""utils.py: Helper Functions for Circuitgraph"""

# System Imports
from typing import Tuple, Union

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2024, DFKI and others"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"


Number = Union[int, float]


def bbox_to_pos(x_min: Number, x_max: Number, y_min: Number, y_max: Number,
                rot: Number = 0, mirror_h: bool = False, mirror_v: bool = False) -> dict:
    """Converts a Bounding Box to a Position as Used in EngGraph"""

    return {"x": (x_min + x_max) / 2.0, "y": (y_min + y_max) / 2.0,
            "width": abs(x_max - x_min), "height": abs(y_max - y_min),
            "rotation": rot, "mirror_horizontal": mirror_h, "mirror_vertical": mirror_v}


def pos_to_bbox(pos: dict) -> Tuple[int, int, int, int, int, bool, bool]:
    """Converts a Position as Used in EngGraph to a Bounding Box"""

    x: Number = pos['x']
    y: Number = pos['y']
    width: Number = pos.get("width", 0)
    height: Number = pos.get("height", 0)

    return round(x - abs(width / 2.0)),  \
           round(x + abs(width / 2.0)),  \
           round(y - abs(height / 2.0)), \
           round(y + abs(height / 2.0)), \
           pos.get('rotation', 0), \
           pos.get('mirror_horizontal', False), \
           pos.get('mirror_vertical', False)


def encode_color(color):
    """Encodes a Color to Hex"""

    if not color:
        return ""

    return f"#{'0' if color[2] < 16 else ''}{hex(color[2])[2:]}" \
           f"{'0' if color[1] < 16 else ''}{hex(color[1])[2:]}" \
           f"{'0' if color[0] < 16 else ''}{hex(color[0])[2:]}"
