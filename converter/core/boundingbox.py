"""boundingbox.py: Bounding Box with Point-Update and Internal Position Representation"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Development"

# System Imports
from typing import Tuple

class BoundingBox:
    """Bounding Box with Point-Update"""

    def __init__(self, x_min=999999, y_min=999999, x_max=-999999, y_max=-999999, points: list = None, rotation=0):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.rotation = rotation

        if points:
            self.x_min = min([p[0] for p in points])
            self.x_max = max([p[0] for p in points])
            self.y_min = min([p[1] for p in points])
            self.y_max = max([p[1] for p in points])


    def update(self, x: int, y: int) -> None:
        """Updates the Bounding Box bv incorporating a new Point"""

        self.x_min = min(self.x_min, x)
        self.x_max = max(self.x_max, x)
        self.y_min = min(self.y_min, y)
        self.y_max = max(self.y_max, y)


    @property
    def size(self) -> Tuple[int, int]:
        """returns Width and Height of the Bounding Box"""

        return abs(self.x_max - self.x_min), abs(self.y_max - self.y_min)

    @property
    def position(self) -> dict:
        """Returns a Standard BB Descriptor of self"""

        return {"x": (self.x_min + self.x_max) // 2, "y": (self.y_min + self.y_max) // 2,
                "width": abs(self.x_max - self.x_min), "height": abs(self.y_max - self.y_min),
                "rotation": self.rotation}
