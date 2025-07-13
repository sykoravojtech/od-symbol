"""pngConverter.py Export to PNG Images"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Development"

# Third-Party Imports
import cv2
import numpy as np

# Project Imports
from converter.core.converter import Converter
from converter.core.engineeringGraph import EngGraph
from converter.core.renderer import Renderer, Line, Rectangle, Circle, Polygon, Text



class PngConverter(Converter):
    """PNG Converter Class"""

    STOREMODE = "wb"
    renderer = Renderer()

    def to_array(self, graph: EngGraph, mode="simple", **kwargs) -> np.ndarray:
        """Renders the Graph to an ndarray"""

        image = kwargs.get("background", None).copy() \
            if kwargs.get("background", None) is not None \
            else np.ones((int(graph.graph["height"]), int(graph.graph["width"]), 3), np.uint8) * 255

        for item in self.renderer.render(graph, mode, **kwargs):

            if type(item) is Line:
                cv2.line(image, (int(item.a.x), int(item.a.y)), (int(item.b.x), int(item.b.y)),
                         item.color, item.stroke, cv2.LINE_AA)

            if type(item) is Rectangle:
                cv2.rectangle(image, (int(item.left), int(item.top)), (int(item.right), int(item.bottom)),
                              item.color, item.stroke)

            if type(item) is Circle:
                if item.fillColor:
                    cv2.circle(image, (int(item.x), int(item.y)), int(item.radius), item.fillColor, -1)

                if item.stroke:
                    cv2.circle(image, (int(item.x), int(item.y)), int(item.radius), item.color, item.stroke)

            if type(item) is Polygon:
                if item.fillColor:
                    cv2.fillPoly(image, np.array([item.points], dtype=np.int32), item.fillColor)

                if item.stroke:
                    cv2.polylines(image, np.array([item.points], dtype=np.int32),
                                  isClosed=True, color=item.color, thickness=int(item.stroke))

            if type(item) is Text:
                FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
                FONT_SCALE = 1.2
                FONT_THICKNESS = 2
                cv2.putText(image, item.text, (int(item.x), int(item.y)),
                            FONT_FACE, FONT_SCALE, item.color, FONT_THICKNESS, cv2.LINE_AA)

        return image


    def _write(self, graph: EngGraph, mode="simple", **kwargs) -> str:
        """Constructs a PNG File String from an Engineering Graph"""

        return cv2.imencode('.png', self.to_array(graph, mode, **kwargs))[1].tostring()
