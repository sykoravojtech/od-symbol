"""svgConverter.py Export to SVG Images"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Development"

# System Imports
import xml.etree.ElementTree as ET

# Project Imports
from converter.core.converter import Converter
from converter.core.engineeringGraph import EngGraph
from converter.core.renderer import Renderer, Line, Rectangle, Circle, Polygon, Text



class SvgConverter(Converter):
    """SVG Converter Class"""

    renderer = Renderer()

    def _write(self, graph: EngGraph, mode="simple", **kwargs) -> str:
        """Constructs a SVG File String from an Engineering Graph"""

        main_element = ET.Element('svg')
        main_element.attrib['xmlns'] = "http://www.w3.org/2000/svg"
        main_element.attrib['height'] = str(graph.graph['height'])
        main_element.attrib['width'] = str(graph.graph['width'])
        tree = ET.ElementTree(main_element)

        for item in self.renderer.render(graph, mode, **kwargs):

            if type(item) is Line:
                xml_element = ET.SubElement(main_element, "line")
                xml_element.attrib['x1'] = str(item.a.x)
                xml_element.attrib['y1'] = str(item.a.y)
                xml_element.attrib['x2'] = str(item.b.x)
                xml_element.attrib['y2'] = str(item.b.y)
                xml_element.attrib['stroke'] = SvgConverter._encode_color(item.color)
                xml_element.attrib['stroke-width'] = str(item.stroke)

            if type(item) is Rectangle:
                xml_element = ET.SubElement(main_element, "rect")
                xml_element.attrib['x'] = str(item.left)
                xml_element.attrib['y'] = str(item.top)
                xml_element.attrib['width'] = str(item.right-item.left)
                xml_element.attrib['height'] = str(item.bottom-item.top)
                xml_element.attrib['stroke'] = SvgConverter._encode_color(item.color)
                xml_element.attrib['stroke-width'] = str(item.stroke)
                xml_element.attrib['fill'] = "none"

            if type(item) is Circle:
                xml_element = ET.SubElement(main_element, "circle")
                xml_element.attrib['cx'] = str(item.x)
                xml_element.attrib['cy'] = str(item.y)
                xml_element.attrib['r'] = str(item.radius)
                xml_element.attrib['stroke'] = SvgConverter._encode_color(item.color)
                xml_element.attrib['stroke-width'] = str(item.stroke)
                xml_element.attrib['fill'] = SvgConverter._encode_color(item.fillColor)

            if type(item) is Polygon:
                xml_element = ET.SubElement(main_element, "polygon")
                xml_element.attrib['points'] = " ".join([f"{p[0]},{p[1]}" for p in item.points])
                xml_element.attrib['stroke'] = SvgConverter._encode_color(item.color)
                xml_element.attrib['stroke-width'] = str(item.stroke)
                xml_element.attrib['fill'] = "none"

            if type(item) is Text:
                xml_element = ET.SubElement(main_element, "text")
                xml_element.attrib['x'] = str(item.x)
                xml_element.attrib['y'] = str(item.y)
                xml_element.attrib['fill'] = SvgConverter._encode_color(item.color)
                xml_element.text = item.text

        return '<?xml version="1.0" encoding="UTF-8" standalone="no"?>' + ET.tostring(tree.getroot()).decode()


    @staticmethod
    def _encode_color(color):

        if not color:
            return "none"

        return f"#{'0' if color[2]<10 else ''}{hex(color[2])[2:]}" \
               f"{'0' if color[1]<10 else ''}{hex(color[1])[2:]}" \
               f"{'0' if color[0]<10 else ''}{hex(color[0])[2:]}"
