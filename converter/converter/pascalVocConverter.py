"""PascalVocConverter.py Import from and Export to Pascal VOC"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022, DFKI"
__status__ = "Prototype"

# System Imports
import os
from lxml import etree as ET

# Project Imports
from converter.core.converter import Converter
from converter.core.engineeringGraph import EngGraph
from converter.core.utils import pos_to_bbox, bbox_to_pos


class PascalVocConverter(Converter):
    """Pascal Voc Converter Class"""

    MARGIN = 2

    def _parse(self, data: str) -> EngGraph:
        """Reads a Pascal VOC File String and Return an Engineering Graph Object"""

        root = ET.fromstring(data)
        graph = EngGraph(name=root.find("filename").text,
                         width=int(root.find("size/width").text),
                         height=int(root.find("size/height").text),
                         image=root.find("path").text)

        for nbr, obj in enumerate(root.findall("object")):
            cls = obj.find("name").text
            name = f"annotation/object{nbr}"  # xpointer(/annotation/object[{nbr}]) Best so far: f"annotation/object[{nbr}]"
            x_min, x_max, y_min, y_max = (float(obj.find(label).text)
                                          for label in ("bndbox/xmin", "bndbox/xmax", "bndbox/ymin", "bndbox/ymax"))
            rotation = int(obj.find("bndbox/rotation").text) if obj.find("bndbox/rotation") is not None else 0
            mirror_tags = [tag.text for tag in obj.findall("bndbox/mirror")]
            mirror_h = "horizontal" in mirror_tags
            mirror_v = "vertical" in mirror_tags
            text = obj.find("text").text if obj.find("text") is not None else ""

            graph.add_node(nbr, name=name, type=cls,
                           position=bbox_to_pos(x_min, x_max, y_min, y_max, rotation, mirror_h, mirror_v),
                           text=text, ports=[], shape=[], properties=[])

        return graph


    def _write(self, graph: EngGraph, props=True, edges=True, ports=True, **kwargs) -> str:
        """Constructs a Pascal VOC File String from an Engineering Graph"""

        main_element = ET.Element('annotation')
        tree = ET.ElementTree(main_element)

        subelement = ET.SubElement(main_element, "folder")
        subelement.text = "images"

        subelement = ET.SubElement(main_element, "filename")
        subelement.text = os.path.split(graph.graph["image"])[-1]

        subelement = ET.SubElement(main_element, "path")
        subelement.text = graph.graph["image"]

        subelement_source = ET.SubElement(main_element, "source")
        subelement = ET.SubElement(subelement_source, "database")
        subelement.text = "CGHD"

        subelement_size = ET.SubElement(main_element, "size")
        subelement = ET.SubElement(subelement_size, "width")
        subelement.text = str(graph.graph["width"])

        subelement = ET.SubElement(subelement_size, "height")
        subelement.text = str(graph.graph["height"])

        subelement = ET.SubElement(subelement_size, "depth")
        subelement.text = "3"

        ET.SubElement(main_element, "segmented").text = "0"

        # Node Export
        for node_id, node_data in graph.nodes.items():
            self._write_object(main_element, node_data['position'],
                               node_data.get('type', 'unknown'), node_data.get('text', None))

            if props:
                for node_prop in node_data['properties']:
                    if node_prop["name"] and node_prop["visibility"]:
                        self._write_object(main_element, node_prop["position"], 'text')

            if ports:
                for node_port in node_data['ports']:
                    self._write_object(main_element, node_port["position"], 'port')

        # Edge Export
        if edges:
            for edge in graph.edgeEndPoints():
                self._write_object(main_element,
                                   {"x": (edge[0][0] + edge[1][0])/2.0,
                                    "y": (edge[0][1] + edge[1][1])/2.0,
                                    "width": abs(edge[0][0] - edge[1][0]) + 2*self.MARGIN,
                                    "height": abs(edge[0][1] - edge[1][1]) + 2*self.MARGIN},
                                   'edge')

        ET.indent(main_element, space="\t")
        return ET.tostring(tree.getroot(), pretty_print=True).decode()


    @staticmethod
    def _write_object(root: ET.Element, pos: dict, cls: str, text: str = None) -> None:
        """Writes a Pascal VOC Annotation Object"""

        x_min, x_max, y_min, y_max, rotation, mirror_horizontal, mirror_vertical = pos_to_bbox(pos)

        object = ET.SubElement(root, "object")

        name = ET.SubElement(object, "name")
        name.text = cls

        subelement = ET.SubElement(object, "pose")
        subelement.text = 'Unspecified'

        subelement = ET.SubElement(object, "truncated")
        subelement.text = '0'

        subelement = ET.SubElement(object, "difficult")
        subelement.text = '0'

        bbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(x_min)
        ET.SubElement(bbox, "ymin").text = str(y_min)
        ET.SubElement(bbox, "xmax").text = str(x_max)
        ET.SubElement(bbox, "ymax").text = str(y_max)

        if rotation:
            ET.SubElement(bbox, "rotation").text = str(rotation)

        if mirror_horizontal:
            ET.SubElement(bbox, "mirror").text = "horizontal"

        if mirror_vertical:
            ET.SubElement(bbox, "mirror").text = "vertical"

        if text:
            ET.SubElement(object, "text").text = text
