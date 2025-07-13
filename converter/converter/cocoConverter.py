"""cocoConverter.py Import from and Export to MS COCO JSON"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Prototype"

# System Imports
import os
import json

# Third-Party Imports
import networkx as nx
import numpy as np
import cv2

# Project Imports
from converter.core.converter import Converter
from converter.core.engineeringGraph import EngGraph


class CocoConverter(Converter):
    """COCO Converter Class"""

    def _parse(self, data: str) -> EngGraph:
        """Reads a COCO File String and Return an Engineering Graph Object"""

        json_data = json.loads(data)
        graph = EngGraph(name=json_data['imagePath'],
                         width=int(json_data['imageHeight']),
                         height=int(json_data['imageHeight']))

        return graph


    def _write(self, graph: EngGraph, props=True, edges=True, ports=True, **kwargs) -> str:
        """Constructs a COCO File String from an Engineering Graph"""

        return json.dumps({"images": [self._write_info(graph)],
                           "categories": self._write_categories(),
                           "annotations": self._write_annotations(graph)},
                          indent=4)


    def _write_info(self, graph: EngGraph, image_id: int = 0) -> dict:
        """Turns the Graph's Metadata into a COCO Image Descriptor"""

        return {"height": graph.graph['height'], "width": graph.graph['width'],
                "id": image_id, "file_name": graph.graph['name']}


    def _write_categories(self) -> list:
        """Generates a COCO Categories Descriptor List"""

        with open(os.path.join("converter", "symbols", "classes_ports.json")) as ports_file:
            classes_ports = json.load(ports_file)

        return [{"supercategory": "component",
                 "id": class_id,
                 "name": class_name,
                 "keypoints": class_ports,
                 "skeleton": [list(range(1, 1+len(class_ports)))]}
                for class_id, (class_name, class_ports) in enumerate(classes_ports.items())]


    def _write_annotations(self, graph: EngGraph, image_id: int = 0) -> list:
        """Turns the Graph's Nodes into a list of COCO Annotations"""

        node_class = nx.get_node_attributes(graph, 'type')
        node_shape = nx.get_node_attributes(graph, 'shape')
        node_pos = nx.get_node_attributes(graph, 'position')
        node_ports = nx.get_node_attributes(graph, 'ports')
        nodes = ((node_id, node_class[node_id], node_shape[node_id], node_pos[node_id], node_ports[node_id])
                 for node_id in node_class.keys())

        edge_shape = nx.get_edge_attributes(graph, 'shape')

        return [{"segmentation": [[ordinate for point in node_shape for ordinate in point]],
                 "num_keypoints": len(node_ports),
                 "iscrowd": 0,
                 "keypoints": [nbr for port in
                               [(port["position"]["x"], port["position"]["y"], 2) for port in node_ports]
                               for nbr in port],
                 "area": cv2.contourArea(np.array(node_shape, dtype=np.int32)),
                 "image_id": image_id,
                 "bbox": [node_pos["x"], node_pos["y"], node_pos["width"], node_pos["height"]],
                 "category_id": 9,
                 "id": 1}
                for node_id, node_class, node_shape, node_pos, node_ports in nodes]
