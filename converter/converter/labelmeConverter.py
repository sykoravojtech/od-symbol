"""labelmeConverter.py Import from and Export to LabelME JSON"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Prototype"

# System Imports
import json
from itertools import chain

# Third-Party Imports
import networkx as nx

# Project Imports
from converter.core.converter import Converter
from converter.core.engineeringGraph import EngGraph
from converter.core.boundingbox import BoundingBox
from converter.core.geometry import Point


class LabelMeConverter(Converter):
    """LabelME Converter Class"""

    def _parse(self, data: str) -> EngGraph:
        """Reads a LabelME File String and Return an Engineering Graph Object"""

        json_data = json.loads(data)
        graph = EngGraph(name=json_data['imagePath'],
                         width=int(json_data['imageHeight']),
                         height=int(json_data['imageHeight']))

        # Add Nodes
        for shape in json_data['shapes']:
            if shape['shape_type'] == "polygon":
                if shape['label'] != "wire":
                    node_id = graph.add_node(shape.get('group_id', None),
                                             name=shape.get('group_id', None),
                                             type=shape['label'],
                                             position=BoundingBox(points=shape['points'],
                                                                  rotation=int(shape.get("rotation", 0))).position,
                                             text=shape.get("text", ""),
                                             shape=shape['points'])

                    for nbr, port in enumerate(json_data['shapes']):
                        if port['shape_type'] == "point" and port['label'].startswith("connector") \
                           and shape['group_id'] == port['group_id']:
                            graph.add_port(node_id=node_id,
                                           position={'x': port['points'][0][0], 'y': port['points'][0][1]},
                                           name=(port['label'].split(".")[1] if "." in port['label'] else "")+str(nbr))

        # Add Edges
        for shape in json_data['shapes']:
            if shape['shape_type'] == "polygon":
                if shape['label'] == "wire":

                    endpoints = [Point(x=port['points'][0][0], y=port['points'][0][1])
                                 for port in json_data['shapes']
                                 if port['shape_type'] == "point" and port['label'].startswith("connector") and
                                 shape['group_id'] == port['group_id']]

                    if len(endpoints) == 2:
                        source, target = endpoints
                        source_node, source_port = graph.closestPort(source)
                        target_node, target_port = graph.closestPort(target)
                        graph.add_edge(source_node, target_node, **{'sourcePort': source_port,
                                                                    'targetPort': target_port,
                                                                    'shape': shape['points']})

                    else:
                        print(f"Can't load Edge with ports: {endpoints}")

        return graph


    def _write(self, graph: EngGraph, props=True, edges=True, ports=True, **kwargs) -> str:
        """Constructs a LabelME File String from an Engineering Graph"""

        node_class = nx.get_node_attributes(graph, 'type')
        node_shape = nx.get_node_attributes(graph, 'shape')
        nodes = ((node_id, node_class[node_id], node_shape[node_id]) for node_id in node_class.keys())
        node_ports = nx.get_node_attributes(graph, 'ports')
        ports = ((node_id, port["position"]["x"], port["position"]["y"])
                 for node_id in node_ports.keys() for port in node_ports[node_id])
        edge_shape = nx.get_edge_attributes(graph, 'shape')
        edges = ((edge_id, "wire", edge_shape[edge_id]) for edge_id in edge_shape.keys())

        return json.dumps({'version': '3.16.7', 'flags': {}, 'lineColor': [0, 255, 0, 128], 'fillColor': [255, 0, 0, 128],
                           'imagePath': graph.graph['name'], 'imageData': None,
                           'imageHeight': graph.graph['height'], 'imageWidth': graph.graph['width'],
                           'shapes': [{'label': item_class, 'line_color': None, 'fill_color': None,
                                       'points': item_shape, 'group_id': item_id, 'shape_type': 'polygon', 'flags': {}}
                                      for item_id, item_class, item_shape in chain(nodes, edges)] +
                                     [{'label': "connector", 'points': [[port_x, port_y]],
                                       'group_id': node_id, 'shape_type': 'point', 'flags': {}}
                                      for node_id, port_x, port_y in ports]},
                          indent=2)
