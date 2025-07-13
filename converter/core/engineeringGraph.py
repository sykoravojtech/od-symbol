"""engineeringGraph.py: Engineering Structures Container Class"""

__author__ = "Johannes Bayer, Vojtěch Sýkora"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Development"

import json

# System Imports
import os
import re
from copy import deepcopy
from math import atan2, pi, sqrt
from random import randint
from typing import Callable, List, Tuple, Union

# Third-Party Imports
import cv2
import networkx as nx
import numpy as np
import torch

from converter.core.geometry import (
    Point,
    length,
    map_points,
    scale,
    shift,
    transform,
    transform_to_local,
)

# Project Imports
from converter.core.symbol import Port, Property, load_symbols
from converter.core.utils import bbox_to_pos, pos_to_bbox

NUMBER = Union[int, float]


class EngGraph(nx.DiGraph):
    """Container for Graph-based Engineering Structures
    like Electrical Engineering Diagrams"""

    with open(os.path.join("converter", "symbols", "properties.json")) as prop_map_file:
        PROP_MAP = json.load(prop_map_file)

    symbols = load_symbols()

    def __init__(self, name="", width=0, height=0, image=""):
        """Protective Wrapper for Enforcing the Datamodel"""

        super().__init__(name=name, width=width, height=height, image=image)

    def add_node(
        self,
        node_id=None,
        name="",
        type="",
        position=None,
        text="",
        ports=None,
        shape=None,
        properties=None,
    ):
        """Protective Wrapper for Enforcing the Datamodel"""

        if node_id is None:
            node_id = self.new_node_id()

        if position is None:
            print("Missing position!")
            position = {"x": 0, "y": 0}

        if shape is None:
            shape = []

        super(EngGraph, self).add_node(
            node_id,
            name=name,
            type=type,
            position=position,
            text=text,
            ports=[],
            shape=shape,
            properties=[],
        )

        if isinstance(ports, list):
            for port in ports:
                if isinstance(port, Port):
                    self.nodes[node_id]["ports"].append(port)
                else:
                    print("WARNING: FALLING BACK TO LEGACY PORT READING")
                    self.add_port(node_id, port["position"], port["name"])

        if isinstance(properties, list):
            for prop in properties:
                if isinstance(prop, Property):
                    self.nodes[node_id]["properties"].append(prop)

        return node_id

    def add_edge(
        self,
        source,
        target,
        sourcePort=None,
        targetPort=None,
        shape=None,
        type="electrical",
    ):
        """Protective Wrapper for Enforcing the Datamodel"""

        attrs = {}

        if type != "electrical":
            attrs["type"] = type

        if sourcePort is not None:
            attrs["sourcePort"] = sourcePort

        if targetPort is not None:
            attrs["targetPort"] = targetPort

        if shape is not None:
            attrs["shape"] = shape

        super(EngGraph, self).add_edge(source, target, **attrs)

    def add_port(
        self, node_id, position: dict = None, name: str = None, port: Port = None
    ):
        """Adds a Port to a node by specifying its node_id and either a Global Position and Name or a Port Instance"""

        ports = self.nodes[node_id]["ports"]
        node_pos = self.nodes[node_id]["position"]

        if port:
            ports.append(port)
            return port.name

        if not name:
            port_names = [port.name for port in ports]
            port_counter = 0

            while str(port_counter) in port_names:
                port_counter += 1

            name = str(port_counter)

        ports.append(
            Port(
                name, transform_to_local(Point(position["x"], position["y"]), node_pos)
            )
        )

        return name

    def replace_port(self, node_id, port_old: Port, port_new: Port):

        self.nodes[node_id]["ports"].remove(port_old)
        self.nodes[node_id]["ports"].append(port_new)

        if not port_old.name == port_new.name:
            for (source, target), attributes in self.edges.items():
                if node_id == source:
                    if attributes.get("sourcePort", None) == port_old.name:
                        attributes["sourcePort"] = port_new.name
                if node_id == target:
                    if attributes.get("targetPort", None) == port_old.name:
                        attributes["targetPort"] = port_new.name

    def add_symbols_ports(self) -> None:
        """Adds Symbol Ports for all Nodes"""

        for node in self.nodes:
            self.add_symbol_ports(node)

    def add_symbol_ports(self, node_id, match_existing: bool = True) -> None:
        """Adds the Ports as Defined in the Node's Type Symbol to the Node"""

        node_type = self.nodes[node_id]["type"]

        if node_type in self.symbols:
            symbol_ports = deepcopy(self.symbols[node_type].ports)
            symbol_ports_pos = {port.position: port for port in symbol_ports}
            existing_port_pos = {
                port.position: port for port in self.nodes[node_id]["ports"]
            }
            point_map = map_points(symbol_ports_pos.keys(), existing_port_pos.keys())

            for symbol_port in symbol_ports:
                if symbol_port.position in point_map.keys():
                    self.replace_port(
                        node_id,
                        existing_port_pos[point_map[symbol_port.position]],
                        symbol_port,
                    )

                else:
                    self.add_port(node_id, port=symbol_port)

    def __iadd__(self, other):
        """Graph Layout Shifting"""

        if type(other) is int or type(other) is float:

            pos = nx.get_node_attributes(self, "position")
            print(pos)

    def __imul__(self, other: Union[int, float, Tuple[NUMBER, NUMBER]]):
        """Graph Layout Scaling"""

        other_x, other_y = geo_resolve(other)

        # Graph Handling
        self.graph["width"] *= other_x
        self.graph["height"] *= other_y

        # Node Handling
        for node_pos in nx.get_node_attributes(self, "position").values():
            geo_mul(node_pos, other_x, other_y)

        # Port Handling
        for node_ports in nx.get_node_attributes(self, "ports").values():
            for port in node_ports:
                geo_mul(port["position"], other_x, other_y)

        # Property Handling
        for node_props in nx.get_node_attributes(self, "properties").values():
            for prop in node_props:
                if "position" in prop:
                    geo_mul(prop["position"], other_x, other_y)

        return self

    def consistency(self) -> bool:
        """Checks whether the Graph complies to the Information Model"""

        # TODO Check Positions (integrity+existence)
        # TODO Check Referenced Ports
        pass

    def closestPort(self, position: Point) -> tuple:
        """Return the node id and Port name in the Graph closest to the given Global Point in Euclidean Metric"""

        node_closest, port_closest = None, None
        dist_closest = 99999

        for node_id, node_attrs in self.nodes.items():
            for port in node_attrs["ports"]:
                dist_test = length(
                    shift(
                        self.port_position(node_attrs["position"], port),
                        scale(position, -1, -1),
                    )
                )

                if dist_test < dist_closest:
                    dist_closest = dist_test
                    node_closest = node_id
                    port_closest = port.name

        return node_closest, port_closest

    def closestNode(
        self, pos: dict, electrical_only: bool = False, ppMetric=False
    ) -> int:
        """Returns the Id of the Node closest to the given Point by Euclidean Metric"""

        node_type = nx.get_node_attributes(self, "type")
        closest_node, closest_dist = 0, float("inf")

        for node_id, node_pos in nx.get_node_attributes(self, "position").items():

            if not electrical_only or (
                electrical_only
                and node_type[node_id] not in ["junction", "crossover", "text"]
            ):

                if ppMetric:
                    dist = sqrt(
                        (node_pos["x"] - pos["x"]) ** 2
                        + (node_pos["y"] - pos["y"]) ** 2
                    )

                else:
                    dist = -cv2.pointPolygonTest(
                        np.array(
                            [
                                (
                                    node_pos["x"] - node_pos["width"] // 2,
                                    node_pos["y"] - node_pos["height"] // 2,
                                ),
                                (
                                    node_pos["x"] + node_pos["width"] // 2,
                                    node_pos["y"] - node_pos["height"] // 2,
                                ),
                                (
                                    node_pos["x"] + node_pos["width"] // 2,
                                    node_pos["y"] + node_pos["height"] // 2,
                                ),
                                (
                                    node_pos["x"] - node_pos["width"] // 2,
                                    node_pos["y"] + node_pos["height"] // 2,
                                ),
                            ],
                            dtype=np.int32,
                        ),
                        (pos["x"], pos["y"]),
                        True,
                    )

                if dist < closest_dist:
                    closest_dist = dist
                    closest_node = node_id

        return closest_node

    def port_position(self, node_pos: dict, port: Port) -> Point:
        """Determines the Global Position of a Port"""

        return transform(port.position, node_pos)

    def edgeEndPoint(self, node_id, port_id=None):
        """Determines the Geometric Endpoints of the Graph's Edges"""

        if port_id is None:
            return (
                self.nodes[node_id]["position"]["x"],
                self.nodes[node_id]["position"]["y"],
            )

        else:
            port = [
                port for port in self.nodes[node_id]["ports"] if port.name == port_id
            ][0]
            port_pos = self.port_position(self.nodes[node_id]["position"], port)
            return port_pos.x, port_pos.y

    def edgeEndPoints(self, usePorts: bool = True, convertToInt: bool = False):
        """Determines the Geometric Endpoints of the Graph's Edges, optionally considering the Node's Ports"""

        edge_points = []

        for source, target, edge_data in self.edges(data=True):
            source_point = self.edgeEndPoint(
                source, edge_data.get("sourcePort", None) if usePorts else None
            )
            target_point = self.edgeEndPoint(
                target, edge_data.get("targetPort", None) if usePorts else None
            )

            if convertToInt:
                source_point = (int(source_point[0]), int(source_point[1]))
                target_point = (int(target_point[0]), int(target_point[1]))

            edge_points += [(source_point, target_point)]

        return edge_points

    def new_node_id(self) -> int:
        """Returns a Number not yet existing as a Node ID in the Given Graph"""

        new_id = 0

        while new_id in self.nodes:
            new_id += 1

        return new_id

    def move_node(
        self, node: int, delta_x: NUMBER, delta_y: NUMBER, update_mask: bool = True
    ):
        """Moves a Node by a Delta in X and Y Position, Optionally Mask and Properties"""

        self.nodes[node]["position"]["x"] += delta_x
        self.nodes[node]["position"]["y"] += delta_y

        if update_mask:
            for point in self.nodes[node]["shape"]:
                point[0] += delta_x
                point[1] += delta_y

    def duplicate_node(self, node_id: int) -> int:
        """Adds a Copy of given Node to the Graph"""

        node_id_new = self.new_node_id()
        self.add_node(
            node_id_new,
            name=deepcopy(self.nodes[node_id]["name"]),
            type=deepcopy(self.nodes[node_id]["type"]),
            position=deepcopy(self.nodes[node_id]["position"]),
            text=deepcopy(self.nodes[node_id]["text"]),
            ports=deepcopy(self.nodes[node_id]["ports"]),
            properties=deepcopy(self.nodes[node_id]["properties"]),
            shape=deepcopy(self.nodes[node_id]["shape"]),
        )

        return node_id_new

    def resolve_text_nodes(self):
        """Removes all text nodes and converts their Content to Graph/Node/Edge/Port Attributes"""

        for node_id, node_data in self.nodes.items():
            if node_data["type"] == "text" and node_data.get("text", ""):
                assignee = self.closestNode(node_data["position"], electrical_only=True)
                prop = Property(
                    name=self.classify_text(node_data["text"]),
                    value=node_data["text"],
                    position=transform_to_local(
                        Point(node_data["position"]["x"], node_data["position"]["y"]),
                        self.nodes[assignee]["position"],
                    ),
                )
                self.nodes[assignee]["properties"].append(prop)

        for node_id, node_type in nx.get_node_attributes(self, "type").items():
            if node_type == "text":
                self.remove_node(node_id)

    def classify_text(self, text: str) -> str:
        """Classifies a Text by matching against RegEx List"""

        for rule in self.PROP_MAP:
            if re.findall(rule["regex"], text):
                return rule["name"]

        return "Unknown"

    def encapsulate_text_nodes(self):

        print("Implement Me")

    def resolve_wire_hops(self):
        """Remove Crossover Nodes by joining their Opposing Edges"""

        for node_id, node_type in nx.get_node_attributes(self, "type").items():
            if node_type == "crossover":
                if self.degree(node_id) == 4:
                    neighbors = [
                        (
                            source,
                            data.get("sourcePort", None),
                            target,
                            data.get("targetPort", None),
                        )
                        for source, target, data in self.in_edges(node_id, True)
                    ] + [
                        (
                            target,
                            data.get("targetPort", None),
                            source,
                            data.get("sourcePort", None),
                        )
                        for source, target, data in self.out_edges(node_id, True)
                    ]
                    endpoints = [
                        (
                            self.edgeEndPoint(other, other_port),
                            self.edgeEndPoint(own, own_port),
                        )
                        for other, other_port, own, own_port in neighbors
                    ]
                    displacement = [
                        (other[0] - own[0], other[1] - own[1])
                        for other, own in endpoints
                    ]
                    angle = [atan2(vector[1], vector[0]) for vector in displacement]
                    angle_dist = [
                        abs(((a - angle[0] + pi) % (2 * pi)) - pi) for a in angle
                    ]
                    zero_partner_index = angle_dist.index(max(angle_dist))
                    neighbors[1], neighbors[zero_partner_index] = (
                        neighbors[zero_partner_index],
                        neighbors[1],
                    )

                    self.add_edge(
                        neighbors[0][0],
                        neighbors[1][0],
                        sourcePort=neighbors[0][1],
                        targetPort=neighbors[1][1],
                    )
                    self.add_edge(
                        neighbors[2][0],
                        neighbors[3][0],
                        sourcePort=neighbors[2][1],
                        targetPort=neighbors[3][1],
                    )
                    self.remove_node(node_id)

                else:
                    print(
                        f"Can't resolve crossover {node_id} with degree {self.degree(node_id)}"
                    )

    def encapsulate_wire_hops(self):

        print("Implement Me")

    def shake(self, std: int = 10):
        """Adds Random Noise to all Node Positions"""

        for node in self.nodes:
            self.move_node(node, randint(-std, std), randint(-std, std))

    def straighten(self):
        """Attempts making all Edge Lines Vertical and Horizontal"""

        self.add_symbols_ports()

        # Straight Rotations
        for node_attrs in self.nodes.values():
            rotation = node_attrs["position"]["rotation"]
            node_attrs["position"]["rotation"] = 45 * ((17 + rotation) // 45)

        # Straight Edges
        # for node in nx.dfs_preorder_nodes(self):

        print("Implement Me")

    def print_nodes(self):
        """Prints nodes each on separate lines."""
        print("--- GRAPH NODES ---")
        node_data = self.nodes(data=True)
        for id, ann in node_data:
            print(f"{id}: {ann}")
        print("-------------------")

    def create_empty_copy(self) -> "EngGraph":
        """Creates a new graph with the same attributes as the current one, but without nodes and edges."""
        new_graph = EngGraph(
            name=self.graph.get("name", ""),
            width=self.graph.get("width", 0),
            height=self.graph.get("height", 0),
            image=self.graph.get("image", ""),
        )

        return new_graph

    def get_boxes_labels(self, get_id_from_str_fn: Callable):
        """"""
        boxes = []
        labels = []

        # Iterate through nodes and collect bbox and label
        for _, node_data in self.nodes(data=True):
            # Convert position to bbox tuple
            x_min, x_max, y_min, y_max, *_ = pos_to_bbox(node_data["position"])
            # Append in [xmin, ymin, xmax, ymax] order
            boxes.append([x_min, y_min, x_max, y_max])

            # Map node type to class id
            ann_type = node_data.get("type", "")
            class_id = get_id_from_str_fn(ann_type)
            labels.append(class_id)

        # Convert to torch tensors
        boxes_tensor = torch.tensor(boxes, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        return {"boxes": boxes_tensor, "labels": labels_tensor}


def geo_resolve(
    other: Union[int, float, Tuple[NUMBER, NUMBER]],
) -> Tuple[NUMBER, NUMBER]:
    """Resolves Incoming Parameters for Uniform Further Processing"""

    other_x, other_y = 1, 1

    # Multiply by number -> Scaling
    if type(other) is int or type(other) is float:
        other_x = other
        other_y = other

    # Multiply by tuple -> Stretching
    elif (
        type(other) is tuple
        and len(other) == 2
        and (type(other[0]) is int or type(other[0]) is float)
        and (type(other[1]) is int or type(other[1]) is float)
    ):
        other_x = other[0]
        other_y = other[1]

    # Other Types are not Supported
    else:
        print(
            "ERROR: Geometric Manipulation only allowed by numbers and pairs of numbers"
        )

    return other_x, other_y


def geo_mul(geo: dict, scale_x: NUMBER, scale_y: NUMBER) -> None:
    """Applies a Geometric Scaling/Stretching to a Point or Box"""

    geo["x"] *= scale_x
    geo["y"] *= scale_y

    if "width" in geo:
        geo["width"] *= scale_x
        geo["height"] *= scale_y


def geo_add(geo: dict, shift_x: NUMBER, shift_y: NUMBER) -> None:
    """Applies a Geometric Shift to a Point or Box"""

    geo["x"] += shift_x
    geo["y"] += shift_y
