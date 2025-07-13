"""kicadConverter.py Import from and Export to KiCad"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Development"

# System Imports
import os
import json
from typing import Tuple

# Third-Party Imports
import networkx as nx

# Project Imports
from converter.core.converter import Converter
from converter.core.engineeringGraph import EngGraph, geo_add
from converter.core.boundingbox import BoundingBox


class KiCadConverter(Converter):
    """KiCad Converter Class"""

    HEAD = 'EESchema Schematic File Version 4\nEELAYER 30 0\nEELAYER END\n$Descr A4 11693 8268\nencoding ' \
           'utf-8\nSheet 1 1\nTitle ""\nDate ""\nRev ""\nComp ""\nComment1 ""\nComment2 ""\nComment3 ""\nComment4 ' \
           '""\n$EndDescr\n'
    TAIL = '$EndSCHEMATC'
    COMP_HEAD = '$Comp\n'
    COMP_TAIL = '$EndComp\n'
    FALLBACK_SIZE = 40
    DIR_LIB = '/usr/share/kicad/library'
    TRANSFORM_MAP = {(1,  0,  0, -1): (0, False, False),
                     (0, -1, -1,  0): (90, False, False),
                     (-1, 0,  0,  1): (180, False, False),
                     (0,  1,  1,  0): (270, False, False),
                     (1,  0,  0,  1): (0, False, True),
                     (0, -1,  1,  0): (90, False, True),
                     (0,  1, -1,  0): (270, False, True),
                     (-1, 0,  0, -1): (0, True, False),
                     (0,  1, -1,  0): (90, True, False),
                     (1,  0,  0,  1): (180, True, False),
                     (0, -1,  1,  0): (270, True, False)}
    TRANSFORM_MAP_INV = {value: key for (key, value) in TRANSFORM_MAP.items()}

    with open(os.path.join("converter", "symbols", "kicad_classes.json")) as classes_file:
        CLASSES = json.load(classes_file)

    CLASSES_INV = {value: key for (key, value) in CLASSES.items()}

    
    def _parse(self, data: str) -> EngGraph:
        """Reads a KiCad eeSchema File String and Return an Engineering Graph Object"""

        lines = data.splitlines()
        width, height = 2000, 2000

        for line in lines:
            if line.startswith("$Descr"):
                _, _, width, height = line.split(" ")
                width = int(width)
                height = int(height)
                break

        graph = EngGraph(name="", width=width, height=height)

        pointer1 = 0
        for element_NoConn in range(data.count('NoConn')):
            pointer1 = data.find('NoConn', pointer1)
            pointer1 = data.find('~ ', pointer1) + 2
            pointer2 = data.find(' ', pointer1)
            x = int(data[pointer1:pointer2])
            pointer1 = pointer2 + 1
            pointer2 = data.find('\n', pointer1)
            y = int(data[pointer1:pointer2])
            self.add_junction(graph, x, y)

        # Symbol Nodes
        x, y = 0, 0
        cls = ""
        properties = []

        for line in lines:
            if line.startswith("L "):
                cls = line.split(" ")[1]

            if line.startswith("P "):
                _, x, y = line.split(" ")
                x, y = int(x), int(y)

            if line.startswith("F "):
                prop_text, prop_rot, prop_x, prop_y,\
                    prop_size, prop_flags, prop_style, prop_just = line.split(" ")[2:10]

                try:
                    prop_text = prop_text[1:-1]
                    prop_rot = 0 if prop_rot == "H" else 90
                    prop_x = int(prop_x)
                    prop_y = int(prop_y)
                    prop_width = int(prop_size)*len(prop_text)
                    prop_height = int(prop_size)
                    prop_vis = prop_style[3] == "0"

                    if prop_rot:
                        prop_width, prop_height = prop_height, prop_width

                    if prop_just == "L":
                        prop_x += prop_width//2

                    if prop_just == "R":
                        prop_x -= prop_width//2

                    properties += [{"name": prop_text, "visibility": prop_vis,
                                    "position": {"x": prop_x, "y": prop_y,
                                                 "width": prop_width, "height": prop_height,
                                                 "rotation": prop_rot}}]

                except ValueError:
                    pass  # TODO >>> F 6 "sin(0 1 1k)" H 2530 2959 50  0001 L CNN "Spice_Model"

            if line.startswith("$EndComp"):
                rotation, mirror_x, mirror_y = self.read_rotation(last_line)
                ports, bb, props = self.get_class_info(cls)

                for port in ports:
                    self.apply_rotation(port["position"], rotation, mirror_x, mirror_y)
                    geo_add(port["position"], x, y)

                for prop in props:
                    self.apply_rotation(prop["position"], rotation, mirror_x, mirror_y)
                    prop["visibility"] = True

                # TODO merge props and properties

                size = {"x": bb.size[0], "y": bb.size[1]}
                self.apply_rotation(size, rotation, mirror_x, mirror_y)
                graph.add_node(graph.number_of_nodes(),
                               name=str(graph.number_of_nodes()),
                               type="integrated_circuit" if cls=="MCU_Microchip_ATmega:ATmega8-16A" else self.CLASSES_INV.get(cls, "unknown"),
                               position={"x": x, "y": y,
                                         "width": size["x"]*(-1 if mirror_x else 1),
                                         "height": size["y"]*(-1 if mirror_y else 1),
                                         "rotation": rotation},
                               text=None, shape=[], ports=ports, properties=properties)
                properties = []

            last_line = line

        # Edges
        for index, wire_line in enumerate(lines):
            if wire_line.startswith("Wire"):
                x1, y1, x2, y2 = [c for c in lines[index+1].split(" ") if c]
                x1, y1, x2, y2 = int(x1[1:]), int(y1), int(x2), int(y2)
                attrs = {}

                node_source, port_source = self.add_junction(graph, x1, y1)
                node_target, port_target = self.add_junction(graph, x2, y2)

                if port_source is not None:
                    attrs['sourcePort'] = port_source["name"]

                if port_target is not None:
                    attrs['targetPort'] = port_target["name"]

                graph.add_edge(node_source, node_target, type="electrical", **attrs)

        return graph


    def _write(self, graph: EngGraph, **kwargs) -> str:
        """Constructs a KiCad eeSchema File String from an Engineering Graph"""

        data = "" + self.HEAD
        node_class = nx.get_node_attributes(graph, 'type')
        node_pos = nx.get_node_attributes(graph, 'position')

        # Write Nodes
        for node in graph.nodes:
            if node_class[node] == "junction":
                data += "Connection" if graph.degree[node] else "NoConn"
                data += f' ~ {node_pos[node]["x"]} {node_pos[node]["y"]}\n'

            else:
                data += self.COMP_HEAD
                data += f'L {self.CLASSES.get(node_class[node], "Device:C")} C{node}\n'
                data += 'U 1 1 61A75608\n'
                data += f'P {node_pos[node]["x"]} {node_pos[node]["y"]}\n'
                trans = self.TRANSFORM_MAP_INV.get((node_pos[node]["rotation"],
                                                    node_pos[node]["width"] < 0,
                                                    node_pos[node]["height"] < 0),
                                                   (1, 0, 0, -1))
                data += f'\t1    {node_pos[node]["x"]} {node_pos[node]["y"]}\n'
                data += f'\t{"   ".join([str(i) for i in trans])}  \n'
                data += self.COMP_TAIL

        # Write Edges
        for source_point, target_point in graph.edgeEndPoints():
            data += "Wire Wire Line\n"
            data += f'\t{source_point[0]} {source_point[1]} ' \
                    f'{target_point[0]} {target_point[1]}\n'

        return data + self.TAIL


    def junction_at_position(self, graph: EngGraph, x: int, y: int):
        """Returns the ID of a junction Node at a Given Location, if such exists in the Graph"""

        node_pos = nx.get_node_attributes(graph, 'position')

        for node_id, node_type in nx.get_node_attributes(graph, 'type').items():
            if node_type == "junction":
                if node_pos[node_id]["x"] == x and node_pos[node_id]["y"] == y:
                    return node_id, None

        return None, None



    def port_at_position(self, graph: EngGraph, x: int, y: int):
        """Returns a Node and Port at a Given Location, if such exists in the Graph"""

        for node, ports in nx.get_node_attributes(graph, 'ports').items():
            for port in ports:
                if port["position"]["x"] == x and port["position"]["y"] == y:
                    return node, port

        return None, None


    def add_junction(self, graph: EngGraph, x: int, y: int):
        """Adds a Junction if no Element is already Present and Returns its IDs"""

        node, port = self.port_at_position(graph, x, y)

        if node is None:
            node, port = self.junction_at_position(graph, x, y)

        if node is None:
            node = graph.number_of_nodes()
            graph.add_node(node, name=str(node), type='junction',
                           position={"x": x, "y": y,
                                     "width": self.FALLBACK_SIZE, "height": self.FALLBACK_SIZE, "rotation": 0},
                           text=None, shape=[], ports={}, properties=[])

        return node, port


    def get_class_info(self, label):
        """Resolves a Component's Classes Ports, Texts and Bounding Box By A KiCad Library Lookup"""

        library, component = label.split(":")
        found = False
        component_head = f"# {component}"
        ports = []
        props = []
        bb = BoundingBox()

        with open(os.path.join(self.DIR_LIB, f"{library}.lib")) as file_lib:
            for lib_line in file_lib:
                if not found and lib_line.startswith(component_head):
                    found = True
                if not found and lib_line.startswith("ALIAS") and (component in lib_line):
                    found = True
                if found:
                    if lib_line.startswith("F"):
                        lib_line_parts = lib_line.split(" ")
                        if not "sin" in lib_line_parts[1]:
                            props += [{"name": lib_line_parts[0],
                                       "position": {"x": int(lib_line_parts[2]), "y": int(lib_line_parts[3])}}]

                    if lib_line.startswith("X"):
                        lib_line_parts = lib_line.split(" ")
                        port_name = lib_line_parts[2]
                        port_x = int(lib_line_parts[3])
                        port_y = -int(lib_line_parts[4])
                        ports += [{'name': port_name, 'position': {'x': port_x, 'y': port_y}}]
                        bb.update(port_x, port_y)
                    if lib_line.startswith("P "):
                        raw = [int(value) for value in lib_line.split(" ")[1:-1]]
                        for x, y in zip(raw[::2], raw[1::2]):
                            bb.update(x, y)
                    if lib_line.startswith("S "):
                        lib_line_parts = lib_line.split(" ")
                        bb.update(int(lib_line_parts[1]), int(lib_line_parts[2]))
                        bb.update(int(lib_line_parts[3]), int(lib_line_parts[4]))
                    if lib_line.startswith("C "):
                        center_x, center_y, radius = [int(part) for part in lib_line.split(" ")[1:4]]
                        bb.update(center_x - radius, center_y - radius)
                        bb.update(center_x + radius, center_y + radius)
                    if lib_line.startswith("A "):
                        center_x, center_y, radius, start, stop = [int(part) for part in lib_line.split(" ")[1:6]]
                        if start < 899 and stop > 901:
                            bb.update(center_x, center_y - radius)
                    if lib_line.startswith("ENDDEF"):
                        break

        return ports, bb, props


    def read_rotation(self, line: str):
        """Interprets Rotation and Mirroring at X and Y Axis"""

        transform = tuple(int(c) for c in line[1:].split(" ") if c)

        return self.TRANSFORM_MAP[transform]


    def apply_rotation(self, pos: dict, rotation: int, mirror_x: bool, mirror_y: bool) -> None:
        """Applies a Transformation to a Point"""

        if rotation == 90:
            pos["x"], pos["y"] = pos["y"], -pos["x"]

        if rotation == 270:
            pos["x"], pos["y"] = -pos["y"], pos["x"]

        if rotation == 180:
            pos["x"], pos["y"] = -pos["x"], -pos["y"]

        if mirror_x:
            pos["x"] = -pos["x"]

        if mirror_y:
            pos["y"] = -pos["y"]
