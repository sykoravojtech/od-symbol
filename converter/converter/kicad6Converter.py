"""kicad6Converter.py Import from and Export to KiCad v6"""

__author__ = "Jaikrishna Patil"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "8-kicad-6-converter"

# System Imports
import os
import json

# Third-Party Imports
import networkx as nx
import pyparsing
import random, string

# Project Imports
from converter.core.converter import Converter
from converter.core.engineeringGraph import EngGraph
from converter.core.boundingbox import BoundingBox


class KiCad6Converter(Converter):
    """KiCad 6 Converter Class"""
    FALLBACK_SIZE = 5
    SYMBOLS_DIR = '/usr/share/kicad/symbols'
    with open(os.path.join("converter", "symbols", "kicad_classes.json")) as classes_file:
        CLASSES = json.load(classes_file)
    CLASSES_INV = {value: key for (key, value) in CLASSES.items()}

    def _parse(self, data: str) -> EngGraph:
        """Reads a KiCad eeSchema File String and Return an Engineering Graph Object"""
        width, height = 2000, 2000
        ports = {}
        symbol_instances={}
        content = pyparsing.Word(pyparsing.alphanums + '_' + '-' + '.')
        parens = pyparsing.nestedExpr('(', ')', content=content)
        x = parens.parseString(data)
        x = x.asList()
        graph = EngGraph("", width=width, height=height)
        bb = BoundingBox()

        for item in x[0]:
            if item[0] == 'symbol_instances':
                for i in range(1, len(item)):
                    path = item[i][1].replace('"', '')
                    path = path.replace('/', '')
                    value = item[i][4][1].replace('"','')
                    symbol_instances[path] = {'symbol_id': path, 'value': value}

        for item in x[0]:

            if item[0] == 'version':
                version = item[1]
            elif item[0] == 'generator':
                generator = item[1]
            elif item[0] == 'paper':
                paper = item[1]
            elif item[0] == 'lib_symbols':
                #print(item,'\n\n')
                ports = self.get_ports(item)
            elif item[0] == 'symbol':
                properties = []
                mirror = ''
                for i in range(1, len(item)):

                    if item[i][0] == 'lib_id':
                        lib_id = item[i][1]
                    elif item[i][0] == 'at':
                        x = float(item[i][1])
                        y = float(item[i][2])
                        rotation = int(item[i][3])
                        bb.update(x, y)
                    elif item[i][0] == 'unit':
                        unit = item[i][1]
                    elif item[i][0] == 'mirror':
                        mirror = item[i][1]
                    elif item[i][0] == 'in_bom':
                        in_bom = item[i][1]
                    elif item[i][0] == 'on_board':
                        on_board = item[i][1]
                    elif item[i][0] == 'uuid':
                        symbol_uuid = item[i][1]
                    elif item[i][0] == 'property':
                        hide = ""
                        justify = ""
                        size = []
                        for j in item[i]:
                            for k in j:
                                if k[0] == 'font':
                                    size = [k[1][1], k[1][2]]
                                if k == 'hide':
                                    hide = 'hide'
                                if k[0] == 'justify':
                                    justify = k[1]
                        property_key = item[i][1].replace('"', '')
                        property_value = item[i][2].replace('"', '')
                        property_id = item[i][3][1]
                        property_x = float(item[i][4][1])
                        property_y = float(item[i][4][2])
                        property_rotation = int(item[i][4][3])
                        property_font_size = size
                        property_justify = justify
                        property_hide = hide
                        bb.update(property_x, property_y)
                        try:
                            properties += [
                                {"key": property_key, "value": property_value, "id": property_id, "x": property_x,
                                 "y": property_y,
                                 "rotation": property_rotation, "font_size": property_font_size,
                                 "justify": property_justify, "hide": property_hide}]
                        except ValueError:
                            print("Error in appending properties")
                ports_node = ports[lib_id]
                for k, v in ports_node.items():
                    ports_node[k]['x'] = x + v['x']
                    ports_node[k]['y'] = y + v['y']
                    bb.update(ports_node[k]['x'], ports_node[k]['y'])
                node_width, node_height = bb.size
                lib_id = lib_id.replace('"', '')

                graph.add_node(graph.number_of_nodes(),
                               name=str(graph.number_of_nodes()),
                               type=self.CLASSES_INV.get(lib_id),
                               path=symbol_instances[symbol_uuid],
                               position={"x": x, "y": y,
                                         "width": node_width,
                                         "height": node_height,
                                         "rotation": rotation, "mirror": mirror},
                               ports=ports_node, properties=properties)
            elif item[0] == 'junction':
                junction_x = float(item[1][1])
                junction_y = float(item[1][2])
                junction_diameter = item[2][1]
                junction_color = [item[3][1], item[3][2], item[3][3], item[3][4]]
                node = graph.number_of_nodes()
                graph.add_node(node, name=str(node), type='junction', path={},
                               position={"x": junction_x, "y": junction_y,
                                         "width": self.FALLBACK_SIZE, "height": self.FALLBACK_SIZE, "rotation": 0, "mirror": ''},
                               ports={}, properties=[])
            elif item[0] == 'wire':
                x1 = float(item[1][1][1])
                y1 = float(item[1][1][2])

                x2 = float(item[1][2][1])
                y2 = float(item[1][2][2])

                wire_stroke_width = float(item[2][1][1])
                wire_stroke_type = item[2][2][1]
                wire_stroke_color = [item[2][3][1], item[2][3][2], item[2][3][3], item[2][3][4]]

                source_node = self.add_junction(graph, x1, y1)
                target_node = self.add_junction(graph, x2, y2)
                attrs = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "wire_stroke_width": wire_stroke_width,
                         "wire_stroke_type": wire_stroke_type, "wire_stroke_color": wire_stroke_color}

                graph.add_edge(source_node, target_node, type="electrical", **attrs)
        return graph

    def get_ports(selfself, item):
        ports = {}
        for i in range(1, len(item)):

            symbol_def = item[i]
            symbol_id = symbol_def[1]
            length_def = len(symbol_def)
            port_definitions = symbol_def[length_def - 1]
            pin = {}
            for j in port_definitions:
                if j[0] == 'pin':
                    number = str(j[6][1]).replace('"', '')
                    pin_x = float(j[3][1])
                    pin_y = float(j[3][2])
                    pin_rotation = int(j[3][3])
                    pin_length = float(j[4][1])
                    pin[number] = {'x': pin_x, 'y': pin_y, 'rotation': pin_rotation, 'length': pin_length}
            ports[symbol_id] = pin
        return ports

    def junction_or_node_at_position(self, graph: EngGraph, x: float, y: float):
        """Returns the ID of a junction Node at a Given Location, if such exists in the Graph"""

        node_pos = nx.get_node_attributes(graph, 'position')
        # print(node_pos)

        for node_id, node_type in nx.get_node_attributes(graph, 'type').items():
            if node_pos[node_id]["x"] == x and node_pos[node_id]["y"] == y:
                return node_id
        return None

    def port_at_position(self, graph: EngGraph, x: float, y: float):
        """Returns a Node and Port ID at a Given Location, if such exists in the Graph"""

        for node_id, ports in nx.get_node_attributes(graph, 'ports').items():
            for key, value in ports.items():
                if value['x'] == x and value['y'] == y:
                    return node_id

    def add_junction(self, graph: EngGraph, x: int, y: int):
        """Adds a Junction if no Element is already Present and Returns its IDs"""
        node = self.junction_or_node_at_position(graph, x, y)
        if node is None:
            node = self.port_at_position(graph, x, y)
        if node is None:
            node = graph.number_of_nodes()
            graph.add_node(node, name=str(node), type='extra_junction', path={},
                           position={"x": x, "y": y,
                                     "width": self.FALLBACK_SIZE, "height": self.FALLBACK_SIZE, "rotation": 0, "mirror": ''},
                           ports={}, properties=[])
        return node

    def _write(self, graph: EngGraph, **kwargs) -> str:
        """Constructs a KiCad eeSchema File String from an Engineering Graph"""

        version = '20211123'
        generator = 'eeschema'
        paper = '"A4"'
        uuid_schematic = ''.join(random.choices(string.ascii_letters + '-' + string.digits, k=16))
        schematic_head = f'(kicad_sch (version {version}) (generator {generator})\n\n  (uuid {uuid_schematic})\n\n  (paper {paper})\n\n'
        data = '' + schematic_head

        node_class = nx.get_node_attributes(graph, 'type')
        node_pos = nx.get_node_attributes(graph, 'position')
        node_prop = nx.get_node_attributes(graph, 'properties')
        node_pin = nx.get_node_attributes(graph, 'ports')
        node_type = nx.get_node_attributes(graph, 'type')
        node_path = nx.get_node_attributes(graph, 'path')
        # Write symbol defs
        schematic_lib_symbols = '  (lib_symbols\n'
        junction_data='\n'
        used_labels = []
        for node in graph.nodes:

            if node_class[node] != "junction" and node_class[node] != "extra_junction":
                label = self.CLASSES.get(node_class[node], "unknown")
                if label not in used_labels:
                    lib_symbol_def = self.get_graph_symbol_data(label)
                    schematic_lib_symbols += '  '
                    schematic_lib_symbols += lib_symbol_def
                    used_labels.append(label)
        schematic_lib_symbols += '  )\n'
        data += schematic_lib_symbols

        # Write Edges/Wires
        schematic_wires = "\n"
        for _, _, attrs in graph.edges.data():
            x1 = attrs['x1']
            y1 = attrs['y1']
            x2 = attrs['x2']
            y2 = attrs['y2']
            wire_stroke_width = attrs['wire_stroke_width']
            wire_stroke_color = attrs['wire_stroke_color']
            uuid_wire = ''.join(random.choices(string.ascii_letters + '-' + string.digits, k=16))
            schematic_wires += f'  (wire (pts (xy {x1} {y1}) (xy {x2} {y2}))\n'
            schematic_wires += '    (stroke (width 0) (type default) (color 0 0 0 0))\n'
            schematic_wires += f'    (uuid {uuid_wire})\n  )\n'
        # Write postions and properties of symbols
        schematic_symbols = '\n'
        symbol_instances = {}
        for node in graph.nodes:

            if node_class[node] == "junction":
                x = node_pos[node]['x']
                y = node_pos[node]['y']
                #diameter = node_pos[node]['diameter']
                #color = node_pos[node]['color']
                junction_uuid = ''.join(random.choices(string.ascii_letters + '-' + string.digits, k=16))
                junction_data += f'  (junction (at {x} {y}) (diameter 0) (color 0 0 0 0)\n'
                junction_data += f'    (uuid {junction_uuid})\n  )\n'
            elif node_class[node] != "extra_junction":
                label = self.CLASSES.get(node_class[node], "unknown")
                x = node_pos[node]['x']
                y = node_pos[node]['y']
                rotation = node_pos[node]['rotation']
                # in_bom= node_pos[node]['in_bom']
                # on_board= node_pos[node]['on_board']
                symbol_uuid = node_path[node]['symbol_id']
                schematic_symbols += '  (symbol (lib_id '
                schematic_symbols += f'"{label}") (at {x} {y} {rotation}) (unit 1)\n'
                schematic_symbols += f'    (in_bom yes) (on_board yes)\n'
                schematic_symbols += f'    (uuid {symbol_uuid})\n'
                for property in node_prop[node]:
                    schematic_symbols += f'    (property "{property["key"]}" "{property["value"]}" (id {property["id"]}) (at {property["x"]} {property["y"]} {property["rotation"]})\n'

                    if property["justify"] != "" and property["hide"] != "":
                        schematic_symbols += f'      (effects (font (size 1.27 1.27)) (justify {property["justify"]}) hide)\n'
                    elif property["justify"] == "" and property["hide"] != "":
                        schematic_symbols += f'      (effects (font (size 1.27 1.27)) hide)\n'
                    elif property["justify"] != "" and property["hide"] == "":
                        schematic_symbols += f'      (effects (font (size 1.27 1.27)) (justify {property["justify"]}))\n'
                    schematic_symbols += '    )\n'

                    if property['key'] == 'Reference':
                        reference = property['value']
                    if property['key'] == 'Value':
                        value = property['value']
                    if property['key'] == 'Footprint':
                        footprint = property['value']

                if node_type[node] != 'junction' and node_type[node] != 'extra_junction':
                    symbol_instances[node] = {'symbol_uuid':symbol_uuid, 'reference': reference, 'unit': 1, 'value': node_path[node]['value'], 'footprint': footprint}

                for pin_number, value in node_pin[node].items():

                    pin_uuid = ''.join(random.choices(string.ascii_letters + '-' + string.digits, k=16))
                    schematic_symbols += f'    (pin "{pin_number}" (uuid {pin_uuid}))\n'
                schematic_symbols += '  )\n\n'

        data += junction_data
        data += schematic_wires
        data += schematic_symbols

        data = data + '  (sheet_instances\n    (path "/" (page "1"))\n  )\n\n'
        data += f'  (symbol_instances\n'
        for key, value in symbol_instances.items():
            data += f'    (path "/{value["symbol_uuid"]}"\n'
            data += f'      (reference "{value["reference"]}") (unit 1) (value "{value["value"]}") (footprint "{value["footprint"]}")\n'
            data += '    )\n'
        data += '  )\n)'
        return data

    def get_graph_symbol_data(self, label):

        library, component = label.split(":")
        found = False
        symbol_definition = ''
        with open(os.path.join(self.SYMBOLS_DIR, f"{library}.kicad_sym")) as file_lib:
            for lib_line in file_lib:
                if not found and lib_line.startswith(f'  (symbol "{component}"'):
                    lib_line = lib_line.replace(f'"{component}"', f'"{library}:{component}"')
                    symbol_definition += lib_line
                    found = True
                    continue
                if found:
                    if lib_line.startswith(f'  (symbol "') and not lib_line.startswith(f'  (symbol "{component}"'):
                        break
                    else:
                        symbol_definition += f'  {lib_line}'
            return symbol_definition