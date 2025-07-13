"""renderer.py Engineering Graphs to Geometric Primitives"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Development"

# System Imports
from math import sin, cos, pi
from typing import Callable, Any

# Third-Party Imports
import cv2
import networkx as nx

# Project Imports
from converter.core.engineeringGraph import EngGraph
from converter.core.geometry import Point, Line, Rectangle, Circle, Polygon, Text, transform
from converter.core.symbol import load_symbols

NODE_COLOR = (180, 50, 50)
NODE_INNER_COLOR = (255, 255, 255)
NODE_RADIUS = 30
NODE_BORDER_THICKNESS = 4
NODE_JUNCTION_RADIUS = 10

PORT_RADIUS = 5
PORT_COLOR = (180, 50, 180)

EDGE_WIDTH = 3
EDGE_COLOR = (50, 50, 180)


FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
FONT_THICKNESS = 2
FONT_COLOR = (180, 50, 50)
FONT_COLOR_PROP = (10, 250, 10)
PROP_MARGIN = 10
PROP_BORDER_COLOR = (0, 0, 0)
PROP_INNER_COLOR = (140, 200, 125)
PROP_FONT_COLOR = (0, 0, 0)
PROP_BORDER_THICKNESS = 3





class Renderer:
    """Graph Renderer Class"""

    STOREMODE = "wb"

    def __init__(self):

        self.symbols = load_symbols()


    def render(self, graph: EngGraph, mode="simple", **kwargs) -> list:
        """Constructs a List of Geometric Primitives from an Engineering Graph"""

        geo = []

        if mode == "simple":
            self._draw_edges(graph, geo, **kwargs)
            self._draw_nodes(graph, geo, self._node_bubble, kwargs)

        else:
            self._draw_edges(graph, geo,  **kwargs)
            self._draw_nodes(graph, geo, self._node_rectangle, kwargs)
            self._draw_nodes(graph, geo, self._node_ports, kwargs)
            self._draw_nodes(graph, geo, self._node_text, kwargs)
            self._draw_nodes(graph, geo, self._node_props, kwargs)
            self._draw_nodes(graph, geo, self._node_symbol, kwargs)

        return geo


    @staticmethod
    def _draw_nodes(graph: EngGraph, geo: list, method: Callable[[list, dict, dict], None], kwargs: Any) -> None:
        """Wrapper for Node Rendering"""

        for node_id, node_data in graph.nodes.items():
            method(geo, {'id': node_id, 'type': node_data['type'],
                         'x': node_data['position']["x"], 'y': node_data['position']["y"],
                         'width': node_data['position']["width"], 'height': node_data['position']["height"],
                         'rotation': node_data['position'].get("rotation", 0),
                         'mirror_horizontal': node_data['position'].get('mirror_horizontal', False),
                         'mirror_vertical': node_data['position'].get('mirror_vertical', False),
                         'shape': node_data.get('shape', []),
                         'ports': node_data.get('ports', []),
                         'props': node_data.get('properties', []),
                         'text': node_data.get('text', None)}, **kwargs)


    @staticmethod
    def _node_bubble(geo: list, node: dict, bubble_scale: float = 1.0, **kwargs) -> None:
        """Renders Node as equally sized circle with its ID in it"""

        geo.append(Circle(node['x'], node['y'], bubble_scale*NODE_RADIUS,
                          NODE_COLOR, NODE_BORDER_THICKNESS, NODE_INNER_COLOR, node['id']))
        geo.append(Text(str(str(node['id'])), node['x'], node['y'], 0, FONT_COLOR, "center", node['id']))
        # FONT_FACE, bubble_scale*FONT_SCALE, FONT_THICKNESS, cv2.LINE_AA)


    @staticmethod
    def _node_text(geo: list, node: dict, **kwargs) -> None:
        """Renders all Node Texts"""

        if kwargs.get("text", True) and node['text']:
            geo.append(Text(node['text'], node['x'], node['y'], node['rotation'],
                            kwargs.get("textColor", PROP_FONT_COLOR), "center", node['id']))


    @staticmethod
    def _node_props(geo: list, node: dict, bubble_scale: float = 0.6,
                    propOwner: bool = False, **kwargs) -> None:

        if node['props']:
            for prop in node['props']:
                prop_text = prop.value if prop.value else prop.name
                (prop_width, prop_height), _ = cv2.getTextSize(prop_text, FONT_FACE,
                                                               bubble_scale*FONT_SCALE, FONT_THICKNESS)
                prop_pos = transform(prop.position, node)

                left = prop_pos.x - prop_width // 2
                right = prop_pos.x + prop_width // 2
                top = prop_pos.y - prop_height // 2
                bottom = prop_pos.y + prop_height // 2
                poly = [[left - PROP_MARGIN, top - PROP_MARGIN],
                        [right + PROP_MARGIN, top - PROP_MARGIN],
                        [right + 3 * PROP_MARGIN, (top + bottom) // 2],
                        [right + PROP_MARGIN, bottom + PROP_MARGIN],
                        [left - PROP_MARGIN, bottom + PROP_MARGIN],
                        [left - 3 * PROP_MARGIN, (top + bottom) // 2]]

                if propOwner:
                    geo.append(Line(Point(node['x'], node['y']), prop_pos,
                                    PROP_BORDER_COLOR, NODE_BORDER_THICKNESS, node['id']))

                geo.append(Polygon(poly, PROP_BORDER_COLOR, PROP_BORDER_THICKNESS, PROP_INNER_COLOR, node['id']))
                geo.append(Text(prop_text, prop_pos.x, prop_pos.y, 0, PROP_FONT_COLOR, "center", node['id']))


    @staticmethod
    def _node_rectangle(geo: list, node: dict, drawNodeId=True, drawNodeType=True,
                        junctionCircles=True, shapes=False, rectangles=True, drawRotation=False, **kwargs) -> None:
        """Renders the Node's Bounding Box Rectangle"""

        if shapes and node["shape"]:
            geo.append(Polygon(node["shape"], NODE_COLOR, NODE_BORDER_THICKNESS, None, node['id']))

        if drawRotation and node["type"] != "junction":
            geo.append(Circle(node['x'], node['y'], NODE_RADIUS, NODE_COLOR, NODE_BORDER_THICKNESS, None, node['id']))
            geo.append(Line(Point(node['x'], node['y']),
                            Point(node['x']+cos(2*pi*node['rotation']/360)*NODE_RADIUS/2,
                                  node['y']-sin(2*pi*node['rotation']/360)*NODE_RADIUS/2),
                            NODE_COLOR, NODE_BORDER_THICKNESS, node['id']))

        if rectangles:
            if junctionCircles and node['type'] == "junction":
                geo.append(Circle(node['x'], node['y'], NODE_JUNCTION_RADIUS, None, 0, NODE_COLOR, node['id']))

            else:
                geo.append(Rectangle(node['x'] - node['width'] / 2, node['y'] - node['height'] / 2,
                                     node['x'] + node['width'] / 2, node['y'] + node['height'] / 2,
                                     kwargs.get("rectangleColor", NODE_COLOR), NODE_BORDER_THICKNESS, None, node['id']))

        if drawNodeId or drawNodeType:
            text = str(node['id']) if drawNodeId else ''

            if drawNodeType and (not junctionCircles or node['type'] != "junction"):
                text += (': ' if drawNodeId else '') + node['type']

            geo.append(Text(text, node['x'] - node['width']/2, node['y'] - node['height']/2, 0,
                            FONT_COLOR_PROP, "sw", node['id']))  # TODO FONT_FACE, FONT_SCALE, FONT_THICKNESS


    @staticmethod
    def _node_ports(geo: list, node: dict, portPosition=True, portOwner=False, portType=False, **kwargs) -> None:
        """Renders all Nodes as Rectangles"""

        for port in node['ports']:
            port_pos = transform(port.position, node)

            if portPosition:
                geo.append(Circle(port_pos.x, port_pos.y, PORT_RADIUS,
                                  None, 0, PORT_COLOR, node['id']))

            if portOwner:
                geo.append(Line(port_pos,
                                Point(node['x'], node['y']),
                                PORT_COLOR, EDGE_WIDTH, node['id']))

            if portType:
                geo.append(Text(port.name, port_pos.x, port_pos.y, 0,
                                FONT_COLOR_PROP, "nw", node['id']))  # TODO FONT_FACE, 0.6 * FONT_SCALE, FONT_THICKNESS


    def _node_symbol(self, geo: list, node: dict, **kwargs) -> None:
        """Renders Node Symbols"""

        if kwargs.get("symbol", False) and node['type'] in self.symbols:
            for item in self.symbols[node['type']].geometry:
                if type(item) is Line:
                    geo.append(Line(transform(item.a, node), transform(item.b, node),
                                    kwargs.get("symbolColor", NODE_COLOR), EDGE_WIDTH, node['id']))

                if type(item) is Circle:
                    trans_c = transform(Point(item.x, item.y), node)  # TODO Circle Center should be Point
                    geo.append(Circle(trans_c.x, trans_c.y, item.radius*max(node['width'], node['height']),
                                      kwargs.get("symbolColor", NODE_COLOR), EDGE_WIDTH,
                                      kwargs.get("symbolColor", NODE_COLOR) if item.fillColor else None,
                                      node['id']))

                if type(item) is Polygon:
                    trans_p = [transform(Point(point[0], point[1]), node) for point in item.points]
                    geo.append(Polygon([[point[0], point[1]] for point in trans_p],
                                      kwargs.get("symbolColor", NODE_COLOR), EDGE_WIDTH,
                                      kwargs.get("symbolColor", NODE_COLOR) if item.fillColor else None,
                                      node['id']))


    def _draw_edges(self, graph: EngGraph, geo: list, drawEdgePorts=True, shapes=False, **kwargs):
        """Draws Lines to Visualize Edges"""

        if shapes:
            for edge_shape in nx.get_edge_attributes(graph, 'shape').values():
                geo.append(Polygon(edge_shape, EDGE_COLOR, EDGE_WIDTH, None, None))

        else:
            for source_point, target_point in graph.edgeEndPoints(drawEdgePorts, convertToInt=True):
                geo.append(Line(Point(source_point[0], source_point[1]),
                                Point(target_point[0], target_point[1]),
                                EDGE_COLOR, EDGE_WIDTH, None))
