"""rdfConverter.py Import from and Export to RDF Turtle"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Development"

# System Imports
import os
import json

# Third-Party Imports
import networkx as nx
import rdflib
from rdflib import Literal, URIRef
from rdflib.namespace import RDF

# Project Imports
from converter.core.converter import Converter
from converter.core.engineeringGraph import EngGraph
from converter.core.symbol import Port, Property
from converter.core.geometry import Point, transform_to_local, transform


class RdfConverter(Converter):
    """RDF/Turtle Converter Class"""

    with open(os.path.join("converter", "symbols", "rdf_classes.json")) as classes_file:
        CLASSES = json.load(classes_file)
        CLASSES_INV = {v: k for k, v in CLASSES.items()}

    with open(os.path.join("converter", "symbols", "rdf_function_classes.json")) as function_classes_file:
        FUNCTION_CLASSES = json.load(function_classes_file)
        FUNCTION_CLASSES_INV = {v: k for k, v in FUNCTION_CLASSES.items()}

    EDGE_CLASSES = {"electrical": "https://www.wikidata.org/wiki/Q5357710"}
    CIRCUIT = URIRef("https://www.wikidata.org/wiki/Q1815901")  # Electric Net: "https://www.wikidata.org/wiki/Q132629"
    TERMINAL = URIRef("https://www.wikidata.org/wiki/Q947546")  #    https://www.wikidata.org/wiki/Q2443617")  # <-Port  Terminal (Conflicing Node Class) -> "https://www.wikidata.org/wiki/Q182610"
    COMPONENT = URIRef("https://www.wikidata.org/wiki/Q11653")  # Fallback for unknown Components

    # Property Ressources
    HAS_PART = URIRef("https://www.wikidata.org/wiki/Property:P527")
    CONNECTS_WITH = URIRef("https://www.wikidata.org/wiki/Property:P2789")
    POSITION_IN_IMAGE = URIRef("https://www.wikidata.org/wiki/Property:P2677")
    ANGLE = URIRef("https://www.wikidata.org/wiki/Property:P4183")
    WIDTH = URIRef("https://www.wikidata.org/wiki/Property:P2049")
    HEIGHT = URIRef("https://www.wikidata.org/wiki/Property:P2048")
    FUNCTION = URIRef("https://www.wikidata.org/wiki/Property:P366")
    NAME = URIRef("https://www.wikidata.org/wiki/Property:P2561")


    def _parse(self, data: str):
        """Reads a File String and Return a NetworkX Graph Object"""

        rdf_doc = rdflib.Graph().parse(data=data, format='ttl')
        rdf_graph = next(rdf_doc.subjects(RDF.type, self.CIRCUIT))

        # Graph Attributes
        graph = EngGraph(name=str(rdf_graph),
                         width=int(next(rdf_doc.objects(rdf_graph, self.WIDTH))),
                         height=int(next(rdf_doc.objects(rdf_graph, self.HEIGHT))))

        # Nodes
        for node_nbr, rdf_node in enumerate(rdf_doc.objects(rdf_graph, self.HAS_PART)):
            node_x, node_y, node_width, node_height = self.decode_position(graph, next(rdf_doc.objects(rdf_node,
                                                                                                       self.POSITION_IN_IMAGE)))
            node_rotation = int(next(rdf_doc.objects(rdf_node, self.ANGLE)))
            node_pos = {"x": node_x, "y": node_y,
                        "width": node_width, "height": node_height, "rotation": node_rotation}
            node_type = self.CLASSES_INV.get(str(next(rdf_doc.objects(rdf_node, RDF.type))), "unknown")
            node_properties = [Property(self.FUNCTION_CLASSES_INV[str(func)], "", Point(0, 1.1))
                               for func in rdf_doc.objects(rdf_node, self.FUNCTION)]

            # Ports
            ports = []

            for rdf_port in rdf_doc.objects(rdf_node, self.HAS_PART):
                port_name = str(next(rdf_doc.objects(rdf_port, self.NAME)))
                port_x, port_y, _, _ = self.decode_position(graph, next(rdf_doc.objects(rdf_port,
                                                                                        self.POSITION_IN_IMAGE)))
                ports += [Port(name=port_name, position=transform_to_local(Point(port_x, port_y), node_pos))]

            graph.add_node(node_nbr, name=str(rdf_node), type=node_type,
                           position=node_pos, ports=ports, properties=node_properties)

        # Edges
        node_name_inv = {v: k for k, v in nx.get_node_attributes(graph, 'name').items()}

        for rdf_source, rdf_target in rdf_doc.subject_objects(self.CONNECTS_WITH):
            attrs = {}

            if next(rdf_doc.objects(rdf_source, RDF.type)) == self.TERMINAL:
                attrs['sourcePort'] = str(next(rdf_doc.objects(rdf_source, self.NAME)))
                source = node_name_inv[str(next(rdf_doc.subjects(self.HAS_PART, rdf_source)))]
            else:
                source = node_name_inv[str(rdf_source)]

            if next(rdf_doc.objects(rdf_target, RDF.type)) == self.TERMINAL:
                attrs['targetPort'] = str(next(rdf_doc.objects(rdf_target, self.NAME)))
                target = node_name_inv[str(next(rdf_doc.subjects(self.HAS_PART, rdf_target)))]
            else:
                target = node_name_inv[str(rdf_target)]

            graph.add_edge(source, target, **attrs)

        return graph


    def _write(self, graph: EngGraph, **kwargs) -> str:
        """Constructs a File String from a NetworkX Graph"""

        rdf_doc = rdflib.Graph()
        rdf_doc.bind("wikidata", "https://www.wikidata.org/wiki/")
        rdf_doc.bind("wikidata-prop", "https://www.wikidata.org/wiki/Property:")
        rdf_doc.bind("circuit", f"file:/{graph.graph['name']}/")

        # Graph Attributes
        rdf_graph = URIRef(f"file:/{graph.graph['name']}")
        rdf_doc.add((rdf_graph,
                     RDF.type,
                     self.CIRCUIT))
        rdf_doc.add((rdf_graph,
                     self.WIDTH,
                     Literal(graph.graph['width'])))
        rdf_doc.add((rdf_graph,
                     self.HEIGHT,
                     Literal(graph.graph['height'])))

        # Node Attributes
        node_name = nx.get_node_attributes(graph, 'name')
        node_class = nx.get_node_attributes(graph, 'type')
        node_pos = nx.get_node_attributes(graph, 'position')
        node_ports = nx.get_node_attributes(graph, 'ports')

        for node in graph.nodes:
            rdf_node = URIRef(f"file:/{graph.graph['name']}#{node_name[node]}")
            rdf_node_class = URIRef(self.CLASSES[node_class[node]]) \
                if node_class[node] in self.CLASSES else self.COMPONENT

            rdf_doc.add((rdf_graph,
                         self.HAS_PART,
                         rdf_node))
            rdf_doc.add((rdf_node,
                         RDF.type,
                         rdf_node_class))
            rdf_doc.add((rdf_node,
                         self.POSITION_IN_IMAGE,
                         self.encode_position(graph, node_pos[node]["x"], node_pos[node]["y"],
                                              node_pos[node]["width"], node_pos[node]["height"])))
            rdf_doc.add((rdf_node,
                         self.ANGLE,
                         Literal(str(node_pos[node]["rotation"]))))

            # Ports
            for port in node_ports[node]:
                port_pos_global = transform(port.position, node_pos[node])
                rdf_port = URIRef(f"file:/{graph.graph['name']}#{node_name[node]}#{port.name}")
                rdf_doc.add((rdf_node,
                             self.HAS_PART,
                             rdf_port))
                rdf_doc.add((rdf_port,
                             RDF.type,
                             self.TERMINAL))
                rdf_doc.add((rdf_port,
                             self.NAME,
                             Literal(port.name)))
                rdf_doc.add((rdf_port,
                             self.POSITION_IN_IMAGE,
                             self.encode_position(graph, port_pos_global.x, port_pos_global.y, 1, 1)))

        # Edge Attributes
        source_port = nx.get_edge_attributes(graph, 'sourcePort')
        target_port = nx.get_edge_attributes(graph, 'targetPort')

        for edge in graph.edges:
            source, target = edge

            rdf_source = URIRef(f"file:/{graph.graph['name']}#{node_name[source]}")
            rdf_target = URIRef(f"file:/{graph.graph['name']}#{node_name[target]}")

            if edge in source_port:
                rdf_source = URIRef(f"file:/{graph.graph['name']}#{node_name[source]}#{source_port[edge]}")

            if edge in target_port:
                rdf_target = URIRef(f"file:/{graph.graph['name']}#{node_name[target]}#{target_port[edge]}")

            rdf_doc.add((rdf_source,
                         self.CONNECTS_WITH,
                         rdf_target))

        #edge_class = nx.get_edge_attributes(graph, 'type')
        #    URIRef(self.EDGE_CLASSES.get(edge_class[(source, target)]))

        return rdf_doc.serialize(format='ttl').decode("utf-8")


    @staticmethod
    def encode_position(graph: EngGraph, x: int, y: int, width: int, height: int) -> Literal:
        """Creates a PCT Literal RDF Node representing an Internal Node's or Port's Position"""

        return Literal(f"pct:{100*x/graph.graph['width']:.4f},"
                       f"{100*y/graph.graph['height']:.4f},"
                       f"{100*width/graph.graph['width']:.4f},"
                       f"{100*height/graph.graph['height']:.4f}")


    @staticmethod
    def decode_position(graph: EngGraph, pos: Literal) -> tuple:
        """Decodes a PCT RDF Literal into an Internal Node's or Port's Position"""

        x, y, width, height = [float(item) for item in pos[4:].split(",")]
        x = int(x * graph.graph['width'] / 100)
        y = int(y * graph.graph['height'] / 100)
        width = int(width * graph.graph['width'] / 100)
        height = int(height * graph.graph['height'] / 100)
        return x, y, width, height
