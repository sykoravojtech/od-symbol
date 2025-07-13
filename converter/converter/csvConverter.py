"""csvConverter.py Import from and Export to CSV"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Development"

# System Imports
import io
import csv

# Third-Party Imports
import networkx as nx

# Project Imports
from converter.core.converter import Converter
from converter.core.engineeringGraph import EngGraph


class CsvConverter(Converter):
    """CSV Converter Class"""

    def _write(self, graph: EngGraph, **kwargs) -> str:
        """Constructs a CSV File String from an Engineering Graph"""

        io_string = io.StringIO()
        writer = csv.writer(io_string, quoting=csv.QUOTE_NONNUMERIC)

        # Graph
        writer.writerow(["Graph"])
        writer.writerow(['name', 'width', 'height'])
        writer.writerow([graph.graph['name'], graph.graph['width'], graph.graph['height']])

        # Nodes
        writer.writerow([])
        writer.writerow(["Nodes"])
        writer.writerow(["ID", "name", "type", "x", "y", "width", "height", "rotation"])
        node_name = nx.get_node_attributes(graph, 'name')
        node_class = nx.get_node_attributes(graph, 'type')
        node_pos = nx.get_node_attributes(graph, 'position')

        for node in graph.nodes:
            writer.writerow([node, node_name[node], node_class[node],
                             node_pos[node]["x"], node_pos[node]["y"],
                             node_pos[node]["width"], node_pos[node]["height"], node_pos[node]["rotation"]])

        # Edges
        writer.writerow([])
        writer.writerow(["Edges"])
        writer.writerow(["source", "target"])

        for source, target in graph.edges:
            writer.writerow([source, target])

        return io_string.getvalue()
