"""jsonConverter.py Import from and Export to JSON"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Development"

# System Imports
import json

# Project Imports
from converter.core.converter import Converter
from converter.core.engineeringGraph import EngGraph
from converter.core.symbol import Port, Property
from converter.core.geometry import Point


class JSONConverter(Converter):
    """CSV Converter Class"""

    def _parse(self, data: str) -> EngGraph:
        """Reads a JSON String and Return a NetworkX Graph Object"""

        data = json.loads(data)
        graph = EngGraph(name=data['graph']['name'], width=data['graph']['width'], height=data['graph']['height'],
                         image=data['graph']['image'])

        for node in data['nodes']:
            graph.add_node(node_id=node['id'], name=node['name'], type=node['type'],
                           position=node['position'], text=node['text'], shape=node['shape'],
                           ports=[Port(name=port['name'], position=Point(port['position']['x'], port['position']['y']))
                                  for port in node['ports']],
                           properties=[Property(name=prop['name'], value=prop['value'],
                                                position=Point(prop['position']['x'], prop['position']['y']))
                                       for prop in node['properties']])

        for edge in data['edges']:
            edge_import = {'source': edge['source'], 'target': edge['target']}

            if 'sourcePort' in edge:
                edge_import['sourcePort'] = edge['sourcePort']

            if 'targetPort' in edge:
                edge_import['targetPort'] = edge['targetPort']

            if 'shape' in edge:
                edge_import['shape'] = edge['shape']

            graph.add_edge(**edge_import)

        return graph


    def _write(self, graph: EngGraph, **kwargs) -> str:
        """Constructs a JSON String from an Engineering Graph"""

        data = {'graph': {}, 'nodes': [], 'edges': []}

        # Graph Attributes
        data['graph']['name'] = graph.graph['name']
        data['graph']['width'] = graph.graph['width']
        data['graph']['height'] = graph.graph['height']
        data['graph']['image'] = graph.graph['image']

        # Nodes
        for node_id, node_data in graph.nodes.items():
            data['nodes'].append({'name': node_data['name'],
                                  'type': node_data['type'],
                                  'position': node_data['position'],
                                  'text': node_data['text'],
                                  'ports': [{'name': port.name,
                                             'position': {'x': port.position.x, 'y': port.position.y}}
                                            for port in node_data['ports']],
                                  'shape': node_data['shape'],
                                  'properties': [{'name': prop.name,
                                                  'value': prop.value,
                                                  'position': {'x': prop.position.x, 'y': prop.position.y}}
                                                 for prop in node_data['properties']],
                                  'id': node_id})

        # Edges
        for source, target in graph.edges:
            edge_original = graph.edges[(source, target)]
            edge_export = {'source': source, 'target': target}

            if 'sourcePort' in edge_original:
                edge_export['sourcePort'] = edge_original['sourcePort']

            if 'targetPort' in edge_original:
                edge_export['targetPort'] = edge_original['targetPort']

            if 'shape' in edge_original:
                edge_export['shape'] = edge_original['shape']

            data['edges'].append(edge_export)

        return json.dumps(data, indent=2)
