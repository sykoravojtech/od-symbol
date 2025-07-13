"""evaluation.py: End-to-End Pipeline Evaluation"""

# System Imports

# Third-Party Imports
import networkx as nx

from converter.converter.labelmeConverter import LabelMeConverter

# Project Imports
from converter.core.engineeringGraph import EngGraph

__author__ = "Johannes Bayer, Vojtěch Sýkora"
__copyright__ = "Copyright 2025, Johannes Bayer"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@mail.de"
__status__ = "Prototype"


def node_ins_cost(node: dict) -> float:

    return 1


def node_del_cost(node: dict) -> float:
    return 1


def node_subst_cost(node_a: dict, node_b: dict) -> float:

    if (
        node_a["position"]["x"] == node_b["position"]["x"]
        and node_a["position"]["y"] == node_b["position"]["y"]
    ):
        return 0

    return 1


def edge_ins_cost(edge: dict) -> float:

    return 1


def edge_del_cost(edge: dict) -> float:
    return 1


def edge_subst_cost(edge_a: dict, edge_b: dict) -> float:

    # Equal
    if (edge_a.get("sourcePort", None) == edge_b.get("sourcePort", None)) and (
        edge_a.get("targetPort", None) == edge_b.get("targetPort", None)
    ):
        return 0

    # Reverse
    if (edge_a.get("sourcePort", None) == edge_b.get("targetPort", None)) and (
        edge_a.get("targetPort", None) == edge_b.get("sourcePort", None)
    ):
        return 0

    return 1


def compare(graph_a: EngGraph, graph_b: EngGraph) -> float:
    ged = nx.graph_edit_distance(
        graph_a,
        graph_b,
        node_ins_cost=node_ins_cost,
        node_del_cost=node_del_cost,
        node_subst_cost=node_subst_cost,
        edge_ins_cost=edge_ins_cost,
        edge_del_cost=edge_del_cost,
        edge_subst_cost=edge_subst_cost,
    )

    return ged


eng_a = LabelMeConverter().load(
    fileName="data/cghd_raw/drafter_30/instances_refined/C349_D2_P4.json"
)
eng_b = LabelMeConverter().load(
    fileName="data/cghd_raw/drafter_30/instances_refined/C349_D2_P4.json"
)
ged_1_2 = compare(eng_a, eng_b)
print(ged_1_2)

"""
prp -m extraction.src.core.evaluation
"""
