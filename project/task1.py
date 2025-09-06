from typing import List, Tuple
from dataclasses import dataclass
import cfpq_data as cfpq
from networkx import MultiDiGraph
from pyparsing import Any
from networkx.drawing.nx_pydot import write_dot


@dataclass
class GraphMetaData:
    nodes_number: int
    edges_number: int
    labels: List[Any]


def _load_graph_by_name(graph_name: str) -> MultiDiGraph:
    graph_path = cfpq.download(graph_name)
    return cfpq.graph_from_csv(graph_path)


def get_graph_meta_data(graph_name: str) -> GraphMetaData:
    graph = _load_graph_by_name(graph_name)
    return GraphMetaData(
        nodes_number=cfpq.graph.number_of_nodes(),
        edges_number=cfpq.graph.number_of_edges(),
        labels=cfpq.get_sorted_labels(graph),
    )


def create_two_cycled_graph(
    fst_c_node_num: int, snd_c_node_num: int, labels: Tuple[str, str]
) -> MultiDiGraph:
    return cfpq.labeled_two_cycles_graph(fst_c_node_num, snd_c_node_num, labels=labels)


def save_graph_to_dot(graph: MultiDiGraph, path: str):
    write_dot(graph, path)
