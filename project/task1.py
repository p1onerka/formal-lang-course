from typing import List, Tuple
from dataclasses import dataclass
from networkx import MultiDiGraph
from pyparsing import Any
from networkx.drawing.nx_pydot import write_dot

import cfpq_data as cfpq


@dataclass
class GraphMetaData:
    nodes_number: int
    edges_number: int
    labels: List[Any]


def _load_graph_by_name(graph_name: str) -> MultiDiGraph:
    graph_path = cfpq.download(graph_name)
    return cfpq.graph_from_csv(graph_path)


def _save_graph_to_dot(graph: MultiDiGraph, path: str):
    write_dot(graph, path)


def get_graph_meta_data(graph_name: str) -> GraphMetaData:
    graph = _load_graph_by_name(graph_name)
    return GraphMetaData(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        cfpq.get_sorted_labels(graph),
    )


def create_two_cycles_graph_and_save_to_dot(
    fst_c_node_num: int, snd_c_node_num: int, labels: Tuple[str, str], file_path: str
):
    graph = cfpq.labeled_two_cycles_graph(fst_c_node_num, snd_c_node_num, labels=labels)
    _save_graph_to_dot(graph, file_path)
