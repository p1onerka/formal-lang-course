from pathlib import Path
from typing import List
from pyparsing import Any
from networkx import is_isomorphic
from networkx.drawing.nx_pydot import read_dot
from scripts.shared import TESTS
from project.task1 import (
    GraphMetaData,
    get_graph_meta_data,
    create_two_cycles_graph_and_save_to_dot,
)

import pytest

EXPECTED_DOTS = TESTS / Path("resources/task1")


def test_get_graph_meta_data_from_name_not_in_list():
    with pytest.raises(FileNotFoundError):
        get_graph_meta_data("this name does not exist")


@pytest.mark.parametrize(
    "graph_name, nodes_number, edges_number, labels",
    [
        pytest.param(
            "generations",
            129,
            273,
            [
                "type",
                "first",
                "rest",
                "onProperty",
                "intersectionOf",
                "equivalentClass",
                "someValuesFrom",
                "hasValue",
                "hasSex",
                "hasChild",
                "hasParent",
                "inverseOf",
                "sameAs",
                "hasSibling",
                "oneOf",
                "range",
                "versionInfo",
            ],
        ),
        pytest.param("bzip", 632, 556, ["d", "a"]),
        pytest.param(
            "eclass",
            239111,
            360248,
            [
                "subClassOf",
                "type",
                "comment",
                "label",
                "hierarchyCode",
                "domain",
                "range",
                "subPropertyOf",
                "creator",
                "imports",
            ],
        ),
    ],
)
def test_get_graph_meta_data_from_valid_name(
    graph_name: str, nodes_number: int, edges_number: int, labels: List[Any]
):
    expected = GraphMetaData(nodes_number, edges_number, labels)
    result = get_graph_meta_data(graph_name)
    assert expected == result


def create_two_cycles_graph_and_save_to_dot_zero_nodes_both_cycles():
    with pytest.raises(IndexError):
        create_two_cycles_graph_and_save_to_dot(
            0, 0, ("label_1", "label_2"), "somepath.dot"
        )


def create_two_cycles_graph_and_save_to_dot_zero_nodes_one_cycle():
    with pytest.raises(IndexError):
        create_two_cycles_graph_and_save_to_dot(
            1, 0, ("label_1", "label_2"), "somepath.dot"
        )


@pytest.mark.parametrize(
    "fst_c_node_num, snd_c_node_num, labels, path_to_expected",
    [
        pytest.param(
            1,
            1,
            ("label_1", "label_2"),
            f"{EXPECTED_DOTS}/2_vertices_two_cycles_graph.dot",
        ),
        pytest.param(
            3,
            3,
            ("label_1", "label_2"),
            f"{EXPECTED_DOTS}/7_vertices_two_cycles_graph.dot",
        ),
        pytest.param(
            5,
            7,
            ("label_1", "label_2"),
            f"{EXPECTED_DOTS}/13_vertices_two_cycles_graph.dot",
        ),
    ],
)
def test_create_two_cycles_graph_and_save_to_dot_valid_cycles(
    fst_c_node_num, snd_c_node_num, labels, path_to_expected, tmp_path
):
    expected = read_dot(path_to_expected)
    path = f"{tmp_path}/graph.dot"
    create_two_cycles_graph_and_save_to_dot(
        fst_c_node_num, snd_c_node_num, labels, path
    )
    result = read_dot(path)
    assert is_isomorphic(
        expected,
        result,
        edge_match=lambda e1, e2: dict(e1) == dict(e2),
        node_match=lambda n1, n2: dict(n1) == dict(n2),
    )
