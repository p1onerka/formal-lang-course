from project.task1 import create_two_cycles_graph_and_save_to_dot, get_graph_meta_data
from project.task2 import regex_to_dfa, graph_to_nfa
from networkx.drawing.nx_pydot import read_dot
from networkx import relabel_nodes
import pytest
import cfpq_data as cfpq
from pyformlang.regular_expression import MisformedRegexError


def test_regex_to_dfa_empty_regex():
    dfa = regex_to_dfa("")
    assert dfa.is_empty()


def test_regex_to_dfa_incorrect_regex_entry():
    with pytest.raises(MisformedRegexError):
        regex_to_dfa("(|)*")


@pytest.mark.parametrize(
    "graph_name, start_states, final_states",
    [
        pytest.param(
            "generations",
            set(),
            set(),
        ),
        pytest.param(
            "generations",
            {1, 2, 3},
            set(),
        ),
        pytest.param(
            "generations",
            set(),
            {1, 2, 3},
        ),
        pytest.param(
            "generations",
            {1, 2, 3},
            {1, 2, 3},
        ),
    ],
)
def test_graph_to_nfa_from_dataset(graph_name, start_states, final_states):
    graph_path = cfpq.download(graph_name)
    graph = cfpq.graph_from_csv(graph_path)
    nfa = graph_to_nfa(graph, start_states, final_states)
    metadata = get_graph_meta_data(graph_name)
    assert metadata.nodes_number == len(nfa.states)
    assert metadata.edges_number == nfa.get_number_transitions()
    if start_states == set():
        assert set(graph.nodes) == nfa.start_states
    else:
        assert start_states == nfa.start_states
    if final_states == set():
        assert set(graph.nodes) == nfa.final_states
    else:
        assert final_states == nfa.final_states
    assert set(metadata.labels) == nfa.symbols


@pytest.mark.parametrize(
    "start_states, final_states, fst_c_node_num, snd_c_node_num, labels",
    [
        pytest.param(
            set(),
            set(),
            3,
            4,
            ("fst", "snd"),
        ),
        pytest.param(
            set(),
            {0, 1, 2},
            3,
            4,
            ("fst", "snd"),
        ),
        pytest.param(
            {0, 1, 2},
            set(),
            3,
            4,
            ("fst", "snd"),
        ),
        pytest.param(
            {0, 1, 2},
            {0, 1, 2},
            3,
            4,
            ("fst", "snd"),
        ),
    ],
)
def test_graph_to_nfa_from_dot(
    start_states, final_states, fst_c_node_num, snd_c_node_num, labels, tmp_path
):
    path = f"{tmp_path}/graph.dot"
    create_two_cycles_graph_and_save_to_dot(
        fst_c_node_num, snd_c_node_num, labels, path
    )
    graph = read_dot(path)
    # read_dot marks vertices with strings, eg ['1', '2'],
    # so without int cast with non-empty set of start/fin states there will be two different states in nfa, marked eg '1' and 1.
    # Assuming that graph_to_nfa takes Set[int] only,
    # it is safe to always cast vertices' marks to int before constructing nfa to prevent vertex "duplication"
    graph = relabel_nodes(graph, lambda x: int(x))
    nfa = graph_to_nfa(graph, start_states, final_states)

    assert graph.number_of_nodes() == len(nfa.states)
    assert graph.number_of_edges() == nfa.get_number_transitions()
    if start_states == set():
        assert set(graph.nodes) == nfa.start_states
    else:
        assert start_states == nfa.start_states
    if final_states == set():
        assert set(graph.nodes) == nfa.final_states
    else:
        assert final_states == nfa.final_states
    assert set(cfpq.get_sorted_labels(graph)) == nfa.symbols
