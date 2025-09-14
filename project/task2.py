from pyformlang.finite_automaton import (
    Symbol,
    State,
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from networkx import MultiDiGraph
from pyformlang.regular_expression import Regex
from typing import Set


EPSILON = "$"


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    reg = Regex(regex)
    nfa_w_eps = reg.to_epsilon_nfa()
    dfa = nfa_w_eps.to_deterministic()
    return dfa.minimize()


def _add_states_to_aut(
    graph: MultiDiGraph,
    start_states: Set[int],
    final_states: Set[int],
) -> NondeterministicFiniteAutomaton:
    nodes = graph.nodes
    if len(start_states) == 0:
        start_states = nodes
    if len(final_states) == 0:
        final_states = nodes
    nfa = NondeterministicFiniteAutomaton(
        states=nodes, start_state=start_states, final_states=final_states
    )
    return nfa


def _add_transitions_to_aut(
    graph: MultiDiGraph, nfa: NondeterministicFiniteAutomaton
) -> NondeterministicFiniteAutomaton:
    edges = graph.edges(data="label")
    for start, end, label in edges:
        if label is None:
            label = EPSILON
        nfa.add_transition(State(start), Symbol(label), State(end))
    return nfa


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    nfa = _add_states_to_aut(graph, start_states, final_states)
    nfa = _add_transitions_to_aut(graph, nfa)
    return nfa
