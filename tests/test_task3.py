from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from project.task3 import AdjacencyMatrixFA, build_AdjMatrixFA_with_artefacts


# -> 0 -a-> 1 <-a---b-> (2)
def make_simple_nfa():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_start_state(State(0))
    nfa.add_final_state(State(2))
    nfa.add_transition(State(0), Symbol("a"), State(1))
    nfa.add_transition(State(1), Symbol("b"), State(2))
    nfa.add_transition(State(2), Symbol("a"), State(1))
    return nfa


def test_simple_words():
    nfa = make_simple_nfa()
    adj = AdjacencyMatrixFA(nfa)

    assert adj.accepts("ab") is True
    assert adj.accepts("ababababababababab") is True
    assert adj.accepts("a") is False
    assert adj.accepts("c") is False
    assert adj.accepts("") is False
    assert adj.accepts("$") is False


def test_no_final_states():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_start_state(State(0))
    nfa.add_transition(State(0), Symbol("a"), State(0))
    adj = AdjacencyMatrixFA(nfa)

    assert adj.is_empty() is True
    assert adj.accepts("a") is False


def test_no_start_states():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_final_state(State(0))
    nfa.add_transition(State(0), Symbol("a"), State(0))
    adj = AdjacencyMatrixFA(nfa)

    assert adj.is_empty() is True
    assert adj.accepts("a") is False


def test_index_mappings_are_consistent():
    nfa = make_simple_nfa()
    adj = AdjacencyMatrixFA(nfa)
    for state in adj.index_of_state:
        index = adj.index_of_state.get(state)
        assert state == adj.state_of_index.get(index)


def test_build_with_artefacts_equivalence():
    nfa = make_simple_nfa()
    adj_exp = AdjacencyMatrixFA(nfa)

    states = nfa.states
    start_states = nfa.start_states
    final_states = nfa.final_states
    index_of_state = adj_exp.index_of_state
    state_of_index = adj_exp.state_of_index
    bool_dec = adj_exp.boolean_decompress
    adj_res = build_AdjMatrixFA_with_artefacts(
        states, start_states, final_states, index_of_state, state_of_index, bool_dec
    )

    words = ["a", "c", "ab", "ababababab", "ba"]
    for word in words:
        assert adj_exp.accepts(word) == adj_res.accepts(word)


def test_boolean_decompress_correctness():
    nfa = make_simple_nfa()
    adj = AdjacencyMatrixFA(nfa)

    mat_a = adj.boolean_decompress[Symbol("a")].toarray()
    mat_b = adj.boolean_decompress[Symbol("b")].toarray()

    idx0 = adj.index_of_state[State(0)]
    idx1 = adj.index_of_state[State(1)]
    idx2 = adj.index_of_state[State(2)]

    assert mat_a[idx0, idx1] == 1
    assert mat_a[idx2, idx1] == 1
    assert mat_a.sum() == 2

    assert mat_b[idx1, idx2] == 1
    assert mat_b.sum() == 1
