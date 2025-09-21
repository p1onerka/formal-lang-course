from typing import Iterable
from networkx import MultiDiGraph
from pyformlang.finite_automaton import State, Symbol, NondeterministicFiniteAutomaton
from scipy.sparse import identity, kron, csr_matrix

from project.task2 import regex_to_dfa, graph_to_nfa


class AdjacencyMatrixFA:
    states: set[State]
    index_of_state: dict[State, int]
    state_of_index: dict[int, State]
    start_states: set[State]
    final_states: set[State]
    labels: set[Symbol]
    boolean_decompress: dict[Symbol, csr_matrix]

    def _print_boolean_decompress_pretty(self):
        for symbol in self.boolean_decompress:
            # * 1 for casting into int for better visibility
            print(f"{symbol}:\n {self.boolean_decompress.get(symbol).toarray() * 1}\n")

    def __init__(
        self,
        fa: NondeterministicFiniteAutomaton,
        index_mapping: (tuple[dict[State, int], dict[int, State]] | None) = None,
    ):
        self.states = fa.states
        self.labels = fa.symbols

        # for DFA: casting it to NFA
        if not (isinstance(fa.start_states, set)):
            self.start_states = {fa.start_states}
        else:
            self.start_states = fa.start_states
        if not (isinstance(fa.final_states, set)):
            self.final_states = {fa.final_states}
        else:
            self.final_states = fa.final_states

        if index_mapping is None:
            enum_states = list(enumerate(self.states))
            self.index_of_state = {state: index for index, state in enum_states}
            self.state_of_index = {index: state for index, state in enum_states}
        else:
            self.index_of_state, self.state_of_index = index_mapping

        # make a dictionary with double array to cast to csr_matrix later
        boolean_dec_dict = dict()
        for symbol in fa.symbols:
            adj_arr = [
                [False for _ in range(len(self.states))]
                for _ in range(len(self.states))
            ]
            boolean_dec_dict.update({symbol: adj_arr})

        # iterate over "start" states, for every one of them iterate over symbols,
        # for every symbol iterate over "final" states and put 1 into corresponding place
        # in its symbol's array
        trans_func: dict = fa.to_dict()
        for fst_state in self.states:
            trans: dict = trans_func.get(fst_state)
            if trans is None:
                continue
            for symbol in trans:
                snd_states = trans.get(symbol)
                if not isinstance(snd_states, set):
                    snd_states = {snd_states}
                cur_arr = boolean_dec_dict.get(symbol)
                for snd_state in snd_states:
                    cur_arr[self.index_of_state.get(fst_state)][
                        self.index_of_state.get(snd_state)
                    ] = True
                    boolean_dec_dict.update({symbol: cur_arr})

        self.boolean_decompress = dict()
        for symbol in boolean_dec_dict:
            arr = boolean_dec_dict.get(symbol)
            mat = csr_matrix(arr)
            self.boolean_decompress.update({symbol: mat})

    def get_trans_closure(self) -> csr_matrix:
        E = identity(len(self.states), format="csr", dtype="bool")
        sum_m = E
        for symbol in self.boolean_decompress:
            mat = self.boolean_decompress.get(symbol)
            sum_m += mat

        trans_closure = sum_m.copy()
        non_zero_elem = 0
        for _ in range(1, len(self.states)):
            trans_closure = trans_closure @ sum_m
            if trans_closure.count_nonzero() == non_zero_elem:
                break
            else:
                non_zero_elem = trans_closure.count_nonzero()
        return trans_closure

    def is_empty(self) -> bool:
        tc = self.get_trans_closure()
        for start_st in self.start_states:
            for final_st in self.final_states:
                if tc[
                    self.index_of_state.get(start_st),
                    self.index_of_state.get(final_st),
                ]:
                    # language is non-empty, ergo false
                    return False
        return True

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current_states = self.start_states
        next_states = set()
        for symbol in word:
            mat = self.boolean_decompress.get(symbol)
            if mat is None:
                return False
            for fst_state in current_states:
                for snd_state in self.states:
                    if mat[
                        self.index_of_state.get(fst_state),
                        self.index_of_state.get(snd_state),
                    ]:
                        next_states.add(snd_state)
            current_states = next_states.copy()
        if current_states.intersection(self.final_states):
            return True
        else:
            return False


def build_AdjMatrixFA_with_artefacts(
    states: set[State],
    start_states: set[State],
    final_states: set[State],
    index_of_state: dict[State, int],
    state_of_index: dict[int, State],
    bool_dec: dict[Symbol, csr_matrix],
) -> AdjacencyMatrixFA:
    nfa = NondeterministicFiniteAutomaton(
        states=states, start_state=start_states, final_states=final_states
    )
    for symbol in bool_dec:
        mat = bool_dec.get(symbol).toarray()
        for fst_state in states:
            for snd_state in states:
                if mat[index_of_state.get(fst_state), index_of_state.get(snd_state)]:
                    nfa.add_transition(fst_state, symbol, snd_state)
    mat = AdjacencyMatrixFA(nfa, index_mapping=(index_of_state, state_of_index))
    return mat


# NB: assume that both FA have the same alphabet (from lecture)
def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    new_states = set()
    aut1_idx = automaton1.index_of_state
    aut2_idx = automaton2.index_of_state
    intersection_idxes = dict()
    intersection_states_of_idx = dict()

    for fst in automaton1.states:
        for snd in automaton2.states:
            intersection_idx = aut1_idx.get(fst) * len(
                automaton2.states
            ) + aut2_idx.get(snd)
            new_states.add(State((fst, snd)))
            intersection_idxes.update({State((fst, snd)): intersection_idx})
            intersection_states_of_idx.update({intersection_idx: State((fst, snd))})

    new_start_states = set()
    for fst in automaton1.start_states:
        for snd in automaton2.start_states:
            new_start_states.add(State((fst, snd)))

    new_final_states = set()
    for fst in automaton1.final_states:
        for snd in automaton2.final_states:
            new_final_states.add(State((fst, snd)))

    shared_labels = automaton1.labels.intersection(automaton2.labels)
    new_bool_dec = dict()
    for symbol in shared_labels:
        fst_bool_dec = automaton1.boolean_decompress.get(symbol)
        snd_bool_dec = automaton2.boolean_decompress.get(symbol)
        intersection = kron(fst_bool_dec, snd_bool_dec, format="csr")
        new_bool_dec.update({symbol: intersection})

    return build_AdjMatrixFA_with_artefacts(
        new_states,
        new_start_states,
        new_final_states,
        intersection_idxes,
        intersection_states_of_idx,
        new_bool_dec,
    )


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    reg_graph = regex_to_dfa(regex)
    aut1 = AdjacencyMatrixFA(reg_graph)
    aut2 = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    intersection = intersect_automata(aut1, aut2)
    ind_of_st = intersection.index_of_state
    int_tc = intersection.get_trans_closure()
    result = set()
    for start in intersection.start_states:
        for final in intersection.final_states:
            if int_tc[ind_of_st.get(start), ind_of_st.get(final)]:
                # values of intersection's states are pairs of initial aut's states
                _, start_gr = start.value
                _, final_gr = final.value
                result.add((start_gr, final_gr))
    return result
