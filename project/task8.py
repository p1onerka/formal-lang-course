from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton, Box
from pyformlang.finite_automaton import State, NondeterministicFiniteAutomaton
import networkx as nx
import scipy.sparse as scsp
from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA, intersect_automata


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def _build_rsm_fa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    rsm_trans = []
    rsm_start = set()
    rsm_fin = set()
    for sym in rsm.boxes:
        box: Box = rsm.get_box(sym)
        for fst_st, snd_st, lbl in box.dfa.to_networkx().edges(data="label"):
            rsm_trans.append((State((sym, fst_st)), lbl, State((sym, snd_st))))
        for st in box.start_state:
            rsm_start.add(State((sym, st.value)))
        for st in box.final_states:
            rsm_fin.add(State((sym, st.value)))

    rsm_fa = NondeterministicFiniteAutomaton(
        start_state=rsm_start, final_states=rsm_fin
    )
    rsm_fa.add_transitions(rsm_trans)
    return rsm_fa


def _ms_bfs_with_paths(intersection: AdjacencyMatrixFA, adj_rsm: AdjacencyMatrixFA):
    n = len(intersection.states)
    matrix_ctor = getattr(scsp, f"{intersection.matrix_format}_matrix", scsp.csr_matrix)

    bool_dec = intersection.boolean_decompress
    for sym in bool_dec:
        mat = bool_dec.get(sym)
        rows, cols = mat.nonzero()
        for i, j in zip(rows, cols):
            s = intersection.state_of_index.get(i)

    reachability = matrix_ctor((n, n), dtype=bool)
    start_st = _define_start_states(intersection, adj_rsm)
    for s in start_st:
        i = intersection.index_of_state[s]
        reachability[i, i] = True

    edge_usage = []

    finished = False
    while not finished:
        finished = True
        for sym, mat in bool_dec.items():
            sym_reach = reachability @ mat
            new_edges = sym_reach > reachability
            if new_edges.count_nonzero() > 0:
                reachability = reachability + sym_reach
                finished = False

                rows, cols = new_edges.nonzero()
                for i, j in zip(rows, cols):
                    s_from = intersection.state_of_index[i]
                    s_to = intersection.state_of_index[j]
                    edge_usage.append((s_from, sym, s_to))

    rows, cols = reachability.nonzero()
    for i, j in zip(rows, cols):
        s = intersection.state_of_index.get(i)
        f = intersection.state_of_index.get(j)

    return reachability, edge_usage


# MSBFS should search all paths from RSM-start to RSM-fin, regardless of graph start and end vertices
def _define_start_states(
    intersection: AdjacencyMatrixFA, adj_rsm: AdjacencyMatrixFA
) -> set[State]:
    rsm_starts = adj_rsm.start_states
    res = []
    for st in intersection.states:
        _, rsm_st = st.value
        if rsm_st in rsm_starts:
            res.append(st)
    return res


def _add_nonterms(
    reachability: scsp.spmatrix,
    adj_graph: AdjacencyMatrixFA,
    adj_rsm: AdjacencyMatrixFA,
    intersection: AdjacencyMatrixFA,
    edges,
) -> tuple[AdjacencyMatrixFA, bool]:
    new_nonterm_added = False
    rows, cols = reachability.nonzero()
    for i, j in zip(rows, cols):
        gr_start, rsm_start = intersection.state_of_index.get(i).value
        gr_fin, rsm_fin = intersection.state_of_index.get(j).value
        start_rsm_box, _ = rsm_start.value
        fin_rsm_box, _ = rsm_fin.value
        if (
            start_rsm_box == fin_rsm_box
            and rsm_start in adj_rsm.start_states
            and rsm_fin in adj_rsm.final_states
        ):
            if start_rsm_box not in adj_graph.boolean_decompress:
                n = len(adj_graph.states)
                matrix_ctor = getattr(
                    scsp, f"{adj_graph.matrix_format}_matrix", scsp.csr_matrix
                )
                new_nonterm_added = True
                nonterm_mat = matrix_ctor((n, n), dtype=bool)
                nonterm_mat[
                    adj_graph.index_of_state.get(gr_start),
                    adj_graph.index_of_state.get(gr_fin),
                ] = True
                adj_graph.boolean_decompress.update({start_rsm_box: nonterm_mat})
                adj_graph.labels.add(start_rsm_box)
            else:
                nonterm_mat = adj_graph.boolean_decompress.get(start_rsm_box)
                if not nonterm_mat[
                    adj_graph.index_of_state.get(gr_start),
                    adj_graph.index_of_state.get(gr_fin),
                ]:
                    new_nonterm_added = True
                    nonterm_mat[
                        adj_graph.index_of_state.get(gr_start),
                        adj_graph.index_of_state.get(gr_fin),
                    ] = True
                    adj_graph.boolean_decompress.update({start_rsm_box: nonterm_mat})
    return (adj_graph, new_nonterm_added)


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
    matrix_format="csr",
) -> set[tuple[int, int]]:
    if start_nodes is None:
        start_nodes = graph.nodes
    if final_nodes is None:
        final_nodes = graph.nodes

    fa_graph = graph_to_nfa(graph, start_nodes, final_nodes)
    adj_graph = AdjacencyMatrixFA(fa_graph, matrix_format=matrix_format)
    fa_rsm = _build_rsm_fa(rsm)
    adj_rsm = AdjacencyMatrixFA(fa_rsm, matrix_format=matrix_format)

    info_added = True
    while info_added:
        info_added = False
        intersection = intersect_automata(adj_graph, adj_rsm)
        reachability, edges_reachable_from_start = _ms_bfs_with_paths(
            intersection, adj_rsm
        )
        adj_graph, info_added = _add_nonterms(
            reachability, adj_graph, adj_rsm, intersection, edges_reachable_from_start
        )

    res = set()
    if rsm.initial_label in adj_graph.boolean_decompress:
        mat = adj_graph.boolean_decompress.get(rsm.initial_label)
        rows, cols = mat.nonzero()
        for i, j in zip(rows, cols):
            s = adj_graph.state_of_index.get(i)
            f = adj_graph.state_of_index.get(j)
            if (s in start_nodes) and (f in final_nodes):
                res.add((s, f))
    return res
