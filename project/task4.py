from networkx import MultiDiGraph
from scipy.sparse import csr_matrix, vstack

from project.task3 import AdjacencyMatrixFA
from project.task2 import regex_to_dfa, graph_to_nfa


def _build_front(
    aut1_len: int, aut2_len: int, aut1_start: set[int], aut2_start: set[int]
) -> csr_matrix:
    fronts_by_starts = list()

    for i in sorted(aut1_start):
        front_arr = [[False for _ in range(aut2_len)] for _ in range(aut1_len)]
        for j in aut2_start:
            front_arr[i][j] = True
        fronts_by_starts.append(csr_matrix(front_arr))
    front = vstack(fronts_by_starts, format="csr", dtype=bool)
    return front


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    aut1 = graph_to_nfa(graph, start_nodes, final_nodes)
    aut2 = regex_to_dfa(regex)
    adj1 = AdjacencyMatrixFA(aut1)
    adj2 = AdjacencyMatrixFA(aut2)

    aut1_start_st_ind = set()
    for start_state in aut1.start_states:
        aut1_start_st_ind.add(adj1.index_of_state.get(start_state))
    aut2_start_st_ind = set()
    for start_state in aut2.start_states:
        aut2_start_st_ind.add(adj2.index_of_state.get(start_state))
    front = _build_front(
        len(aut1.states),
        len(aut2.states),
        sorted(list(aut1_start_st_ind)),
        aut2_start_st_ind,
    )

    bool_dec_transposed = dict()
    for symbol in adj1.boolean_decompress:
        bool_dec_transposed.update(
            {symbol: adj1.boolean_decompress.get(symbol).transpose()}
        )

    visited = front
    shared_labels = adj1.labels.intersection(adj2.labels)
    finished = False
    while not finished:
        current_front_sum = csr_matrix(
            (len(aut1.states) * len(start_nodes), len(aut2.states)), dtype=bool
        )
        for label in shared_labels:
            aut1_mat = bool_dec_transposed.get(label)
            aut2_mat = adj2.boolean_decompress.get(label)
            blocks = []
            for b_num in range(len(start_nodes)):
                cur_b = front[
                    b_num * len(aut1.states) : (b_num + 1) * len(aut1.states), :
                ]
                new_block = aut1_mat @ cur_b
                blocks.append(new_block)
            symbol_front = vstack(blocks, format="csr")
            result = symbol_front @ aut2_mat
            current_front_sum += result
        front = current_front_sum
        finished = not (visited < current_front_sum).toarray().any()
        visited += front

    result = set()
    start_list = sorted(list(start_nodes))
    for start_num in range(len(start_nodes)):
        cur_start = start_list[start_num]
        cur_visited = visited[
            start_num * len(aut1.states) : (start_num + 1) * len(aut1.states), :
        ]
        for i in adj1.states:
            for j in adj2.states:
                if (
                    cur_visited[adj1.index_of_state.get(i), adj2.index_of_state.get(j)]
                    and (i in adj1.final_states)
                    and (j in adj2.final_states)
                ):
                    result.add((cur_start, i))
    return result
