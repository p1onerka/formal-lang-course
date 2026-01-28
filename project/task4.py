from networkx import MultiDiGraph
from scipy.sparse import csr_matrix, vstack
import scipy.sparse as scsp

from project.task3 import AdjacencyMatrixFA
from project.task2 import regex_to_dfa, graph_to_nfa
# from task1 import _load_graph_by_name

import time


def _build_front(
    aut1_len: int,
    aut2_len: int,
    aut1_start: set[int],
    aut2_start: set[int],
    matrix_format: str = "csr",
) -> scsp.spmatrix:
    fronts_by_starts = list()
    matrix_ctor = getattr(scsp, f"{matrix_format}_matrix", csr_matrix)

    for i in sorted(aut1_start):
        front_arr = [[False for _ in range(aut2_len)] for _ in range(aut1_len)]
        for j in aut2_start:
            front_arr[i][j] = True
        fronts_by_starts.append(matrix_ctor(front_arr))
    front = vstack(fronts_by_starts, format=matrix_format, dtype=bool)
    return front


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    matrix_format: str = "csr",
) -> set[tuple[int, int]]:
    aut1 = graph_to_nfa(graph, start_nodes, final_nodes)
    aut2 = regex_to_dfa(regex)
    adj1 = AdjacencyMatrixFA(aut1, matrix_format=matrix_format)
    adj2 = AdjacencyMatrixFA(aut2, matrix_format=matrix_format)

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
        matrix_format=matrix_format,
    )

    bool_dec_transposed = dict()
    for symbol in adj1.boolean_decompress:
        bool_dec_transposed.update(
            {symbol: adj1.boolean_decompress.get(symbol).transpose()}
        )

    matrix_ctor = getattr(scsp, f"{matrix_format}_matrix", csr_matrix)
    visited = front
    shared_labels = adj1.labels.intersection(adj2.labels)
    finished = False
    # time_for_sum = 0
    # time_for_mul = 0
    # time_for_pick = 0
    start_bfs = time.perf_counter()
    while not finished:
        current_front_sum = matrix_ctor(
            (len(aut1.states) * len(start_nodes), len(aut2.states)), dtype=bool
        )
        for label in shared_labels:
            aut1_mat = bool_dec_transposed.get(label)
            aut2_mat = adj2.boolean_decompress.get(label)
            blocks = []
            for b_num in range(len(start_nodes)):
                # start_pick = time.perf_counter()
                cur_b = front[
                    b_num * len(aut1.states) : (b_num + 1) * len(aut1.states), :
                ]
                # end_pick = time.perf_counter()
                # time_for_pick += end_pick - start_pick
                # start_mul_fst = time.perf_counter()
                new_block = aut1_mat @ cur_b
                # end_mul_fst = time.perf_counter()
                blocks.append(new_block)
            # print(f"BLOCKS IS {blocks} LEN START IS {len(start_nodes)}")
            symbol_front = vstack(blocks, format=matrix_format)
            # start_mul_snd = time.perf_counter()
            result = symbol_front @ aut2_mat
            # end_mul_snd = time.perf_counter()
            # time_for_mul += end_mul_fst - start_mul_fst + end_mul_snd - start_mul_snd
            # start_sum_fst = time.perf_counter()
            current_front_sum += result
            # end_sum_fst = time.perf_counter()
        front = current_front_sum
        finished = not (visited < current_front_sum).toarray().any()
        # start_sum_snd = time.perf_counter()
        visited += front
        # end_sum_snd = time.perf_counter()
        # time_for_sum += end_sum_fst - start_sum_fst + end_sum_snd - start_sum_snd
    end_bfs = time.perf_counter()
    print(f"time inside BFS is {end_bfs - start_bfs}")

    # print(f"pick: {time_for_pick}, mul: {time_for_mul}, sum: {time_for_sum}")

    result = set()
    start_list = sorted(list(start_nodes))
    start_cycle = time.perf_counter()
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
    end_cycle = time.perf_counter()
    print(f"cycle {end_cycle - start_cycle}")
    return result


"""
graph = _load_graph_by_name("wc")
print("Vertex picking cycle with DOK")
for _ in range(1, 5):
    res = ms_bfs_based_rpq("(d | a)* d", graph, graph.nodes, graph.nodes, matrix_format="dok")
print("Vertex picking cycle with LIL")
for _ in range(1, 5):
    res = ms_bfs_based_rpq("(d | a)* d", graph, graph.nodes, graph.nodes, matrix_format="lil")
print("Vertex picking cycle with CSC")
for _ in range(1, 5):
    res = ms_bfs_based_rpq("(d | a)* d", graph, graph.nodes, graph.nodes, matrix_format="csc")
print("Vertex picking cycle with CSR")
for _ in range(1, 5):
    res = ms_bfs_based_rpq("(d | a)* d", graph, graph.nodes, graph.nodes, matrix_format="csr")
"""
