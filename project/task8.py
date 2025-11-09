from collections import defaultdict
from networkx import MultiDiGraph
from pyformlang.cfg import Variable, Terminal, CFG, Production, Epsilon
from pyformlang.rsa import RecursiveAutomaton, Box
from pyformlang.finite_automaton import State, Symbol, NondeterministicFiniteAutomaton
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as scsp
import scipy as sc
from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA, intersect_automata

def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    #print(cfg.to_text())
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

    rsm_fa = NondeterministicFiniteAutomaton(start_state=rsm_start, final_states=rsm_fin)
    rsm_fa.add_transitions(rsm_trans)
    return rsm_fa

def _ms_bfs_with_paths(intersection: AdjacencyMatrixFA, adj_rsm: AdjacencyMatrixFA):
    n = len(intersection.states)
    matrix_ctor = getattr(scsp, f"{intersection.matrix_format}_matrix", scsp.csr_matrix)

    bool_dec = intersection.boolean_decompress
    #intersection._print_boolean_decompress_pretty()
    for sym in bool_dec:
        mat = bool_dec.get(sym)
        rows, cols = mat.nonzero()
        for i, j in zip(rows, cols):
            #print(f"Non-zero elems of intersection with symbol {sym}")
            s = intersection.state_of_index.get(i)
            f = intersection.state_of_index.get(j)
            #print(f"{s} {f}")

    # represents start vertices
    reachability = matrix_ctor((n, n), dtype=bool)
    start_st = _define_start_states(intersection, adj_rsm)
    for s in start_st:
        i = intersection.index_of_state[s]
        reachability[i, i] = True

    #print(reachability.toarray() * 1)
    edge_usage = []

    finished = False
    while not finished:
        finished = True
        for sym, mat in bool_dec.items():
            sym_reach = reachability @ mat
            new_edges = sym_reach > reachability
            #print(sym)
            #print(sym_reach.toarray() * 1)
            if new_edges.count_nonzero() > 0:
                reachability = reachability + sym_reach
                finished = False

                rows, cols = new_edges.nonzero()
                for i, j in zip(rows, cols):
                    s_from = intersection.state_of_index[i]
                    s_to = intersection.state_of_index[j]
                    edge_usage.append((s_from, sym, s_to))

    #print(reachability.toarray() * 1)
    rows, cols = reachability.nonzero()
    for i, j in zip(rows, cols):
        s = intersection.state_of_index.get(i)
        f = intersection.state_of_index.get(j)
        #print(f"{s} {f}")
    #print(f"edge_usage is {edge_usage}")

    return reachability, edge_usage

# MSBFS should search all paths from RSM-start to RSM-fin, regardless of graph start and end vertices
def _define_start_states(intersection: AdjacencyMatrixFA, adj_rsm: AdjacencyMatrixFA) -> set[State]:
    rsm_starts = adj_rsm.start_states
    res = []
    for st in intersection.states:
        _, rsm_st = st.value
        if rsm_st in rsm_starts:
            res.append(st)
    return res


def _add_nonterms(reachability: scsp.spmatrix, adj_graph: AdjacencyMatrixFA, adj_rsm: AdjacencyMatrixFA, intersection: AdjacencyMatrixFA, edges) -> tuple[AdjacencyMatrixFA, bool]:
    new_nonterm_added = False
    rows, cols = reachability.nonzero()
    for i, j in zip(rows, cols):
        gr_start, rsm_start = intersection.state_of_index.get(i).value
        gr_fin, rsm_fin = intersection.state_of_index.get(j).value
        #print(f"{gr_start} {rsm_start} {gr_fin} {rsm_fin}")
        start_rsm_box, _ = rsm_start.value
        fin_rsm_box, _ = rsm_fin.value
        if start_rsm_box == fin_rsm_box and rsm_start in adj_rsm.start_states and rsm_fin in adj_rsm.final_states:
            #print(f"{gr_start} {rsm_start} {gr_fin} {rsm_fin}")
            '''
            if ((gr_start, rsm_start), start_rsm_box, (gr_fin, rsm_fin)) not in edges:
                print("IM NOT HERE")
                continue'''
            if not start_rsm_box in adj_graph.boolean_decompress:
                n = len(adj_graph.states)
                matrix_ctor = getattr(scsp, f"{adj_graph.matrix_format}_matrix", scsp.csr_matrix)
                new_nonterm_added = True
                nonterm_mat = matrix_ctor((n, n), dtype=bool)
                nonterm_mat[adj_graph.index_of_state.get(gr_start), adj_graph.index_of_state.get(gr_fin)] = True
                #print(f" I PUT THIS IN MAT {gr_start} {rsm_start} {gr_fin} {rsm_fin}")
                adj_graph.boolean_decompress.update({start_rsm_box: nonterm_mat})
                adj_graph.labels.add(start_rsm_box)
            else:
                nonterm_mat = adj_graph.boolean_decompress.get(start_rsm_box)
                if not nonterm_mat[adj_graph.index_of_state.get(gr_start), adj_graph.index_of_state.get(gr_fin)]:
                    new_nonterm_added = True
                    nonterm_mat[adj_graph.index_of_state.get(gr_start), adj_graph.index_of_state.get(gr_fin)] = True
                    #print(f" I PUT THIS IN MAT {gr_start} {rsm_start} {gr_fin} {rsm_fin}")
                    adj_graph.boolean_decompress.update({start_rsm_box: nonterm_mat})
    return (adj_graph, new_nonterm_added)





def tensor_based_cfpq(
  rsm: RecursiveAutomaton,
  graph: nx.DiGraph,
  start_nodes: set[int] = None,
  final_nodes: set[int] = None,
  matrix_format="csr",
) -> set[tuple[int, int]]:
    matrix_ctor = getattr(scsp, f"{matrix_format}_matrix", scsp.csr_matrix)
    # maybe change it to "if start_nodes is None or s in start_nodes" in last if?
    if start_nodes is None:
        start_nodes = graph.nodes
    if final_nodes is None:
        final_nodes = graph.nodes

    # add nullable symbols into graph
    '''for sym in rsm.boxes:
        box: Box = rsm.get_box(sym)
        if box.start_state.issubset(box.final_states):
            print(f"{sym} IS NULL")
            for node in graph.nodes:
                graph.add_edge(node, node, label=sym)'''

    fa_graph = graph_to_nfa(graph, start_nodes, final_nodes)
    adj_graph = AdjacencyMatrixFA(fa_graph, matrix_format=matrix_format)
    #print("BOOL DEC BEFORE ALGO")
    #adj_graph._print_boolean_decompress_pretty()
    fa_rsm = _build_rsm_fa(rsm)
    #print(fa_rsm._transition_function.to_dict())
    adj_rsm = AdjacencyMatrixFA(fa_rsm, matrix_format=matrix_format)
    #print(adj_rsm.start_states)
    #print(adj_rsm.final_states)


    info_added = True
    while info_added:
        info_added = False
        intersection = intersect_automata(adj_graph, adj_rsm)
        #print(adj_rsm.labels)
        #print(intersection.labels)
        #intersection._print_boolean_decompress_pretty()
        reachability, edges_reachable_from_start = _ms_bfs_with_paths(intersection, adj_rsm)
        adj_graph, info_added = _add_nonterms(reachability, adj_graph, adj_rsm, intersection, edges_reachable_from_start)
        #adj_graph._print_boolean_decompress_pretty()
        #adj_graph._print_boolean_decompress_pretty()
        #print(edges_reachable_from_start)
        '''for fst, lbl, snd in edges_reachable_from_start:
            bool_dec_mat = adj_graph.boolean_decompress.get(lbl)
            fst_gr, _ = fst.value
            snd_gr, _ = snd.value
            if not bool_dec_mat[adj_graph.index_of_state.get(fst_gr), adj_graph.index_of_state.get(snd_gr)]:
                # TODO: on  example, maybe S shouldnt be there?
                # TODO: maybe check if all vertices are in start->end path?
                #print(f"{fst} {lbl} {snd}")
                bool_dec_mat[adj_graph.index_of_state.get(fst_gr), adj_graph.index_of_state.get(snd_gr)] = True
                info_added = True'''



    res = set()
    if rsm.initial_label in adj_graph.boolean_decompress:
        mat = adj_graph.boolean_decompress.get(rsm.initial_label)
        rows, cols = mat.nonzero()
        for i, j in zip(rows, cols):
            s = adj_graph.state_of_index.get(i)
            f = adj_graph.state_of_index.get(j)
            if (s in start_nodes) and (f in final_nodes):
                res.add((s, f))
    #print(f"res is {res}")
    return res






'''
cfg = CFG.from_text("S -> a S b | a b")
baa_graph = MultiDiGraph()
baa_graph.add_edges_from(
    [(0, 0, {"label": "b"}), (0, 1, {"label": "a"}), (1, 0, {"label": "a"})]
)
res = cfg_to_rsm(cfg)

res_tensor = tensor_based_cfpq(res, baa_graph)
print(res_tensor)

box: Box = res.get_box("S")
edges = box.dfa.minimize().to_networkx().edges(data="label")
s_graph = box.dfa.minimize().to_networkx()

edge_labels = defaultdict(list)
for u, v, data in s_graph.edges(data=True):
    if 'label' in data:
        edge_labels[(u, v)].append(data['label'])
edge_labels_combined = {}
for (u, v), labels in edge_labels.items():
    edge_labels_combined[(u, v)] = ', '.join(map(str, labels))
pos = nx.spring_layout(s_graph)
plt.figure(figsize=(12, 8))
nx.draw(s_graph, pos, with_labels=True,
        node_color='lightblue',
        node_size=500,
        font_size=10,
        font_weight='bold',
        arrows=True,
        edge_color='gray')
nx.draw_networkx_edge_labels(s_graph, pos, edge_labels=edge_labels_combined)
plt.show()
'''

'''
# Creation of variables
var_S = Variable("S")

# Creation of terminals
ter_a = Terminal("a")
ter_b = Terminal("b")
eps = Epsilon()

# Creation of productions
p0 = Production(var_S, [eps])
p1 = Production(var_S, [ter_a, var_S, ter_b])
p2 = Production(var_S, [var_S, var_S])


# Creation of the CFG
cfg = CFG(
    {var_S}, {ter_a, ter_b}, var_S, {p0, p1, p2}
)
#print(cfg.to_text())

# Creation of graph
graph = nx.MultiDiGraph()
graph.add_edge(1, 0, label="a")  # 1 -a-> 0
graph.add_edge(0, 1, label="a")  # 0 -a-> 1
graph.add_edge(0, 0, label="b")  # 0 -b-> 0

res = cfg_to_rsm(cfg)

res_tensor = tensor_based_cfpq(res, graph, {1}, {0, 1})
print(res_tensor)

box: Box = res.get_box(var_S)
edges = box.dfa.minimize().to_networkx().edges(data="label")
s_graph = box.dfa.minimize().to_networkx()

edge_labels = defaultdict(list)
for u, v, data in s_graph.edges(data=True):
    if 'label' in data:
        edge_labels[(u, v)].append(data['label'])
edge_labels_combined = {}
for (u, v), labels in edge_labels.items():
    edge_labels_combined[(u, v)] = ', '.join(map(str, labels))
pos = nx.spring_layout(s_graph)
plt.figure(figsize=(12, 8))
nx.draw(s_graph, pos, with_labels=True,
        node_color='lightblue',
        node_size=500,
        font_size=10,
        font_weight='bold',
        arrows=True,
        edge_color='gray')
nx.draw_networkx_edge_labels(s_graph, pos, edge_labels=edge_labels_combined)
plt.show()
'''
