from collections import defaultdict
import networkx as nx
from pyparsing import Any, Set
import scipy.sparse as scsp
from pyformlang.cfg import Variable, Terminal, CFG
from project.task6 import cfg_to_weak_normal_form


def _matrix_terminal_rules(
    cfg: CFG,
    graph: nx.DiGraph,
    bool_dec: dict[Variable, scsp.spmatrix],
    index_of_node: dict[Any, int],
) -> dict[Variable, scsp.spmatrix]:
    term_to_nonterm = defaultdict(list)
    for rule in cfg.productions:
        if len(rule.body) > 0 and isinstance(rule.body[0], Terminal):
            term_to_nonterm[rule.body[0].value].append(rule.head)

    for fst, snd, lbl in graph.edges(data="label"):
        if term_to_nonterm.get(lbl):
            for nonterm in term_to_nonterm.get(lbl):
                cur_m = bool_dec.get(nonterm)
                cur_m[index_of_node[fst], index_of_node[snd]] = True

    eps_nonterms = cfg.get_nullable_symbols()
    for nonterm in eps_nonterms:
        cur_m = bool_dec.get(nonterm)
        for node in graph.nodes:
            cur_m[index_of_node[node], index_of_node[node]] = True
    return bool_dec


def _matrix_nonterminal_rules(
    cfg: CFG,
    bool_dec: dict[Variable, scsp.spmatrix],
) -> dict[Variable, scsp.spmatrix]:
    body_of_head = dict()
    var_to_its_body_rules = defaultdict(list)
    changed_nonterms = []
    for rule in cfg.productions:
        if len(rule.body) == 2:
            body_of_head.update({rule.head: rule.body})
            var_to_its_body_rules[rule.body[0]].append(rule)
            var_to_its_body_rules[rule.body[1]].append(rule)
        # if length of body < 2, then this rule is either A -> a or Ni -> eps
        elif len(rule.body) == 1:
            changed_nonterms.append(rule.head)

    while changed_nonterms:
        cur_nonterm = changed_nonterms.pop()
        if var_to_its_body_rules.get(cur_nonterm):
            for rule in var_to_its_body_rules.get(cur_nonterm):
                mat_nonterm_new = bool_dec.get(rule.body[0]) @ bool_dec.get(
                    rule.body[1]
                )
                mat_nonterm = bool_dec.get(rule.head)
                mat_nonterm_new += mat_nonterm
                if (mat_nonterm < mat_nonterm_new).toarray().any():
                    changed_nonterms.append(rule.head)
                    bool_dec.update({rule.head: mat_nonterm_new})
    return bool_dec


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
    matrix_format="csr",
) -> set[tuple[int, int]]:
    cfg_wnf = cfg_to_weak_normal_form(cfg)
    enum_nodes = list(enumerate(graph.nodes))
    index_of_node = {node: index for index, node in enum_nodes}
    node_of_index = {index: node for index, node in enum_nodes}

    matrix_ctor = getattr(scsp, f"{matrix_format}_matrix", scsp.csr_matrix)
    bool_dec = dict()
    size = len(graph.nodes)
    for nonterm in cfg_wnf.variables:
        bool_dec.update({nonterm: matrix_ctor((size, size), dtype="bool")})

    bool_dec = _matrix_terminal_rules(cfg_wnf, graph, bool_dec, index_of_node)
    bool_dec = _matrix_nonterminal_rules(cfg_wnf, bool_dec)

    if start_nodes is None:
        start_nodes = graph.nodes
    if final_nodes is None:
        final_nodes = graph.nodes

    res = set()
    for fst, snd in zip(*bool_dec.get(cfg_wnf.start_symbol).nonzero()):
        if (
            node_of_index.get(fst) in start_nodes
            and node_of_index.get(snd) in final_nodes
        ):
            res.add((node_of_index.get(fst), node_of_index.get(snd)))

    return res
