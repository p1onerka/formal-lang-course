from pyformlang.cfg import Production, Variable, Terminal, CFG, Epsilon
from networkx import DiGraph
from collections import defaultdict


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    eps_rules = set()
    for nonterm in cfg.get_nullable_symbols():
        eps_rules.add(Production(nonterm, [Epsilon()]))
    cfg = cfg.to_normal_form()
    cfg = CFG(
        cfg.variables, cfg.terminals, cfg.start_symbol, cfg.productions | eps_rules
    )

    return cfg.remove_useless_symbols()


# for rules of types (A -> a) and (Ni -> Eps)
def _hellings_terminal_rules(
    cfg: CFG, graph: DiGraph
) -> set[tuple[int, Variable, int]]:
    term_to_nonterm = defaultdict(list)
    for rule in cfg.productions:
        if len(rule.body) > 0 and isinstance(rule.body[0], Terminal):
            term_to_nonterm[rule.body[0].value].append(rule.head)

    new_edges = set()
    for fst, snd, lbl in graph.edges(data="label"):
        if term_to_nonterm.get(lbl) is not None:
            for nonterm in term_to_nonterm.get(lbl):
                new_edges.add((fst, nonterm, snd))

    eps_nonterms = cfg.get_nullable_symbols()
    for node in graph.nodes:
        for nonterm in eps_nonterms:
            new_edges.add((node, nonterm, node))
    return new_edges


def _hellings_nonterminal_rules(
    cfg: CFG, edges: set[tuple[int, Variable, int]]
) -> set[tuple[int, Variable, int]]:
    two_nonterm_rules = []
    for rule in cfg.productions:
        if len(rule.body) == 2:
            two_nonterm_rules.append(rule)

    edges_are_added = True
    while edges_are_added:
        edges_are_added = False
        added_edges = set()
        for fst1, N1, snd1 in edges:
            for fst2, N2, snd2 in edges:
                if snd1 == fst2:
                    for rule in two_nonterm_rules:
                        if (
                            N1 == rule.body[0]
                            and N2 == rule.body[1]
                            and (fst1, rule.head, snd2) not in edges
                        ):
                            added_edges.add((fst1, rule.head, snd2))
        if added_edges:
            edges_are_added = True
            edges = edges | added_edges
    return edges


def hellings_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    cfg_wnf = cfg_to_weak_normal_form(cfg)
    new_edges = _hellings_terminal_rules(cfg_wnf, graph)
    new_edges = _hellings_nonterminal_rules(cfg_wnf, new_edges)

    if start_nodes is None:
        start_nodes = graph.nodes
    if final_nodes is None:
        final_nodes = graph.nodes

    res = set()
    for fst, var, snd in new_edges:
        if var == cfg_wnf.start_symbol and fst in start_nodes and snd in final_nodes:
            res.add((fst, snd))
    return res
