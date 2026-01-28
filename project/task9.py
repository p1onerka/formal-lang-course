from pyformlang.finite_automaton import State, Symbol
from pyformlang.rsa import RecursiveAutomaton
import networkx as nx
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoxState:
    box: Symbol
    state: State


@dataclass(frozen=True)
class Descriptor:
    vertex: int
    box_state: BoxState
    frame: "StackFrame"


class StackFrame:
    box_state: BoxState
    graph_node: int
    edges: dict[BoxState, set["StackFrame"]]
    visited_vertices: set[int]

    def __init__(self, box_state: BoxState, graph_node: int):
        self.box_state = box_state
        self.graph_node = graph_node
        self.edges = {}
        self.visited_vertices = set()


def link_frames(frame: StackFrame, edge_state: BoxState, parent_frame: StackFrame) -> set[Descriptor]:
    bucket = frame.edges.setdefault(edge_state, set())
    if parent_frame in bucket:
        return set()
    bucket.add(parent_frame)
    return {Descriptor(n, edge_state, parent_frame) for n in frame.visited_vertices}


def pop(frame: StackFrame, node: int) -> set[Descriptor]:
    if node in frame.visited_vertices:
        return set()
    frame.visited_vertices.add(node)
    result = set()
    for box_state, frames in frame.edges.items():
        for frame in frames:
            result.add(Descriptor(node, box_state, frame))
    return result


@dataclass
class BoxTransitions:
    # represents terminal and the following state of box after it is parsed
    terminals: dict[str, BoxState] = field(default_factory=dict)
    # tuple represents the pair of start state of inner box and the following state of outer box after nonterm will be parsed
    nonterminals: dict[str, tuple[BoxState, BoxState]] = field(default_factory=dict)
    is_final: bool = False


class GLLContext:
    graph: nx.DiGraph
    graph_edges: dict[int, dict[str, set[int]]]
    stack_frames: dict[tuple[BoxState, int], StackFrame]
    start_state: BoxState
    fin_frame: StackFrame
    processing_queue: set[Descriptor] #doesn't really need to be strictly queue because the order of descriptors processing is arbitrary
    visited: set[Descriptor]
    result: set[tuple[int, int]]

    def __init__(self, rsm: RecursiveAutomaton, graph: nx.DiGraph):
        self.graph = graph
        self.stack_frames = {}
        self.start_state = BoxState(
            rsm.initial_label,
            rsm.boxes[rsm.initial_label].dfa.start_state.value
        )
        self.fin_frame = StackFrame(BoxState(Symbol("$"), State("fin")), -1)
        self.processing_queue = set()
        self.visited = set()
        self.result = set()


        self.graph_edges = {}
        for fst, snd, lbl in graph.edges(data="label"):
            self.graph_edges.setdefault(fst, {}).setdefault(lbl, set()).add(snd)


        self.rsm_data = {}
        for sym, box in rsm.boxes.items():
            data = {}
            graph_box = box.dfa.to_networkx()
            for state in graph_box.nodes:
                data[state] = BoxTransitions(is_final=(state in box.dfa.final_states))
            for fst, snd, lbl in graph_box.edges(data="label"):
                #print(f"i have label {lbl}")
                if Symbol(lbl) not in rsm.boxes:
                    data[fst].terminals[lbl] = BoxState(sym, snd)
                else:
                    # if lbl is nonterm, find start state of its box
                    start_of_inner_box = rsm.boxes[Symbol(lbl)].dfa.start_state.value
                    data[fst].nonterminals[lbl] = (
                        BoxState(Symbol(lbl), start_of_inner_box),
                        BoxState(sym, snd),
                    )
            self.rsm_data[sym] = data


def create_stack_frame(gll: GLLContext, st: BoxState, node: int) -> StackFrame:
    key = (st, node)
    if key not in gll.stack_frames:
        gll.stack_frames[key] = StackFrame(st, node)
    return gll.stack_frames[key]


def push_new_descriptors(gll: GLLContext, items: set[Descriptor]):
    fresh = set(items) - gll.visited
    gll.visited |= fresh
    gll.processing_queue |= fresh


def handle_popped(gll: GLLContext, popped: set[Descriptor], prev: Descriptor):
    for desc in popped:
        if desc.frame is gll.fin_frame:
            gll.result.add((prev.frame.graph_node, desc.vertex))
        else:
            push_new_descriptors(gll, {desc})


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:

    gll = GLLContext(rsm, graph)
    if not start_nodes:
        start_nodes = graph.nodes()
    if not final_nodes:
        final_nodes = graph.nodes()

    # create start configs and link them to final frame via service state. push all start descriptors into queue
    for start_node in start_nodes:
        frame = create_stack_frame(gll, gll.start_state, start_node)
        link_frames(frame, BoxState(Symbol("$"), State("fin")), gll.fin_frame)
        push_new_descriptors(gll, {Descriptor(start_node, gll.start_state, frame)})

    while gll.processing_queue:
        cur_desc = gll.processing_queue.pop()
        node, state, frame = cur_desc.vertex, cur_desc.box_state, cur_desc.frame
        rsm_info = gll.rsm_data[state.box][state.state]

        # case 1: terminal transitions
        for term, to_state in rsm_info.terminals.items():
            for nxt in gll.graph_edges.get(node, {}).get(term, ()):
                push_new_descriptors(gll, {Descriptor(nxt, to_state, frame)})

        # case 2: nonterminal transitions
        for _, (inner_start_state, return_state) in rsm_info.nonterminals.items():
            new_frame = create_stack_frame(gll, inner_start_state, node)
            popped = link_frames(new_frame, return_state, frame)
            handle_popped(gll, popped, cur_desc)
            push_new_descriptors(gll, {Descriptor(node, inner_start_state, new_frame)})

        # case 3: final state of current box
        if rsm_info.is_final:
            popped = pop(frame, node)
            handle_popped(gll, popped, cur_desc)

    return {(fst, snd) for (fst, snd) in gll.result if snd in final_nodes}
