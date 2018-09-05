import collections

import networkx as nx

from graphs.indoor import IndoorMap
from .planner_abs import Planner


def _add_to_line_graph(layer, graph, line_graph, to_node=None, from_node=None, point=None):
    node = to_node or from_node
    state = layer.states[node]
    if point is None:
        point = state.geometry
    line_graph_node = (node, node, point.coords[0])
    for node_2 in graph.neighbors(node):
        state_2 = layer.states[node_2]
        t = state.transitionsTo(state_2)[0]
        if t and t.duality and t.duality.geometry:
            point_2 = t.duality.geometry.centroid
            delta = point.distance(point_2)
            if line_graph.has_node((node_2, node, t.id)):
                y = (node_2, node, t.id)
            else:
                y = (node, node_2, t.id)
            line_graph.add_edge(line_graph_node, y, length=delta)

    return line_graph_node


def _traversability(layer, node, data):
    c = layer.states[node].duality
    if not c:
        return False
    n = c.className
    if n and ('T' in n or 'O' in n):
        data['hidden'] = True
        data['p_t'] = float(n[1:])
        data['traversable'] = ('T' in c.className)
    else:
        data['hidden'] = False
        data['p_t'] = 1.0
        data['traversable'] = True
    return data['hidden']


def _traversability_graph(layer, hidden_nodes=None, not_traversable_states=[]):
    graph = nx.MultiGraph(layer.graph)

    not_traversable_states = (not_traversable_states +
                              [s.id for s in layer.states.values()
                               if not s.duality or s.duality.type == 'CellSpace'])
    graph.remove_nodes_from(not_traversable_states)
    if hidden_nodes is None:
        hidden_nodes = [node for node, data in graph.nodes(data=True)
                        if _traversability(layer, node, data)]
    return graph, hidden_nodes


def _state_for_line_graph_edge(e):
    n1, n2, data = e
    (s11, s12, ti1) = n1
    (s21, s22, ti2) = n2
    if s11 in n2:
        return s11
    else:
        return s12


def _dual_position(layer, t_id):
    try:
        return layer.transitions[t_id].geometry.coords[1]
    except KeyError:
        return t_id


def _dual_positions(layer, line_graph):
    return {node: _dual_position(layer, node[2]) for node in line_graph.nodes()}


def line_graph(layer, target_state, hidden_nodes=None, not_traversable_states=[]):
    graph, hidden_nodes = _traversability_graph(
        layer, hidden_nodes=hidden_nodes, not_traversable_states=not_traversable_states)
    line_graph = nx.line_graph(graph)
    line_graph.edges_in_state = collections.defaultdict(list)
    for e in line_graph.edges(data=True):
        n1, n2, data = e
        (_, so, ti1) = n1
        (_, _, ti2) = n2
        s = _state_for_line_graph_edge(e)
        p1 = layer.transitions[ti1].geometry.centroid
        p2 = layer.transitions[ti2].geometry.centroid
        data['length'] = p1.distance(p2)
        line_graph.edges_in_state[s].append(e[:2])
    target = _add_to_line_graph(layer, graph, line_graph, to_node=target_state)
    hidden_state = [line_graph.edges_in_state[node] for node in hidden_nodes]
    for n, data in line_graph.nodes(data=True):
        data['observe'] = []
    for i, edges in enumerate(hidden_state):
        for x, y in edges:
            line_graph[x][y][0]['hidden'] = True
            line_graph.node[x]['observe'].append(i)
            line_graph.node[y]['observe'].append(i)
    return graph, nx.Graph(line_graph), target, hidden_state


class CellPlanner(Planner):
    def __init__(self, path='', layer_id='L0',
                 target_id='L0S45'):
        m = IndoorMap.loadFromFile(path)
        self._layer = m.space_layers['L0']
        self._cell_graph, lg, target, hidden_state = line_graph(
            self._layer, target_id)
        super(CellPlanner, self).__init__(lg, target, hidden_state)

    def _add_to_visibility_graph(self, graph, from_node=None):
        try:
            node, point = from_node
        except ValueError:
            node = from_node
            point = None
        return _add_to_line_graph(self._layer, self._cell_graph, graph,
                                  from_node=node, point=point)
