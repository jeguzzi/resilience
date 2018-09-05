import random

import networkx as nx

from graphs import prune_bridges, prune_fringes, prune_loops
from .planner_abs import Planner
from .planner_indoor import CellPlanner, IndoorMap, line_graph


def st_ap(g, s, t):
    aps = []
    ps = list(nx.articulation_points(g))
    for p in ps:
        if p not in [s, t]:
            g1 = nx.Graph(g)
            g1.remove_node(p)
            if not nx.has_path(g1, s, t):
                aps.append(p)
    return aps


def candidate_nodes(g, s, t):
    aps = set(st_ap(g, s, t))
    shortest_path = set(nx.shortest_path(g, s, t, weight='length')) - aps - set([s, t])
    all_nodes = set(g.nodes()) - aps - set([s, t])
    return shortest_path, all_nodes


def hidden_doors(layer, s, t, n_hidden=None, closed_doors=[], open_doors=[]):
    g = nx.Graph(layer.graph)
    g.remove_nodes_from(closed_doors)
    all_doors = set([c.duality.id for c in layer.cells.values()
                     if c.function and 'Door' in c.function])
    all_doors = all_doors - set(open_doors) - set(closed_doors)
    prune_fringes(g, s, t)
    prune_bridges(g, s, t)
    prune_loops(g, s, t)
    ne, ae = candidate_nodes(g, s, t)
    ne = ne.intersection(all_doors)
    ae = ae.intersection(all_doors)
    if n_hidden is None:
        return list(ae)
    if n_hidden > 0 and len(ne) > 0:
        e = random.choice(list(ne))
        ae.remove(e)
        hidden_nodes = [e]
        n_hidden = n_hidden - 1
    else:
        hidden_nodes = []

    if n_hidden > 0:
        es = list(ae)
        random.shuffle(es)
        hidden_nodes = hidden_nodes + es[:n_hidden]
    return hidden_nodes


class DoorPlanner(CellPlanner):
    def __init__(self, path='m_como.xml', layer_id='W',
                 target_id='WS12', source_id='WS119', n_hidden=None, closed_doors=[],
                 open_doors=[]):
        m = IndoorMap.loadFromFile(path)
        self._layer = m.space_layers[layer_id]
        hns = hidden_doors(self._layer, source_id, target_id, n_hidden, closed_doors, open_doors)
        self._cell_graph, lg, target, hidden_state = line_graph(
            self._layer, target_id, hidden_nodes=hns, not_traversable_states=closed_doors)
        Planner.__init__(self, lg, target, hidden_state)
