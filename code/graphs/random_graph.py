import random

import networkx as nx
import numpy as np

from .bridge import bridges


def uniform(a, b):
    def f():
        return random.uniform(a, b)
    return f


def _random_length(max_length):
    length = uniform(1, max_length)

    def f(x, y):
        return length()
    return f


def _length(pos=None):
    def f(x, y):
        if pos is not None:
            return np.linalg.norm(pos[x] - pos[y])
        return np.linalg.norm(np.array(x) - np.array(y))
    return f

# TODO refactor: split weight assignement and hidden state


def hidden_graph(g, n_hidden, s=None, t=None, pos=None, weight='length', max_length=3):

    if weight == 'length':
        length = _length(pos)
    else:
        length = _random_length(max_length)
    for x, y, data in g.edges(data=True):
        data['length'] = length(x, y)

    if t is not None:
        bridge_edges = set(st_bridges(g, s, t))
    else:
        bridge_edges = set()
    shortest_path_edges = set(shortest_path(g, s, t, weight='length')) - bridge_edges
    all_edges = set(g.edges()) - bridge_edges
    if s is not None:
        all_edges = set([e for e in all_edges if s not in e])
        shortest_path_edges = set([e for e in shortest_path_edges if s not in e])
    # 1. do not place uncertainty on bridge_edges
    # 2. place at least 1 uncertainty on shortest_path_edges
    # 3. place the rest on all_edges
    #
    # print('se', len(shortest_path_edges))
    # print('ae', len(all_edges))

    if n_hidden > 0 and len(shortest_path_edges) > 0:
        e = random.choice(list(shortest_path_edges))
        all_edges.remove(e)
        hidden_edges = [e]
        n_hidden = n_hidden - 1
    else:
        hidden_edges = []

    if n_hidden > 0:
        es = list(all_edges)
        random.shuffle(es)
        hidden_edges = hidden_edges + es[:n_hidden]

    hidden_state = [[e] for e in hidden_edges]

    for _, data in g.nodes(data=True):
        data['observe'] = []
    for i, es in enumerate(hidden_state):
        for x, y in es:
            g.node[x]['observe'].append(i)
            g.node[y]['observe'].append(i)
            g[x][y]['hidden'] = True
    return g, hidden_state


def prune_fringes(g, s, t):
    while True:
        n = g.number_of_nodes()
        g.remove_nodes_from(
            [n for n, d in dict(g.degree()).items() if d == 1 and n not in [s, t]])
        if g.number_of_nodes() == n:
            break


def prune_bridges(g, s, t):
    while True:
        n = g.number_of_nodes()
        for b in bridges(g):
            g2 = nx.Graph(g)
            g2.remove_edge(*b)
            g3, g4 = nx.connected_component_subgraphs(g2)
            if s in g3 and t in g3:
                g.remove_nodes_from(g4.nodes())
                break
            if s in g4 and t in g4:
                g.remove_nodes_from(g3.nodes())
                break
        if g.number_of_nodes() == n:
            break


def prune_loops(g, s, t):
    ps = list(nx.articulation_points(g))
    for p in ps:
        if p in g and p not in [s, t]:
            g1 = nx.Graph(g)
            g1.remove_node(p)
            cs = nx.node_connected_component(g1, s)
            ct = nx.node_connected_component(g1, t)
            c = cs.union(ct)
            c.add(p)
            g.remove_nodes_from([n for n in g if n not in c])


def cut_across(g, n_t, s, t):
    # st = dt.now()
    cut = nx.Graph()
    pruned = nx.Graph()
    n = g.number_of_edges()
    while True:
        # print(dt.now()-st)
        o_n = g.number_of_edges()
        e_t = n - o_n
        if e_t >= n_t:
            # print('Has cut enough edges')
            break
        es = list(g.edges())[:]
        random.shuffle(es)
        # print('ra',dt.now()-st)
        for e in es:
            g1 = nx.Graph(g)
            g1.remove_edge(*e)
            if nx.has_path(g1, s, t):
                g = g1
                cut.add_edge(*e)
                o_edges = set(g.edges())
                g.remove_nodes_from(
                    [n for n in g if n not in nx.node_connected_component(g, s)])
                # print('r',dt.now()-st)
                prune_fringes(g, s, t)
                # print('f',dt.now()-st)
                prune_bridges(g, s, t)
                # print('b',dt.now()-st)
                prune_loops(g, s, t)
                # print('l',dt.now()-st)
                pruned.add_edges_from(o_edges - set(g.edges()))
                break
        else:
            # print('Could not cut any edge')
            break

    return g, cut, pruned


def random_realization(g, hidden_state, s, t):
    es = sum(hidden_state, [])
    while True:
        r = random.choices([0, 1], k=len(hidden_state))
        g1 = nx.Graph(g)
        g1.remove_edges_from([e for t, e in zip(r, es) if t == 0])
        if nx.has_path(g1, s, t):
            return r


def _edge_colors(g, realization=None, hidden_state=None):
    if realization is not None and hidden_state is not None:
        res = dict(sum([[(e, 0.5 if r else 1.0) for e in es]
                        for r, es in zip(realization, hidden_state)], []))
        return [res.get(e, 0) for e in g.edges()]
    return [0.75 if ('hidden' in d) else 0 for _, _, d in g.edges(data=True)]


def st_bridges(g, s, t):
    bs = []
    for e in g.edges():
        g2 = nx.Graph(g)
        g2.remove_edge(*e)
        if not nx.has_path(g2, s, t):
            bs.append(e)
    return bs


def shortest_path(g, s, t, weight='length'):
    es = set(g.edges())
    ns = nx.shortest_path(g, s, t, weight=weight)
    cs = set(zip(ns[:-1], ns[1:]))
    cs = cs.union(zip(ns[1:], ns[:-1]))
    return cs.intersection(es)


def plot(g, weight='length', pos=None, realization=None, hidden_state=None, s=None, t=None,
         with_egde_labels=True, cut=None, pruned=None, **kwargs):
    from matplotlib import pyplot as plt
    if pos is None:
        try:
            pos_g = g.pos.tolist()
        except AttributeError:
            pos_g = {n: d.get('pos', n) for n, d in g.nodes(data=True)}
    else:
        pos_g = pos
    cs = [1 if (len(d.get('observe', {})) > 0) else 0 for n, d in g.nodes(data=True)]
    ecs = _edge_colors(g, realization, hidden_state)
    nx.draw_networkx(g, pos_g, node_color=cs,
                     edge_color=ecs, node_size=10, cmap='coolwarm', **kwargs)
    if with_egde_labels:
        nx.draw_networkx_edge_labels(g, pos_g, {(x, y): round(data.get(weight, 0), 1)
                                                for x, y, data in g.edges(data=True)})
    if s is not None:
        plt.plot(*pos_g[s], 'go')
    if t is not None:
        plt.plot(*pos_g[t], 'bo')

    if pruned is not None:
        if pos is None:
            pos_p = {n: d.get('pos', n) for n, d in pruned.nodes(data=True)}
        else:
            pos_p = pos
        nx.draw_networkx(pruned, pos_p, edge_color='grey', node_size=2, alpha=0.2, **kwargs)
    if cut is not None:
        if pos is None:
            pos_c = {n: d.get('pos', n) for n, d in cut.nodes(data=True)}
        else:
            pos_c = pos
        nx.draw_networkx(cut, pos_c, edge_color='red', node_size=2, alpha=0.2, **kwargs)

    plt.axis('off')
