import networkx as nx
import numpy as np

_b = 10000000.0


def _mix(x, y, data, graph, weight):
    if graph.has_edge(x, y):
        o_data = graph[x][y]
        if o_data[weight] < data[weight]:
            data = o_data
    return data


def copy_with_edges(graph, edges, weight='length'):
    new_graph = nx.Graph(graph)
    new_graph.remove_edges_from(edges)
    new_graph.add_edges_from([(x, y, _mix(x, y, data, graph, weight)) for x, y, data in edges])
    return new_graph


def initial_value(k, n, e, b=np.infty):
    shape = [n] + [3] * k
    v = np.full(shape, b)
    v[e] = 0
    return v


def slice_index(n, i, m):
    r = [slice(0, 3)] * (m - 1)
    r.insert(n, slice(i, i + 1))
    return r


def matrix_from_graph(graph, nodes=None, weight='length', ):
    m = np.asarray(nx.to_numpy_matrix(graph, dtype=np.float64, nodelist=nodes,
                                      weight=weight, nonedge=np.infty))
    for k in range(len(graph)):
        m[k, k] = 0
    return m


def _a(p, y, threshold):
    if y == 0:
        if p == 1:
            return 2
        if p > threshold:
            return 2
        return 1
    return y


def information_policy(state, policy, ps=[], threshold=0.0):
    _, r = state
    if policy == 'optimal':
        return r
    return [_a(p, y, threshold) for y, p in zip(r, ps)]


def visibility_graphs(graph, hidden_state, target=None, weight='length'):
    hidden_edges = set(sum(hidden_state, []))
    nodes_incident_to_hidden_edges = set(sum(hidden_edges, ()))
    # observation_node = [n for n, data in graph.nodes(data=True) if data.get(obeserve_key, [])]
    visible_graph = nx.Graph(graph)
    visible_graph.remove_edges_from(hidden_edges)
    sparse_visible_graph = nx.Graph()
    sparse_visible_graph.add_nodes_from(nodes_incident_to_hidden_edges)
    nodes = nodes_incident_to_hidden_edges
    sparse_visible_graph.add_node(target)
    if target:
        nodes.add(target)
    for x in nodes:
        ls = nx.single_source_dijkstra_path_length(visible_graph, x, weight=weight)
        xs = nx.single_source_dijkstra_path(visible_graph, x, weight=weight)
        for y in nodes:
            if y != x and y in ls and len(nodes & set(xs[y][1:-1])) == 0:
                sparse_visible_graph.add_edge(x, y, length=ls[y], path=xs[y])
    hidden_edges_with_data = [[(x, y, graph[x][y]) for x, y in es] for es in hidden_state]
    hidden_graphs = [copy_with_edges(sparse_visible_graph, es, weight=weight)
                     for es in hidden_edges_with_data]
    return visible_graph, sparse_visible_graph, hidden_graphs


def pessimistic_weight(t, p, th):
    # print('t p th', t,p,th)
    if t == 0:
        if p > th:
            return 0
        else:
            return -np.log(p)
    elif t == 1:
        return np.inf
    else:
        return 0


class Planner(object):

    # Every node in the graph store the list of uncertain states ids (int) it can observe
    # Hidden hidden_edges is a list of <list of edges>, one for every state
    # When the state is set to True, all edges in the list are traversable
    # else all are not traversables

    def __init__(self, graph, target, hidden_state=[]):
        self._graph = graph
        self.visible_graph, self.sparse_visible_graph, self.hidden_graphs = visibility_graphs(
            graph, hidden_state, target)
        self.sparse_total_graph = self.sparse_explored_graph
        self.hidden_state = hidden_state
        self.nodes = list(self.sparse_visible_graph.nodes())
        self._node_index = {n: i for i, n in enumerate(self.nodes)}
        self.target = target
        self.target_index = self._node_index[self.target]
        self._m_o = matrix_from_graph(self.sparse_visible_graph)
        self._m_t = np.array([matrix_from_graph(g) for g in self.hidden_graphs])
        self._observable = {n: self.observable_state(node=n) for n in self.nodes}
        self._min_cost = {}
        self._values = {}

    def is_connected(self, node, realization):
        visible_graph, node = self.visibility_graph_from(node)
        t_edges = sum([es for es, r in zip(self.hidden_state, realization) if r == 1], [])
        visible_edges = set(list(visible_graph.edges()) + t_edges)
        g = nx.Graph(list(visible_edges))
        g.add_nodes_from(visible_graph.nodes())
        return nx.has_path(g, node, self.target)

    def observable_state(self, index=None, node=None):
        if index is not None:
            node = self.nodes[index]
        return self.visible_graph.node[node].get('observe', [])

    def get_value(self, traversability_probability):
        # key = tuple(traversability_probability)
        return self._compute_value(traversability_probability)
        # if key not in self._values:
        #     self._values[key] = self._compute_value(traversability_probability)
        # return self._values[key]

    def _compute_value(self, ps):
        t = self._node_index[self.target]
        v = initial_value(len(self.hidden_state), len(self.sparse_visible_graph), t, _b)
        i = 0
        while True:
            i += 1
            old_v = np.copy(v)
            self._iterate(v, ps)
            if np.allclose(old_v, v, atol=1e-100):
                if i == 1:
                    raise NameError("Could not compute value")
                return v

    def _iterate(self, v, ps):
        le = len(ps)
        for n in range(len(v)):
            for face in self.observable_state(index=n):
                p = ps[face]
                us = slice_index(face, 0, le)
                os = slice_index(face, 1, le)
                ts = slice_index(face, 2, le)
                v[n][us] = p * v[n][ts] + (1 - p) * v[n][os]
                fos = [Ellipsis] + os
                fts = [Ellipsis] + ts
                s = [Ellipsis] + [np.newaxis] * le
                v[n][os] = np.min(v[fos] + self._m_o[n][s], axis=0)
                v[n][ts] = np.min(v[fts] + self._m_t[face][n][s], axis=0)

    def visible_neighbors(self, state):
        node, information = state
        index = self._node_index[node]
        states = self.observable_state(node=node)
        if len(states) == 0:
            return self._m_o[index]
        indices = [i for i in states if information[i] == 2]
        if len(indices) == 0:
            return self._m_o[index]
        return np.min(self._m_t[indices, index], axis=0)

    def _add_to_visibility_graph(self, graph, from_node=None):
        return graph, from_node

    def visibility_graph_from(self, source):
        if source in self.visible_graph:
            return self.visible_graph, source

        g = nx.Graph(self.visible_graph)
        source = self._add_to_visibility_graph(g, from_node=source)
        return g, source

    def _sparse_visible_edges(self, source, weight='length'):
        visible_graph, source = self.visibility_graph_from(source)
        ws = nx.single_source_dijkstra_path_length(visible_graph, source, weight=weight)
        paths = nx.single_source_dijkstra_path(visible_graph, source, weight=weight)
        es = []
        for n in self.nodes:
            w = ws.get(n, np.infty)
            path = paths.get(n, [])
            if len(path) > 2 and len(set(path[1:-1]) & set(self.nodes)) != 0:
                w = np.infty
            es.append(w)
        # es = [ws.get(n, np.infty) for n in self.nodes]
        return es

    def explored_graph(self, information):
        g = nx.Graph(self._graph)
        e_1 = sum([e for e, t in zip(self.hidden_state, information) if t == 1], [])
        e_2 = sum([e for e, t in zip(self.hidden_state, information) if t == 2], [])
        g.remove_edges_from(e_1)
        for x, y in e_2:
            del g[x][y]['hidden']
        return g

    def sparse_explored_graph(self, information):
        g = nx.Graph(self.sparse_visible_graph)
        edges = sum([g.edges(data=True) for g, r
                     in zip(self.hidden_graphs, information) if r == 2], [])
        g.add_edges_from(edges)
        return g

    def sparse_optimistic_graph(self, information, ps, optimistic_threshold):
        # print(ps)
        g = nx.Graph(self.sparse_visible_graph)
        edges = sum([[(x, y, dict(**data, weight=pessimistic_weight(t, p, optimistic_threshold)))
                      for x, y, data in g.edges(data=True)]
                     for g, t, p in zip(self.hidden_graphs, information, ps) if t != 1], [])
        # print('PS', ps, 'EEE', edges)
        g.add_edges_from(edges)
        return g

    def pessimistic_1(self, state, ps, edges, a_information, optimistic_threshold,
                      weight='length'):
        # Choose unexplored node with highest p
        # among the one that were assumed to be not traversable
        # print('Go for alt node from', node)
        node, information = state
        g = self.sparse_explored_graph(a_information)
        if node not in g:
            g.add_edges_from([(node, self.nodes[e], {weight: cost})
                              for e, cost in enumerate(edges) if cost < np.infty])
        conn_nodes = nx.node_connected_component(g, node)
        nodes = sum([sum([[(p, x), (p, y)] for (x, y) in es], [])
                     for es, p, t in zip(self.hidden_state, ps, information)
                     if p <= optimistic_threshold and t == 0], [])
        # TODO: do not jump over nodes, only direct connected,
        # sparse negihtbrs intead of conn components
        nodes = sorted([(p, x) for (p, x) in nodes if x in conn_nodes and x != node])
        n = nodes[0][1]
        path = nx.shortest_path(g, node, n, weight=weight)
        next_node = path[1]
        return next_node, g[node][next_node][weight]

    def pessimistic_0(self, state, ps, edges, optimistic_threshold, weight='length'):
        node, information = state
        g = self.sparse_optimistic_graph(information, ps, optimistic_threshold)
        if node not in g:
            g.add_edges_from([(node, self.nodes[e], {'weight': 0, weight: cost})
                              for e, cost in enumerate(edges) if cost < np.infty])
        path = nx.shortest_path(g, node, self.target, weight='weight')
        if len(path) == 0:
            raise NameError('Could not find pessimistic path')
            return (None, 0)
        else:
            return path[1], g[path[0]][path[1]][weight]

    def policy(self, value, state, ps=[], optimistic_threshold=0.0, policy='optimal',
               weight='length'):
        node, information = state
        if node == self.target:
            return (state, 0)
        index = self._node_index.get(node, None)

        a_information = information_policy(state, policy, ps=ps, threshold=optimistic_threshold)
        if index is not None:
            edges = self.visible_neighbors(state)
            edges = np.copy(edges)
            edges[index] = np.infty
        else:
            edges = self._sparse_visible_edges(node)
        cost = value[([Ellipsis] + a_information)] + edges
        next_node = np.argmin(cost)
        next_cost = cost[next_node]
        c = edges[next_node]
        next_node = self.nodes[next_node]
        if next_cost >= _b:
            if policy == 'optimal':
                return (None, 0)
            # next_node, c = self.pessimistic_1(state, ps, edges, a_information,
            #                                   optimistic_threshold, weight=weight)
            next_node, c = self.pessimistic_0(state, ps, edges,
                                              optimistic_threshold, weight=weight)
        return (next_node, information), c

    def unknown(self):
        return [0] * len(self.hidden_state)

    def min_cost(self, realization, source, weight='length'):
        r = tuple(realization)
        if (r, source) not in self._min_cost:
            g = self.explored_graph(np.array(realization) + 1)
            if source not in g:
                source = self._add_to_visibility_graph(g, from_node=source)
            self._min_cost[(r, source)] = nx.shortest_path_length(g, source, self.target,
                                                                  weight=weight)
        return self._min_cost[(r, source)]

    def cost(self, value_key, realization, source, policy='optimal',
             optimistic_threshold=0.0, **kwargs):
        value = self.get_value(value_key)
        state = (source, self.unknown())
        cost, success = self.do_path(value, realization, state, 0, policy=policy, ps=value_key,
                                     optimistic_threshold=optimistic_threshold, **kwargs)
        return cost, success

    def observe(self, realization, state):
        node, information = state
        information = information[:]
        for i in self.observable_state(node=node):
            information[i] = 2 if realization[i] else 1
        return (node, information)

    def do_path(self, value, realization, state, cost, debug=False, **kwargs):
        node, _ = state
        if node == self.target:
            return (cost, True)
        next_state, next_cost = self.policy(value, state, **kwargs)
        if not next_state:
            return (cost, False)
        cost = cost + next_cost
        if next_state[0] == self.target:
            if debug:
                print(f'FINAL {state} -> ({next_cost}) {next_state}')
            return (cost, True)
        next_state = self.observe(realization, next_state)
        if debug:
            print(f'{state} -> ({next_cost}) {next_state}')
        return self.do_path(value, realization, next_state, cost, debug=debug, **kwargs)
