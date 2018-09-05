"""Generate random grids
"""

import networkx as nx

from .random_graph import cut_across, hidden_graph


def random_grid(side,
                p_not_traversable=0.5,
                n_hidden=0,
                max_length=3):
    """ Generate a subgraph of a grid by removing some edges and declaring other edges as hidden

    Args:
        side (int): The side length of the original (side x side) grid.
        p_not_traversable (float, optional): The total number of edges to remove
                                             (comprised pruned edges).
        n_hidden (int, optional): The (maximal) number of hidden edges.
        max_length (float, optional): The upper bound of random (uniform) weights
                                      assigned to the edges. Should be larger than 1.
    Returns:
        A tuple consisting in
        final graph, hidden state, source, target, cut graph and pruned graph.

    """
    g = nx.grid_2d_graph(side, side)
    s = list(g.nodes())[0]
    t = list(g.nodes())[-1]
    n_t = round(p_not_traversable * g.number_of_edges())
    g, cut, pruned = cut_across(g, n_t, s, t)
    g, hidden_state = hidden_graph(g, n_hidden, s=s, t=t, max_length=max_length,
                                   weight='random')
    return g, hidden_state, s, t, cut, pruned
