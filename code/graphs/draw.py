import networkx as nx


def draw_graph(g, s, t, pos, removed_edges=[], ax=None, hidden_edges=dict(),
               marker_size=12):
    from matplotlib import pyplot as plt
    if ax is None:
        ax = plt.subplot(111)
    if removed_edges:
        rem = nx.Graph(removed_edges)
        nx.draw_networkx(rem, g.pos, edge_color='black', node_size=15, alpha=0.15,
                         with_labels=False, node_color='black', ax=ax)
    g_t = nx.Graph([e for e in g.edges() if e not in hidden_edges])
    g_h_t = nx.Graph([e for e in g.edges() if e in hidden_edges and hidden_edges[e] == 1])
    g_h_n = nx.Graph([e for e in g.edges() if e in hidden_edges and hidden_edges[e] == 0])
    g_h_u = nx.Graph([e for e in g.edges() if e in hidden_edges and hidden_edges[e] not in [0, 1]])

    nx.draw_networkx(g_t, pos, edge_color='black', node_size=15, alpha=1.0,
                     with_labels=False, node_color='black', ax=ax)
    nx.draw_networkx(g_h_t, g.pos, edge_color='green', node_size=0, alpha=1.0,
                     with_labels=False, node_color='black', style='dashed', ax=ax)
    nx.draw_networkx(g_h_n, g.pos, edge_color='red', node_size=10, alpha=1.0,
                     with_labels=False, node_color='black', style='dashed', ax=ax)
    nx.draw_networkx(g_h_u, g.pos, edge_color='black', node_size=10, alpha=1.0,
                     with_labels=False, node_color='black', style='dashed', ax=ax)
    ax.plot(*pos[s], 'ko', markersize=marker_size)
    ax.plot(*pos[t], 'wo', markersize=marker_size * 3 / 4, markeredgecolor='black',
            markeredgewidth=marker_size / 3)
    ax.axis('equal')
    ax.axis('off')


colorForCellType = {'CellSpace': 'k', 'Navigable': 'g'}
colorForBorderType = {'CellSpaceBoundary': 'k', 'NavigableBoundary': 'g'}


def draw_indoor_layer(layer, with_labels=False):
    for border_id, border in layer.boundaries.items():
        plotBorder(border)
    if with_labels:
        for cell_id, cell in layer.cells.items():
            plotLabelCell(cell)


def colorForCell(cell):
    return colorForCellType.get(cell.type, 'b')


def colorForBoundary(border):
    return colorForBorderType.get(border.type, 'b')


def plotBorder(border, color=None):
    from matplotlib import pyplot as plt
    (x, y) = border.geometry.xy
    if color is None:
        plt.plot(x, y, color=colorForBoundary(border))
    else:
        plt.plot(x, y, color=color)


def plotLabelCell(cell):
    from matplotlib import pyplot as plt
    x, y = cell.duality.geometry.coords[0]
    plt.text(x, y, cell.duality.id)
