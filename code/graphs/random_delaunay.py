import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from shapely.geometry import Polygon

from .random_graph import cut_across, hidden_graph


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def lloyd(points, iters=1):
    for _ in range(iters):
        vor = Voronoi(points)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        r = Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
        ps = [Polygon(vertices[region]).intersection(r) for region in regions]
        points = np.array([np.array(p.centroid) for p in ps])
    return points


def delaunay_graph(size, iters=1):
    points = np.random.rand(size, 2)
    points = lloyd(points, iters=iters)
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    a_points = np.concatenate([points, corners])
    tri = Delaunay(a_points)
    edges = sum([[(x, y), (x, z), (y, z)] for x, y, z in tri.vertices], [])
    g = nx.Graph(edges)
    corners_indices = range(len(points), len(a_points))
    g.remove_nodes_from(corners_indices)
    return g, points


def random_delaunay(size, p_not_traversable=0.5, n_hidden=0, weight='length', max_length=3,
                    iters=1):
    g, pos = delaunay_graph(size, iters=iters)
    n_o = g.number_of_edges()
    b = np.sum(pos, axis=1)
    s = np.argmin(b)
    t = np.argmax(b)
    n_t = round(p_not_traversable * g.number_of_edges())
    g, cut, pruned = cut_across(g, n_t, s, t)
    g, hidden_state = hidden_graph(g, n_hidden, s=s, t=t, weight=weight, pos=pos,
                                   max_length=max_length)
    g.pos = pos
    g.original_number_of_edges = n_o
    return g, hidden_state, s, t, cut, pruned
