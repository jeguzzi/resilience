from collections import defaultdict

import networkx as nx


def bridges(G):
    """
    Bridge detection algorithm based on WGB-DFS.

    """

    if G.is_directed():
        raise nx.NetworkXError('This function is for undirected graphs.\n'
                               'Use directed_wgb_dfs() for directed graphs.')

    class WhiteGrayBlackDFS:

        def __init__(self, G):
            # white: empty
            # gray: 1
            # black: 2

            self.visited = set()
            self.dfs_num = {}
            self.num = 0
            self.G = G
            self.back_edges = defaultdict(set)

        def bridges(self, parent, current):
            # print '{'
            # print 'parent, current:', parent, current
            # print 'dfs_num:', self.dfs_num
            self.visited.add(current)
            current_lowpoint = self.dfs_num[current] = self.num

            self.num += 1
            # print 'dfs_num:', self.dfs_num

            for child in G.neighbors(current):
                if child != parent:
                    # print 'current, child:', current, child
                    if (current not in self.back_edges or
                       (current in self.back_edges and child not in self.back_edges[current])):
                        if child in self.visited:
                            current_lowpoint = min(
                                current_lowpoint, self.dfs_num[child])
                        else:
                            for x in self.bridges(current, child):
                                yield x
                            if self.child_lowpoint > self.dfs_num[current]:
                                # print '>>> bridge:', current, child
                                yield (current, child)
                            current_lowpoint = min(
                                current_lowpoint, self.child_lowpoint)

            # print 'parent, current, current_lowpoint:', parent, current, current_lowpoint
            # print 'dfs_num:', self.dfs_num
            # print '}'
            self.child_lowpoint = current_lowpoint

    dfs = WhiteGrayBlackDFS(G)

    for x in G:
        if x not in dfs.visited:
            # print x
            for e in dfs.bridges(x, x):
                yield e
