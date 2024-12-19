from union_find import UnionFind
import cupy as cp
import numpy as np
import pandas as pd
import heapq


cp.cuda.Device(0).use()


class UnionFindGPU:
    """
    Optimized GPU-based Union-Find data structure with path compression and union by rank.
    """
    def __init__(self, nodes):
        self.parent = cp.arange(nodes, dtype=cp.int32)
        self.rank = cp.zeros(nodes, dtype=cp.int32)


    @cp.fuse()
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]


    @cp.fuse()
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


def kruskal_mst_gpu(edges, nodes):
    """
    Fully GPU-accelerated Kruskal's MST algorithm.
    """
    edges_cp = cp.array(edges, dtype=cp.float64)
    weights = edges_cp[:, 0]  # Extract weights
    sorted_indices = cp.argsort(weights)  # GPU-based sorting
    sorted_edges = edges_cp[sorted_indices]  # Sorted edges on GPU


    uf = UnionFindGPU(nodes)  # GPU-based Union-Find
    mst = []


    for edge in sorted_edges.get():  # Transfer to CPU only at the end
        weight, u, v = edge
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst.append((u, v, weight))


    return mst


