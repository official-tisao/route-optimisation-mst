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


