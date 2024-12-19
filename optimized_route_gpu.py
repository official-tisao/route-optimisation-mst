from union_find import UnionFind
import cupy as cp
import numpy as np
import pandas as pd
import heapq
import networkx as nx


cp.cuda.Device(0).use()


class UnionFindGPU:
  """
  Optimized GPU-based Union-Find data structure with path compression and union by rank.
  """
  def __init__(self, num_nodes):
    self.parent = cp.arange(num_nodes, dtype=cp.int32)
    self.rank = cp.zeros(num_nodes, dtype=cp.int32)


  def find(self, x):
    if self.parent[x] != x:
      self.parent[x] = self.find(self.parent[x])
    return self.parent[x]


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
  weights = edges_cp[:, 0]  # Extract weights (time)
  sorted_indices = cp.argsort(weights)
  sorted_edges = edges_cp[sorted_indices]
  uf = UnionFindGPU(len(nodes))
  mst = []
  for edge in sorted_edges.get():
    weight, u, v = edge
    if uf.find(u) != uf.find(v):
      uf.union(u, v)
      mst.append((u, v, weight))
  return mst


def boruvka_mst_gpu(graph, edges):
  """
  Fully GPU-optimized Borůvka’s MST using CuPy arrays.
  """
  nodes = cp.array(list(graph.keys()), dtype=cp.int32)
  components = cp.arange(len(nodes), dtype=cp.int32)
  num_components = len(nodes)
  mst = []
  while num_components > 1:
    cheapest = cp.full((len(nodes), 3), -1, dtype=cp.float32)
    for weight, u, v in edges:
      root_u = components[u]
      root_v = components[v]
      if root_u != root_v:
        if cheapest[root_u, 0] == -1 or edges[int(cheapest[root_u, 0])][0] > weight:
          cheapest[root_u] = (u, v, weight)
        if cheapest[root_v, 0] == -1 or edges[int(cheapest[root_v, 0])][0] > weight:
          cheapest[root_v] = (u, v, weight)
    for u, v, weight in cheapest:
      root_u = components[u]
      root_v = components[v]
      if root_u != root_v:
        mst.append((u, v, weight))
        components[components == root_v] = root_u
        num_components -= 1
  return mst


def calculate_time_weight(shape_length, speed_limit, default_speed=30):
  """
  Calculate time (in seconds) for a road segment based on shape_length (in meters)
  and speed_limit (in km/h). If speed is missing or exceeds 30 km/h, use default 30 km/h.
  """
  if pd.notna(speed_limit) and speed_limit > 0 and speed_limit <= 30:
    speed = speed_limit
  else:
    speed = default_speed


  speed_mps = (speed * 1000) / 3600  # Convert km/h to m/s
  time_seconds = shape_length / speed_mps
  return time_seconds


