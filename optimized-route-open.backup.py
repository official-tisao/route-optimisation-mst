import pandas as pd
import heapq

# Load CSV files
def load_data(road_segments_file, intersections_file):
  road_segments = pd.read_csv(road_segments_file)
  intersections = pd.read_csv(intersections_file)
  return road_segments, intersections

# Build the graph from road segments
def build_graph(road_segments):
  graph = {}
  edges = []

  for _, row in road_segments.iterrows():
    from_node = row['FROMNODE']
    to_node = row['TONODE']
    weight = row['Shape_Length']  # Distance as weight

    # Add bidirectional edges
    if from_node not in graph:
      graph[from_node] = []
    if to_node not in graph:
      graph[to_node] = []

    graph[from_node].append((weight, to_node))
    graph[to_node].append((weight, from_node))
    edges.append((weight, from_node, to_node))

  return graph, edges

# Kruskal's Algorithm (Union-Find)
class UnionFind:
  def __init__(self, nodes):
    self.parent = {node: node for node in nodes}
    self.rank = {node: 0 for node in nodes}

  def find(self, node):
    if self.parent[node] != node:
      self.parent[node] = self.find(self.parent[node])  # Path compression
    return self.parent[node]

  def union(self, u, v):
    root_u = self.find(u)
    root_v = self.find(v)

    if root_u != root_v:
      if self.rank[root_u] > self.rank[root_v]:
        self.parent[root_v] = root_u
      elif self.rank[root_u] < self.rank[root_v]:
        self.parent[root_u] = root_v
      else:
        self.parent[root_v] = root_u
        self.rank[root_u] += 1

def kruskal_mst(edges, nodes):
  edges.sort()  # Sort by weight
  uf = UnionFind(nodes)
  mst = []
  for weight, u, v in edges:
    if uf.find(u) != uf.find(v):
      uf.union(u, v)
      mst.append((u, v, weight))
  return mst

# Prim's Algorithm
def prim_mst(graph):
  start_node = next(iter(graph))  # Pick any starting node
  min_heap = [(0, start_node, None)]  # (weight, node, parent)
  visited = set()
  mst = []

  while min_heap:
    weight, node, parent = heapq.heappop(min_heap)
    if node in visited:
      continue
    visited.add(node)
    if parent is not None:
      mst.append((parent, node, weight))

    for neighbor_weight, neighbor in graph[node]:
      if neighbor not in visited:
        heapq.heappush(min_heap, (neighbor_weight, neighbor, node))

  return mst

# Borůvka’s Algorithm
def boruvka_mst(graph, edges):
  components = {node: node for node in graph}
  num_components = len(graph)
  mst = []

  while num_components > 1:
    cheapest = {}
    for weight, u, v in edges:
      root_u = components[u]
      root_v = components[v]
      if root_u != root_v:
        if root_u not in cheapest or cheapest[root_u][2] > weight:
          cheapest[root_u] = (u, v, weight)
        if root_v not in cheapest or cheapest[root_v][2] > weight:
          cheapest[root_v] = (u, v, weight)

    for u, v, weight in cheapest.values():
      root_u = components[u]
      root_v = components[v]
      if root_u != root_v:
        mst.append((u, v, weight))
        for node in graph:
          if components[node] == root_v:
            components[node] = root_u
        num_components -= 1

  return mst

# Main Execution
road_segments_file = "road_segments.csv"
intersections_file = "intersections.csv"

road_segments, intersections = load_data(road_segments_file, intersections_file)
graph, edges = build_graph(road_segments)
nodes = set(graph.keys())

kruskal_result = kruskal_mst(edges, nodes)
prim_result = prim_mst(graph)
boruvka_result = boruvka_mst(graph, edges)

# Print Results
print("Kruskal's MST:", kruskal_result)
print("Prim's MST:", prim_result)
print("Borůvka’s MST:", boruvka_result)
