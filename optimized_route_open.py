import pandas as pd
import heapq
from union_find import UnionFind  # Assuming you have a UnionFind implementation

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

def kruskal_mst(edges, nodes):
  edges.sort()  # Sort by weight
  uf = UnionFind(nodes)
  mst = []
  for weight, u, v in edges:
    if uf.find(u) != uf.find(v):
      uf.union(u, v)
      mst.append((u, v, weight))
  return mst

def kruskal_mst_from(edges, nodes, start, end):
  mst = kruskal_mst(edges, nodes)
  filtered_mst = [edge for edge in mst if edge[0] == start or edge[1] == end]
  return filtered_mst if filtered_mst else "No path found between start and end"

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

def prim_mst_from(graph, start, end):
  min_heap = [(0, start, None)]
  visited = set()
  mst = []

  while min_heap:
    weight, node, parent = heapq.heappop(min_heap)
    if node in visited:
      continue
    visited.add(node)
    if parent is not None:
      mst.append((parent, node, weight))
    if node == end:
      return mst  # Stop once we reach the end

    for neighbor_weight, neighbor in graph[node]:
      if neighbor not in visited:
        heapq.heappush(min_heap, (neighbor_weight, neighbor, node))

  return "No path found between start and end"

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

def boruvka_mst_from(graph, edges, start, end):
  mst = boruvka_mst(graph, edges)
  filtered_mst = [edge for edge in mst if edge[0] == start or edge[1] == end]
  return filtered_mst if filtered_mst else "No path found between start and end"

# Main Execution
road_segments_file = "road_segment.csv"
intersections_file = "intersection.csv"

road_segments, intersections = load_data(road_segments_file, intersections_file)
graph, edges = build_graph(road_segments)
nodes = set(graph.keys())

# Set specific start and end AssetIDs
start_asset_id = 142196  # Change this to your desired start
end_asset_id = 142195  # Change this to your desired end


prim_result = prim_mst(graph)
prim_result_from = prim_mst_from(graph, start_asset_id, end_asset_id)

boruvka_result = boruvka_mst(graph, edges)
boruvka_result_from = boruvka_mst_from(graph, edges, start_asset_id, end_asset_id)
#
# kruskal_result = kruskal_mst(edges, nodes)
# kruskal_result_from = kruskal_mst_from(edges, nodes, start_asset_id, end_asset_id)

print("Prim's MST:", prim_result)
print("Prim's MST (from start to end):", prim_result_from)
print("Borůvka’s MST:", boruvka_result)
print("Borůvka’s MST (from start to end):", boruvka_result_from)
#
# # Print Results
# print("Kruskal's MST:", kruskal_result)
# print("Kruskal's MST (from start to end):", kruskal_result_from)
