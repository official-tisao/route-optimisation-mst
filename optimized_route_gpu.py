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


def build_graph(road_segments):
  graph = {}
  edges = []
  # Identify and print rows with missing FROMNODE or TONODE
  nan_rows = road_segments[road_segments["FROMNODE"].isna() | road_segments["TONODE"].isna()]
  if not nan_rows.empty:
    print("\n🚨 The following records contain NaN values and will be removed:\n")
    print(nan_rows.to_string(index=False))
    # Drop rows with NaN values in FROMNODE or TONODE


  road_segments = road_segments.dropna(subset=["FROMNODE", "TONODE"]).copy()


  # Convert node IDs to integers safely
  road_segments.loc[:, "FROMNODE"] = road_segments["FROMNODE"].astype(int)
  road_segments.loc[:, "TONODE"] = road_segments["TONODE"].astype(int)


  unique_nodes = set(road_segments["FROMNODE"]).union(set(road_segments["TONODE"]))
  node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}


  for _, row in road_segments.iterrows():
    from_node = node_mapping[row["FROMNODE"]]
    to_node = node_mapping[row["TONODE"]]
    weight = calculate_time_weight(row["Shape_Length"], row["SPEEDLIMIT"])  # Time-based weight


    if from_node not in graph:
      graph[from_node] = []
    if to_node not in graph:
      graph[to_node] = []


    graph[from_node].append((weight, to_node))
    graph[to_node].append((weight, from_node))
    edges.append((weight, from_node, to_node))


    return graph, edges, node_mapping  # Return node mapping


def load_data(road_segments_file, intersections_file):
    """
    Load CSV data using pandas.
    """
    road_segments = pd.read_csv(road_segments_file)
    intersections = pd.read_csv(intersections_file)
    return road_segments, intersections


def chinese_postman_problem(road_segments):
  """
  Solve the Chinese Postman Problem to find the shortest route covering all edges.
  Returns the total time (seconds) and route.
  """
  # Build undirected graph with NetworkX
  G = nx.Graph()
  for _, row in road_segments.iterrows():
    from_node = int(row["FROMNODE"])
    to_node = int(row["TONODE"])
    weight = calculate_time_weight(row["Shape_Length"], row["SPEEDLIMIT"])
    G.add_edge(from_node, to_node, weight=weight)


  # Check if Eulerian
  odd_nodes = [node for node, degree in G.degree() if degree % 2 != 0]
  if not odd_nodes:
    # Fully Eulerian, find circuit
    route = list(nx.eulerian_circuit(G))
  else:
    # Add edges to make Eulerian
    G_euler = nx.Graph(G)
    odd_pairs = nx.min_weight_matching(G.subgraph(odd_nodes))
    for u, v in odd_pairs:
      weight = nx.shortest_path_length(G, u, v, weight="weight")
      G_euler.add_edge(u, v, weight=weight)
    route = list(nx.eulerian_circuit(G_euler))


  # Calculate total time
  total_time = sum(G[u][v]["weight"] if (u, v) in G.edges else G[v][u]["weight"] for u, v in route)
  return route, total_time


def kruskal_mst_from_gpu(edges, nodes, start, end):
  mst = kruskal_mst_gpu(edges, nodes)
  filtered_mst = [edge for edge in mst if edge[0] == start or edge[1] == end]
  return filtered_mst if filtered_mst else "No path found between start and end"


def kruskal_mst_cpu(edges, nodes):
    """
    Kruskal's MST using CPU-based sorting with NumPy.
    """
    edges_np = np.array(edges)
    if edges_np.size == 0:
        return []
    weights = edges_np[:, 0].astype(np.float64)
    sorted_indices = np.argsort(weights)
    sorted_edges = edges_np[sorted_indices]
    uf = UnionFind(nodes)
    mst = []
    for edge in sorted_edges:
        weight, u, v = edge
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst.append((u, v, weight))
    return mst


def prim_mst(graph):
    """
    Prim's MST algorithm (CPU-based).
    """
    start_node = next(iter(graph))
    min_heap = [(0, start_node, None)]
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


def boruvka_mst(graph, edges):
    """
    Borůvka’s MST algorithm (CPU-based).
    """
    components = {node: node for node in graph}
    num_components = len(graph)
    mst = []
    while num_components > 1:
        cheapest = {}
        for weight, u, v in edges:
            root_u = components[u]
            root_v = components[v]
            if root_u != root_v:
                if (root_u not in cheapest) or (cheapest[root_u][2] > weight):
                    cheapest[root_u] = (u, v, weight)
                if (root_v not in cheapest) or (cheapest[root_v][2] > weight):
                    cheapest[root_v] = (u, v, weight)
        for u, v, weight in cheapest.values():
            root_u = components[u]
            root_v = components[v]
            if root_u != root_v:
                mst.append((u, v, weight))
                for node in components:
                    if components[node] == root_v:
                        components[node] = root_u
                num_components -= 1
    return mst


def save_to_csv(mst, filename):
    """
    Save the MST result to a CSV file.
    """
    df = pd.DataFrame(mst, columns=['FromNode', 'ToNode', 'Weight'])
    df.to_csv(filename, index=False)


if __name__ == "__main__":
  road_segments_file = "./road_segment.csv"
  intersections_file = "./intersection.csv"
  road_segments, intersections = load_data(road_segments_file, intersections_file)
  # graph, edges = build_graph(road_segments)


  start_asset_id = 142196  # Change this to your desired start
  end_asset_id = 142195  # Change this to your desired end


  # Build graph with time-based weights
  graph, edges, node_mapping = build_graph(road_segments)
  nodes = set(graph.keys())


  # Map start and end nodes
  start_asset_id = node_mapping.get(start_asset_id, -1)
  end_asset_id = node_mapping.get(end_asset_id, -1)
  if start_asset_id == -1 or end_asset_id == -1:
    raise ValueError("Invalid start or end asset IDs after mapping")


  # Chinese Postman Problem
  cpp_route, cpp_total_time = chinese_postman_problem(road_segments)
  print("CPP Route:", cpp_route)
  print("Total Time (seconds):", cpp_total_time)
  save_to_csv([(u, v, G[u][v]["weight"]) for u, v in cpp_route], "./result/cpp_route.csv")


  # MST Algorithms with time-based weights
  mst_boruvka_gpu = boruvka_mst_gpu(graph, edges)
  save_to_csv(mst_boruvka_gpu, "./result/boruvka_mst_gpu.csv")
  print("GPU-based Borůvka’s MST:", mst_boruvka_gpu)


  # GPU accelerated Kruskal's MST from start to end
  mst_kruskal_from_gpu = kruskal_mst_from_gpu(edges, nodes, start_asset_id, end_asset_id)
  save_to_csv(mst_kruskal_from_gpu, "./result/kruskal_mst_from_gpu.csv")
  print("GPU accelerated Kruskal's MST from start to end:", mst_kruskal_from_gpu)


  # Prim's MST (CPU-based)
  mst_prim = prim_mst(graph)
  save_to_csv(mst_prim, "./result/prim_mst.csv")
  print("CPU-based Prim's MST:", mst_prim)


  # Kruskal's MST (CPU-based)
  mst_kruskal_cpu = kruskal_mst_cpu(edges, nodes)
  save_to_csv(mst_kruskal_cpu, "./result/kruskal_mst_cpu.csv")
  print("CPU-based Kruskal's MST:", mst_kruskal_cpu)


