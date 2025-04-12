import pandas as pd
import networkx as nx

def calculate_time_weight(shape_length, speed_limit, default_speed=30):
  """
  Calculate edge weight as time (seconds) based on Shape_Length (meters) and SPEEDLIMIT (km/h).
  If speed_limit is NaN, use default_speed (km/h).
  """
  if(pd.notna(speed_limit) and speed_limit > 0):
    speed = speed_limit
  else:
    speed = default_speed
  speed_mps = speed * (5 / 18)  # Convert km/h to m/s
  print (f"Speed: {speed} km/h, Shape_Length: {shape_length}, speed_mps: {speed_mps} m")
  return (shape_length * 18) / (5 * speed)  # Time in seconds

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

def cpp(road_segments):
  """
  Solve the Chinese Postman Problem to find the shortest route covering all edges.
  Returns the total time (seconds) and route.
  """
  # Build undirected graph with NetworkX
  G = nx.Graph()
  road_segments = road_segments.dropna(subset=["FROMNODE", "TONODE"]).copy()
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

  start_asset_id = 142196
  end_asset_id = 142195

  # Build graph with time-based weights
  graph, edges, node_mapping = build_graph(road_segments)
  nodes = set(graph.keys())

  # Map start and end nodes
  start_asset_id = node_mapping.get(start_asset_id, -1)
  end_asset_id = node_mapping.get(end_asset_id, -1)
  if start_asset_id == -1 or end_asset_id == -1:
    raise ValueError("Invalid start or end asset IDs after mapping")

  # Chinese Postman Problem
  cpp_route, cpp_total_time = cpp(road_segments)
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

  # Borůvka's MST (CPU-based)
  mst_boruvka = boruvka_mst(graph, edges)
  save_to_csv(mst_boruvka, "./result/boruvka_mst_cpu.csv")
  print("CPU-based Borůvka’s MST:", mst_boruvka)
