import networkx as nx
import pandas as pd

def create_graph_from_roads(road_segments_df, intersections_df):
  """
  Creates a weighted graph representing road segments and intersections.

  Args:
      road_segments_df (pd.DataFrame): DataFrame of road segment data.
      intersections_df (pd.DataFrame): DataFrame of intersection data.

  Returns:
      nx.Graph: A weighted graph where nodes are intersection ASSETIDs
                and edges are road segments weighted by their length.
  """
  G = nx.Graph()

  # Create a dictionary for quick lookup of intersection coordinates (optional, but can be useful)
  intersection_coords = {}
  for index, row in intersections_df.iterrows():
    if 'ASSETID' in row and 'longitude' in row and 'latitude' in row:
      intersection_coords[row['ASSETID']] = (row['longitude'], row['latitude'])

  for index, row in road_segments_df.iterrows():
    from_node = row.get('FROMNODE')
    to_node = row.get('TONODE')
    length = row.get('Shape_Length')

    if from_node is not None and to_node is not None and length is not None:
      if from_node not in G:
        G.add_node(from_node)
      if to_node not in G:
        G.add_node(to_node)
      G.add_edge(from_node, to_node, weight=length)
    else:
      print(f"Warning: Skipping road segment with missing node or length information (ASSETID: {row.get('ASSETID')})")

  return G

def kruskal_mst(graph):
  """
  Computes the Minimum Spanning Tree using Kruskal's algorithm.

  Args:
      graph (nx.Graph): The weighted graph.

  Returns:
      nx.Graph: The Minimum Spanning Tree.
  """
  mst = nx.Graph()
  edges = sorted(graph.edges(data=True), key=lambda t: t[2].get('weight', float('inf')))
  parent = {node: node for node in graph.nodes()}

  def find_set(v):
    if v == parent[v]:
      return v
    parent[v] = find_set(parent[v])
    return parent[v]

  def unite_sets(a, b):
    a = find_set(a)
    b = find_set(b)
    if a != b:
      parent[b] = a
      return True
    return False

  num_edges = 0
  for u, v, data in edges:
    if unite_sets(u, v):
      mst.add_edge(u, v, weight=data['weight'])
      num_edges += 1
      if num_edges == len(graph.nodes()) - 1:
        break
  return mst

def prim_mst(graph, start_node=None):
  """
  Computes the Minimum Spanning Tree using Prim's algorithm.

  Args:
      graph (nx.Graph): The weighted graph.
      start_node: The node to start the Prim's algorithm from (optional).
                  If None, an arbitrary node will be chosen.

  Returns:
      nx.Graph: The Minimum Spanning Tree.
  """
  if not graph.nodes():
    return nx.Graph()

  mst = nx.Graph()
  visited = set()
  if start_node is None:
    start_node = list(graph.nodes())[0]
  visited.add(start_node)
  pq = []  # Priority queue of (weight, u, v)

  for neighbor, data in graph[start_node].items():
    weight = data.get('weight', float('inf'))
    pq.append((weight, start_node, neighbor))
  pq.sort(key=lambda item: item[0])

  while pq and len(visited) < len(graph.nodes()):
    weight, u, v = pq.pop(0)
    if v not in visited:
      visited.add(v)
      mst.add_edge(u, v, weight=weight)
      for neighbor, data in graph[v].items():
        if neighbor not in visited:
          weight = data.get('weight', float('inf'))
          pq.append((weight, v, neighbor))
      pq.sort(key=lambda item: item[0])

  return mst

def boruvka_mst(graph):
  """
  Computes the Minimum Spanning Tree using Borůvka's algorithm.

  Args:
      graph (nx.Graph): The weighted graph.

  Returns:
      nx.Graph: The Minimum Spanning Tree.
  """
  mst = nx.Graph()
  num_nodes = len(graph.nodes())
  if num_nodes <= 1:
    return graph

  parent = {node: node for node in graph.nodes()}

  def find_set(v):
    if v == parent[v]:
      return v
    parent[v] = find_set(parent[v])
    return parent[v]

  def unite_sets(a, b):
    a = find_set(a)
    b = find_set(b)
    if a != b:
      parent[b] = a
      return True
    return False

  while len(mst.nodes()) < num_nodes - 1:
    best_edges = {}
    for u, v, data in graph.edges(data=True):
      weight = data.get('weight', float('inf'))
      root_u = find_set(u)
      root_v = find_set(v)
      if root_u != root_v:
        if root_u not in best_edges or weight < best_edges[root_u][0]:
          best_edges[root_u] = (weight, u, v)
        if root_v not in best_edges or weight < best_edges[root_v][0]:
          best_edges[root_v] = (weight, u, v)

    if not best_edges:
      break

    edges_added = 0
    for root in list(best_edges.keys()):
      weight, u, v = best_edges[root]
      if unite_sets(u, v):
        mst.add_edge(u, v, weight=weight)
        edges_added += 1
    if edges_added == 0:
      break

  return mst

if __name__ == "__main__":
  # Sample DataFrames (replace with your actual data loading)
  road_segments_data = [
    {'ASSETID': 140484, 'FROMNODE': 142196, 'TONODE': 142195, 'Shape_Length': 465.27176680113035},
    {'ASSETID': 136637, 'FROMNODE': 142448, 'TONODE': 142444, 'Shape_Length': 68.40856452880844},
    {'ASSETID': 136711, 'FROMNODE': 142630, 'TONODE': 142585, 'Shape_Length': 341.74867938607866},
    {'ASSETID': 778875, 'FROMNODE': 142522, 'TONODE': 142502, 'Shape_Length': 191.45519401179487},
    # Add more road segment data here
  ]
  road_segments_df = pd.DataFrame(road_segments_data)

  intersections_data = [
    {'ASSETID': 142573},
    {'ASSETID': 716730},
    {'ASSETID': 716731},
    {'ASSETID': 145675},
    {'ASSETID': 142196},
    {'ASSETID': 142195},
    {'ASSETID': 142448},
    {'ASSETID': 142444},
    {'ASSETID': 142630},
    {'ASSETID': 142585},
    {'ASSETID': 142522},
    {'ASSETID': 142502},
    # Add more intersection data here
  ]
  intersections_df = pd.DataFrame(intersections_data)

  # Create the weighted graph
  graph = create_graph_from_roads(road_segments_df, intersections_df)

  if not graph.nodes():
    print("Warning: No valid road segments found to build the graph.")
  else:
    print("\n--- Kruskal's Algorithm ---")
    mst_kruskal = kruskal_mst(graph)
    print(f"Number of edges in MST (Kruskal): {mst_kruskal.number_of_edges()}")
    print(f"Total weight of MST (Kruskal): {sum(data['weight'] for u, v, data in mst_kruskal.edges(data=True))}")
    # You can further analyze the mst_kruskal graph here (e.g., print edges)

    print("\n--- Prim's Algorithm ---")
    mst_prim = prim_mst(graph)
    print(f"Number of edges in MST (Prim): {mst_prim.number_of_edges()}")
    print(f"Total weight of MST (Prim): {sum(data['weight'] for u, v, data in mst_prim.edges(data=True))}")
    # You can further analyze the mst_prim graph here

    print("\n--- Borůvka's Algorithm ---")
    mst_boruvka = boruvka_mst(graph)
    print(f"Number of edges in MST (Borůvka): {mst_boruvka.number_of_edges()}")
    print(f"Total weight of MST (Borůvka): {sum(data['weight'] for u, v, data in mst_boruvka.edges(data=True))}")
    # You can further analyze the mst_boruvka graph here
