# File: optimized_route_gpu.py
import cupy as cp
import numpy as np
import pandas as pd
import heapq
from union_find import UnionFind

def load_data(road_segments_file, intersections_file):
    """
    Load CSV data using pandas.
    """
    road_segments = pd.read_csv(road_segments_file)
    intersections = pd.read_csv(intersections_file)
    return road_segments, intersections

def build_graph(road_segments):
    """
    Build a bidirectional graph and edge list from road segments.
    """
    graph = {}
    edges = []
    for _, row in road_segments.iterrows():
        from_node = row['FROMNODE']
        to_node = row['TONODE']
        weight = row['Shape_Length']
        if from_node not in graph:
            graph[from_node] = []
        if to_node not in graph:
            graph[to_node] = []
        graph[from_node].append((weight, to_node))
        graph[to_node].append((weight, from_node))
        edges.append((weight, from_node, to_node))
    return graph, edges

def kruskal_mst_gpu(edges, nodes):
    """
    Kruskal's MST using GPU-accelerated sorting with CuPy.
    """
    edges_np = np.array(edges)
    if edges_np.size == 0:
        return []
    weights = edges_np[:, 0].astype(np.float64)
    sorted_indices = cp.asnumpy(cp.argsort(cp.asarray(weights)))
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
    Bor\u016fvka\u2019s MST algorithm (CPU-based).
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

if __name__ == "__main__":
    # Sample execution for testing; no code runs on import.
    road_segments_file = "./road_segment.csv"
    intersections_file = "./intersection.csv"
    road_segments, intersections = load_data(road_segments_file, intersections_file)
    graph, edges = build_graph(road_segments)
    nodes = set(graph.keys())

    mst_kruskal_gpu = kruskal_mst_gpu(edges, nodes)
    mst_prim = prim_mst(graph)
    mst_boruvka = boruvka_mst(graph, edges)

    print("GPU accelerated Kruskal's MST:", mst_kruskal_gpu)
    print("Prim's MST:", mst_prim)
    print("Bor\u016fvka\u2019s MST:", mst_boruvka)