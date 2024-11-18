# main.py
import importlib.util
from commons_function import (calculate_time_weight, build_graph, load_data, save_to_csv)
from chinese_postman_algorithm import cpp
# Check if GPU is available using CuPy
gpu_available = False
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        gpu_available = True
except Exception:
    gpu_available = False


if gpu_available:
    print("GPU detected. Using GPU optimized script.")
    from optimized_route_gpu import (boruvka_mst_gpu, kruskal_mst_from_gpu, kruskal_mst_gpu, prim_mst, boruvka_mst)
                                    # load_data, save_to_csv, build_graph,
else:
    print("GPU not detected. Using CPU script.")
    from optimized_route_open import kruskal_mst, prim_mst, boruvka_mst
                                     # , load_data, build_graph


def main():
  road_segments_file = "./road_segment.csv"
  intersections_file = "./intersection.csv"
  road_segments, intersections = load_data(road_segments_file, intersections_file)
  # graph, edges = build_graph(road_segments)


  start_asset_id = 142196  # Change this to your desired start
  end_asset_id = 142195  # Change this to your desired end


  # Build graph with time-based weights


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


  if gpu_available:
    # GPU accelerated Kruskal's MST from start to end
    mst_kruskal_from_gpu = kruskal_mst_from_gpu(edges, nodes, start_asset_id, end_asset_id)
    save_to_csv(mst_kruskal_from_gpu, "./result/kruskal_mst_from_gpu.csv")
    print("GPU accelerated Kruskal's MST from start to end:", mst_kruskal_from_gpu)


    # Borůvka's MST (GPU-based)
    mst_boruvka_gpu = boruvka_mst_gpu(graph, edges)
    save_to_csv(mst_boruvka_gpu, "./result/boruvka_mst_gpu.csv")
    print("GPU-based Borůvka’s MST:", mst_boruvka_gpu)


    # Prim's MST (CPU-based)
    mst_prim = prim_mst(graph)
    save_to_csv(mst_prim, "./result/prim_mst.csv")
    print("CPU-based Prim's MST:", mst_prim)


