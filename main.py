# main.py
import importlib.util

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
    from optimized_route_gpu import load_data, build_graph, kruskal_mst_gpu, prim_mst, boruvka_mst
else:
    print("GPU not detected. Using CPU script.")
    from optimized_route_open import load_data, build_graph, kruskal_mst, prim_mst, boruvka_mst

def main():
    road_segments_file = "./road_segment.csv"
    intersections_file = "./intersection.csv"
    road_segments, intersections = load_data(road_segments_file, intersections_file)
    graph, edges = build_graph(road_segments)
    nodes = set(graph.keys())

    if gpu_available:
        mst_kruskal = kruskal_mst_gpu(edges, nodes)
    else:
        mst_kruskal = kruskal_mst(edges, nodes)

    mst_prim = prim_mst(graph)
    mst_boruvka = boruvka_mst(graph, edges)

    print("Kruskal's MST:", mst_kruskal)
    print("Prim's MST:", mst_prim)
    print("Borůvka’s MST:", mst_boruvka)

if __name__ == "__main__":
    main()