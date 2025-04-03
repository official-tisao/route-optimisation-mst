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
    from optimized_route_gpu import (load_data, save_to_csv, build_graph, boruvka_mst_gpu,
                                     kruskal_mst_from_gpu, kruskal_mst_gpu, prim_mst, boruvka_mst)
else:
    print("GPU not detected. Using CPU script.")
    from optimized_route_open import load_data, build_graph, kruskal_mst, prim_mst, boruvka_mst

def main():
    if gpu_available:
        road_segments_file = "./road_segment.csv"
        intersections_file = "./intersection.csv"
        road_segments, intersections = load_data(road_segments_file, intersections_file)
        # graph, edges = build_graph(road_segments)

        start_asset_id = 142196  # Change this to your desired start
        end_asset_id = 142195  # Change this to your desired end

        graph, edges, node_mapping = build_graph(road_segments)
        nodes = set(graph.keys())

        # Convert start and end node IDs
        start_asset_id = node_mapping.get(start_asset_id, -1)
        end_asset_id = node_mapping.get(end_asset_id, -1)

        if start_asset_id == -1 or end_asset_id == -1:
            raise ValueError("Invalid start or end asset IDs after mapping")

        # Set specific start and end AssetIDs

        # Prim's MST (CPU-based)
        mst_prim = prim_mst(graph)
        save_to_csv(mst_prim, "./result/prim_mst.csv")
        print("CPU-based Prim's MST:", mst_prim)


        # GPU accelerated Kruskal's MST from start to end
        mst_kruskal_from_gpu = kruskal_mst_from_gpu(edges, nodes, start_asset_id, end_asset_id)
        save_to_csv(mst_kruskal_from_gpu, "./result/kruskal_mst_from_gpu.csv")
        print("GPU accelerated Kruskal's MST from start to end:", mst_kruskal_from_gpu)

        # Borůvka's MST (GPU-based)
        mst_boruvka_gpu = boruvka_mst_gpu(graph, edges)
        save_to_csv(mst_boruvka_gpu, "./result/boruvka_mst_gpu.csv")
        print("GPU-based Borůvka’s MST:", mst_boruvka_gpu)

        # Kruskal's MST (CPU-based)
        mst_kruskal_cpu = kruskal_mst_cpu(edges, nodes)
        save_to_csv(mst_kruskal_cpu, "./result/kruskal_mst_cpu.csv")
        print("CPU-based Kruskal's MST:", mst_kruskal_cpu)

        # Borůvka's MST (CPU-based)
        mst_boruvka = boruvka_mst(graph, edges)
        save_to_csv(mst_boruvka, "./result/boruvka_mst_cpu.csv")
        print("CPU-based Borůvka’s MST:", mst_boruvka)

    else:
        road_segments_file = "./road_segment.csv"
        intersections_file = "./intersection.csv"
        road_segments, intersections = load_data(road_segments_file, intersections_file)
        graph, edges = build_graph(road_segments)
        nodes = set(graph.keys())
        mst_kruskal = kruskal_mst(edges, nodes)

        mst_prim = prim_mst(graph)
        mst_boruvka = boruvka_mst(graph, edges)

        print("Kruskal's MST:", mst_kruskal)
        print("Prim's MST:", mst_prim)
        print("Borůvka’s MST:", mst_boruvka)

if __name__ == "__main__":
    main()