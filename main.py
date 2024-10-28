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


