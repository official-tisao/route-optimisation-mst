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


