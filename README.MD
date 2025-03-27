## Project Name

This project is a Route Optimization project originally developed
to help save snow plowing cost for Ontario City of Sudbury.

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/official-tisao/route-optimisation-mst.git
    ```
2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```
    OR (if above command does not work)

   ```bash
   python3 -m venv .venv
   ```
3. **Activate the virtual environment:**

   ```bash
    source .venv/bin/activate
    ```
4. **Install the package:**

    ```bash
    pip install .
    ```
5. **Run the package:**

    5.1 **Run the main from the command line:**

    ```bash
    python main.py
    ```

    5.2 **Run the view map from the command line:**

    ```bash
    python view-map.py
    ```
6. **Run the tests(if you add any unit test):**

    ```bash
    python -m unittest discover -s tests
    ```
   
7. **When you are done, Deactivate the virtual environment:**

   ```bash
   deactivate
   ```

# Minimum Spanning Tree (MST) Algorithms

This project implements three Minimum Spanning Tree (MST) algorithms:

- **Kruskal's Algorithm**
- **Prim's Algorithm**
- **Borůvka’s Algorithm**

It loads road network data from **CSV files** (`road_segments.csv`, `intersections.csv`) and builds a graph to compute MSTs.

---

## Features

- ✅ **Load CSV data for road segments and intersections**
- ✅ **Compute MST using Kruskal, Prim, and Borůvka**
- ✅ **Set a specific start and end intersection (ASSETID)**
- ✅ **Preserve original MST implementations**
- ✅ **If no path exists, return an appropriate error message**

---

## New Functionality

| Algorithm | Method | Finds Full MST? | Finds MST for Specific Path? |
|-----------|--------|----------------|------------------------------|
| **Kruskal** | `kruskal_mst()` | ✅ Yes | ❌ No |
| **Kruskal** | `kruskal_mst_from(start, end)` | ❌ No | ✅ Yes |
| **Prim** | `prim_mst()` | ✅ Yes | ❌ No |
| **Prim** | `prim_mst_from(start, end)` | ❌ No | ✅ Yes |
| **Borůvka** | `boruvka_mst()` | ✅ Yes | ❌ No |
| **Borůvka** | `boruvka_mst_from(start, end)` | ❌ No | ✅ Yes |

---

## Usage

1. **Set CSV file paths**  
   Ensure you have the correct paths for `road_segments.csv` and `intersections.csv`.

2. **Modify the Start & End Intersection IDs**
   ```python
   start_asset_id = 142196  # Change this to your desired start ASSETID
   end_asset_id = 142195  # Change this to your desired end ASSETID
