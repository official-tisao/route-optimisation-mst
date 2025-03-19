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
    OR (if above command does not work)

   ```bash
    .venv\Scripts\Activate.ps1
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
