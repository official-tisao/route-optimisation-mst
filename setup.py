from setuptools import setup, find_packages

setup(
    name="your_project_name",  # Replace with your project name
    version="0.1.0",  # Replace with your project version
    packages=find_packages(),
    install_requires=[
        "geopandas",
        "networkx",
        "matplotlib",
        "scipy",
        "geopy",
        "numpy",
        # "cupy",
        "cupy-cuda12x"
    ],
    # Add other metadata here (optional)
    author="Saheed Oluwatosin Tiamiyu",
    author_email="tiamiyusaheedoluwatosin@gmail.com",
    description="Snow Route Optimization using Minimum Spanning Tree algorithms",
    license="MIT",  # Choose an appropriate license
    # ...
)
