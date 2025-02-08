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
        "geopy"
    ],
    # Add other metadata here (optional)
    author="Your Name",
    author_email="your_email@example.com",
    description="A brief description of your project",
    license="MIT",  # Choose an appropriate license
    # ...
)
