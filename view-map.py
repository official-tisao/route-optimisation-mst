import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from geopy.distance import geodesic

# Step 1: Load the Intersection_Point feature class
intersections = gpd.read_file('Winter_Control_Data.gdb', layer='Intersection_Point')

# Step 2: Extract coordinates
intersections['longitude'] = intersections.geometry.x
intersections['latitude'] = intersections.geometry.y

# Step 3: Create a graph and add nodes
G = nx.Graph()
for idx, row in intersections.iterrows():
  G.add_node(idx, pos=(row['longitude'], row['latitude']))

# Step 4: Calculate pairwise distances
coords = list(zip(intersections['latitude'], intersections['longitude']))
distances = squareform(pdist(coords, lambda u, v: geodesic(u, v).meters))

# Step 5: Add weighted edges
for i in range(len(distances)):
  for j in range(i + 1, len(distances)):
    G.add_edge(i, j, weight=distances[i][j])

# Step 6: Compute the MST
mst = nx.minimum_spanning_tree(G, algorithm='kruskal')

# Step 7: Visualize the MST and save as SVG
pos = nx.get_node_attributes(G, 'pos')  # Get node positions

# Create a plot
plt.figure(figsize=(12, 8))

# Draw the full graph (optional, for context)
nx.draw(G, pos, with_labels=False, node_size=20, edge_color='gray', alpha=0.5)

# Draw the MST edges
nx.draw_networkx_edges(mst, pos, edge_color='red', width=2)

# Draw the nodes
nx.draw_networkx_nodes(mst, pos, node_size=50, node_color='blue')

# Add edge labels (weights)
edge_labels = nx.get_edge_attributes(mst, 'weight')
nx.draw_networkx_edge_labels(mst, pos, edge_labels=edge_labels, font_size=8, font_color='green')

# Save the plot as an SVG file
plt.savefig('mst_visualization.svg', format='svg')

# Show the plot (optional)
plt.show()
