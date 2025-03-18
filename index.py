import sys
import geopandas as gpd
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from geopy.distance import geodesic
from pyproj import Proj, Transformer

# Define UTM Zone (e.g., UTM Zone 17N for Ontario, Canada)
utm_proj = Proj(proj="utm", zone=17, ellps="WGS84", south=False)
wgs84_proj = Proj(proj="latlong", datum="WGS84")

# Convert coordinates
transformer = Transformer.from_proj(utm_proj, wgs84_proj)


# Load the Intersection_Point feature class
intersections = gpd.read_file('Sudbury_Data/Winter_Control_Data.gdb', layer='Intersection_Point')


# Convert the geometry column to latitude and longitude
intersections['longitude'] = intersections.geometry.x
intersections['latitude'] = intersections.geometry.y

# Create an empty graph
G = nx.Graph()

intersections[['longitude', 'latitude']] = intersections.apply(
  lambda row: transformer.transform(row['longitude'], row['latitude']), axis=1, result_type="expand"
)

# Inspect the data
print(intersections.head())

for idx, row in intersections.iterrows():
  G.add_node(idx, pos=(row['longitude'], row['latitude']))

# Check for NaN values
if intersections[['latitude', 'longitude']].isnull().values.any():
  raise ValueError("Latitude or Longitude contains NaN values. Please clean the data.")

# Check for out-of-range latitude/longitude
invalid_lat = intersections[(intersections['latitude'] < -90) | (intersections['latitude'] > 90)]
invalid_lon = intersections[(intersections['longitude'] < -180) | (intersections['longitude'] > 180)]
if not invalid_lat.empty or not invalid_lon.empty:
  raise ValueError(f"Invalid latitude/longitude values detected:\n{invalid_lat}\n{invalid_lon}")

# Extract coordinates as a list of tuples
coords = list(zip(intersections['latitude'], intersections['longitude']))

distances = squareform(pdist(coords, lambda u, v: geodesic(u, v).meters))
# Calculate the pairwise distances
# Swap the order to (latitude, longitude) inside pdist()
# distances = squareform(pdist(coords, lambda u, v: geodesic((u[0], u[1]), (v[0], v[1])).meters))




for i in range(len(distances)):
  for j in range(i + 1, len(distances)):
    G.add_edge(i, j, weight=distances[i][j])

sys.exit(0);
