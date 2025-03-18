import requests
from bs4 import BeautifulSoup
import io

def decode_secret_message(doc_url):
  try:
    response = requests.get(doc_url)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    print(response.content)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the data (adjust if needed)
    table = soup.find('table')
    if table is None:
      raise ValueError("Table not found in the document.")

    data = []
    for row in table.find_all('tr'):
      cells = row.find_all('td')
      if cells:  # Skip empty rows
        x_axis = int(cells[0].text.strip())
        character = cells[1].text.strip()
        y_axis = int(cells[2].text.strip())
        data.append((x_axis, y_axis, character))

    # Determine grid dimensions
    max_x = max(x for x, _, _ in data)
    max_y = max(y for _, _, y in data)

    # Create the grid filled with spaces
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Populate the grid with characters from the document
    for x_axis, y_axis, character in data:
      grid[y_axis][x_axis] = character

    # Print the grid
    for row in grid:
      print(''.join(row))

  except requests.exceptions.RequestException as e:
    print(f"Error fetching URL: {e}")
  except ValueError as e:
    print(f"Error processing document: {e}")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Example usage (replace with the actual URL)
doc_url = "https://docs.google.com/document/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub"  # Or use the image URL you provided if it works directly.
decode_secret_message(doc_url)
