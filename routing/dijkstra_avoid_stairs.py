# Shortest route while excluding stairs


import csv
import heapq
from pathlib import Path


def load_graph():
    """Build an undirected graph from the sidewalk segments."""
    csv_path = Path(__file__).resolve().parents[1] / "data" / "segments.csv"
    graph = {}

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if not row["segment_id"].strip():
                continue

            start = row["start_point"].strip()
            end = row["end_point"].strip()

            edge = {
                "segment_id": row["segment_id"].strip(),
                "distance": float(row["distance_m"]),
                "stairs": row["stairs"].strip()
            }

            graph.setdefault(start, []).append((end, edge))
            graph.setdefault(end, []).append((start, edge))

    return graph


def dijkstra_avoid_stairs(graph, start, destination):
    """Find the shortest route without using segments containing stairs."""
    if start not in graph or destination not in graph:
        return None

    distances = {node: float("inf") for node in graph}
    previous = {}
    distances[start] = 0

    queue = [(0, start)]

    while queue:
        current_distance, current = heapq.heappop(queue)

        if current_distance > distances[current]:
            continue

        if current == destination:
            break

        for neighbor, edge in graph[current]:

            # Skip sidewalk segments containing stairs
            if edge["stairs"].strip().lower() == "yes":
                continue

            new_distance = current_distance + edge["distance"]

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = (current, edge["segment_id"])
                heapq.heappush(queue, (new_distance, neighbor))

    if distances[destination] == float("inf"):
        return None

    nodes = [destination]
    segments = []
    current = destination

    while current != start:
        prior_node, segment_id = previous[current]
        nodes.append(prior_node)
        segments.append(segment_id)
        current = prior_node

    return {
        "nodes": nodes[::-1],
        "segments": segments[::-1],
        "distance_m": distances[destination]
    }


if __name__ == "__main__":
    graph = load_graph()
    result = dijkstra_avoid_stairs(graph, "N005", "N018")

    if result:
        print("Route:", " -> ".join(result["nodes"]))
        print("Segments:", " -> ".join(result["segments"]))
        print(f"Distance: {result['distance_m']:.2f} meters")
    else:
        print("No accessible route found.")
