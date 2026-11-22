# Routing with accessibility penalties


import csv
import heapq
from pathlib import Path


# Experimental accessibility penalties, measured in equivalent meters
DRAIN_GRATE_PENALTY = 20
GENTLE_SLOPE_PENALTY = 10
MODERATE_SLOPE_PENALTY = 30


def load_graph():
    """Build an undirected graph with accessibility information."""
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
                "stairs": row["stairs"].strip().lower(),
                "drain_grate": row["drain_grate_present"].strip().lower(),
                "slope": row["slope_level"].strip().lower()
            }

            graph.setdefault(start, []).append((end, edge))
            graph.setdefault(end, []).append((start, edge))

    return graph


def calculate_accessibility_cost(edge):
    """Calculate distance plus experimental accessibility penalties."""
    if edge["stairs"] == "yes":
        return float("inf")

    cost = edge["distance"]

    if edge["drain_grate"] == "yes":
        cost += DRAIN_GRATE_PENALTY

    if edge["slope"] == "gentle":
        cost += GENTLE_SLOPE_PENALTY
    elif edge["slope"] == "moderate":
        cost += MODERATE_SLOPE_PENALTY

    return cost


def dijkstra_accessibility(graph, start, destination):
    """Find a route minimizing distance and accessibility penalties."""
    if start not in graph or destination not in graph:
        return None

    costs = {node: float("inf") for node in graph}
    distances = {node: float("inf") for node in graph}
    previous = {}

    costs[start] = 0
    distances[start] = 0

    queue = [(0, start)]

    while queue:
        current_cost, current = heapq.heappop(queue)

        if current_cost > costs[current]:
            continue

        if current == destination:
            break

        for neighbor, edge in graph[current]:
            edge_cost = calculate_accessibility_cost(edge)

            if edge_cost == float("inf"):
                continue

            new_cost = current_cost + edge_cost

            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                distances[neighbor] = distances[current] + edge["distance"]
                previous[neighbor] = (current, edge["segment_id"])

                heapq.heappush(queue, (new_cost, neighbor))

    if costs[destination] == float("inf"):
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
        "distance_m": distances[destination],
        "routing_cost": costs[destination]
    }


if __name__ == "__main__":
    graph = load_graph()
    result = dijkstra_accessibility(graph, "N005", "N018")

    if result:
        print("Route:", " -> ".join(result["nodes"]))
        print("Segments:", " -> ".join(result["segments"]))
        print(f"Walking distance: {result['distance_m']:.2f} meters")
        print(f"Routing cost: {result['routing_cost']:.2f}")
    else:
        print("No accessible route found.")
