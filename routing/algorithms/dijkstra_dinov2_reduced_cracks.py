# # DINOv2-informed accessibility routing

import csv
import heapq
from pathlib import Path

# Existing experimental accessibility penalties (equivalent meters)
DRAIN_GRATE_PENALTY = 20
GENTLE_SLOPE_PENALTY = 10
MODERATE_SLOPE_PENALTY = 30

# DINOv2 category thresholds from the evaluation
CRACKS_THRESHOLD = 0.0259
BRICK_THRESHOLD = 0.5112

# Maximum experimental DINOv2 penalties (equivalent meters)
CRACKS_PENALTY = 15
BRICK_PENALTY = 20


ROOT = Path(__file__).resolve().parents[1]


def load_segment_predictions():
    """Load DINOv2 scores aggregated at the segment level."""
    csv_path = ROOT / "results" / "accessai_segment_predictions.csv"
    predictions = {}

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            segment_id = row["segment_id"].strip()

            predictions[segment_id] = {
                "cracks": float(row["score_cracks"] or 0),
                "brick_cobblestone": float(
                    row["score_brick_cobblestone"] or 0
                )
            }

    return predictions


def load_graph():
    """Build an undirected graph with manual and DINOv2 information."""
    segments_path = ROOT / "data" / "segments.csv"
    predictions = load_segment_predictions()
    graph = {}

    with open(
        segments_path, "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if not row["segment_id"].strip():
                continue

            start = row["start_point"].strip()
            end = row["end_point"].strip()
            segment_id = row["segment_id"].strip()

            edge = {
                "segment_id": segment_id,
                "distance": float(row["distance_m"]),
                "stairs": row["stairs"].strip().lower(),
                "drain_grate": row["drain_grate_present"].strip().lower(),
                "slope": row["slope_level"].strip().lower(),
                "cracks_score": predictions.get(
                    segment_id, {}
                ).get("cracks", 0),
                "brick_score": predictions.get(
                    segment_id, {}
                ).get("brick_cobblestone", 0)
            }

            graph.setdefault(start, []).append((end, edge))
            graph.setdefault(end, []).append((start, edge))

    return graph


def normalized_risk(score, threshold):
    """Gradually scale risk relative to the category threshold."""
    return score / (score + threshold) if score > 0 else 0


def calculate_accessibility_cost(edge):
    """Calculate distance plus manual and DINOv2 penalties."""
    if edge["stairs"] == "yes":
        return float("inf")

    cost = edge["distance"]

    if edge["drain_grate"] == "yes":
        cost += DRAIN_GRATE_PENALTY

    if edge["slope"] == "gentle":
        cost += GENTLE_SLOPE_PENALTY
    elif edge["slope"] == "moderate":
        cost += MODERATE_SLOPE_PENALTY

    cracks_risk = normalized_risk(
        edge["cracks_score"], CRACKS_THRESHOLD
    )
    brick_risk = normalized_risk(
        edge["brick_score"], BRICK_THRESHOLD
    )

    cost += cracks_risk * CRACKS_PENALTY
    cost += brick_risk * BRICK_PENALTY

    return cost


def print_penalty_breakdown(graph, route_segments):
    """Print the distance and penalty contributions for each route segment."""

    print("\nGRADUAL DINOv2 PENALTY BREAKDOWN")
    print("=" * 65)

    totals = {
        "distance": 0,
        "drain_grate": 0,
        "slope": 0,
        "cracks": 0,
        "brick_cobblestone": 0
    }

    for segment_id in route_segments:
        edge = next(
            edge
            for neighbors in graph.values()
            for neighbor, edge in neighbors
            if edge["segment_id"] == segment_id
        )

        distance = edge["distance"]
        grate = (
            DRAIN_GRATE_PENALTY
            if edge["drain_grate"] == "yes" else 0
        )

        if edge["slope"] == "gentle":
            slope = GENTLE_SLOPE_PENALTY
        elif edge["slope"] == "moderate":
            slope = MODERATE_SLOPE_PENALTY
        else:
            slope = 0

        cracks = (
            normalized_risk(
                edge["cracks_score"], CRACKS_THRESHOLD
            ) * CRACKS_PENALTY
        )

        brick = (
            normalized_risk(
                edge["brick_score"], BRICK_THRESHOLD
            ) * BRICK_PENALTY
        )

        segment_cost = distance + grate + slope + cracks + brick

        print(f"\nSegment: {segment_id}")
        print(f"  Distance: {distance:.2f}")
        print(f"  Drain grate penalty: {grate:.2f}")
        print(f"  Slope penalty: {slope:.2f}")
        print(f"  Cracks penalty: {cracks:.2f}")
        print(f"  Brick/cobblestone penalty: {brick:.2f}")
        print(f"  Total segment cost: {segment_cost:.2f}")

        totals["distance"] += distance
        totals["drain_grate"] += grate
        totals["slope"] += slope
        totals["cracks"] += cracks
        totals["brick_cobblestone"] += brick

    print("\nTOTAL CONTRIBUTIONS")
    print("-" * 65)

    for category, value in totals.items():
        print(f"{category.replace('_', ' ').title()}: {value:.2f}")

    print("-" * 65)
    print(f"Total routing cost: {sum(totals.values()):.2f}")



def dijkstra_dinov2(graph, start, destination):
    """Find a route minimizing distance and experimental risk penalties."""
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
                distances[neighbor] = (
                    distances[current] + edge["distance"]
                )
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
    result = dijkstra_dinov2(graph, "N005", "N018")

    if result:
        print("DINOv2-informed route")
        print("Nodes:", " -> ".join(result["nodes"]))
        print("Segments:", " -> ".join(result["segments"]))
        print(f"Walking distance: {result['distance_m']:.2f} meters")
        print(f"Routing cost: {result['routing_cost']:.2f}")
        print_penalty_breakdown(graph, result["segments"])
    else:
        print("No accessible route found.")