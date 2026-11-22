# # DINOv2-informed accessibility routing

import csv
import heapq
from pathlib import Path

# Existing experimental accessibility penalties (equivalent meters)
DRAIN_GRATE_PENALTY = 20
GENTLE_SLOPE_PENALTY = 10
MODERATE_SLOPE_PENALTY = 30


ROOT = Path(__file__).resolve().parents[2]


def load_segment_predictions():
    """Load the seven strong-match DINOv2 scores for each segment."""
    csv_path = ROOT / "results" / "accessai_segment_predictions.csv"
    predictions = {}

    categories = [
        "brick_cobblestone",
        "bumpy",
        "cracks",
        "height_difference",
        "sand_gravel",
        "uneven_slanted",
        "very_broken"
    ]

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            segment_id = row["segment_id"].strip()

            if not segment_id:
                continue

            predictions[segment_id] = {
                category: float(row[f"score_{category}"] or 0)
                for category in categories
            }

    return predictions


def load_graph():
    """Build the sidewalk graph and attach manual attributes and DINOv2 scores."""
    graph = {}

    predictions = load_segment_predictions()
    segments_path = ROOT / "data" / "segments.csv"

    categories = [
        "brick_cobblestone",
        "bumpy",
        "cracks",
        "height_difference",
        "sand_gravel",
        "uneven_slanted",
        "very_broken",
    ]

    with open(segments_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            segment_id = row["segment_id"].strip()
            start = row["start_point"].strip()
            end = row["end_point"].strip()

            prediction = predictions.get(segment_id, {})

            edge = {
                "segment_id": segment_id,
                "distance": float(row["distance_m"]),
                "stairs": row["stairs"],
                "drain_grate": row["drain_grate_present"],
                "slope": row["slope_level"],
                "surface": row["surface"],
                "obstacle": row["obstacle"],
            }

            for category in categories:
                edge[f"{category}_score"] = prediction.get(category, 0.0)

            graph.setdefault(start, []).append((end, edge))
            graph.setdefault(end, []).append((start, edge))

    return graph



def calculate_accessibility_cost(edge, mobility_profile, use_stairs=False):
    """Calculate personalized routing cost using research-derived surface weights."""

    # Apply profile-specific stair restrictions
    stairs_present = (
        str(edge["stairs"]).strip().lower() in ("yes", "true", "1")
    )

    if stairs_present and (
        mobility_profile != "Walking cane" or not use_stairs
    ):
        return float("inf")
    elif stairs_present:
        stair_penalty = 50
    else:
        stair_penalty = 0

    # Research-derived weights for each mobility profile
    surface_weights = {
        "Walking cane": 0.413,
        "Walker": 0.551,
        "Mobility scooter": 0.603,
        "Manual wheelchair": 0.457,
        "Motorized wheelchair": 0.505,
    }

    if mobility_profile not in surface_weights:
        raise ValueError(f"Unknown mobility profile: {mobility_profile}")

    # Seven strong-match DINOv2 categories
    categories = [
        "brick_cobblestone",
        "bumpy",
        "cracks",
        "height_difference",
        "sand_gravel",
        "uneven_slanted",
        "very_broken",
    ]

    # Combine the model scores for the segment
    surface_score = sum(
        float(edge.get(f"{category}_score", 0.0) or 0.0)
        for category in categories
    )

    # Apply the research-derived surface penalty
    weight = surface_weights[mobility_profile]
    surface_penalty = 0.10 * weight * surface_score

    # Preserve the existing manual penalties
    drain_grate_penalty = (
        DRAIN_GRATE_PENALTY
        if str(edge["drain_grate"]).strip().lower() in ("yes", "true", "1")
        else 0
    )

    slope = str(edge["slope"]).strip().lower()

    if slope == "gentle":
        slope_penalty = GENTLE_SLOPE_PENALTY
    elif slope == "moderate":
        slope_penalty = MODERATE_SLOPE_PENALTY
    else:
        slope_penalty = 0

    return (
        edge["distance"]
        + drain_grate_penalty
        + slope_penalty
        + stair_penalty
        + edge["distance"] * surface_penalty
    )



def print_penalty_breakdown(graph, route_segments, mobility_profile, use_stairs = False):
    """Print distance and accessibility penalties for each route segment."""

    categories = [
        "brick_cobblestone",
        "bumpy",
        "cracks",
        "height_difference",
        "sand_gravel",
        "uneven_slanted",
        "very_broken",
    ]

    surface_weights = {
        "Walking cane": 0.413,
        "Walker": 0.551,
        "Mobility scooter": 0.603,
        "Manual wheelchair": 0.457,
        "Motorized wheelchair": 0.505,
    }

    weight = surface_weights[mobility_profile]

    totals = {
        "distance": 0,
        "drain_grate": 0,
        "slope": 0,
        "stairs": 0,
        "surface": 0,
    }

    print(f"\nPERSONALIZED ROUTE BREAKDOWN — {mobility_profile}")
    print("=" * 65)

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
            if str(edge["drain_grate"]).strip().lower() in ("yes", "true", "1")
            else 0
        )

        slope_level = str(edge["slope"]).strip().lower()

        stairs_present = (
            str(edge["stairs"]).strip().lower() in ("yes", "true", "1")
        )

        if stairs_present and mobility_profile == "Walking cane" and use_stairs:
            stair_penalty = 50
        else:
            stair_penalty = 0

        if slope_level == "gentle":
            slope = GENTLE_SLOPE_PENALTY
        elif slope_level == "moderate":
            slope = MODERATE_SLOPE_PENALTY
        else:
            slope = 0

        surface_score = sum(
            float(edge.get(f"{category}_score", 0) or 0)
            for category in categories
        )

        surface_penalty = distance * 0.10 * weight * surface_score

        segment_cost = (
            distance + grate + slope + stair_penalty + surface_penalty
        )
        print(f"\nSegment: {segment_id}")
        print(f"  Distance:          {distance:.2f} m")
        print(f"  Drain grate:       {grate:.2f}")
        print(f"  Slope:              {slope:.2f}")
        print(f"  Stairs:             {stair_penalty:.2f}")
        print(f"  Combined surface:  {surface_penalty:.2f}")
        print(f"  Routing cost:      {segment_cost:.2f}")

        totals["distance"] += distance
        totals["drain_grate"] += grate
        totals["slope"] += slope
        totals["stairs"] += stair_penalty
        totals["surface"] += surface_penalty

    print("\nTOTAL CONTRIBUTIONS")
    print("-" * 65)

    for category, value in totals.items():
        print(f"{category.replace('_', ' ').title()}: {value:.2f}")

    print("-" * 65)
    print(f"Total routing cost: {sum(totals.values()):.2f}")



def dijkstra_dinov2(graph, start, destination, mobility_profile, use_stairs = False):
    """Find the lowest-cost route for a selected mobility profile."""

    distances = {node: float("inf") for node in graph}
    distances[start] = 0

    previous = {}
    priority_queue = [(0, start)]

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)

        if current_cost > distances[current_node]:
            continue

        if current_node == destination:
            break

        for neighbor, edge in graph.get(current_node, []):
            edge_cost = calculate_accessibility_cost(edge, mobility_profile, use_stairs)
            new_cost = current_cost + edge_cost

            if new_cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_cost
                previous[neighbor] = (current_node, edge)
                heapq.heappush(priority_queue, (new_cost, neighbor))

    if distances.get(destination, float("inf")) == float("inf"):
        return None

    # Reconstruct the route
    nodes = [destination]
    route_edges = []
    current = destination

    while current != start:
        previous_node, edge = previous[current]
        route_edges.append(edge)
        nodes.append(previous_node)
        current = previous_node

    nodes.reverse()
    route_edges.reverse()

    route_segments = [edge["segment_id"] for edge in route_edges]
    distance_m = sum(edge["distance"] for edge in route_edges)

    return {
        "nodes": nodes,
        "segments": route_segments,
        "distance_m": distance_m,
        "routing_cost": distances[destination],
    }


if __name__ == "__main__":
    graph = load_graph()
    start = "N005"
    destination = "N018"
    mobility_profile = "Walking cane"
    use_stairs = True

    result = dijkstra_dinov2(
        graph,
        start,
        destination,
        mobility_profile,
        use_stairs
        )
    if result:
        print("DINOv2-informed route")
        print("Nodes:", " -> ".join(result["nodes"]))
        print("Segments:", " -> ".join(result["segments"]))
        print(f"Walking distance: {result['distance_m']:.2f} meters")
        print(f"Routing cost: {result['routing_cost']:.2f}")
        print_penalty_breakdown(graph, result["segments"], mobility_profile, use_stairs)
    else:
        print("No accessible route found.")