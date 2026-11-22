
import csv
import sys
from pathlib import Path
from itertools import permutations
from collections import Counter

# Project paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from routing.algorithms.dijkstra_dinov2_reduced_both import (
    load_graph,
    dijkstra_dinov2,
)

ENTRANCES_FILE = ROOT / "data" / "entrances.csv"
NODES_FILE = ROOT / "data" / "nodes.csv"
SEGMENTS_FILE = ROOT / "data" / "segments.csv"
REPORT_FILE = ROOT / "results" / "all_route_tests.csv"

# All six mobility scenarios
SCENARIOS = [
    ("Walking cane", True),
    ("Walking cane", False),
    ("Walker", False),
    ("Mobility scooter", False),
    ("Manual wheelchair", False),
    ("Motorized wheelchair", False),
]


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main():
    print("Loading entrance and node data...")

    entrances = load_csv(ENTRANCES_FILE)
    nodes = load_csv(NODES_FILE)
    segments = load_csv(SEGMENTS_FILE)

    # Map entrance IDs to their routing node IDs
    entrance_nodes = {
        row["entrance_id"]: row["node_id"]
        for row in nodes
        if row["node_type"].strip().lower() == "entrance"
        and row["entrance_id"].strip().upper() != "N/A"
    }

    # Keep only entrances that have a corresponding routing node
    valid_entrances = [
        entrance for entrance in entrances
        if entrance["entrance_id"] in entrance_nodes
    ]

    unmapped = [
        entrance["entrance_id"] for entrance in entrances
        if entrance["entrance_id"] not in entrance_nodes
    ]

    print(f"Entrances found: {len(entrances)}")
    print(f"Entrances with routing nodes: {len(valid_entrances)}")

    if unmapped:
        print("WARNING - Entrances without routing nodes:", unmapped)

    pairs = list(permutations(valid_entrances, 2))
    total_tests = len(pairs) * len(SCENARIOS)

    print(f"Ordered entrance pairs: {len(pairs)}")
    print(f"Mobility scenarios: {len(SCENARIOS)}")
    print(f"Total route tests: {total_tests}")
    print("\nLoading sidewalk graph...")

    graph = load_graph()

    # Segment lookup for route validation
    segment_lookup = {
        row["segment_id"]: row for row in segments
    }

    results = []
    segment_usage = Counter()
    successful = 0
    unsuccessful = 0
    restriction_violations = 0
    errors = 0

    print("\nStarting route tests...\n")

    for mobility_profile, use_stairs in SCENARIOS:
        scenario_name = (
            f"{mobility_profile} - "
            f"{'stairs allowed' if use_stairs else 'stairs avoided'}"
            if mobility_profile == "Walking cane"
            else mobility_profile
        )

        print(f"Testing: {scenario_name}")

        for start, destination in pairs:
            start_id = start["entrance_id"]
            destination_id = destination["entrance_id"]

            start_node = entrance_nodes[start_id]
            destination_node = entrance_nodes[destination_id]

            try:
                route = dijkstra_dinov2(
                    graph,
                    start_node,
                    destination_node,
                    mobility_profile,
                    use_stairs=use_stairs,
                )

                if route is None:
                    unsuccessful += 1
                    results.append({
                        "scenario": scenario_name,
                        "start_entrance": start_id,
                        "start_building": start["building_id"],
                        "destination_entrance": destination_id,
                        "destination_building": destination["building_id"],
                        "status": "No route",
                        "distance_m": "",
                        "routing_cost": "",
                        "segments": "",
                        "restriction_violation": "",
                        "error": "",
                    })
                    continue

                successful += 1
                route_segments = route["segments"]
                violation = False

                for segment_id in route_segments:
                    segment_usage[segment_id] += 1
                    segment = segment_lookup.get(segment_id)

                    if segment is None:
                        continue

                    has_stairs = segment["stairs"].strip().lower() in (
                        "yes", "true", "1"
                    )

                    if has_stairs and (
                        mobility_profile != "Walking cane" or not use_stairs
                    ):
                        violation = True

                if violation:
                    restriction_violations += 1

                results.append({
                    "scenario": scenario_name,
                    "start_entrance": start_id,
                    "start_building": start["building_id"],
                    "destination_entrance": destination_id,
                    "destination_building": destination["building_id"],
                    "status": "Route found",
                    "distance_m": round(route["distance_m"], 2),
                    "routing_cost": round(route["routing_cost"], 2),
                    "segments": " -> ".join(route_segments),
                    "restriction_violation": violation,
                    "error": "",
                })

            except Exception as error:
                errors += 1
                results.append({
                    "scenario": scenario_name,
                    "start_entrance": start_id,
                    "start_building": start["building_id"],
                    "destination_entrance": destination_id,
                    "destination_building": destination["building_id"],
                    "status": "Error",
                    "distance_m": "",
                    "routing_cost": "",
                    "segments": "",
                    "restriction_violation": "",
                    "error": str(error),
                })

    # Save detailed results
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print("\n" + "=" * 55)
    print("ROUTE TEST SUMMARY")
    print("=" * 55)
    print(f"Total tests:             {total_tests}")
    print(f"Routes found:            {successful}")
    print(f"No route found:          {unsuccessful}")
    print(f"Errors:                  {errors}")
    print(f"Restriction violations:  {restriction_violations}")
    print(f"Unique segments used:    {len(segment_usage)}")
    print(f"Total segments:          {len(segments)}")

    unused_segments = [
        row["segment_id"] for row in segments
        if row["segment_id"] not in segment_usage
    ]

    print(f"\nSegments never used: {len(unused_segments)}")
    if unused_segments:
        print(", ".join(unused_segments))

    print(f"\nDetailed report saved to:\n{REPORT_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    main()
