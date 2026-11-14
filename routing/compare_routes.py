# Compare distances and segments across four routing implementations

from dijkstra import load_graph as load_distance_graph, dijkstra
from dijkstra_avoid_stairs import (
    load_graph as load_stairs_graph,
    dijkstra_avoid_stairs
)
from dijkstra_accessibility import (
    load_graph as load_accessibility_graph,
    dijkstra_accessibility
)
from dijkstra_dinov2 import (
    load_graph as load_dinov2_graph,
    dijkstra_dinov2
)


def compare_routes(start, destination):
    # Load each graph
    distance_graph = load_distance_graph()
    stairs_graph = load_stairs_graph()
    accessibility_graph = load_accessibility_graph()
    dinov2_graph = load_dinov2_graph()

    # Run all four routing methods
    results = {
        "1. Standard Dijkstra": dijkstra(
            distance_graph, start, destination
        ),
        "2. Avoid Stairs": dijkstra_avoid_stairs(
            stairs_graph, start, destination
        ),
        "3. Accessibility-Weighted": dijkstra_accessibility(
            accessibility_graph, start, destination
        ),
        "4. DINOv2-Informed": dijkstra_dinov2(
            dinov2_graph, start, destination
        )
    }

    print(f"\nROUTE COMPARISON: {start} -> {destination}")
    print("=" * 70)

    for method, result in results.items():
        print(f"\n{method}")

        if result is None:
            print("No route found.")
            continue

        print("Route:", " -> ".join(result["nodes"]))
        print("Segments:", " -> ".join(result["segments"]))
        print(f"Walking distance: {result['distance_m']:.2f} meters")

        routing_cost = result.get("routing_cost", result["distance_m"])
        print(f"Routing cost: {routing_cost:.2f}")


if __name__ == "__main__":
    compare_routes("N005", "N018")
    compare_routes("N001", "N020")
    compare_routes("N003", "N017")
    compare_routes("N008", "N012")
    compare_routes("N010", "N019")