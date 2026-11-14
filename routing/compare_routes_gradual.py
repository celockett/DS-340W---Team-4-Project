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
from dijkstra_dinov2_gradual import (
    load_graph as load_gradual_graph,
    dijkstra_dinov2 as dijkstra_dinov2_gradual
)
from dijkstra_dinov2_reduced_cracks import (
    load_graph as load_reduced_graph,
    dijkstra_dinov2 as dijkstra_reduced
)
from dijkstra_dinov2_reduced_both import (
    load_graph as load_reduced_both_graph,
    dijkstra_dinov2 as dijkstra_reduced_both
)


def compare_routes(start, destination):
    # Load each graph
    distance_graph = load_distance_graph()
    stairs_graph = load_stairs_graph()
    accessibility_graph = load_accessibility_graph()
    dinov2_graph = load_dinov2_graph()
    gradual_graph = load_gradual_graph()
    reduced_graph = load_reduced_graph()
    reduced_both_graph = load_reduced_both_graph()

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
        ),
        "5. DINOv2-Gradual": dijkstra_dinov2_gradual(
            gradual_graph, start, destination
        ),
        "6. DINOv2-Reduced Cracks": dijkstra_reduced(
            reduced_graph, start, destination
        ),
        "7. DINOv2-Reduced Both": dijkstra_reduced_both(
            reduced_both_graph, start, destination
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