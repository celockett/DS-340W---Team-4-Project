
import csv
from pathlib import Path
from collections import defaultdict, deque


def load_graph(avoid_stairs=False):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "segments.csv"
    graph = defaultdict(list)

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if not row["segment_id"].strip():
                continue

            start = row["start_point"].strip()
            end = row["end_point"].strip()
            stairs = row["stairs"].strip().lower()

            if avoid_stairs and stairs == "yes":
                continue

            graph[start].append(end)
            graph[end].append(start)

    return graph


def find_components(graph):
    visited = set()
    components = []

    for node in graph:
        if node in visited:
            continue

        component = []
        queue = deque([node])
        visited.add(node)

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(sorted(component))

    return components


if __name__ == "__main__":
    original_graph = load_graph()
    no_stairs_graph = load_graph(avoid_stairs=True)

    original_components = find_components(original_graph)
    accessible_components = find_components(no_stairs_graph)

    print("NETWORK CONNECTIVITY ANALYSIS")
    print("=" * 45)

    print(f"\nOriginal network nodes: {len(original_graph)}")
    print(f"Original connected components: {len(original_components)}")

    print(f"\nNetwork after excluding stairs:")
    print(f"Remaining nodes: {len(no_stairs_graph)}")
    print(f"Connected components: {len(accessible_components)}")

    for i, component in enumerate(
        sorted(accessible_components, key=len, reverse=True), start=1
    ):
        print(f"\nComponent {i} ({len(component)} nodes):")
        print(", ".join(component))

    isolated_nodes = [
        node for node, neighbors in no_stairs_graph.items()
        if len(neighbors) == 0
    ]

    print("\nNodes with no remaining connections:")
    print(", ".join(sorted(isolated_nodes)) if isolated_nodes else "None")
