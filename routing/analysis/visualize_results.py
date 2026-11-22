from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
# -----------------------------------
# 1. N005 -> N018: Seven approaches
# -----------------------------------

results = {
    "Approach": [
        "Standard",
        "Avoid Stairs",
        "Accessibility Weighted",
        "DINOv2",
        "DINOv2 Gradual",
        "Reduced Cracks",
        "Reduced Both"
    ],
    "Distance (m)": [
        209.09, 452.02, 474.57, 460.86,
        460.86, 460.86, 460.86
    ],
    "Routing Cost": [
        209.09, 452.02, 544.57, 671.10,
        662.53, 606.81, 606.69
    ]
}

df = pd.DataFrame(results)

print("\nN005 -> N018 RESULTS")
print(df.to_string(index=False))

# Save table
df.to_csv(RESULTS_DIR / "N005_N018_results.csv", index=False)

# -----------------------------------
# 2. Gradual vs. Original DINOv2
# -----------------------------------

gradual_results = {
    "Route": [
        "N005 -> N018",
        "N001 -> N020",
        "N008 -> N012"
    ],
    "DINOv2": [671.10, 303.61, 454.83],
    "Gradual": [662.53, 300.51, 444.07]
}

gradual_df = pd.DataFrame(gradual_results)

print("\nDINOv2 VS. GRADUAL SCORING")
print(gradual_df.to_string(index=False))

# Save table
gradual_df.to_csv(RESULTS_DIR / "gradual_comparison.csv", index=False)

# Create bar graph
ax = gradual_df.set_index("Route")[["DINOv2", "Gradual"]].plot(
    kind="bar",
    figsize=(9, 5),
    width=0.75
)

ax.set_title("DINOv2 vs. Gradual Scoring")
ax.set_xlabel("Origin and Destination Pair")
ax.set_ylabel("Routing Cost")
ax.legend(title="Approach")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "gradual_comparison.png", dpi=300)
plt.show()

# -----------------------------------
# 3. Route availability
# -----------------------------------

availability = pd.DataFrame({
    "Approach": [
        "Standard",
        "Avoid Stairs",
        "Accessibility Weighted",
        "DINOv2",
        "DINOv2 Gradual",
        "Reduced Cracks",
        "Reduced Both"
    ],
    "Routes Found": [5, 3, 3, 3, 3, 3, 3],
    "Total Tested": [5, 5, 5, 5, 5, 5, 5]
})

print("\nROUTE AVAILABILITY")
print(availability.to_string(index=False))

# Save table
availability.to_csv(RESULTS_DIR / "route_availability.csv", index=False)

# Create availability graph
plt.figure(figsize=(9, 5))
plt.bar(
    availability["Approach"],
    availability["Routes Found"]
)

plt.title("Route Availability Across Five Test Pairs")
plt.xlabel("Routing Approach")
plt.ylabel("Number of Routes Found")
plt.ylim(0, 5.5)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "route_availability.png", dpi=300)
plt.show()
