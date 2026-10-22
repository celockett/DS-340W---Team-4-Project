import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

manual = pd.read_csv(ROOT / "data/manual_labels.csv", na_values=["NA", ""])
preds = pd.read_csv(ROOT / "results/accessai_model_predictions.csv")

data = manual.merge(
    preds, on=["photo_id", "photo_filename"], validate="one_to_one"
)
assert len(data) == len(manual) == len(preds), "Photo mismatch!"


tags = [
    "brick_cobblestone", "bumpy", "construction", "cracks",
    "grass", "height_difference", "narrow_sidewalk",
    "rail_tram_track", "sand_gravel", "uneven_slanted",
    "utility_panel", "very_broken"
]

results = []

for tag in tags:
    actual = pd.to_numeric(data[tag], errors="coerce")
    predicted = data[f"pred_{tag}"]
    valid = actual.isin([0, 1]) & predicted.isin([0, 1])
    y, p = actual[valid].astype(int), predicted[valid].astype(int)

    tp = ((y == 1) & (p == 1)).sum()
    fp = ((y == 0) & (p == 1)).sum()
    fn = ((y == 1) & (p == 0)).sum()

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * tp / (2 * tp + fp + fn)
          if 2 * tp + fp + fn else float("nan"))

    results.append({
        "condition": tag,
        "photos_checked": len(y),
        "manual_positive": int(y.sum()),
        "model_positive": int(p.sum()),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

report = pd.DataFrame(results)
report.to_csv(ROOT / "results/accessai_evaluation.csv", index=False)
print(report.round(3).to_string(index=False))
print("\nSaved to results/accessai_evaluation.csv")

