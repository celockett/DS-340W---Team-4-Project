from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "accessai_merged_scores_labels.csv"
OUT = ROOT / "results" / "accessai_plots"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

# Tags with enough positive and negative examples to make a useful plot.
tags = [
    "cracks",
    "utility_panel",
    "brick_cobblestone",
    "uneven_slanted",
]

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for tag in tags:
    y = df[tag].astype(int)
    scores = df[f"logit_{tag}"]
    if y.nunique() < 2:
        continue
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    ax.plot(fpr, tpr, label=f"{tag.replace('_', ' ')} (AUC={auc:.2f})")

ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate (Recall)",
    title="AccessAI Baseline: ROC Curves",
)
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "roc_curves.png", dpi=200)
plt.close(fig)

# Precision–Recall curves
fig, ax = plt.subplots(figsize=(7, 6))
for tag in tags:
    y = df[tag].astype(int)
    scores = df[f"logit_{tag}"]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    baseline = y.mean()
    ax.plot(
        recall, precision,
        label=f"{tag.replace('_', ' ')} (AP={ap:.2f})"
    )
    ax.axhline(baseline, linestyle=":", alpha=0.35)

ax.set(
    xlabel="Recall",
    ylabel="Precision",
    title="AccessAI Baseline: Precision–Recall Curves",
    xlim=(0, 1),
    ylim=(0, 1.02),
)
ax.legend(loc="best", fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "precision_recall_curves.png", dpi=200)
plt.close(fig)

# Per-tag AUC bar chart
rows = []
for tag in [
    "brick_cobblestone", "construction", "cracks",
    "height_difference", "narrow_sidewalk",
    "sand_gravel", "uneven_slanted",
    "utility_panel", "very_broken",
]:
    y = df[tag].astype(int)
    if y.nunique() < 2:
        continue
    rows.append((
        tag.replace("_", " "),
        int(y.sum()),
        roc_auc_score(y, df[f"logit_{tag}"])
    ))

metrics = pd.DataFrame(rows, columns=["tag", "positives", "auc"])
metrics = metrics.sort_values("auc")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(metrics["tag"], metrics["auc"])
ax.axvline(0.5, color="black", linestyle="--", label="Chance (0.50)")
for bar, count in zip(bars, metrics["positives"]):
    ax.text(
        bar.get_width() + 0.01,
        bar.get_y() + bar.get_height()/2,
        f"n={count}",
        va="center",
        fontsize=8,
    )
ax.set(
    xlabel="AUC",
    title="AccessAI Baseline Performance by Condition",
    xlim=(0, 1.12),
)
ax.legend()
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "auc_by_condition.png", dpi=200)
plt.close(fig)

print(f"Charts saved to: {OUT}")
for path in sorted(OUT.glob("*.png")):
    print(path.name)
