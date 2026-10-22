from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
 
ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "results/accessai_model_predictions.csv"
LABELS = ROOT / "data/manual_labels.csv"
OUT_METRICS = ROOT / "results/accessai_tag_metrics.csv"
OUT_MERGED = ROOT / "results/accessai_merged_scores_labels.csv"
 
TAGS = [
    "brick_cobblestone", "bumpy", "construction", "cracks", "grass",
    "height_difference", "narrow_sidewalk", "rail_tram_track", "sand_gravel",
    "uneven_slanted", "utility_panel", "very_broken",
]
 
METRIC_KEYS = [
    "mean_score_pos", "mean_score_neg", "max_score",
    "tp@0.5", "fp@0.5", "fn@0.5", "precision@0.5", "recall@0.5",
    "auc", "avg_precision", "baseline_precision",
    "best_threshold", "best_threshold_logit", "best_f1",
    "precision@best", "recall@best",
]
 
 
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
 
 
def evaluate(name, y, prob, rank=None, rank_is_logit=False):
    """prob: probabilities (for @0.5 columns). rank: score used for ranking
    metrics (logits if available, otherwise the probabilities)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    rank = prob if rank is None else np.asarray(rank, dtype=float)
    n, n_pos = len(y), int(y.sum())
 
    row = {"tag": name, "n_photos": n, "n_positive": n_pos}
    row.update({k: np.nan for k in METRIC_KEYS})
 
    row["max_score"] = float(prob.max())
    if n_pos > 0:
        row["mean_score_pos"] = float(prob[y == 1].mean())
    if n_pos < n:
        row["mean_score_neg"] = float(prob[y == 0].mean())
 
    # Performance at the default 0.5 cutoff (always on probabilities)
    pred = prob >= 0.5
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    row["tp@0.5"], row["fp@0.5"], row["fn@0.5"] = tp, fp, fn
    if tp + fp > 0:
        row["precision@0.5"] = tp / (tp + fp)
    if tp + fn > 0:
        row["recall@0.5"] = tp / (tp + fn)
 
    # Threshold-free metrics need both classes present
    if 0 < n_pos < n:
        row["auc"] = roc_auc_score(y, rank)
        row["avg_precision"] = average_precision_score(y, rank)
        row["baseline_precision"] = n_pos / n
        p, r, t = precision_recall_curve(y, rank)
        f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
        k = int(np.argmax(f1))
        if rank_is_logit:
            row["best_threshold_logit"] = float(t[k])
            row["best_threshold"] = float(sigmoid(t[k]))
        else:
            row["best_threshold"] = float(t[k])
        row["best_f1"] = float(f1[k])
        row["precision@best"] = float(p[k])
        row["recall@best"] = float(r[k])
    return row
 
 
def main():
    pred = pd.read_csv(PRED)
    labels = pd.read_csv(LABELS)
 
    df = labels.merge(pred, on="photo_id", how="inner", suffixes=("", "_pred"))
    print(f"labels: {len(labels)} rows | predictions: {len(pred)} rows | matched: {len(df)}")
    missing = set(labels["photo_id"]) - set(pred["photo_id"])
    if missing:
        print("WARNING: labeled photos with no prediction:", sorted(missing))
    if pred["photo_id"].duplicated().any():
        print("WARNING: duplicate photo_id in predictions:",
              sorted(pred.loc[pred["photo_id"].duplicated(), "photo_id"]))
 
    has_logits = all(f"logit_{t}" in df.columns for t in TAGS)
    print("Ranking metrics use:", "logits" if has_logits else "probabilities (no logit_ columns found)")
 
    def rank_for(tag):
        return df[f"logit_{tag}"] if has_logits else None
 
    rows = [
        evaluate(t, df[t], df[f"score_{t}"], rank_for(t), has_logits)
        for t in TAGS
    ]
 
    # Extra check for your campus photos: round manhole covers were labeled
    # utility_panel by you, but the model may express them as uncovered_manhole.
    um_ok = "score_uncovered_manhole" in df.columns
    if um_ok:
        um_logit = "logit_uncovered_manhole" in df.columns and has_logits
        rows.append(evaluate(
            "utility_panel [uncovered_manhole score]",
            df["utility_panel"], df["score_uncovered_manhole"],
            df["logit_uncovered_manhole"] if um_logit else None, um_logit,
        ))
        both_prob = df[["score_utility_panel", "score_uncovered_manhole"]].max(axis=1)
        both_rank = (df[["logit_utility_panel", "logit_uncovered_manhole"]].max(axis=1)
                     if um_logit else None)
        rows.append(evaluate(
            "utility_panel [max of both scores]",
            df["utility_panel"], both_prob, both_rank, um_logit,
        ))
 
    # Grate check: does the model's utility_panel / uncovered_manhole output
    # line up with your separate "grate" label (rectangular slotted fixtures)?
    if "grate" in df.columns:
        g = df["grate"].astype(int)
        up = df["utility_panel"].astype(int)
        either = ((g + up) > 0).astype(int)
        print(f"grate positives: {int(g.sum())} | utility_panel positives: {int(up.sum())} "
              f"| both: {int(((g == 1) & (up == 1)).sum())} | either: {int(either.sum())}")
        um_logit2 = um_ok and has_logits and "logit_uncovered_manhole" in df.columns
        for name, y in [("grate", g), ("utility_panel OR grate", either)]:
            rows.append(evaluate(
                f"{name} [utility_panel score]", y,
                df["score_utility_panel"], rank_for("utility_panel"), has_logits,
            ))
            if um_ok:
                rows.append(evaluate(
                    f"{name} [uncovered_manhole score]", y,
                    df["score_uncovered_manhole"],
                    df["logit_uncovered_manhole"] if um_logit2 else None, um_logit2,
                ))
 
    metrics = pd.DataFrame(rows)
 
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_METRICS, index=False)
    df.to_csv(OUT_MERGED, index=False)
 
    show = ["tag", "n_positive", "auc", "avg_precision", "baseline_precision",
            "precision@0.5", "recall@0.5", "best_threshold", "best_f1"]
    print()
    print(metrics[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print()
    print("Notes:")
    print(" - auc: 0.5 = no better than chance, 1.0 = perfect ranking.")
    print(" - avg_precision should be compared to baseline_precision (the share of positives).")
    print(" - best_threshold / best_f1 are chosen on these same photos, so they are")
    print("   optimistic. With few positives, treat them as rough guides only.")
    print(" - Tags with no positives (or all positives) have no auc / avg_precision.")
    print("Saved:", OUT_METRICS)
    print("Saved:", OUT_MERGED)
 
 
if __name__ == "__main__":
    main()


 