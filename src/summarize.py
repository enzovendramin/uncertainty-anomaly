"""
Turn a results csv into the summary table and one plot.

Usage:
    python src/summarize.py results/baseline_representative.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(path: Path) -> None:
    df = pd.read_csv(path)
    q = df["q"].iloc[0]

    # average over seeds first, then over datasets, so every dataset counts once
    per_dataset = df.groupby(["bench", "dataset", "score"]).mean(numeric_only=True).reset_index()

    cols = ["auroc", "auprc", "precision_at_n_anom",
            "ebh_n_flagged", "ebh_fdp", "ebh_recall", "normal_false_alarm_rate_at_q"]
    table = per_dataset.groupby(["bench", "score"])[cols].mean().round(3)
    table["n_datasets"] = per_dataset.groupby(["bench", "score"]).size()
    table["frac_datasets_with_alerts"] = (
        per_dataset.assign(any=per_dataset["ebh_n_flagged"] > 0)
        .groupby(["bench", "score"])["any"].mean().round(2)
    )
    print(f"\n=== {path.name}   (q = {q}; means over datasets, seeds averaged first) ===")
    print(table.to_string())

    # the two guarantees, dataset by dataset
    print("\nGuarantee checks (should hold on average):")
    for score, g in per_dataset.groupby("score"):
        with_alerts = g[g["ebh_n_flagged"] > 0]
        print(f"  {score:8s} false-alarm rate among normals at p<=q: mean {g['normal_false_alarm_rate_at_q'].mean():.3f} "
              f"(max {g['normal_false_alarm_rate_at_q'].max():.3f}) ; "
              f"e-BH realised FDP where it flagged something: mean {with_alerts['ebh_fdp'].mean():.3f} "
              f"over {len(with_alerts)} datasets, {(with_alerts['ebh_fdp'] > q).mean():.0%} of them above q")

    # plot: AUROC vs e-BH recall, one point per dataset
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, label in [(axes[0], "ebh_recall", "e-BH recall (fraction of anomalies flagged)"),
                              (axes[1], "ebh_fdp", "e-BH realised false-alarm fraction")]:
        for score, g in per_dataset.groupby("score"):
            ax.scatter(g["auroc"], g[metric], s=18, alpha=0.7, label=score)
        ax.set_xlabel("AUROC of the score (Q1)")
        ax.set_ylabel(label)
        ax.axvline(0.5, color="grey", lw=0.8, ls=":")
        ax.grid(alpha=0.3)
    axes[1].axhline(q, color="red", lw=1, ls="--", label=f"target q = {q}")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(f"{path.stem}: {per_dataset['dataset'].nunique()} datasets")
    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"\nplot: {out}")


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
