"""
Q2: at a fixed review budget, which score catches more real anomalies?

Runs offline from the per-row scores saved by run.py (results/<tag>/scores/).
kNN is recomputed here on exactly the same rows (same split indices), so all
arms are compared on identical calibration and test rows.

Arms (fixed in advance, no tuning on test labels):
    knn        distance to the nearest known-normal rows          (baseline)
    std        TabPFN regressor predictive std                     (uncertainty)
    nll        TabPFN regressor negative log-density               (error)
    knn+std, knn+nll, std+nll, knn+std+nll
               combinations: each part becomes a conformal p-value against its
               own calibration scores and the parts are merged as the mean of
               -log p (equal weights, fixed in advance). Scale-free and
               label-free; see arms.combined_score.

Budgets: k = number of anomalies in the test set, and k = 1%, 5%, 10% of the
test set (at least 1). Metrics: precision@k, AUPRC (baseline = anomaly rate).

Usage:
    python src/q2.py --tag tabpfnreg_representative
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

from arms import ARMS, arm_scores, load_records
from run import RESULTS_DIR, git_commit, precision_at_k


def load_rows(tag: str, seed: int):
    """One record per dataset: test labels and per-row test scores for every arm."""
    for rec in load_records(tag, seed):
        rng = np.random.default_rng(seed)
        test_scores = {arm: arm_scores(rec, arm, rng)[1] for arm in ARMS}
        yield rec.bench, rec.dataset, rec.y, test_scores, rec.n_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tabpfnreg_representative")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for bench, name, y, scores, n_features in load_rows(args.tag, args.seed):
        n, n_anom = len(y), int(y.sum())
        if n_anom == 0 or n_anom == n:
            continue
        budgets = {"n_anom": n_anom, "1pct": max(1, n // 100), "5pct": max(1, n // 20), "10pct": max(1, n // 10)}
        for arm in ARMS:
            s = scores[arm]
            row = {"bench": bench, "dataset": name, "arm": arm, "n_test": n, "n_anomalies": n_anom,
                   "anomaly_rate": n_anom / n, "n_features": n_features,
                   "auroc": roc_auc_score(y, s), "auprc": average_precision_score(y, s)}
            for b, k in budgets.items():
                row[f"prec@{b}"] = precision_at_k(s, y, k)
            rows.append(row)
        print(f"{bench:8s} {name[:36]:36s} anomalies {n_anom:5d}/{n}", flush=True)

    df = pd.DataFrame(rows)
    df["commit"] = git_commit()
    out = RESULTS_DIR / f"q2_{args.tag}.csv"
    df.to_csv(out, index=False)

    # ---- summary -----------------------------------------------------------
    metrics = ["auroc", "auprc", "prec@n_anom", "prec@1pct", "prec@5pct", "prec@10pct"]
    w = df.pivot_table(index="dataset", columns="arm", values=metrics)
    print(f"\n=== Q2 · {w.shape[0]} datasets · mean anomaly rate {df.drop_duplicates('dataset').anomaly_rate.mean():.3f} ===")
    table = pd.DataFrame({m: {a: w[(m, a)].mean() for a in ARMS} for m in metrics}).round(3)
    print(table.to_string())

    print("\nPaired against kNN (wins out of datasets, sign-test p):")
    for m in ["auprc", "prec@n_anom", "prec@5pct"]:
        line = []
        for a in ARMS[1:]:
            d = w[(m, a)] - w[(m, "knn")]
            wins, ties = int((d > 0).sum()), int((d == 0).sum())
            n_eff = len(d) - ties
            p = binomtest(wins, n_eff).pvalue if n_eff else float("nan")
            line.append(f"{a}: {wins}/{n_eff} (p={p:.2g}, mean {d.mean():+.3f})")
        print(f"  {m:12s} " + " | ".join(line))

    print("\nUncertainty against error (std vs nll):")
    for m in ["auprc", "prec@n_anom", "prec@5pct"]:
        d = w[(m, "std")] - w[(m, "nll")]
        print(f"  {m:12s} std wins {(d > 0).sum()}/{(d != 0).sum()}, mean {d.mean():+.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
