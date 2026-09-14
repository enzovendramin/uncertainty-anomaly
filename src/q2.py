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
               combinations: each score becomes a conformal p-value against its
               own calibration scores, p is turned into an e-value with the
               calibrator e = 0.5 / sqrt(p), and the e-values are AVERAGED with
               equal weights. Scale-free, label-free, and still a valid e-value.

Budgets: k = number of anomalies in the test set, and k = 1%, 5%, 10% of the
test set (at least 1). Metrics: precision@k, AUPRC (baseline = anomaly rate).

Usage:
    python src/q2.py --tag tabpfnreg_representative
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from conformal import conformal_pvalues, p_to_e
from data import load
from run import RESULTS_DIR, ROOT, git_commit, precision_at_k
from scores import knn_score

ARMS = ["knn", "std", "nll", "knn+std", "knn+nll", "std+nll", "knn+std+nll"]


def load_rows(tag: str, seed: int):
    """One record per dataset: test labels and per-row scores for every arm."""
    for f in sorted((RESULTS_DIR / tag / "scores").glob(f"*__s{seed}__tabpfn_reg.npz")):
        bench, name, _, _ = f.name.split("__")
        z = np.load(f)
        d = load(ROOT / "data" / bench / "representative" / f"{name}.npz")
        n_calib = int(z["n_calib"])
        calib_idx, test_idx, y = z["calib_idx"], z["test_idx"], z["y_test"]

        # the fit rows are the complement of the (uncapped) calibration split
        fit_idx, _ = train_test_split(np.arange(len(d.X_train)), test_size=0.3, random_state=seed)
        knn = knn_score(d.X_train[fit_idx], np.vstack([d.X_train[calib_idx], d.X_test[test_idx]]), seed)

        base = {"knn": knn, "std": z["tabpfnreg_std"], "nll": z["tabpfnreg_nll"]}
        rng = np.random.default_rng(seed)
        # p-values for every base score, then e-values; combinations average the e-values
        evals = {}
        for k, s in base.items():
            p = conformal_pvalues(s[:n_calib], s[n_calib:], rng=rng)
            evals[k] = p_to_e(p, kappa=0.5)
        test_scores = {k: s[n_calib:] for k, s in base.items()}
        for combo in ARMS[3:]:
            parts = combo.split("+")
            test_scores[combo] = np.mean([evals[p] for p in parts], axis=0)
        yield bench, name, y, test_scores, d.n_features


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
