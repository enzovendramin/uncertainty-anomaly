"""
The driver: for every dataset and every score function,

    1. split the known-normal train set into fit (70%) and calibration (30%)
    2. compute scores on calibration and test
    3. p-values, e-values, BH and e-BH at level q
    4. evaluate on the test labels

and write one row per (dataset, score, seed) to results/<tag>.csv.
Test labels are used ONLY in step 4. See splits.md.

Usage:
    python src/run.py --scores knn iforest --q 0.1 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from conformal import (
    bh,
    conformal_evalues,
    conformal_pvalues,
    ebh,
    false_discovery_proportion,
    recall,
)
from data import BENCHES, Dataset, load_all
from scores import SCORES

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def precision_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    """Among the k highest-scored points, what fraction are anomalies?"""
    if k <= 0:
        return float("nan")
    top = np.argsort(-scores)[:k]
    return float(y[top].mean())


def evaluate_one(d: Dataset, score_name: str, seed: int, q: float,
                 calib_frac: float = 0.3, max_test: int | None = None,
                 max_calib: int | None = None) -> list[dict]:
    """
    Returns one row per score. A score function may return a single array or
    a dict of named arrays (e.g. TabPFN gives error, entropy and set size from
    one pass); each named array becomes its own row.
    """
    score_fn = SCORES[score_name]
    t0 = time.time()

    # 1. split the normal training points: fit / calibration
    X_fit, X_calib = train_test_split(d.X_train, test_size=calib_frac, random_state=seed)
    # optional cap on calibration rows: a random subset of normal points is
    # still a valid calibration set, and a few thousand is plenty for p-values
    if max_calib is not None and len(X_calib) > max_calib:
        X_calib = X_calib[np.random.default_rng(seed).choice(len(X_calib), max_calib, replace=False)]

    # optional: random subset of the test rows (labels are NOT looked at)
    X_test, y = d.X_test, d.y_test
    if max_test is not None and len(y) > max_test:
        keep = np.random.default_rng(seed).choice(len(y), max_test, replace=False)
        X_test, y = X_test[keep], y[keep]

    # 2. scores (larger = less normal), one pass over calibration + test rows
    out = score_fn(X_fit, np.vstack([X_calib, X_test]), seed)
    if not isinstance(out, dict):
        out = {score_name: out}
    seconds = round(time.time() - t0, 2)

    rows = []
    for name, s_all in out.items():
        s_calib, s_test = s_all[: len(X_calib)], s_all[len(X_calib):]
        rows.append(_evaluate_scores(d, name, seed, q, X_fit, X_calib, s_calib, s_test, y, seconds))
    return rows


def _evaluate_scores(d, score_name, seed, q, X_fit, X_calib, s_calib, s_test, y, seconds) -> dict:
    # 3. the conformal / e-value layer
    p = conformal_pvalues(s_calib, s_test)
    e = conformal_evalues(s_calib, s_test, q=q)
    flag_bh = bh(p, q=q)
    flag_ebh = ebh(e, q=q)

    # 4. evaluation with the test labels
    n_anom = int(y.sum())
    row = {
        "bench": d.bench,
        "dataset": d.name,
        "score": score_name,
        "seed": seed,
        "q": q,
        "n_fit": len(X_fit),
        "n_calib": len(X_calib),
        "n_test": len(y),
        "n_anomalies": n_anom,
        "n_features": d.n_features,
        # Q1: does the score rank anomalies above normals?
        "auroc": roc_auc_score(y, s_test) if 0 < n_anom < len(y) else float("nan"),
        "auprc": average_precision_score(y, s_test) if 0 < n_anom < len(y) else float("nan"),
        "precision_at_n_anom": precision_at_k(s_test, y, n_anom),
        # Q3: alert lists with a guarantee
        "bh_n_flagged": int(flag_bh.sum()),
        "bh_fdp": false_discovery_proportion(flag_bh, y),
        "bh_recall": recall(flag_bh, y),
        "ebh_n_flagged": int(flag_ebh.sum()),
        "ebh_fdp": false_discovery_proportion(flag_ebh, y),
        "ebh_recall": recall(flag_ebh, y),
        # sanity: false alarms among normals at p <= q (should be <= q)
        "normal_false_alarm_rate_at_q": float((p[y == 0] <= q).mean()),
        "seconds": seconds,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs="+", default=["knn", "iforest"], choices=list(SCORES))
    parser.add_argument("--benches", nargs="+", default=list(BENCHES))
    parser.add_argument("--subset", default="representative")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument("--tag", default=None, help="name of the output csv (default: date + scores)")
    parser.add_argument("--max-datasets", type=int, default=None, help="for quick tests")
    parser.add_argument("--max-test", type=int, default=None,
                        help="random subset of test rows per dataset (for slow scores); labels are not used")
    parser.add_argument("--max-calib", type=int, default=None,
                        help="random subset of calibration rows per dataset (for slow scores)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = args.tag or f"{datetime.now():%Y%m%d}_{'_'.join(args.scores)}"
    out_path = RESULTS_DIR / f"{tag}.csv"

    datasets = []
    for bench in args.benches:
        datasets += load_all(bench, args.subset)
    if args.max_datasets:
        datasets = datasets[: args.max_datasets]
    print(f"{len(datasets)} datasets, scores={args.scores}, seeds={args.seeds}, q={args.q}")

    rows = []
    for i, d in enumerate(datasets, 1):
        for score_name in args.scores:
            for seed in args.seeds:
                try:
                    new_rows = evaluate_one(d, score_name, seed, args.q,
                                            max_test=args.max_test, max_calib=args.max_calib)
                except Exception as exc:  # keep going, record the failure
                    print(f"[{i:3d}/{len(datasets)}] {d.bench:8s} {d.name[:32]:32s} {score_name:8s} seed={seed} FAILED: {exc!r}")
                    continue
                for row in new_rows:
                    row["max_test"] = args.max_test
                    row["max_calib"] = args.max_calib
                    rows.append(row)
                    print(
                        f"[{i:3d}/{len(datasets)}] {d.bench:8s} {d.name[:32]:32s} {row['score']:15s} seed={seed} "
                        f"auroc={row['auroc']:.3f} ebh: flagged={row['ebh_n_flagged']:5d} "
                        f"fdp={row['ebh_fdp']:.3f} recall={row['ebh_recall']:.3f} ({row['seconds']}s)",
                        flush=True,
                    )
        # write after every dataset so a crash loses nothing
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
