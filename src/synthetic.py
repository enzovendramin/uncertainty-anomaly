"""
Synthetic diagnostic: which KIND of anomaly does each score see?

We build normal data whose structure we know, plant four kinds of anomaly,
and measure every score on each kind separately. Real benchmarks mix the
kinds, so this is the only way to say "score X sees global outliers but not
broken relationships" as a fact rather than a guess.

Normal rows (6 columns):
    a, b            independent standard normals
    c = a + b + noise            a linear relationship
    d = sin(2a) + 0.5 b + noise  a curved relationship
    e = 0.8 c + noise            depends on c
    f               independent noise (an uninformative column)

Anomaly kinds (each keeps everything else normal):
    extreme        one column pushed 3-4 std out, the rest untouched
    global_shift   every column moved together, relationships preserved (ECG-like)
    broken_link    column c replaced by the c of another normal row: marginals
                   unchanged, the relationship with a and b broken
    subspace       a and b both moved into a region normal rows never visit
                   (each still within range on its own, the pair is not)

Usage:
    python src/synthetic.py --seeds 0 1 2 --scores knn gp catboost tabpfn_reg tabpfn_ens
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run import RESULTS_DIR, git_commit
from scores import SCORES

KINDS = ("extreme", "global_shift", "broken_link", "subspace")


def normal_rows(n: int, rng: np.random.Generator) -> np.ndarray:
    a, b = rng.normal(size=n), rng.normal(size=n)
    c = a + b + 0.3 * rng.normal(size=n)
    d = np.sin(2 * a) + 0.5 * b + 0.3 * rng.normal(size=n)
    e = 0.8 * c + 0.3 * rng.normal(size=n)
    f = rng.normal(size=n)
    return np.column_stack([a, b, c, d, e, f])


def plant(kind: str, X_normal: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return n anomalous rows of the given kind, built from fresh normal rows."""
    X = normal_rows(n, rng)
    if kind == "extreme":
        col = rng.integers(0, 5, size=n)
        sign = rng.choice([-1, 1], size=n)
        X[np.arange(n), col] = sign * rng.uniform(3, 4, size=n) * X_normal.std(axis=0)[col]
    elif kind == "global_shift":
        # move the whole row along the direction of the data: scale every column
        # by a common factor 1.8-2.2 so relationships (c = a + b, ...) survive
        X = X * rng.uniform(1.8, 2.2, size=(n, 1))
    elif kind == "broken_link":
        donor = normal_rows(n, rng)
        X[:, 2] = donor[:, 2]           # c no longer matches its own a and b
    elif kind == "subspace":
        # a and b each within their normal range, but the pair sits where
        # normal rows never are: a high and b = -a (normal a, b are independent,
        # so (2, -2) is a corner they rarely visit); c, d, e recomputed so
        # the row is internally consistent
        a = rng.uniform(1.8, 2.2, size=n) * rng.choice([-1, 1], size=n)
        b = -a + 0.1 * rng.normal(size=n)
        X[:, 0], X[:, 1] = a, b
        X[:, 2] = a + b + 0.3 * rng.normal(size=n)
        X[:, 3] = np.sin(2 * a) + 0.5 * b + 0.3 * rng.normal(size=n)
        X[:, 4] = 0.8 * X[:, 2] + 0.3 * rng.normal(size=n)
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--scores", nargs="+", default=["knn", "gp", "catboost", "tabpfn_reg", "tabpfn_ens"])
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test-normal", type=int, default=600)
    ap.add_argument("--n-per-kind", type=int, default=60)
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        X_train = normal_rows(args.n_train, rng)
        X_fit, X_calib = X_train[: int(0.7 * args.n_train)], X_train[int(0.7 * args.n_train):]
        X_test_normal = normal_rows(args.n_test_normal, rng)
        anomalies = {k: plant(k, X_train, args.n_per_kind, rng) for k in KINDS}
        X_query = np.vstack([X_calib, X_test_normal] + [anomalies[k] for k in KINDS])
        for score_name in args.scores:
            out = SCORES[score_name](X_fit, X_query, seed)
            if not isinstance(out, dict):
                out = {score_name: out}
            for name, s in out.items():
                s_test_normal = s[len(X_calib): len(X_calib) + args.n_test_normal]
                start = len(X_calib) + args.n_test_normal
                for k in KINDS:
                    s_anom = s[start: start + args.n_per_kind]
                    start += args.n_per_kind
                    y = np.r_[np.zeros(len(s_test_normal)), np.ones(len(s_anom))]
                    rows.append({"seed": seed, "score": name, "kind": k,
                                 "auroc": roc_auc_score(y, np.r_[s_test_normal, s_anom])})
            print(f"seed {seed} · {score_name} done", flush=True)

    df = pd.DataFrame(rows)
    df["commit"] = git_commit()
    df.to_csv(RESULTS_DIR / "synthetic.csv", index=False)
    table = df.pivot_table(index="score", columns="kind", values="auroc", aggfunc="mean")[list(KINDS)].round(2)
    print("\nAUROC by anomaly kind (mean over seeds); 0.5 = blind, 1.0 = perfect\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
