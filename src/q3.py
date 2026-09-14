"""
Q3: can the alerts carry a guarantee?

Offline from the saved per-row scores (see arms.py). Four parts:

  A. Guaranteed alert lists.  For every arm, conformal e-values (Ren & Barber
     construction) -> e-BH at level q. A combination is first merged into one
     score (mean of -log p over its parts, see arms.combined_score) and then
     calibrated like any other score. Per dataset: number of alerts, realised
     false-discovery proportion (FDP), recall.

  B. Derandomisation.  With several seeds (different fit/calibration splits),
     average the e-values across seeds -> e-BH. Still valid, and the alert
     list should be more stable than any single seed's. Measured by the
     Jaccard overlap between alert sets, and FDP of the averaged list.

  C. Stream mode.  The test set arrives in batches of `batch` rows (a
     "morning" of transactions). e-BH per batch with the same calibration
     set; pooled FDP and recall over all batches. Compared with the whole
     test set as one batch.

  D. Power vs calibration size.  Same kNN scores, calibration subsampled to
     n in {100, 300, 1000, 3000}: how many datasets get any alert, and the
     recall. Explains the "all or nothing" behaviour.

Usage:
    python src/q3.py --tag tabpfnreg_representative --q 0.1 --batch 200
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from arms import ARMS, arm_scores, load_records, seeds_available
from conformal import conformal_evalues, ebh, false_discovery_proportion, recall
from run import RESULTS_DIR, git_commit


def evalues_for(rec, arm: str, q: float, calib_n: int | None = None, test_mask=None) -> np.ndarray:
    """Ren & Barber e-values for one arm (a combination is first merged into one score)."""
    c, t = arm_scores(rec, arm, np.random.default_rng(rec.seed))
    if calib_n is not None:
        c = c[:calib_n]
    if test_mask is not None:
        t = t[test_mask]
    return conformal_evalues(c, t, q=q)


def summarise_lists(rows: pd.DataFrame, label: str, q: float) -> None:
    g = rows.groupby("arm")
    tab = pd.DataFrame({
        "datasets with alerts": g["n_alerts"].apply(lambda s: (s > 0).mean()),
        "mean recall": g["recall"].mean(),
        "FDP where flagged": rows[rows.n_alerts > 0].groupby("arm")["fdp"].mean(),
        "share of datasets FDP>q": rows[rows.n_alerts > 0].groupby("arm")["fdp"].apply(lambda s: (s > q).mean()),
        "pooled true alerts": g["true_alerts"].sum(),
        "pooled false alerts": g["false_alerts"].sum(),
    }).reindex(list(ARMS)).round(3)
    tab["pooled FDP"] = (tab["pooled false alerts"] / (tab["pooled true alerts"] + tab["pooled false alerts"])).round(3)
    print(f"\n--- {label} (q = {q}) ---")
    print(tab.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tabpfnreg_representative")
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="seeds to use (default: every seed that covers all datasets)")
    args = ap.parse_args()
    q = args.q

    records = {s: {r.dataset: r for r in load_records(args.tag, s)} for s in (args.seeds or seeds_available(args.tag))}
    full = max(len(v) for v in records.values())
    seeds = [s for s, v in records.items() if len(v) == full]
    dropped = sorted(set(records) - set(seeds))
    if dropped:
        print(f"ignoring incomplete seeds {dropped}")
    records = {s: records[s] for s in seeds}
    datasets = sorted(set.intersection(*[set(v) for v in records.values()]))
    print(f"seeds used: {seeds} · {len(datasets)} datasets")

    rows_a, rows_b, rows_c, rows_d = [], [], [], []
    for name in datasets:
        recs = [records[s][name] for s in seeds]
        rec0 = recs[0]
        y = rec0.y
        n_anom = int(y.sum())
        if n_anom == 0:
            continue
        # A. one list per arm, first seed
        for arm in ARMS:
            flagged = ebh(evalues_for(rec0, arm, q), q=q)
            rows_a.append({"dataset": name, "arm": arm, "seed": seeds[0], "n_alerts": int(flagged.sum()),
                           "true_alerts": int((flagged & (y == 1)).sum()), "false_alerts": int((flagged & (y == 0)).sum()),
                           "fdp": false_discovery_proportion(flagged, y), "recall": recall(flagged, y)})

        # B. derandomisation across seeds (needs the same test rows in every seed)
        same_rows = all(len(r.y) == len(y) and (r.y == y).all() for r in recs)
        if len(seeds) > 1 and same_rows:
            for arm in ARMS:
                per_seed = [ebh(evalues_for(r, arm, q), q=q) for r in recs]
                avg = ebh(np.mean([evalues_for(r, arm, q) for r in recs], axis=0), q=q)
                jac = []
                for i in range(len(per_seed)):
                    for j in range(i + 1, len(per_seed)):
                        u = (per_seed[i] | per_seed[j]).sum()
                        jac.append((per_seed[i] & per_seed[j]).sum() / u if u else np.nan)
                rows_b.append({"dataset": name, "arm": arm,
                               "mean_alerts_single": float(np.mean([f.sum() for f in per_seed])),
                               "jaccard_between_seeds": float(np.nanmean(jac)) if jac else np.nan,
                               "n_alerts": int(avg.sum()), "true_alerts": int((avg & (y == 1)).sum()),
                               "false_alerts": int((avg & (y == 0)).sum()),
                               "fdp": false_discovery_proportion(avg, y), "recall": recall(avg, y)})

        # C. stream mode: batches of `batch` rows, first seed
        rng = np.random.default_rng(0)
        order = rng.permutation(len(y))
        for arm in ARMS:
            flagged = np.zeros(len(y), dtype=bool)
            for start in range(0, len(y), args.batch):
                idx = order[start:start + args.batch]
                flagged[idx] = ebh(evalues_for(rec0, arm, q, test_mask=idx), q=q)
            rows_c.append({"dataset": name, "arm": arm, "n_alerts": int(flagged.sum()),
                           "true_alerts": int((flagged & (y == 1)).sum()), "false_alerts": int((flagged & (y == 0)).sum()),
                           "fdp": false_discovery_proportion(flagged, y), "recall": recall(flagged, y)})

        # D. power vs calibration size, kNN only
        for n_cal in (100, 300, 1000, 3000):
            if n_cal > len(rec0.calib["knn"]):
                continue
            flagged = ebh(evalues_for(rec0, "knn", q, calib_n=n_cal), q=q)
            rows_d.append({"dataset": name, "n_calib": n_cal, "n_alerts": int(flagged.sum()),
                           "fdp": false_discovery_proportion(flagged, y), "recall": recall(flagged, y)})

    A, B, C, D = (pd.DataFrame(r) for r in (rows_a, rows_b, rows_c, rows_d))
    commit = git_commit()
    for df, part in ((A, "A"), (B, "B"), (C, "C"), (D, "D")):
        if len(df):
            df["commit"] = commit
            df.to_csv(RESULTS_DIR / f"q3{part}_{args.tag}.csv", index=False)

    summarise_lists(A, "A. guaranteed alert lists, whole test set, one seed", q)
    if len(B):
        summarise_lists(B, f"B. derandomised over {len(seeds)} seeds", q)
        stab = B.groupby("arm")[["mean_alerts_single", "jaccard_between_seeds"]].mean().reindex(list(ARMS)).round(3)
        print("\n   stability of single-seed lists (Jaccard overlap between seeds):")
        print(stab.to_string())
    summarise_lists(C, f"C. stream mode, batches of {args.batch} rows", q)
    if len(D):
        print("\n--- D. power vs calibration size (kNN) ---")
        print(D.groupby("n_calib").agg(datasets=("dataset", "size"),
                                       with_alerts=("n_alerts", lambda s: (s > 0).mean()),
                                       mean_recall=("recall", "mean"),
                                       fdp_where_flagged=("fdp", lambda s: s[D.loc[s.index, "n_alerts"] > 0].mean())).round(3).to_string())


if __name__ == "__main__":
    main()
