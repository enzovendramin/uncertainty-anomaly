"""
Shared loader for the offline analyses (q2.py, q3.py).

Rebuilds, for every dataset and seed, the calibration and test scores of the
three base arms on exactly the same rows:

    knn   distance to the nearest known-normal rows      (baseline)
    std   TabPFN regressor predictive std                (uncertainty)
    nll   TabPFN regressor negative log-density          (error)

std and nll come from the per-row files saved by run.py; kNN is recomputed on
the saved split indices (it is cheap and deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from data import load
from run import RESULTS_DIR, ROOT
from scores import knn_score

BASE_ARMS = ("knn", "std", "nll")
COMBOS = ("knn+std", "knn+nll", "std+nll", "knn+std+nll")
ARMS = BASE_ARMS + COMBOS


@dataclass
class Record:
    bench: str
    dataset: str
    seed: int
    y: np.ndarray                      # test labels, 0 normal / 1 anomaly
    calib: dict[str, np.ndarray]       # base arm -> calibration scores
    test: dict[str, np.ndarray]        # base arm -> test scores
    n_features: int


def _loo_pvalues(calib: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    p-value of each calibration point against the OTHER calibration points
    (leave-one-out), with random tie-breaking. A test point's p-value is
    computed against all n calibration points; a calibration point's against
    the other n-1. Both are then uniform for normal points (on grids of n+1
    and n values; the difference is negligible for n >= 1000, and the
    combined score is calibrated again anyway).
    """
    n = len(calib)
    # number of OTHER calibration scores >= this one, ties broken at random
    sorted_c = np.sort(calib)
    n_greater = n - np.searchsorted(sorted_c, calib, side="right")
    n_equal = np.searchsorted(sorted_c, calib, side="right") - np.searchsorted(sorted_c, calib, side="left") - 1
    ties = np.floor(rng.uniform(size=n) * (n_equal + 1)).astype(int)
    return (1.0 + n_greater + ties) / n


def combined_score(rec: "Record", parts: list[str], rng: np.random.Generator):
    """
    Combine several arms into ONE score, then it can be calibrated once like
    any other score: mean over arms of -log p, where p is the conformal
    p-value of the row for that arm (leave-one-out for calibration rows).
    Equal weights, fixed in advance. Returns (calib_scores, test_scores).
    """
    from conformal import conformal_pvalues
    c_parts, t_parts = [], []
    for p in parts:
        c_parts.append(-np.log(_loo_pvalues(rec.calib[p], rng)))
        t_parts.append(-np.log(conformal_pvalues(rec.calib[p], rec.test[p], rng=rng)))
    return np.mean(c_parts, axis=0), np.mean(t_parts, axis=0)


def arm_scores(rec: "Record", arm: str, rng: np.random.Generator):
    """(calibration scores, test scores) for a base arm or a combination."""
    if arm in BASE_ARMS:
        return rec.calib[arm], rec.test[arm]
    return combined_score(rec, arm.split("+"), rng)


def seeds_available(tag: str) -> list[int]:
    files = (RESULTS_DIR / tag / "scores").glob("*__tabpfn_reg.npz")
    return sorted({int(f.name.split("__")[2][1:]) for f in files})


def load_records(tag: str, seed: int):
    for f in sorted((RESULTS_DIR / tag / "scores").glob(f"*__s{seed}__tabpfn_reg.npz")):
        bench, name, _, _ = f.name.split("__")
        z = np.load(f)
        d = load(ROOT / "data" / bench / "representative" / f"{name}.npz")
        n_calib = int(z["n_calib"])
        calib_idx, test_idx, y = z["calib_idx"], z["test_idx"], z["y_test"]
        fit_idx, _ = train_test_split(np.arange(len(d.X_train)), test_size=0.3, random_state=seed)
        knn = knn_score(d.X_train[fit_idx], np.vstack([d.X_train[calib_idx], d.X_test[test_idx]]), seed)
        base = {"knn": knn, "std": z["tabpfnreg_std"].astype(float), "nll": z["tabpfnreg_nll"].astype(float)}
        yield Record(
            bench=bench, dataset=name, seed=seed, y=y,
            calib={k: v[:n_calib] for k, v in base.items()},
            test={k: v[n_calib:] for k, v in base.items()},
            n_features=d.n_features,
        )
