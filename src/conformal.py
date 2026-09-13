"""
The conformal / e-value layer.

Everything here is plain numpy. The idea, in one line:
compare a new point's score to the scores of points we know are normal,
and turn that comparison into something with a guarantee.

Two ways to use it:

  Route 1 (classification): probabilities -> prediction SET of plausible labels.
      lac_threshold(), lac_sets()

  Route 2 (anomaly detection): anomaly score -> p-value or e-value -> alert list.
      conformal_pvalues(), conformal_evalues(), bh(), ebh()

Conventions:
  * a "score" is a number where LARGER means "looks less normal".
  * calibration scores come from points we KNOW are normal and that the
    model did not train on.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Route 1: prediction sets for a classifier (what the HPLR paper does)
# ---------------------------------------------------------------------------

def lac_threshold(calib_probs: np.ndarray, calib_labels: np.ndarray, alpha: float = 0.1) -> float:
    """
    Find the threshold q_hat from calibration data.

    calib_probs  : (n, K) predicted probabilities for n calibration points
    calib_labels : (n,)   the TRUE label of each calibration point
    alpha        : 1 - target coverage (0.1 means 90% coverage)

    For each calibration point we look at how badly the model scored the
    label that turned out to be correct: score = 1 - p(true label).
    q_hat is (roughly) the 90th percentile of these scores.
    """
    n = len(calib_labels)
    scores = 1.0 - calib_probs[np.arange(n), calib_labels]
    # the (n+1)(1-alpha)-th smallest score, with a ceiling; this small
    # correction is what makes the 90% guarantee exact and not approximate
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def lac_sets(probs: np.ndarray, q_hat: float) -> np.ndarray:
    """
    Build prediction sets: a label is IN the set if 1 - p(label) <= q_hat.

    probs : (m, K) predicted probabilities for m new points
    returns a boolean (m, K) matrix; row i, column k is True if label k is
    plausible for point i. A row of all False is an EMPTY set.
    """
    return (1.0 - probs) <= q_hat


def set_sizes(sets: np.ndarray) -> np.ndarray:
    """Number of labels in each set. 0 means empty set."""
    return sets.sum(axis=1)


def coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of points whose true label is inside their set."""
    return float(sets[np.arange(len(labels)), labels].mean())


# ---------------------------------------------------------------------------
# Route 2: conformal p-values and e-values for "is this point normal?"
# ---------------------------------------------------------------------------

def conformal_pvalues(calib_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """
    p-value for the hypothesis "this test point is normal".

        p = (1 + number of calibration scores >= test score) / (n + 1)

    Small p = the point looks weirder than almost every known-normal point.
    Guarantee: if the point really is normal, P(p <= a) <= a for any a.
    (Bates, Candes, Lei, Romano, Sesia 2023.)
    """
    calib_sorted = np.sort(np.asarray(calib_scores, dtype=float))
    n = len(calib_sorted)
    test_scores = np.asarray(test_scores, dtype=float)
    # number of calibration scores strictly smaller than each test score
    n_smaller = np.searchsorted(calib_sorted, test_scores, side="left")
    n_greater_or_equal = n - n_smaller
    return (1.0 + n_greater_or_equal) / (n + 1.0)


def conformal_evalues(calib_scores: np.ndarray, test_scores: np.ndarray, q: float = 0.1) -> np.ndarray:
    """
    e-value for the hypothesis "this test point is normal"
    (Ren & Barber 2023, "derandomized novelty detection with FDR control").

    Step 1: run BH at level q on the conformal p-values of the whole batch.
            This picks a score threshold T (the smallest flagged score).
    Step 2: every test point above T gets the same e-value,

                e = (n + 1) / (1 + number of calibration scores >= T)

            and every point below T gets e = 0.

    Read e as the payout of a 1-euro bet against "normal".
    Guarantee: if a point really is normal, its AVERAGE payout is <= 1.
    Running e-BH on these e-values at the same q gives back exactly the BH
    alert list. The reason to go through e-values anyway: e-values from
    different random splits, seeds or models can be AVERAGED and stay valid,
    which p-values cannot do.

    If BH flags nothing, T = +inf and all e-values are 0.
    """
    calib_scores = np.asarray(calib_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    n = len(calib_scores)

    p = conformal_pvalues(calib_scores, test_scores)
    flagged = bh(p, q=q)
    if not flagged.any():
        return np.zeros(len(test_scores))
    threshold = test_scores[flagged].min()
    n_calib_above = (calib_scores >= threshold).sum()
    e = np.where(test_scores >= threshold, (n + 1.0) / (1.0 + n_calib_above), 0.0)
    return e


def conformal_evalues_ratio(calib_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """
    The simplest possible conformal e-value:

        e = (n + 1) * s / (s + sum of calibration scores)

    Valid (average <= 1 for a normal point, because its score is exchangeable
    with the calibration scores and so takes 1/(n+1) of the total on average),
    but WEAK: e can never be much larger than (n+1) * s / sum, so in practice
    it rarely clears the e-BH bar. Kept for teaching and for comparison.
    Scores must be >= 0.
    """
    calib_scores = np.asarray(calib_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    if (calib_scores < 0).any() or (test_scores < 0).any():
        raise ValueError("e-values need non-negative scores; shift your score first.")
    n = len(calib_scores)
    total = calib_scores.sum()
    return (n + 1.0) * test_scores / (test_scores + total)


def p_to_e(pvalues: np.ndarray, kappa: float = 0.5) -> np.ndarray:
    """
    Turn p-values into e-values with the calibrator  e = kappa * p^(kappa - 1).

    This is the other way to get an e-value (Vovk & Wang 2021, eq. 1).
    kappa in (0, 1). Kept here so we can compare the two constructions later.
    """
    p = np.asarray(pvalues, dtype=float)
    return kappa * p ** (kappa - 1.0)


# ---------------------------------------------------------------------------
# Deciding where to cut the alert list
# ---------------------------------------------------------------------------

def ebh(evalues: np.ndarray, q: float = 0.1) -> np.ndarray:
    """
    e-BH (Wang & Ramdas 2022): choose the alert list with FDR <= q.

    Sort e-values from largest to smallest. Cutting after rank k is allowed if
        e_(k) >= m / (q * k)
    i.e. "the number of false alarms we could expect by chance (m / e_(k)),
    divided by the list length k, stays under q".
    We take the LARGEST k that passes and flag the top k.

    Holds under arbitrary dependence between the points.

    evalues : (m,)
    returns : boolean (m,) mask, True = flagged as anomaly
    """
    e = np.asarray(evalues, dtype=float)
    m = len(e)
    order = np.argsort(-e)            # largest first
    e_sorted = e[order]
    k = np.arange(1, m + 1)
    passes = e_sorted >= m / (q * k)
    if not passes.any():
        return np.zeros(m, dtype=bool)
    k_star = int(np.max(np.nonzero(passes)[0])) + 1
    flagged = np.zeros(m, dtype=bool)
    flagged[order[:k_star]] = True
    return flagged


def bh(pvalues: np.ndarray, q: float = 0.1) -> np.ndarray:
    """
    Ordinary Benjamini-Hochberg on p-values, for comparison with e-BH.

    Sort p-values from smallest to largest, find the largest k with
        p_(k) <= q * k / m
    and flag the k smallest. Its FDR guarantee needs independence (or a
    correction), which is exactly what e-BH does not need.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    k = np.arange(1, m + 1)
    passes = p_sorted <= q * k / m
    if not passes.any():
        return np.zeros(m, dtype=bool)
    k_star = int(np.max(np.nonzero(passes)[0])) + 1
    flagged = np.zeros(m, dtype=bool)
    flagged[order[:k_star]] = True
    return flagged


# ---------------------------------------------------------------------------
# Checking ourselves on labelled test data
# ---------------------------------------------------------------------------

def false_discovery_proportion(flagged: np.ndarray, is_anomaly: np.ndarray) -> float:
    """Among the flagged points, what fraction were actually normal? (0 if nothing flagged)"""
    flagged = np.asarray(flagged, dtype=bool)
    is_anomaly = np.asarray(is_anomaly, dtype=bool)
    n_flagged = flagged.sum()
    if n_flagged == 0:
        return 0.0
    return float((flagged & ~is_anomaly).sum() / n_flagged)


def recall(flagged: np.ndarray, is_anomaly: np.ndarray) -> float:
    """Among the true anomalies, what fraction did we flag?"""
    flagged = np.asarray(flagged, dtype=bool)
    is_anomaly = np.asarray(is_anomaly, dtype=bool)
    n_anom = is_anomaly.sum()
    if n_anom == 0:
        return 0.0
    return float((flagged & is_anomaly).sum() / n_anom)
