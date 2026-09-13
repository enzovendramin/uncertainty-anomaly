"""
Tests for the conformal / e-value layer.

Most tests SIMULATE data where we know who is normal and who is not,
run the layer many times, and check that the promised guarantee holds
on average. If a test here fails, nothing downstream can be trusted.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conformal import (  # noqa: E402
    bh,
    conformal_evalues,
    conformal_evalues_ratio,
    conformal_pvalues,
    coverage,
    ebh,
    false_discovery_proportion,
    lac_sets,
    lac_threshold,
    p_to_e,
    recall,
    set_sizes,
)


# ---------------------------------------------------------------------------
# Route 1: prediction sets
# ---------------------------------------------------------------------------

def test_lac_sets_worked_example():
    # the example from our discussion: q_hat = 0.40
    probs = np.array([
        [0.95, 0.05],   # confident normal -> {normal}
        [0.30, 0.70],   # confident fraud  -> {fraud}
        [0.55, 0.45],   # unsure           -> empty set
    ])
    sets = lac_sets(probs, q_hat=0.40)
    assert sets.tolist() == [[True, False], [False, True], [False, False]]
    assert set_sizes(sets).tolist() == [1, 1, 0]


def test_lac_coverage_is_at_least_90_percent():
    # a deliberately bad, overconfident model: does coverage still hold? it must.
    rng = np.random.default_rng(0)
    n_calib, n_test, n_classes = 500, 2000, 3
    covs = []
    for _ in range(50):
        labels_c = rng.integers(0, n_classes, n_calib)
        labels_t = rng.integers(0, n_classes, n_test)
        # noisy probabilities: right label gets a bump only sometimes
        probs_c = rng.dirichlet(np.ones(n_classes) * 0.5, n_calib)
        probs_t = rng.dirichlet(np.ones(n_classes) * 0.5, n_test)
        q_hat = lac_threshold(probs_c, labels_c, alpha=0.1)
        covs.append(coverage(lac_sets(probs_t, q_hat), labels_t))
    assert np.mean(covs) >= 0.90 - 0.01


# ---------------------------------------------------------------------------
# Route 2: p-values and e-values
# ---------------------------------------------------------------------------

def test_pvalue_worked_example():
    calib = np.array([0.02, 0.03, 0.05, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
    p = conformal_pvalues(calib, np.array([0.05, 0.40]))
    # 7 calibration scores are >= 0.05 -> (1+7)/10 ; none >= 0.40 -> 1/10
    assert p.tolist() == pytest.approx([0.8, 0.1])


def test_pvalues_are_valid_for_normal_points():
    # normal test points from the same distribution as calibration:
    # the fraction with p <= a must be <= a (up to noise)
    rng = np.random.default_rng(1)
    n_calib, n_test = 200, 200
    hits = {0.05: [], 0.10: [], 0.20: []}
    for _ in range(300):
        calib = rng.exponential(size=n_calib)
        test = rng.exponential(size=n_test)
        p = conformal_pvalues(calib, test)
        for a in hits:
            hits[a].append((p <= a).mean())
    for a, h in hits.items():
        assert np.mean(h) <= a + 0.01, f"false alarm rate {np.mean(h):.3f} > {a}"


def test_ratio_evalue_worked_example():
    calib = np.array([0.02, 0.03, 0.05, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30])  # sums to 0.98
    e = conformal_evalues_ratio(calib, np.array([0.05, 0.40, 0.90]))
    assert e.tolist() == pytest.approx([10 * 0.05 / 1.03, 10 * 0.40 / 1.38, 10 * 0.90 / 1.88])


def test_ratio_evalues_average_at_most_one_for_normal_points():
    rng = np.random.default_rng(2)
    means = []
    for _ in range(500):
        calib = rng.exponential(size=100)
        test = rng.exponential(size=100)
        means.append(conformal_evalues_ratio(calib, test).mean())
    assert np.mean(means) <= 1.0 + 0.02


def test_ratio_evalues_reject_negative_scores():
    with pytest.raises(ValueError):
        conformal_evalues_ratio(np.array([1.0, -1.0]), np.array([0.5]))


def test_evalues_average_at_most_one_for_normal_points():
    # batch of only normal points: the average e-value must stay <= 1
    rng = np.random.default_rng(6)
    means = []
    for _ in range(500):
        calib = rng.exponential(size=100)
        test = rng.exponential(size=100)
        means.append(conformal_evalues(calib, test, q=0.1).mean())
    assert np.mean(means) <= 1.0 + 0.02


def test_evalues_are_zero_when_bh_flags_nothing():
    calib = np.linspace(0, 1, 50)
    test = np.full(20, 0.5)          # nothing stands out
    assert (conformal_evalues(calib, test, q=0.1) == 0).all()


def test_p_to_e_is_valid():
    # for uniform p-values the calibrator must have mean <= 1
    rng = np.random.default_rng(3)
    p = rng.uniform(size=200_000)
    assert p_to_e(p, kappa=0.5).mean() <= 1.0 + 0.02


# ---------------------------------------------------------------------------
# e-BH and BH
# ---------------------------------------------------------------------------

def test_ebh_toy_example_flags_three():
    # m=5, q=0.5 -> thresholds 10/k = 10, 5, 3.3, 2.5, 2
    e = np.array([0.3, 30.0, 0.5, 12.0, 6.0])   # shuffled on purpose
    flagged = ebh(e, q=0.5)
    assert flagged.tolist() == [False, True, False, True, True]


def test_ebh_flags_nothing_when_all_small():
    assert not ebh(np.array([0.1, 0.5, 0.9]), q=0.1).any()


def _simulate_batch(rng, n_calib=1000, n_normal=400, n_anomaly=100, shift=5.0):
    """
    Normal scores ~ Exp(1); anomalies ~ Exp(1) + shift. Returns scores + labels.

    Note: with a small calibration set (say 300) the smallest possible p-value
    is 1/301, which caps how many points BH / e-BH can ever flag. Power then
    switches sharply from "nothing" to "everything" as anomalies get clearer.
    The defaults here are on the "clear" side so the power tests are meaningful.
    """
    calib = rng.exponential(size=n_calib)
    test_normal = rng.exponential(size=n_normal)
    test_anom = rng.exponential(size=n_anomaly) + shift
    scores = np.concatenate([test_normal, test_anom])
    is_anomaly = np.concatenate([np.zeros(n_normal, bool), np.ones(n_anomaly, bool)])
    return calib, scores, is_anomaly


def test_ebh_controls_fdr_and_has_power():
    rng = np.random.default_rng(4)
    q = 0.1
    fdps, recs = [], []
    for _ in range(300):
        calib, scores, is_anomaly = _simulate_batch(rng)
        e = conformal_evalues(calib, scores, q=q)
        flagged = ebh(e, q=q)
        fdps.append(false_discovery_proportion(flagged, is_anomaly))
        recs.append(recall(flagged, is_anomaly))
    assert np.mean(fdps) <= q + 0.02, f"FDR {np.mean(fdps):.3f} > {q}"
    assert np.mean(recs) > 0.5, f"recall {np.mean(recs):.3f}: e-BH barely flags anything"


def test_ebh_on_evalues_matches_bh_on_pvalues():
    rng = np.random.default_rng(7)
    calib, scores, _ = _simulate_batch(rng)
    p_flags = bh(conformal_pvalues(calib, scores), q=0.1)
    e_flags = ebh(conformal_evalues(calib, scores, q=0.1), q=0.1)
    assert (p_flags == e_flags).all()


def test_ratio_evalues_have_no_power_in_a_batch():
    # documents WHY we do not use the ratio e-value for alert lists
    rng = np.random.default_rng(8)
    calib, scores, _ = _simulate_batch(rng)
    assert not ebh(conformal_evalues_ratio(calib, scores), q=0.1).any()


def test_averaged_evalues_over_two_splits_still_control_fdr():
    # the property we care about: average e-values from two different
    # calibration sets (two random splits) -> still valid -> e-BH still safe
    rng = np.random.default_rng(9)
    q = 0.1
    fdps, recs = [], []
    for _ in range(300):
        calib_a, scores, is_anomaly = _simulate_batch(rng)
        calib_b = rng.exponential(size=len(calib_a))
        e = 0.5 * (conformal_evalues(calib_a, scores, q=q) + conformal_evalues(calib_b, scores, q=q))
        flagged = ebh(e, q=q)
        fdps.append(false_discovery_proportion(flagged, is_anomaly))
        recs.append(recall(flagged, is_anomaly))
    assert np.mean(fdps) <= q + 0.02
    assert np.mean(recs) > 0.5


def test_bh_controls_fdr_with_independent_pvalues():
    rng = np.random.default_rng(5)
    q = 0.1
    fdps = []
    for _ in range(300):
        calib, scores, is_anomaly = _simulate_batch(rng)
        p = conformal_pvalues(calib, scores)
        flagged = bh(p, q=q)
        fdps.append(false_discovery_proportion(flagged, is_anomaly))
    # conformal p-values sharing one calibration set are not independent, but
    # BH is known to still hold here (Bates et al. 2023); we allow a bit of slack
    assert np.mean(fdps) <= q + 0.03


def test_fdp_and_recall_helpers():
    flagged = np.array([True, True, False, False])
    is_anomaly = np.array([True, False, True, False])
    assert false_discovery_proportion(flagged, is_anomaly) == 0.5
    assert recall(flagged, is_anomaly) == 0.5
    assert false_discovery_proportion(np.zeros(4, bool), is_anomaly) == 0.0
