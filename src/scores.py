"""
Anomaly scores. Every score function has the same shape:

    score_fn(X_fit, X_query, seed) -> array of length len(X_query)

    X_fit   : known-normal points the score may learn from
    X_query : points to score (calibration or test)
    larger score = looks less normal

Baselines here are classical detectors. The TabPFN uncertainty score will be
added in the same format so the driver does not change.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def knn_score(X_fit: np.ndarray, X_query: np.ndarray, seed: int = 0, k: int = 5) -> np.ndarray:
    """
    Average distance to the k nearest known-normal points (after standardizing
    each feature with the fit set). Far from every normal point = anomalous.
    This is the classic KNN detector, also the strongest simple baseline in
    the MacrOData paper.
    """
    scaler = StandardScaler().fit(X_fit)
    nn = NearestNeighbors(n_neighbors=k).fit(scaler.transform(X_fit))
    dist, _ = nn.kneighbors(scaler.transform(X_query))
    return dist.mean(axis=1)


def iforest_score(X_fit: np.ndarray, X_query: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Isolation Forest: random trees isolate unusual points in few splits.
    sklearn returns "higher = more normal", so we flip the sign.
    """
    model = IsolationForest(n_estimators=200, random_state=seed).fit(X_fit)
    return -model.score_samples(X_query)


SCORES = {
    "knn": knn_score,
    "iforest": iforest_score,
}


# ---------------------------------------------------------------------------
# The main experiment: TabPFN's uncertainty as an anomaly score
# ---------------------------------------------------------------------------

def _bin_column(values_fit: np.ndarray, values_query: np.ndarray, n_bins: int):
    """
    Turn one column into class labels so TabPFN can predict it as a
    classification problem.

    Few distinct values (<= n_bins): each value is its own class.
    Otherwise: n_bins quantile bins computed on the fit rows ("which decile
    does the value fall in?").

    Returns (labels_fit, labels_query, n_classes).
    """
    uniq = np.unique(values_fit)
    if len(uniq) <= n_bins:
        # nearest known value -> class index
        mid = (uniq[:-1] + uniq[1:]) / 2
        return np.searchsorted(mid, values_fit), np.searchsorted(mid, values_query), len(uniq)
    edges = np.unique(np.quantile(values_fit, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.digitize(values_fit, edges), np.digitize(values_query, edges), len(edges) + 1


def tabpfn_scores(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    seed: int = 0,
    max_context: int = 1000,
    max_target_columns: int = 10,
    max_input_features: int = 100,
    n_bins: int = 10,
    n_estimators: int = 2,
    alpha: float = 0.1,
    query_batch: int = 2000,
    device: str = "auto",
    cache: "Path | None" = None,
) -> dict[str, np.ndarray]:
    """
    Ask TabPFN to predict each column from the other columns, using only
    known-normal rows as context. From its predicted probabilities we get,
    for every query row, three numbers averaged over the predicted columns:

      tabpfn_error    : 1 - p(observed value)          "was the value surprising?"
                        (this is what TabPFN-OD in the MacrOData paper uses)
      tabpfn_entropy  : entropy of the prediction / log K
                        "was the model unsure, whatever the value?"
      tabpfn_setsize  : size of the conformal set / K   (LAC, level alpha)
                        same idea as entropy but through the conformal lens

    Note the difference: `error` looks at the actual value in the column,
    `entropy` and `setsize` do NOT - they only measure how sure the model is.
    That is the "uncertainty vs raw probability" contrast of Q2.

    Budget knobs (they make the run feasible, and are logged in results):
      max_context         rows of X_fit used as TabPFN context
      max_target_columns  how many columns we predict (random subset)
      max_input_features  how many other columns are given as input
    A slice (20%) of the context rows is held out to pick the conformal
    threshold for `setsize`, so it never sees the query rows.

    If `cache` is given, the raw per-column probabilities are saved there
    (rows x columns x bins) so any other measure can be computed offline.
    """
    from tabpfn import TabPFNClassifier

    from conformal import lac_threshold, lac_sets

    rng = np.random.default_rng(seed)
    n_fit, d = X_fit.shape

    # rows: context for TabPFN and a small held-out slice for the LAC threshold
    rows = rng.permutation(n_fit)[: min(n_fit, max_context)]
    n_hold = max(20, int(0.2 * len(rows)))
    hold_rows, ctx_rows = rows[:n_hold], rows[n_hold:]

    # which columns to predict, and which columns may be used as input
    targets = rng.permutation(d)[: min(d, max_target_columns)]

    sums = {k: np.zeros(len(X_query)) for k in ("tabpfn_error", "tabpfn_entropy", "tabpfn_setsize")}
    raw = {"targets": [], "K": [], "q_hat": [], "probs": [], "y_bin": []}
    n_done = 0
    for j in targets:
        y_ctx, y_rest, K = _bin_column(
            X_fit[ctx_rows, j], np.concatenate([X_fit[hold_rows, j], X_query[:, j]]), n_bins
        )
        if len(np.unique(y_ctx)) < 2:
            continue  # constant column in the context: nothing to predict
        y_hold, y_query = y_rest[:n_hold], y_rest[n_hold:]

        others = np.array([c for c in range(d) if c != j])
        if len(others) > max_input_features:
            others = rng.choice(others, max_input_features, replace=False)

        clf = TabPFNClassifier(
            n_estimators=n_estimators, device=device, random_state=seed,
            ignore_pretraining_limits=True,
        )
        clf.fit(X_fit[np.ix_(ctx_rows, others)], y_ctx)

        def proba_full(X):
            """predict_proba over ALL K classes (classes absent from context get 0)."""
            out = np.zeros((len(X), K))
            for start in range(0, len(X), query_batch):
                p = clf.predict_proba(X[start:start + query_batch])
                out[start:start + query_batch][:, clf.classes_.astype(int)] = p
            return out

        p_hold = proba_full(X_fit[np.ix_(hold_rows, others)])
        p_query = proba_full(X_query[:, others])

        q_hat = lac_threshold(p_hold, y_hold, alpha=alpha)
        sets = lac_sets(p_query, q_hat)

        sums["tabpfn_error"] += 1.0 - p_query[np.arange(len(X_query)), y_query]
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(p_query * np.log(np.where(p_query > 0, p_query, 1))).sum(axis=1)
        sums["tabpfn_entropy"] += ent / np.log(K)
        sums["tabpfn_setsize"] += sets.sum(axis=1) / K
        n_done += 1
        raw["targets"].append(j); raw["K"].append(K); raw["q_hat"].append(q_hat)
        raw["probs"].append(p_query.astype(np.float32)); raw["y_bin"].append(y_query.astype(np.int16))

    if n_done == 0:
        raise RuntimeError("no column could be predicted (all constant?)")
    if cache is not None:
        _save_raw(cache, "tabpfn", raw)
    return {k: v / n_done for k, v in sums.items()}


def _save_raw(cache, name, raw):
    """Save per-column raw outputs, padding ragged arrays with NaN."""
    from pathlib import Path
    out = {}
    for k, v in raw.items():
        if k in ("targets", "K", "q_hat"):
            out[k] = np.asarray(v)
        else:
            width = max(a.shape[1] if a.ndim > 1 else 1 for a in v)
            stacked = np.full((len(v), len(v[0]), width), np.nan, dtype=np.float32)
            for c, a in enumerate(v):
                a = a if a.ndim > 1 else a[:, None]
                stacked[c, :, : a.shape[1]] = a
            out[k] = stacked          # (columns, rows, width)
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(cache).with_name(Path(cache).name + f"__{name}_raw.npz"), **out)


SCORES["tabpfn"] = tabpfn_scores


def tabpfn_reg_scores(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    seed: int = 0,
    max_context: int = 1000,
    max_target_columns: int = 10,
    max_input_features: int = 100,
    min_unique: int = 11,
    n_estimators: int = 2,
    query_batch: int = 2000,
    device: str = "auto",
    cache: "Path | None" = None,
) -> dict[str, np.ndarray]:
    """
    Same idea as `tabpfn_scores` but each numeric column is predicted with
    TabPFN's REGRESSOR, which outputs a full predictive distribution. No
    binning, so "how far outside the normal range" is preserved, and the
    predictive spread is free to grow away from the data (it does: on toy
    data the std goes from 0.3 at the centre to >100 far away).

    Per row, averaged over the predicted columns (targets are z-scored with
    the context mean/std so columns are comparable):

      tabpfnreg_std      predictive standard deviation        uncertainty
      tabpfnreg_width90  width of the 90% predictive interval uncertainty
      tabpfnreg_nll      -log density of the observed value   error
      tabpfnreg_pit      |2 * CDF(observed) - 1|              error (rank-based, robust)

    Columns with fewer than `min_unique` distinct values are skipped
    (they are categorical-like; use `tabpfn_scores` for those).
    """
    import torch
    from tabpfn import TabPFNRegressor

    rng = np.random.default_rng(seed)
    n_fit, d = X_fit.shape
    ctx_rows = rng.permutation(n_fit)[: min(n_fit, max_context)]
    targets = [j for j in rng.permutation(d) if len(np.unique(X_fit[ctx_rows, j])) >= min_unique]
    targets = targets[:max_target_columns]

    names = ("tabpfnreg_std", "tabpfnreg_width90", "tabpfnreg_nll", "tabpfnreg_pit")
    sums = {k: np.zeros(len(X_query)) for k in names}
    raw = {"targets": [], "std": [], "width90": [], "nll": [], "cdf": [], "mean": []}
    for j in targets:
        mu, sd = X_fit[ctx_rows, j].mean(), X_fit[ctx_rows, j].std() + 1e-12
        z_ctx = (X_fit[ctx_rows, j] - mu) / sd
        z_query = (X_query[:, j] - mu) / sd

        others = np.array([c for c in range(d) if c != j])
        if len(others) > max_input_features:
            others = rng.choice(others, max_input_features, replace=False)

        reg = TabPFNRegressor(
            n_estimators=n_estimators, device=device, random_state=seed,
            ignore_pretraining_limits=True,
        )
        reg.fit(X_fit[np.ix_(ctx_rows, others)], z_ctx)

        std = np.empty(len(X_query)); width = np.empty(len(X_query))
        nll = np.empty(len(X_query)); cdf = np.empty(len(X_query)); mean = np.empty(len(X_query))
        for start in range(0, len(X_query), query_batch):
            sl = slice(start, start + query_batch)
            out = reg.predict(X_query[sl][:, others], output_type="full")
            logits, crit = out["logits"], out["criterion"]
            with torch.no_grad():
                z = torch.as_tensor(z_query[sl], dtype=logits.dtype, device=logits.device)
                std[sl] = crit.variance(logits).clamp_min(0).sqrt().cpu().numpy()
                q = crit.quantile(logits, center_prob=0.9)
                width[sl] = (q[..., 1] - q[..., 0]).cpu().numpy()
                nll[sl] = -torch.log(crit.pdf(logits, z).clamp_min(1e-12)).cpu().numpy()
                cdf[sl] = crit.cdf(logits, z[:, None]).squeeze(-1).cpu().numpy()
                mean[sl] = crit.mean(logits).cpu().numpy()

        sums["tabpfnreg_std"] += std
        sums["tabpfnreg_width90"] += width
        sums["tabpfnreg_nll"] += nll
        sums["tabpfnreg_pit"] += np.abs(2 * cdf - 1)
        raw["targets"].append(j)
        for k, v in (("std", std), ("width90", width), ("nll", nll), ("cdf", cdf), ("mean", mean)):
            raw[k].append(v.astype(np.float32))

    if not targets:
        raise RuntimeError("no numeric column to predict")
    if cache is not None:
        _save_raw(cache, "tabpfnreg", raw)
    return {k: v / len(targets) for k, v in sums.items()}


SCORES["tabpfn_reg"] = tabpfn_reg_scores
