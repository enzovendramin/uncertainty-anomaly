"""
Model zoo: every model's uncertainty (and error) in the same column-prediction
frame, on the same rows. Offline from the per-row score files.

Usage:
    python src/zoo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

from arms import load_records
from run import RESULTS_DIR, git_commit

# (tag, file suffix) -> which score columns to read from those files
SOURCES = {
    ("zoo_representative", "gp"): ["gp_std", "gp_nll"],
    ("zoo_representative", "catboost"): ["cb_knowledge", "cb_data", "cb_nll"],
    ("zoo_representative", "tabpfn_ens"): ["tpe_disagree", "tpe_aleatoric", "tpe_total"],
    ("tabpfnreg_representative", "tabpfn_reg"): ["tabpfnreg_std", "tabpfnreg_nll"],
}
UNCERTAINTY = ["gp_std", "cb_knowledge", "cb_data", "tabpfnreg_std", "tpe_disagree", "tpe_aleatoric", "tpe_total"]
ERROR = ["gp_nll", "cb_nll", "tabpfnreg_nll"]


def main() -> None:
    seed = 0
    rows = []
    # kNN on the same rows, via the shared loader (it recomputes kNN from the split indices)
    knn = {r.dataset: (r.y, r.test["knn"]) for r in load_records("tabpfnreg_representative", seed)}
    for (tag, suffix), cols in SOURCES.items():
        for f in sorted((RESULTS_DIR / tag / "scores").glob(f"*__s{seed}__{suffix}.npz")):
            bench, name = f.name.split("__")[:2]
            z = np.load(f)
            y, n_calib = z["y_test"], int(z["n_calib"])
            if y.sum() == 0:
                continue
            for c in cols:
                s = z[c][n_calib:]
                rows.append({"bench": bench, "dataset": name, "score": c,
                             "auroc": roc_auc_score(y, s), "auprc": average_precision_score(y, s)})
    for name, (y, s) in knn.items():
        if y.sum() > 0:
            rows.append({"bench": name and "", "dataset": name, "score": "knn",
                         "auroc": roc_auc_score(y, s), "auprc": average_precision_score(y, s)})
    df = pd.DataFrame(rows)
    df["commit"] = git_commit()
    df.to_csv(RESULTS_DIR / "zoo_metrics.csv", index=False)

    w = df.pivot_table(index="dataset", columns="score", values="auroc").dropna()
    order = ["knn"] + UNCERTAINTY + ERROR
    order = [c for c in order if c in w]
    summ = pd.DataFrame({
        "mean AUROC": w[order].mean(),
        "median": w[order].median(),
        "share < 0.5": (w[order] < 0.5).mean(),
        "share > 0.8": (w[order] > 0.8).mean(),
        "beats kNN": (w[order].gt(w["knn"], axis=0)).mean(),
    }).round(3)
    print(f"model zoo · {len(w)} datasets · seed {seed}\n")
    print(summ.to_string())
    print("\npaired vs kNN (sign test):")
    for c in order[1:]:
        d = w[c] - w["knn"]
        n = int((d != 0).sum()); wins = int((d > 0).sum())
        print(f"  {c:14s} wins {wins:2d}/{n}  mean diff {d.mean():+.3f}  p={binomtest(wins, n).pvalue:.2g}")
    print("\nrank correlation between uncertainty scores across datasets:")
    print(w[[c for c in ["knn", "gp_std", "cb_knowledge", "tabpfnreg_std", "tpe_disagree"] if c in w]].corr("spearman").round(2).to_string())


if __name__ == "__main__":
    main()
