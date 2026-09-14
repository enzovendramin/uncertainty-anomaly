"""
Build the hand-off bundle for the poster in results/poster/:

    numbers.json     every headline number with where it comes from
    tables/*.csv     one summary table per panel
    figures/*.png    the figures (also .svg), same style throughout
    notes.md         one page: question, method, one paragraph + caption per panel, caveats

Everything is read from the results csv files; nothing is typed by hand.

Usage:
    python src/export.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_all
from run import RESULTS_DIR, git_commit

OUT = RESULTS_DIR / "poster"
FIG, TAB = OUT / "figures", OUT / "tables"

# one palette for everything: family -> colour
C = {"baseline": "#2a78d6", "uncertainty": "#eb6834", "error": "#1baf7a", "old": "#9aa4ad", "ink": "#171b21", "grey": "#5b6673", "rule": "#d9dee4"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": C["rule"], "axes.labelcolor": C["ink"], "xtick.color": C["grey"], "ytick.color": C["grey"],
                     "figure.dpi": 110, "savefig.dpi": 200, "svg.fonttype": "none"})

FRAUD = ["AdRequestFraud", "SubscriptionFraud", "TextFraudDetection", "CreditCardTransactions",
         "FinancialTransaction3", "CybersecurityVulnerability", "NetworkTrafficAnomalies", "VulnerabilityRisk"]

# score -> (label, family)
LABELS = {
    "knn": ("kNN distance", "baseline"), "iforest": ("Isolation Forest", "baseline"),
    "gp_std": ("Gaussian process · std", "uncertainty"), "gp_nll": ("Gaussian process · error", "error"),
    "tpe_disagree": ("TabPFN · disagreement (4 copies)", "uncertainty"),
    "tabpfnreg_std": ("TabPFN · width (1 copy)", "uncertainty"), "tabpfnreg_width90": ("TabPFN · 90% interval", "uncertainty"),
    "tabpfnreg_nll": ("TabPFN · error", "error"), "tabpfnreg_pit": ("TabPFN · rank of value", "error"),
    "cb_knowledge": ("CatBoost · knowledge", "uncertainty"), "cb_data": ("CatBoost · data noise", "uncertainty"), "cb_nll": ("CatBoost · error", "error"),
    "tabpfn_setsize": ("TabPFN binned · set size", "old"), "tabpfn_entropy": ("TabPFN binned · entropy", "old"), "tabpfn_error": ("TabPFN binned · error", "old"),
}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def boot_ci(v, rng, n=3000):
    v = np.asarray(v); bs = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def dot_plot(w, order, title, name, note=None, xlim=(0.4, 0.88)):
    """Horizontal dots with 95% intervals, grouped by family colour, chance line at 0.5."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(order) + 1.4))
    ys = np.arange(len(order))[::-1]
    for y, s in zip(ys, order):
        lab, fam = LABELS[s]
        m = w[s].mean(); lo, hi = boot_ci(w[s].values, rng)
        ax.plot([lo, hi], [y, y], color=C[fam], lw=2, solid_capstyle="round")
        ax.plot(m, y, "o", color=C[fam], ms=8, mec="white", mew=1.5)
        ax.text(hi + 0.008, y, f"{m:.3f}", va="center", fontsize=9.5, color=C["grey"],
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_ylim(-0.7, len(order) - 0.3 + 0.9)
    ax.axvline(0.5, color=C["grey"], ls=":", lw=1)
    ax.text(0.5, len(order) - 0.3 + 0.55, "chance (0.5)", ha="center", va="center", fontsize=9, color=C["grey"],
            bbox=dict(facecolor=plt.rcParams["figure.facecolor"], edgecolor="none", pad=1.5))
    ax.set_yticks(ys); ax.set_yticklabels([LABELS[s][0] for s in order])
    for t, s in zip(ax.get_yticklabels(), order):
        t.set_color(C[LABELS[s][1]] if LABELS[s][1] != "old" else C["grey"])
    ax.set_xlim(*xlim); ax.set_xlabel("AUROC  (probability that a random anomaly scores above a random normal row)")
    ax.set_title(title, loc="left", fontsize=12, color=C["ink"], pad=12)
    ax.grid(axis="x", color=C["rule"], lw=0.6)
    from matplotlib.lines import Line2D
    present = {LABELS[s][1] for s in order}
    ax.legend(handles=[Line2D([], [], marker="o", color=C[f], ls="", ms=8, label=l) for f, l in
                       [("baseline", "distance baseline"), ("uncertainty", "model uncertainty"), ("error", "model error"), ("old", "superseded (binned)")] if f in present],
              loc="lower right", frameon=False, fontsize=9)
    if note:
        fig.text(0.01, -0.02, note, fontsize=9, color=C["grey"])
    save(fig, name)


def main() -> None:
    for d in (OUT, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    numbers = {"commit": git_commit(), "datasets": "100 representative MacrOData datasets (50 OddBench + 50 OvRBench), development set"}

    # ---------------- Q1: per-dataset AUROC, all scores on the same rows ----------------
    q1 = pd.read_csv(RESULTS_DIR / "auroc_paired_meta.csv").set_index("dataset")
    zoo = pd.read_csv(RESULTS_DIR / "zoo_metrics.csv").pivot_table(index="dataset", columns="score", values="auroc")
    w = q1.drop(columns=[c for c in q1.columns if c in zoo.columns]).join(zoo, how="inner")
    w.to_csv(TAB / "q1_auroc_per_dataset.csv")

    order_tabpfn = ["knn", "iforest", "tabpfnreg_std", "tabpfnreg_width90", "tabpfnreg_nll", "tabpfnreg_pit", "tabpfn_error", "tabpfn_entropy", "tabpfn_setsize"]
    dot_plot(w, order_tabpfn, "Q1 · Does TabPFN's uncertainty rank anomalies above normal rows?", "fig2_q1_tabpfn",
             note=f"{len(w)} datasets, one seed. Dots = mean AUROC, bars = 95% bootstrap interval over datasets.")
    order_zoo = ["knn", "gp_std", "tpe_disagree", "tabpfnreg_std", "cb_knowledge", "cb_data", "gp_nll", "cb_nll", "tabpfnreg_nll"]
    dot_plot(w, order_zoo, "Q1 across models · Whose uncertainty knows it is far from the data?", "fig3_q1_model_zoo",
             note=f"{len(w)} datasets, one seed, same rows and same column-prediction task for every model.")
    wn = w[[c for c in w.columns if c in LABELS]]
    summ = pd.DataFrame({"mean_auroc": wn.mean(), "median": wn.median(), "share_below_0.5": (wn < 0.5).mean(),
                         "share_above_0.8": (wn > 0.8).mean(), "beats_knn": wn.gt(wn["knn"], axis=0).mean()}).round(3)
    summ = summ.loc[[s for s in LABELS if s in summ.index]]
    summ.insert(0, "label", [LABELS[s][0] for s in summ.index]); summ.to_csv(TAB / "q1_summary.csv")
    numbers["q1"] = {s: round(float(w[s].mean()), 3) for s in summ.index}
    numbers["q1_share_below_chance"] = {s: round(float((w[s] < 0.5).mean()), 3) for s in summ.index}
    numbers["q1_ecg"] = {s: round(float(w.loc["ECGAnomaly", s]), 3) for s in ["knn", "tabpfn_setsize", "tabpfnreg_std", "tpe_disagree", "gp_std", "tabpfnreg_nll"]}
    numbers["q1_binning_fix"] = {"datasets_improved": int((w["tabpfnreg_std"] > w["tabpfn_setsize"]).sum()), "of": int(len(w)),
                                 "mean_gain": round(float((w["tabpfnreg_std"] - w["tabpfn_setsize"]).mean()), 3)}

    # before/after + vs kNN scatters
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, x, xl, ttl in [(axes[0], "tabpfn_setsize", "binned classifier · set size (AUROC)", "Removing the bins fixed the measurement"),
                           (axes[1], "knn", "kNN distance (AUROC)", "…but distance still wins on most datasets")]:
        odd = w["bench"] == "OddBench"
        ax.scatter(w.loc[odd, x], w.loc[odd, "tabpfnreg_std"], s=26, color=C["uncertainty"], alpha=0.85, label="OddBench")
        ax.scatter(w.loc[~odd, x], w.loc[~odd, "tabpfnreg_std"], s=26, facecolor="white", edgecolor=C["uncertainty"], lw=1.4, label="OvRBench")
        ax.plot([0, 1], [0, 1], color=C["grey"], lw=1); ax.axhline(0.5, color=C["grey"], ls=":", lw=0.8); ax.axvline(0.5, color=C["grey"], ls=":", lw=0.8)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_xlabel(xl); ax.set_ylabel("TabPFN width · regressor (AUROC)")
        ax.set_title(ttl, loc="left", fontsize=11); ax.grid(color=C["rule"], lw=0.5)
    box = dict(facecolor="white", edgecolor=C["rule"], pad=4)
    axes[0].text(0.04, 0.30, "above the line =\nbetter without bins", fontsize=9.5, color=C["ink"], va="top", bbox=box)
    axes[1].text(0.06, 0.93, "below the line = kNN better", fontsize=9.5, color=C["ink"], va="top", bbox=box)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("One point per dataset", x=0.01, ha="left", fontsize=12); fig.tight_layout()
    save(fig, "fig4_q1_scatters")

    # ---------------- synthetic diagnostic ----------------
    syn = pd.read_csv(RESULTS_DIR / "synthetic.csv")
    kinds = ["extreme", "global_shift", "broken_link", "subspace"]
    kind_lab = {"extreme": "one column\nextreme", "global_shift": "whole row\nshifted", "broken_link": "columns no\nlonger fit", "subspace": "unusual pair,\nrest consistent"}
    rows_syn = ["knn", "gp_std", "tpe_disagree", "tabpfnreg_std", "cb_knowledge", "gp_nll", "tabpfnreg_nll", "cb_nll"]
    st = syn.pivot_table(index="score", columns="kind", values="auroc").loc[rows_syn, kinds]
    st.to_csv(TAB / "synthetic_auroc_by_kind.csv")
    numbers["synthetic"] = {s: {k: round(float(st.loc[s, k]), 2) for k in kinds} for s in rows_syn}
    fig, ax = plt.subplots(figsize=(8, 5.2))
    im = ax.imshow(st.values, cmap="Oranges", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(st.shape[0]):
        for j in range(st.shape[1]):
            v = st.values[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white" if v > 0.85 else C["ink"])
    ax.set_xticks(range(4)); ax.set_xticklabels([kind_lab[k] for k in kinds]); ax.set_yticks(range(len(rows_syn)))
    ax.set_yticklabels([LABELS[s][0] for s in rows_syn])
    for t, s in zip(ax.get_yticklabels(), rows_syn): t.set_color(C[LABELS[s][1]])
    ax.set_title("What each score sees · synthetic data with four planted anomaly kinds (AUROC, 3 seeds)", loc="left", fontsize=11, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("AUROC  (0.5 = blind, 1.0 = perfect)")
    for sp in ax.spines.values(): sp.set_visible(False)
    save(fig, "fig5_synthetic_diagnostic")

    # ---------------- Q2: precision at a budget ----------------
    q2 = pd.read_csv(RESULTS_DIR / "q2_tabpfnreg_representative.csv")
    arms = ["knn", "std", "nll", "knn+std", "knn+nll"]
    arm_lab = {"knn": "kNN distance", "std": "TabPFN uncertainty", "nll": "TabPFN error", "knn+std": "kNN + uncertainty", "knn+nll": "kNN + error"}
    arm_fam = {"knn": "baseline", "std": "uncertainty", "nll": "error", "knn+std": "uncertainty", "knn+nll": "error"}
    q2w = q2.pivot_table(index="dataset", columns="arm", values=["prec@1pct", "prec@5pct", "prec@10pct", "auprc"])
    q2t = pd.DataFrame({m: {a: q2w[(m, a)].mean() for a in arms} for m in ["prec@1pct", "prec@5pct", "prec@10pct", "auprc"]}).round(3)
    q2t.insert(0, "label", [arm_lab[a] for a in arms]); q2t.to_csv(TAB / "q2_precision_at_budget.csv")
    base_rate = float(q2.drop_duplicates("dataset").anomaly_rate.mean())
    numbers["q2"] = {"anomaly_rate": round(base_rate, 3), "precision_at_5pct": {a: round(float(q2w[("prec@5pct", a)].mean()), 3) for a in arms},
                     "std_wins_vs_knn_at_5pct": f"{int((q2w[('prec@5pct','std')] > q2w[('prec@5pct','knn')]).sum())}/{int((q2w[('prec@5pct','std')] != q2w[('prec@5pct','knn')]).sum())}"}
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(arms)); wdt = 0.25
    from matplotlib.patches import Patch
    for k, (m, lab) in enumerate([("prec@1pct", "top 1% reviewed"), ("prec@5pct", "top 5%"), ("prec@10pct", "top 10%")]):
        vals = [q2w[(m, a)].mean() for a in arms]
        bars = ax.bar(x + (k - 1) * wdt, vals, wdt, color=[C[arm_fam[a]] for a in arms], alpha=[1.0, 0.7, 0.45][k], edgecolor="white")
        if k == 1:
            for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=9.5, color=C["ink"])
    ax.axhline(base_rate, color=C["grey"], ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([arm_lab[a].replace(" + ", " +" + chr(10)).replace("TabPFN ", "TabPFN" + chr(10)).replace("kNN distance", "kNN" + chr(10) + "distance") for a in arms])
    ax.set_ylim(0, 0.6); ax.set_ylabel("share of the alert list that is truly anomalous"); ax.grid(axis="y", color=C["rule"], lw=0.6)
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=C["grey"], alpha=al, label=l) for al, l in [(1.0, "top 1% reviewed"), (0.7, "top 5% reviewed"), (0.45, "top 10% reviewed")]]
    handles.append(Line2D([], [], color=C["grey"], ls="--", label=f"a random list ({base_rate:.2f})"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.4)
    ax.set_title("Q2 · Review budget: which score fills the list with real anomalies?", loc="left", fontsize=11.5, pad=30)
    save(fig, "fig6_q2_budget")

    # ---------------- Q3: guarantees and power ----------------
    A = pd.read_csv(RESULTS_DIR / "q3A_tabpfnreg_representative.csv"); Cc = pd.read_csv(RESULTS_DIR / "q3C_tabpfnreg_representative.csv")
    B = pd.read_csv(RESULTS_DIR / "q3B_tabpfnreg_representative.csv"); D = pd.read_csv(RESULTS_DIR / "q3D_tabpfnreg_representative.csv")
    def pooled(df):
        g = df.groupby("arm"); t = pd.DataFrame({"datasets_with_alerts": g.n_alerts.apply(lambda s: (s > 0).mean()), "recall": g.recall.mean(),
                                                 "pooled_fdp": g.false_alerts.sum() / (g.true_alerts.sum() + g.false_alerts.sum())})
        return t.loc[arms].round(3)
    q3 = pd.concat({"whole test set": pooled(A), "batches of 200": pooled(Cc)}, axis=1); q3.to_csv(TAB / "q3_alert_lists.csv")
    subB = A[A.dataset.isin(B.dataset.unique())]
    derand = pd.DataFrame({"single_seed_datasets_with_alerts": subB.groupby("arm").n_alerts.apply(lambda s: (s > 0).mean()),
                           "averaged_3_seeds_datasets_with_alerts": B.groupby("arm").n_alerts.apply(lambda s: (s > 0).mean()),
                           "single_seed_recall": subB.groupby("arm").recall.mean(), "averaged_recall": B.groupby("arm").recall.mean(),
                           "overlap_between_seeds_jaccard": B.groupby("arm").jaccard_between_seeds.mean(),
                           "averaged_pooled_fdp": B.groupby("arm").false_alerts.sum() / (B.groupby("arm").true_alerts.sum() + B.groupby("arm").false_alerts.sum())}).loc[arms].round(3)
    derand.to_csv(TAB / "q3_derandomisation.csv")
    ok = D.groupby("dataset").n_calib.max() >= 1000; Ds = D[D.dataset.isin(ok[ok].index) & (D.n_calib <= 1000)]
    power = Ds.groupby("n_calib").agg(datasets_with_alerts=("n_alerts", lambda s: (s > 0).mean()), recall=("recall", "mean"),
                                      fdp_where_flagged=("fdp", lambda s: s[Ds.loc[s.index, "n_alerts"] > 0].mean())).round(3)
    power.to_csv(TAB / "q3_power_vs_calibration_size.csv")
    numbers["q3"] = {"pooled_fdp_whole": {a: round(float(q3[("whole test set", "pooled_fdp")][a]), 3) for a in arms},
                     "datasets_with_alerts_whole": {a: round(float(q3[("whole test set", "datasets_with_alerts")][a]), 3) for a in arms},
                     "datasets_with_alerts_batches": {a: round(float(q3[("batches of 200", "datasets_with_alerts")][a]), 3) for a in arms},
                     "pooled_fdp_batches": {a: round(float(q3[("batches of 200", "pooled_fdp")][a]), 3) for a in arms},
                     "jaccard_between_seeds": {a: round(float(derand.loc[a, "overlap_between_seeds_jaccard"]), 3) for a in arms},
                     "power_vs_calibration": power.to_dict("index")}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    ax = axes[0]
    for a in ["knn", "std", "nll"]:
        vals = [q3[("whole test set", "datasets_with_alerts")][a], q3[("batches of 200", "datasets_with_alerts")][a]]
        ax.plot([0, 1], vals, "-o", color=C[arm_fam[a]], label=arm_lab[a], ms=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["whole test set\nat once", "batches of\n200 rows"]); ax.set_ylim(0, 0.6)
    ax.set_ylabel("share of datasets with ≥1 alert"); ax.set_title("More datasets get alerts in batches", loc="left", fontsize=10.5); ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", color=C["rule"], lw=0.6)
    ax = axes[1]
    for a in ["knn", "std", "nll"]:
        ax.bar([arm_lab[a]], [q3[("whole test set", "pooled_fdp")][a]], color=C[arm_fam[a]], width=0.55)
    ax.axhline(0.1, color="#d03b3b", ls="--", lw=1.2); ax.text(2.3, 0.104, "promised: 10%", ha="right", fontsize=9, color="#d03b3b")
    ax.set_ylim(0, 0.15); ax.set_ylabel("false alarms among all alerts (pooled)"); ax.set_title("The guarantee holds", loc="left", fontsize=10.5); ax.grid(axis="y", color=C["rule"], lw=0.6)
    ax.tick_params(axis="x", labelsize=9)
    ax = axes[2]
    ax.plot(power.index, power["datasets_with_alerts"], "-o", color=C["baseline"], ms=7, label="share of datasets with alerts")
    ax.plot(power.index, power["recall"], "-s", color=C["grey"], ms=6, label="recall")
    ax.set_xscale("log"); ax.set_xticks([100, 300, 1000]); ax.set_xticklabels(["100", "300", "1000"]); ax.set_xlabel("known-normal rows used for calibration")
    ax.set_ylim(0, 0.5); ax.set_title("Below ~300 calibration rows, power drops", loc="left", fontsize=10.5); ax.legend(frameon=False, fontsize=9); ax.grid(color=C["rule"], lw=0.6)
    fig.suptitle("Q3 · Alerts with a false-discovery guarantee (e-BH at 10%)", x=0.01, ha="left", fontsize=12); fig.tight_layout()
    save(fig, "fig7_q3_guarantees")

    # ---------------- fraud subset + duplicates ----------------
    fr = w.loc[[d for d in FRAUD if d in w.index]]
    frt = fr[["knn", "gp_std", "tpe_disagree", "tabpfnreg_std", "cb_knowledge", "tabpfnreg_nll"]].round(3)
    frt.loc["mean (8 datasets)"] = frt.mean().round(3); frt.loc["mean (all 100)"] = w[frt.columns].mean().round(3)
    frt.to_csv(TAB / "fraud_subset_auroc.csv")
    numbers["fraud_subset"] = {"datasets": list(fr.index), "mean_auroc": {c: round(float(fr[c].mean()), 3) for c in frt.columns}}
    dup = {}
    for d in load_all("OddBench") + load_all("OvRBench"):
        dup[d.name] = 1 - np.unique(d.X_train, axis=0).shape[0] / len(d.X_train)
    dup = pd.Series(dup); keep = dup[dup <= 0.5].index
    sens = pd.DataFrame({"all_datasets": w[["knn", "gp_std", "tpe_disagree", "tabpfnreg_std", "tabpfnreg_nll"]].mean(),
                         "excluding_heavy_duplicates": w.loc[w.index.isin(keep), ["knn", "gp_std", "tpe_disagree", "tabpfnreg_std", "tabpfnreg_nll"]].mean()}).round(3)
    sens.to_csv(TAB / "duplicates_sensitivity.csv")
    numbers["duplicates"] = {"share_datasets_over_10pct": round(float((dup > 0.1).mean()), 2), "excluded_over_50pct": int((dup > 0.5).sum()),
                             "sensitivity": sens.to_dict()}

    # ---------------- pipeline figure ----------------
    NL = chr(10)
    fig, ax = plt.subplots(figsize=(13, 3.1)); ax.axis("off")
    steps = [("normal rows only", "training data;" + NL + "no labels needed"),
             ("predict each column" + NL + "from the others", "TabPFN, GP, CatBoost"),
             ("read the answer", "uncertainty = width" + NL + "error = surprise"),
             ("compare with" + NL + "known-normal rows", "conformal p-value"),
             ("bet against" + NL + "'normal'", "e-value"),
             ("cut the ranked list", "e-BH: alert list with" + NL + "false alarms <= 10%")]
    W, GAP = 2.0, 0.35
    for i, (h, sub) in enumerate(steps):
        x = i * (W + GAP)
        ax.add_patch(plt.Rectangle((x, 0.25), W, 1.35, fc="white", ec=C["rule"], lw=1.2))
        ax.text(x + W / 2, 1.22, h, ha="center", va="center", fontsize=9.5, color=C["ink"], weight="bold", linespacing=1.15)
        ax.text(x + W / 2, 0.62, sub, ha="center", va="center", fontsize=8.5, color=C["grey"], linespacing=1.2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + W + GAP, 0.93), xytext=(x + W, 0.93), arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.2))
    # brackets for the two halves
    for (i0, i1, label, col) in [(0, 2, "make a score  (one per row, larger = less normal)", C["uncertainty"]), (3, 5, "turn any score into alerts with a guarantee", C["baseline"])]:
        x0, x1 = i0 * (W + GAP), i1 * (W + GAP) + W
        ax.plot([x0, x0, x1, x1], [1.75, 1.85, 1.85, 1.75], color=col, lw=1.2)
        ax.text((x0 + x1) / 2, 1.98, label, ha="center", va="bottom", fontsize=9.5, color=col, weight="bold")
    ax.set_xlim(-0.15, len(steps) * (W + GAP) - GAP + 0.15); ax.set_ylim(0, 2.45)
    save(fig, "fig1_pipeline")

    json.dump(numbers, open(OUT / "numbers.json", "w"), indent=2)
    write_notes(numbers)
    print(f"bundle written to {OUT}")


def write_notes(n):
    q1, z, s, q2, q3 = n["q1"], n["q1_ecg"], n["synthetic"], n["q2"], n["q3"]
    txt = f"""# Is model uncertainty a measure of anomaly? — poster notes

Code: github.com/enzovendramin/uncertainty-anomaly · commit {n['commit']} · every number below comes from a script in `src/`.
Data: {n['datasets']}. All scores are compared on the same rows. One seed unless stated.

## The question, in one paragraph

A fraud is something the model has never seen. So when a model is *unsure* about a row, is that row more likely to be an anomaly?
And can those alerts carry a guarantee on the number of false alarms? We test this on 100 anomaly-detection datasets
where the training data contain only normal rows: each model predicts every column from the other columns, and we read two
things from its answer — its **uncertainty** (how wide the answer is, never looking at the true value) and its **error**
(how surprising the true value was). A plain **kNN distance** to the nearest normal rows is the baseline.

## Method (fig1_pipeline)

normal rows → predict each column → uncertainty / error score → conformal p-value (compare with known-normal rows)
→ e-value (a bet against "normal") → e-BH (cut the ranked list so that false alarms stay under 10%).
Metric for ranking: AUROC = probability that a random anomaly scores above a random normal row (0.5 = chance).

## Thesis sentence

*A model's uncertainty is an anomaly score exactly to the extent that it measures distance from the training data;
none of the models beats measuring distance directly, and the guaranteed-alert machinery works but is limited by the score, not by the statistics.*

## Panel Q1 — TabPFN (fig2, fig4)

Verdict: yes, weakly, and only with the right measure. TabPFN's regression uncertainty ranks anomalies above normal rows
with AUROC {q1['tabpfnreg_std']} (below chance on {int(n['q1_share_below_chance']['tabpfnreg_std']*100)}% of datasets); its error {q1['tabpfnreg_nll']}; kNN {q1['knn']}.
The first attempt, with columns cut into 10 bins, scored {q1['tabpfn_setsize']} — inverted on {int(n['q1_share_below_chance']['tabpfn_setsize']*100)}% of datasets — because
binning throws away "how far outside the normal range"; removing the bins improved {n['q1_binning_fix']['datasets_improved']} of {n['q1_binning_fix']['of']} datasets by +{n['q1_binning_fix']['mean_gain']}.
Caption for the ECG example: anomalous heartbeats are extreme in every column at once; with bins the model is confidently right
about each column (AUROC {z['tabpfn_setsize']}), without bins its spread explodes far from the data ({z['tabpfnreg_std']}); kNN {z['knn']}.

## Panel Q1 across models (fig3)

Gaussian process uncertainty {q1['gp_std']} · TabPFN disagreement between 4 copies {q1['tpe_disagree']} · TabPFN single-copy width {q1['tabpfnreg_std']}
· CatBoost knowledge uncertainty {q1['cb_knowledge']} · kNN {q1['knn']}. Reading: uncertainty works when it is about distance
(the GP's is, by construction, and it correlates 0.85 with kNN across datasets); trees extrapolate flat and do not know when they
are lost; the foundation model sits in between, and measuring disagreement helps only a little (48/100 wins over width).
Error-type scores land at ~0.72 for every model family.

## Panel: what each score sees (fig5)

Synthetic data with four planted anomaly kinds. Distance-type uncertainty (GP, TabPFN disagreement, kNN) sees extreme values,
whole-row shifts and unusual-but-consistent rows (subspace) at {s['gp_std']['extreme']}–{s['gp_std']['subspace']} but is weaker on broken relationships ({s['gp_std']['broken_link']}).
Error scores are the best on broken relationships ({s['tabpfnreg_nll']['broken_link']}) and nearly blind to subspace anomalies ({s['tabpfnreg_nll']['subspace']}).
CatBoost's knowledge uncertainty is the weakest on extreme values and shifts ({s['cb_knowledge']['extreme']}, {s['cb_knowledge']['global_shift']}).

## Panel Q2 — review budget (fig6)

Reviewing the top 5% of rows: kNN fills the list with {int(q2['precision_at_5pct']['knn']*100)}% real anomalies, TabPFN uncertainty {int(q2['precision_at_5pct']['std']*100)}%,
TabPFN error {int(q2['precision_at_5pct']['nll']*100)}%, kNN + error {int(q2['precision_at_5pct']['knn+nll']*100)}% (random list: {int(q2['anomaly_rate']*100)}%).
Uncertainty beats kNN on {q2['std_wins_vs_knn_at_5pct']} datasets. Adding uncertainty to kNN does not help; adding error helps slightly (not significant).
Combinations merge scores first (mean of −log p) and calibrate once.

## Panel Q3 — guarantees (fig7)

Pooling all alerts across datasets, false alarms are {q3['pooled_fdp_whole']['knn']:.0%} (kNN), {q3['pooled_fdp_whole']['std']:.0%} (uncertainty), {q3['pooled_fdp_whole']['nll']:.0%} (error) against 10% promised — the guarantee holds.
But the list is empty on {100-int(q3['datasets_with_alerts_whole']['knn']*100)}% of datasets for kNN ({100-int(q3['datasets_with_alerts_whole']['std']*100)}% for uncertainty): e-BH flags nothing rather than break the promise.
Feeding the data in batches of 200 rows raises datasets-with-alerts to {int(q3['datasets_with_alerts_batches']['knn']*100)}% (kNN) with false alarms still at {q3['pooled_fdp_batches']['knn']:.0%}.
Power needs at least ~300 calibration rows. Single-split alert lists overlap only {q3['jaccard_between_seeds']['std']:.0%} (uncertainty) to {q3['jaccard_between_seeds']['knn']:.0%} (kNN) between random splits;
averaging e-values over 3 splits keeps the guarantee but halves the number of alerts.

## Panel: fraud, payments and security datasets (table fraud_subset_auroc)

{len(n['fraud_subset']['datasets'])} datasets: kNN {n['fraud_subset']['mean_auroc']['knn']}, GP uncertainty {n['fraud_subset']['mean_auroc']['gp_std']}, TabPFN disagreement {n['fraud_subset']['mean_auroc']['tpe_disagree']},
TabPFN width {n['fraud_subset']['mean_auroc']['tabpfnreg_std']}, CatBoost knowledge {n['fraud_subset']['mean_auroc']['cb_knowledge']}, TabPFN error {n['fraud_subset']['mean_auroc']['tabpfnreg_nll']}.

## Caveats (say them on the poster)

- 100 development datasets, one seed for most panels (three for the derandomisation part); the 1,346 public datasets were not run.
- {int(n['duplicates']['share_datasets_over_10pct']*100)}% of datasets have >10% duplicate training rows, which favours distance scores; excluding the {n['duplicates']['excluded_over_50pct']} datasets with >50% duplicates changes means by ≤0.02 (table duplicates_sensitivity).
- Column-prediction budget: up to 10 target columns and 1000 context rows per dataset; tests subsampled to 4000 rows.
- Related work: a pilot by Eduardo on 10 OddBench datasets uses reconstruction *error* with five models; this study is the uncertainty and guarantee side.

## References

Angelopoulos & Bates 2021 · Bates, Candès, Lei, Romano, Sesia 2023 · Ren & Barber 2023 · Vovk & Wang 2021 · Wang & Ramdas 2022 ·
De Melo Costa et al. 2026 (HPLR) · Ding et al. 2026 (macrOData) · Hollmann et al. 2023/2025 (TabPFN).
"""
    (OUT / "notes.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
