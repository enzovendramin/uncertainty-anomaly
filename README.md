Status: work in progress (2026–27 research project). Results below are preliminary and will change.

# Is model uncertainty a measure of anomaly?

Research project, Filière Métiers de la Recherche 2026–27, CentraleSupélec.
Extends *High Performance, Low Reliability* (De Melo Costa et al., ESANN 2026)
from prediction sets to anomaly detection with false-discovery-rate control.

## Questions

- **Q1** Does a model's uncertainty (conformal set size, empty sets) rank anomalies above normal points?
- **Q2** At equal review budget, does an uncertainty score catch more anomalies than the raw probability?
- **Q3** Can the alerts carry a guarantee? Conformal p-/e-values → e-BH → alert list with FDR ≤ q.

## Method

Let $s(x)$ be an anomaly score (larger = less normal) and $s_1,\dots,s_n$ its values on
$n$ calibration points known to be normal and unseen by the model.
For a test point with score $s$, under $H_0$: "the point is normal",

$$
p = \frac{1 + \lvert\{i : s_i \ge s\}\rvert}{n+1}, \qquad P_{H_0}(p \le a) \le a .
$$

E-values follow Ren & Barber (2023): run BH at level $q$ on the $p$-values of the batch,
let $T$ be the smallest flagged score, and set

$$
e = \frac{n+1}{1 + \lvert\{i : s_i \ge T\}\rvert}\,\mathbb{1}\{s \ge T\}, \qquad \mathbb{E}_{H_0}[e] \le 1 .
$$

e-BH (Wang & Ramdas 2022) sorts $e_{(1)} \ge \dots \ge e_{(m)}$ and flags the top
$k^{*} = \max\{k : e_{(k)} \ge m/(qk)\}$, which controls $\mathrm{FDR} \le q$ under arbitrary
dependence. Averages of e-values (over seeds, splits or models) remain e-values.

Scores compared:

| score | what it measures |
|---|---|
| `knn`, `iforest` | classical baselines |
| `tabpfn_error` | $1 - \hat p(\text{observed bin})$, TabPFN predicting each column from the others (as TabPFN-OD) |
| `tabpfn_entropy` | $H(\hat p)/\log K$ — uncertainty, blind to the observed value |
| `tabpfn_setsize` | $\lvert\{y : 1-\hat p(y) \le \hat q\}\rvert / K$ — the same through a conformal set (LAC, $\alpha = 0.1$) |
| `tabpfnreg_std`, `tabpfnreg_width90` | predictive std / 90% interval width of the TabPFN *regressor* (no binning): uncertainty that can grow away from the data |
| `tabpfnreg_nll`, `tabpfnreg_pit` | $-\log \hat p(x_j)$ and $\lvert 2F(x_j)-1\rvert$ from the same predictive distribution: error-type |
| `gp_std`, `gp_nll` | Gaussian process in the same frame — uncertainty that is distance to the data by construction (positive control) |
| `cb_knowledge`, `cb_data`, `cb_nll` | CatBoost virtual ensembles: epistemic, aleatoric, error |
| `tpe_disagree`, `tpe_aleatoric`, `tpe_total` | disagreement between single-estimator TabPFN members (epistemic proxy), mean member variance, total |

## Data

MacrOData (Ding et al., KDD 2026): OddBench (semantic anomalies) and OvRBench (one-vs-rest).
One `.npz` per dataset; `train` holds normal points only, `test` holds normal points and
anomalies with labels. The normal train set is split 70/30 into fit and calibration;
test labels are used only for evaluation (`splits.md`).

Per-row scores and raw model outputs are saved under `results/<tag>/` so new
measures and combinations are computed offline. Conformal p-values break ties
at random (seeded), and every results row records the git commit.

Known data caveat: about 18% of the representative datasets have more than 10%
duplicate rows in `train`, a few over 90%; this favours distance-based scores.

## Usage

```bash
pip install -r requirements.txt
python -m pytest tests                       # guarantees on simulated data
python src/data.py                           # download the representative subsets
python src/run.py --scores knn iforest --seeds 0 1 2 --tag baseline
python src/run.py --scores tabpfn tabpfn_reg --seeds 0 --max-calib 3000 --max-test 4000 --tag tabpfn
python src/summarize.py results/baseline.csv
python src/q2.py --tag tabpfn                # Q2: precision at a budget, offline from saved scores
```

## Findings so far (100 representative datasets)

- Q1: regression uncertainty (`tabpfnreg_std`) AUROC 0.66, error (`nll`) 0.72, kNN 0.76;
  the binned classifier (0.45) was a measurement artefact.
- Q2: at a fixed review budget uncertainty does not beat kNN; kNN + error gains ~0.02 (n.s.).
- Q3: FDR guarantee holds (pooled FDP 0.07–0.08 at q = 0.1) but the list is empty on ~75%
  of datasets; batches of 200 rows raise that to 45%; keep ≥ 300 calibration rows.
  Combining scores: merge first (mean of −log p), then calibrate once — averaging
  Ren & Barber e-values across scores or splits keeps validity but halves power.
  Single-split alert lists overlap only 29–56% between splits.

## References

Angelopoulos & Bates (2021) · Bates, Candès, Lei, Romano, Sesia (2023) ·
Ren & Barber (2023) · Vovk & Wang (2021) · Wang & Ramdas (2022) ·
De Melo Costa et al. (2026) · Ding et al. (2026) · Hollmann et al. (2023, 2025).
