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
p = \frac{1 + \#\{i : s_i \ge s\}}{n+1}, \qquad P_{H_0}(p \le a) \le a .
$$

E-values follow Ren & Barber (2023): run BH at level $q$ on the $p$-values of the batch,
let $T$ be the smallest flagged score, and set

$$
e = \frac{n+1}{1 + \#\{i : s_i \ge T\}}\,\mathbb{1}\{s \ge T\}, \qquad \mathbb{E}_{H_0}[e] \le 1 .
$$

e-BH (Wang & Ramdas 2022) sorts $e_{(1)} \ge \dots \ge e_{(m)}$ and flags the top
$k^\* = \max\{k : e_{(k)} \ge m/(qk)\}$, which controls $\mathrm{FDR} \le q$ under arbitrary
dependence. Averages of e-values (over seeds, splits or models) remain e-values.

Scores compared:

| score | what it measures |
|---|---|
| `knn`, `iforest` | classical baselines |
| `tabpfn_error` | $1 - \hat p(\text{observed bin})$, TabPFN predicting each column from the others (as TabPFN-OD) |
| `tabpfn_entropy` | $H(\hat p)/\log K$ — uncertainty, blind to the observed value |
| `tabpfn_setsize` | $|\{y : 1-\hat p(y) \le \hat q\}| / K$ — the same through a conformal set (LAC, $\alpha = 0.1$) |

## Data

MacrOData (Ding et al., KDD 2026): OddBench (semantic anomalies) and OvRBench (one-vs-rest).
One `.npz` per dataset; `train` holds normal points only, `test` holds normal points and
anomalies with labels. The normal train set is split 70/30 into fit and calibration;
test labels are used only for evaluation (`splits.md`).

## Usage

```bash
pip install -r requirements.txt
python -m pytest tests                       # guarantees on simulated data
python src/data.py                           # download the representative subsets
python src/run.py --scores knn iforest --seeds 0 1 2 --tag baseline
python src/run.py --scores tabpfn --seeds 0 --max-calib 3000 --max-test 4000 --tag tabpfn
python src/summarize.py results/baseline.csv
```

## References

Angelopoulos & Bates (2021) · Bates, Candès, Lei, Romano, Sesia (2023) ·
Ren & Barber (2023) · Vovk & Wang (2021) · Wang & Ramdas (2022) ·
De Melo Costa et al. (2026) · Ding et al. (2026) · Hollmann et al. (2023, 2025).
