# Who sees what

The supervisor's rule: keep a table with one row per step, saying which split
that step is allowed to look at. Calibration and test must never overlap,
and the test labels are used only in the last row.

Each MacrOData dataset gives us `train` (normal only) and `test` (normal +
anomalies, labelled). We further cut `train` into two parts with a fixed seed.

| step | fit (70% of train) | calibration (30% of train) | test features | test labels |
|---|---|---|---|---|
| fit the model / kNN index | yes | no | no | no |
| compute calibration scores | (model) | yes | no | no |
| compute test scores | (model) | no | yes | no |
| p-values, e-values, e-BH | (calibration scores) | (calibration scores) | (test scores) | no |
| evaluation: AUROC, AUPRC, FDP, recall | no | no | no | yes |

Rules

- The split of `train` into fit/calibration uses `seed`; the same seed gives
  the same split for every score function, so comparisons are paired.
- Nothing is ever tuned on `test`. If we ever add a hyper-parameter, it is
  chosen on the fit part only.
- Every number in the report is produced by `src/run.py` and written to
  `results/`. Nothing is typed by hand.
