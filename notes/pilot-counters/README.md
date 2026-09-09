Paper assets in `pilot-counters.tex`:

- Figure: Experiment 1, H100, whole-tensor canonical layouts, all-scope
  stratification; native issue Q_32B versus L1 global-load sectors.
- Table: Experiments 1 (G_C) and 3 (GL(p,2)), H100 and MI300A,
  all-scope stratification; J_area with each device's l1_to_l2 tau profile
  versus L1-to-L2 read requests. Shows only the median across six kernels for each layout family and device.
  Uses the recorded pre-packet pilot results.

Bias+ReLU is excluded from both assets.

From the repository root, regenerate the figure, table, and source snapshot:

```bash
.venv/bin/python notes/pilot-counters/generate.py
```

Compile from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error pilot-counters.tex
pdflatex -interaction=nonstopmode -halt-on-error pilot-counters.tex
```

The generator reuses the existing quotient plot's font selection and the
Stage-1 average-rank Spearman implementation. `source-data.json` records the
30 source report paths and SHA-256 hashes, observations, table tau maps, and
full-precision correlations, separately for the figure and table.
`l2-correlations.csv` records the four aggregate table values and their
experiment, stratification, tau, predictor, and counter fields. The 24
underlying per-kernel correlations are retained in `source-data.json`.
Medians are checked against `../pilot-l2-correlations/medians.csv`.
Inputs are read only; no GPU profiling or score fitting is performed.
