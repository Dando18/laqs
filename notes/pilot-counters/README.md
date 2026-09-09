Paper assets in `pilot-counters.tex`:

- Figure: Experiment 1, H100, whole-tensor canonical layouts, all-scope
  stratification; native issue Q_32B versus L1 global-load sectors.
- Table: Experiment 2, H100 and MI300A, canonical inner-tile layouts,
  all-scope stratification; J_area with each device's l1_to_l2 tau profile
  versus L1-to-L2 read requests. Includes the median across six kernels.

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
18 source report paths and SHA-256 hashes, observations, table tau maps, and
full-precision correlations, separately for the figure and table.
`l2-correlations.csv` records all 12 per-kernel table values and their
experiment, stratification, tau, predictor, and counter fields. These values
were checked against the complete pilot L2 correlation report in
`../pilot-l2-correlations/per-kernel.csv`. Inputs are read only; no GPU
profiling or score fitting is performed.
