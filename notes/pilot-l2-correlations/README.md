# Pilot L2 correlation report

Open `correlations.html` for filterable tables, or read `correlations.md` for
all tables in a single document. Both contain per-kernel Spearman rho and the
median across kernels with defined rho. Filters cover device, tau profile,
experiment, and stratification. No LaTeX files are changed by this analysis.

- `per-kernel.csv`: full-precision correlations, layout counts, and reasons for
  undefined values.
- `medians.csv`: full-precision medians and contributing kernel counts.
- `provenance.json`: source report paths, SHA-256 hashes, and active tau maps.

Reproduce from the repository root:

```bash
.venv/bin/python notes/pilot-l2-correlations/generate.py
```

The generator recomputes correlations from saved J_area and steady-state
counter observations, checks all matching existing `spearman.csv` values,
and adds collected L2-related summaries omitted by the older analysis.
