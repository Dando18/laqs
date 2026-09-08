#!/usr/bin/env python3
"""Refresh all per-device diagnostics and the full cross-platform coverage audit."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--suite-root", type=Path, required=True)
p.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
a = p.parse_args()
for experiment in (4, 5, 6):
    for tau in ("expert", "l1_to_l2", "speedup"):
        subprocess.run([sys.executable, str(HERE / "analyze-search.py"), "--experiment", str(experiment),
            "--platform", a.platform, "--tau-name", tau, "--results-root", str(a.suite_root),
            "--plots-root", str(a.suite_root / "plots")], check=True)
subprocess.run([sys.executable, str(HERE / "preflight-search.py"), "--suite-root", str(a.suite_root),
                "--platform", a.platform], check=True)
