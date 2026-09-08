#!/usr/bin/env python3
"""Submit shared capture, CPU search, full preflight, and measurement jobs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiment_support import source_identity
from tritonbench_cases import CASE_BY_ID


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    p.add_argument("--suite-root", type=Path, default=Path(os.environ.get("RELAY_FINAL_RESULTS_ROOT", HERE / "results")) / "search-v2")
    p.add_argument("--cases", nargs="+", default=os.environ.get("RELAY_SEARCH_CASES", " ".join(CASE_BY_ID)).split())
    p.add_argument("--tau-names", nargs="+", choices=("expert", "l1_to_l2", "speedup"), default=os.environ.get("RELAY_FINAL_TAU_NAMES", "expert l1_to_l2 speedup").split())
    p.add_argument("--experiments", nargs="+", type=int, choices=(4, 5, 6), default=[4, 5, 6])
    p.add_argument("--dry-run", action="store_true")
    args, forwarded = p.parse_known_args()
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit("Run from the RELAY repository root")
    if set(args.cases) - set(CASE_BY_ID):
        raise SystemExit(f"Unknown cases: {set(args.cases) - set(CASE_BY_ID)}")
    args.suite_root = args.suite_root.resolve()
    manifest = {"platform": args.platform, "source_identity": source_identity(), "cases": args.cases,
                "experiments": args.experiments, "tau_names": args.tau_names, "jobs": []}
    manifest_path = args.suite_root / "shared" / args.platform / "submission.json"
    queue = os.environ.get("RELAY_SEARCH_TUOLUMNE_QUEUE" if args.platform == "tuolumne" else "RELAY_SEARCH_MATRIX_PARTITION", "pbatch")
    common = ["--platform", args.platform, "--suite-root", str(args.suite_root),
              "--experiments", *map(str, args.experiments), "--tau-names", *args.tau_names, *forwarded]

    def submit(stage, case=None, dependencies=(), after="afterany", cpu=False, command=None):
        name = f"relay-v2-{stage}-{case or args.platform}"
        logs = args.suite_root / "shared" / args.platform / "logs"
        if not args.dry_run:
            logs.mkdir(parents=True, exist_ok=True)
        default_time = {"capture": "1h", "search": "4h", "validate": "2h", "measure": "4h", "preflight": "10m", "summary": "10m"}[stage]
        duration = os.environ.get(f"RELAY_SEARCH_TIME_{stage.upper()}", default_time)
        if args.platform == "tuolumne":
            scheduler = ["flux", "submit", "-N", "1", "-n", "1", "-c", "1", "-q", queue,
                "-t", duration, "--cwd", str(ROOT), "--job-name", name,
                "--output", str(logs / f"{name}.out"), "--error", str(logs / f"{name}.err")]
            if not cpu:
                scheduler += ["-g", "1"]
            for dependency in dependencies:
                scheduler += ["--dependency", f"{after}:{dependency}"]
        else:
            # Slurm accepts integer minutes; convert the common h/m duration.
            minutes = int(duration[:-1]) * (60 if duration.endswith("h") else 1)
            scheduler = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
                f"--partition={queue}", f"--time={minutes}", f"--chdir={ROOT}", f"--job-name={name}",
                f"--output={logs / (name + '-%j.out')}", f"--error={logs / (name + '-%j.err')}"]
            if not cpu:
                scheduler += ["--gpus=1"]
            if dependencies:
                scheduler += [f"--dependency={after}:" + ":".join(dependencies)]
        if command is None:
            command = ([str(ROOT / ".venv/bin/python"), str(HERE / "run-search-suite.py")]
                       if cpu else [str(HERE / f"run-search-suite-{args.platform}-job.bash")])
            command += ["--stage", stage, "--case", case, *common]
        if args.platform == "matrix" and cpu:
            scheduler += ["--wrap", shlex.join(command)]
            invocation = scheduler
        else:
            invocation = scheduler + command
        if args.dry_run:
            job = f"DRY{len(manifest['jobs']) + 1}"
            print(shlex.join(invocation))
        else:
            job = subprocess.check_output(invocation, text=True).strip().split(";", 1)[0]
            print(f"{stage} {case or args.platform}: {job}")
        manifest["jobs"].append({"stage": stage, "case": case, "job_id": job, "command": invocation})
        if not args.dry_run:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temp = manifest_path.with_suffix(".tmp")
            temp.write_text(json.dumps(manifest, indent=2) + "\n")
            temp.replace(manifest_path)
        return job

    validations = {}
    for case in args.cases:
        capture = submit("capture", case)
        search = submit("search", case, [capture], cpu=True)
        validations[case] = submit("validate", case, [search])
    preflight = submit("preflight", dependencies=list(validations.values()), after="afterany", cpu=True,
        command=[str(ROOT / ".venv/bin/python"), str(HERE / "preflight-search.py"),
                 "--suite-root", str(args.suite_root), "--platform", args.platform])
    measurements = [submit("measure", case, [preflight, validation]) for case, validation in validations.items()]
    submit("summary", dependencies=measurements, after="afterany", cpu=True,
        command=[str(ROOT / ".venv/bin/python"), str(HERE / "summarize-search-suite.py"),
                 "--suite-root", str(args.suite_root), "--platform", args.platform])
    print(f"Suite: {args.suite_root}")


if __name__ == "__main__":
    main()
