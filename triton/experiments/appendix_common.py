"""Shared helpers for the final appendix experiments."""

from __future__ import annotations

from dataclasses import replace
from importlib import metadata
import json
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlparse


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRITON_ROOT = EXPERIMENT_ROOT.parent
REPOSITORY = TRITON_ROOT.parent
sys.path[:0] = (
    str(EXPERIMENT_ROOT),
    str(TRITON_ROOT / "tritonbench"),
    str(TRITON_ROOT),
    str(REPOSITORY),
)

from relay import HardwareProfile
from tritonbench_cases import OPERATORS, selected_cases


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError("value must be positive")
    return result


def fraction(value: str) -> float:
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ValueError("perturbation magnitudes must be between zero and one")
    return result


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def selected_case(operator: str, config: str):
    matches = selected_cases(operator, config)
    if len(matches) != 1:
        raise ValueError(f"expected one case, found {len(matches)}")
    return matches[0]


def activate_triton_source(platform: str) -> None:
    """Put the platform-specific editable Triton checkout first on sys.path."""

    import os

    configured = os.environ.get("RELAY_TRITON_PYTHON_ROOT")
    if configured:
        source = Path(configured)
    elif platform == "tuolumne":
        source = TRITON_ROOT / "triton-lang" / "python"
    else:
        direct = metadata.distribution("triton").read_text("direct_url.json")
        if direct is None:
            raise RuntimeError("Matrix Triton install has no editable-source metadata")
        checkout = Path(unquote(urlparse(json.loads(direct)["url"]).path))
        source = checkout / "python"
    if not (source / "triton" / "__init__.py").is_file():
        raise RuntimeError(f"invalid platform Triton Python source: {source}")
    sys.path.insert(0, str(source.resolve()))


def profile_with_tau(
    baseline: HardwareProfile,
    *,
    profile_id: str,
    tau: Mapping[str, float],
) -> HardwareProfile:
    return HardwareProfile(
        profile_id=profile_id,
        device=baseline.device,
        byte_scales=baseline.byte_scales,
        fine_component=baseline.fine_component,
        tau=tau,
        kappa=baseline.kappa,
        resource_maps=baseline.resource_maps,
    )


def perturbed_profiles(
    baseline: HardwareProfile,
    magnitudes: Sequence[float],
    trials_per_magnitude: int,
    seed: int,
) -> tuple[tuple[dict[str, Any], HardwareProfile], ...]:
    """Build deterministic, globally reusable independent tau perturbations."""

    if trials_per_magnitude <= 0:
        raise ValueError("trials_per_magnitude must be positive")
    ordered_magnitudes = tuple(float(value) for value in magnitudes)
    if not ordered_magnitudes or any(
        not 0.0 < magnitude < 1.0 for magnitude in ordered_magnitudes
    ):
        raise ValueError("magnitudes must be nonempty and between zero and one")

    active_tau = {
        name: float(value)
        for name, value in sorted(baseline.tau.items())
        if value > 0
    }
    if not active_tau:
        raise ValueError("the hardware profile has no positive tau entries")

    result: list[tuple[dict[str, Any], HardwareProfile]] = [
        (
            {
                "trial_id": "nominal",
                "magnitude": 0.0,
                "trial_index": 0,
                "factors": {name: 1.0 for name in active_tau},
                "tau": active_tau,
            },
            profile_with_tau(
                baseline,
                profile_id=f"{baseline.profile_id}-nominal",
                tau=active_tau,
            ),
        )
    ]
    generator = random.Random(seed)
    for magnitude in ordered_magnitudes:
        label = format(magnitude, ".6g").replace(".", "p")
        for trial_index in range(1, trials_per_magnitude + 1):
            factors = {
                name: generator.uniform(1.0 - magnitude, 1.0 + magnitude)
                for name in active_tau
            }
            tau = {
                name: active_tau[name] * factors[name]
                for name in active_tau
            }
            trial_id = f"m{label}-trial-{trial_index:02d}"
            result.append(
                (
                    {
                        "trial_id": trial_id,
                        "magnitude": magnitude,
                        "trial_index": trial_index,
                        "factors": factors,
                        "tau": tau,
                    },
                    profile_with_tau(
                        baseline,
                        profile_id=f"{baseline.profile_id}-{trial_id}",
                        tau=tau,
                    ),
                )
            )
    return tuple(result)


def output_arguments(analysis) -> tuple[int, ...]:
    argument_names = {
        str(name): int(index)
        for index, name in analysis.bound_arguments.get("__names__", {}).items()
    }
    return tuple(
        sorted(
            {
                (
                    int(allocation.argument)
                    if isinstance(allocation.argument, int)
                    else argument_names[str(allocation.argument)]
                )
                for allocation in analysis.allocations
                if allocation.role != "read"
                and not allocation.path
                and (
                    isinstance(allocation.argument, int)
                    or str(allocation.argument) in argument_names
                )
            }
        )
    )


def runtime_layout_key(layouts: Sequence[Mapping[str, Any]]) -> str:
    import hashlib

    payload = json.dumps(list(layouts), sort_keys=True, separators=(",", ":"))
    return "layout-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _run(launch, layouts) -> None:
    from layout_runtime import rewrite_layouts

    if layouts:
        with rewrite_layouts(layouts):
            launch.run()
    else:
        launch.run()


def _validate_outputs(baseline, selected, outputs: Sequence[int]) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize()
    records = []
    for argument in outputs:
        expected = baseline.values[argument]
        observed = selected.values[argument]
        close = torch.allclose(
            observed,
            expected,
            rtol=1e-2,
            atol=5e-2,
            equal_nan=True,
        )
        error = float((observed.float() - expected.float()).abs().max().item())
        records.append(
            {
                "argument": argument,
                "allclose": bool(close),
                "max_abs_error": error,
            }
        )
    if not records or not all(record["allclose"] for record in records):
        raise RuntimeError(
            f"transformed layout failed numerical validation: {records}"
        )
    return {"correct": True, "outputs": records}


def time_layout_pair(
    spec,
    selected_config: Mapping[str, Any],
    runtime_layout_values: Sequence[Mapping[str, Any]],
    outputs: Sequence[int],
    *,
    warmup: int,
    samples: int,
    iterations: int,
    order_offset: int = 0,
) -> dict[str, Any]:
    """Validate and time baseline/selected launches in alternating order."""

    import torch
    from layout_runtime import (
        RuntimeLayout,
        freeze_launch,
        fresh_outputs,
        replace_inputs,
        rewrite_layouts,
    )

    layouts = tuple(
        RuntimeLayout.from_dict(value) for value in runtime_layout_values
    )
    frozen = freeze_launch(spec, selected_config)
    baseline = fresh_outputs(frozen, outputs)
    selected = fresh_outputs(replace_inputs(frozen, layouts), outputs)
    _run(baseline, ())
    _run(selected, layouts)
    validation = _validate_outputs(baseline, selected, outputs)

    for _ in range(warmup):
        _run(baseline, ())
        _run(selected, layouts)
    torch.cuda.synchronize()

    values = {"baseline": [], "selected": []}
    launches = {"baseline": baseline, "selected": selected}
    labels = ("baseline", "selected")
    for sample in range(samples):
        order = (
            labels
            if (sample + order_offset) % 2 == 0
            else tuple(reversed(labels))
        )
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if label == "selected" and layouts:
                with rewrite_layouts(layouts):
                    for _ in range(iterations):
                        launches[label].run()
            else:
                for _ in range(iterations):
                    launches[label].run()
            end.record()
            end.synchronize()
            values[label].append(float(start.elapsed_time(end)) / iterations)

    summaries = {
        label: {
            "median_ms": statistics.median(observations),
            "mean_ms": statistics.fmean(observations),
            "min_ms": min(observations),
            "samples_ms": observations,
        }
        for label, observations in values.items()
    }
    return {
        "schema": "relay.triton.appendix_timing.v1",
        "configuration": {
            "warmup": warmup,
            "samples": samples,
            "iterations": iterations,
        },
        "validation": validation,
        "baseline": summaries["baseline"],
        "selected": summaries["selected"],
        "speedup": (
            summaries["baseline"]["median_ms"]
            / summaries["selected"]["median_ms"]
        ),
    }


def timed_repetitions(
    function: Callable[[], Any], repeats: int
) -> tuple[Any, dict[str, Any]]:
    """Run a deterministic CPU phase repeatedly and summarize wall time."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    values = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        values.append(perf_counter() - start)
    return result, {
        "repeats": repeats,
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.fmean(values),
        "min_seconds": min(values),
        "samples_seconds": values,
    }


def exclusion_report(
    *,
    experiment: int,
    platform: str,
    case,
    category: str,
    message: str,
    search_experiment: int,
    trace_capture_seconds: float | None = None,
) -> dict[str, Any]:
    result = {
        "schema": f"relay.triton.experiment_{experiment}.v1",
        "experiment": experiment,
        "platform": platform,
        "operator": case.operator,
        "config": case.config,
        "description": case.description,
        "search_experiment": search_experiment,
        "status": "excluded",
        "exclusion": {"category": category, "message": message},
    }
    if trace_capture_seconds is not None:
        result["trace_capture_seconds"] = trace_capture_seconds
    return result


def replace_target_matrices(analysis):
    """Apply the same direct, dense, read-only eligibility gate as search."""

    argument_names = {
        str(name): int(index)
        for index, name in analysis.bound_arguments.get("__names__", {}).items()
    }

    def direct_argument(allocation):
        if allocation.path:
            return None
        if isinstance(allocation.argument, int):
            return allocation.argument
        return argument_names.get(str(allocation.argument))

    eligible_names = {
        allocation.name
        for allocation in analysis.allocations
        if allocation.eligible
        and allocation.dense_status == "dense"
        and direct_argument(allocation) is not None
    }
    matrices = tuple(
        replace(matrix, target=matrix.name in eligible_names)
        for matrix in analysis.matrices
    )
    if not any(matrix.target for matrix in matrices):
        raise RuntimeError("no read-only ordinary-dense direct pointer is realizable")
    return matrices


__all__ = [
    "EXPERIMENT_ROOT",
    "OPERATORS",
    "REPOSITORY",
    "activate_triton_source",
    "exclusion_report",
    "fraction",
    "output_arguments",
    "perturbed_profiles",
    "positive",
    "replace_target_matrices",
    "runtime_layout_key",
    "selected_case",
    "time_layout_pair",
    "timed_repetitions",
    "write_json",
]
