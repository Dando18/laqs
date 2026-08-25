#!/usr/bin/env python3
"""Build an exhaustive shared-word G_S or G_C scoring result corpus.

The experiment applies one complete canonical word to every target matrix in
each kernel.  It computes the exact analytical frontiers on a CPU, then
correctness-checks and times every word to obtain a genuine runtime oracle.
Raw timing records are append-only JSONL checkpoints; the compact JSONL output
contains one plot-oriented summary record per kernel and matrix size.

Prepare the analytical plan on a login node, optionally seeding the existing
73-layout corpus::

    .venv/bin/python experiments/scoring_results.py \
        --prepare-only --seed-timings results/layout_ranking.json

Resume a bounded timing chunk inside an MI300A allocation::

    flux run -n1 -g1 -t 5m -q pdebug \
        .venv/bin/python experiments/scoring_results.py --resume \
        --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
        --max-benchmarks 40
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations
import json
from math import comb
import os
from pathlib import Path
import random
import sys
from typing import Iterable, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.layout_ranking import (
    KERNEL_SPECS,
    KernelSpec,
    LayoutCase,
    benchmark_case,
)
from relay import (
    HARDWARE_PROFILES,
    CanonicalLayout,
    UniversalScopeObjectives,
    canonical_layout_from_word,
    get_hardware_profile,
)
from relay.objectives import build_objectives
from relay.simple_solver import (
    FrontierCost,
    _add_scores,
    _canonical_word_scorer,
    _context,
    _pareto,
    _raw_cost,
)


EXPERIMENT_NAME = "canonical-scoring-results"
SCHEMA_VERSION = 1
DEFAULT_SIZES = (512, 1024)
DEFAULT_OUTPUT = Path("results/canonical_scoring_mi300a.jsonl")
DEFAULT_RAW_OUTPUT = Path("results/canonical_scoring_mi300a.raw.jsonl")
DEFAULT_PLAN = Path("results/canonical_scoring_mi300a.plan.json")
GRAMMARS = {
    "canonical": "G_C",
    "standard": "G_S",
}


@dataclass(frozen=True)
class ScoredWord:
    """One shared canonical word and its five analytical costs."""

    word: str
    cost: FrontierCost


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kernel",
        action="append",
        choices=tuple(KERNEL_SPECS),
        default=None,
        help="kernel to include; repeat as needed (default: all five)",
    )
    parser.add_argument(
        "--size",
        action="append",
        type=positive_integer,
        default=None,
        metavar="N",
        help="square matrix size; repeat as needed (default: 512 and 1024)",
    )
    parser.add_argument(
        "--grammar",
        choices=tuple(GRAMMARS),
        default="canonical",
        help="layout grammar to enumerate exhaustively (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=positive_integer,
        default=5,
        help="independent HIP timing samples per layout (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=positive_integer,
        default=3,
        help="kernel launches per timing sample (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup",
        type=nonnegative_integer,
        default=2,
        help="untimed launches per layout (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=nonnegative_integer,
        default=0,
        help="HIP device ordinal (default: %(default)s)",
    )
    parser.add_argument(
        "--block-size",
        type=positive_integer,
        default=128,
        help="workgroup size for one-dimensional kernels (default: %(default)s)",
    )
    parser.add_argument(
        "--block-x",
        type=positive_integer,
        default=32,
        help="workgroup width for two-dimensional kernels (default: %(default)s)",
    )
    parser.add_argument(
        "--block-y",
        type=positive_integer,
        default=32,
        help="workgroup height for two-dimensional kernels (default: %(default)s)",
    )
    parser.add_argument(
        "--compiler",
        default="/opt/rocm-7.0.2/bin/hipcc",
        help="HIP compiler passed to kernel evaluators (default: %(default)s)",
    )
    parser.add_argument(
        "--arch",
        default="gfx942",
        help="GPU architecture passed to kernel evaluators (default: %(default)s)",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=tuple(HARDWARE_PROFILES),
        default="mi300a",
        help="hardware response used for scoring (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="deterministic residual-sweep ordering seed (default: %(default)s)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write the exact analytical plan and summaries without using a GPU",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing compatible plan and raw timing checkpoint",
    )
    parser.add_argument(
        "--seed-timings",
        type=Path,
        action="append",
        default=None,
        metavar="REPORT",
        help=(
            "import compatible full-word timings from a completed multi-kernel "
            "layout-ranking JSON report"
        ),
    )
    parser.add_argument(
        "--max-benchmarks",
        type=positive_integer,
        default=None,
        metavar="COUNT",
        help="time at most COUNT missing layouts before checkpointing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="JSONL",
        help="compact paper-summary JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_OUTPUT,
        metavar="JSONL",
        help="append-only raw timing checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN,
        metavar="JSON",
        help="exact analytical selection plan (default: %(default)s)",
    )
    return parser, parser.parse_args(argv)


def canonical_words(n: int) -> Iterator[str]:
    """Yield every full rank-two canonical word in lexical position order."""

    if n < 2 or n & (n - 1):
        raise ValueError("matrix sizes must be powers of two greater than one")
    bits = n.bit_length() - 1
    width = 2 * bits
    for first_positions in combinations(range(width), bits):
        positions = set(first_positions)
        yield "".join("i" if index in positions else "j" for index in range(width))


def standard_words(n: int) -> tuple[str, ...]:
    """Return the unique four-form cut-point words that define G_S."""

    validate_matrix_size(n)
    bits = n.bit_length() - 1
    words = set()
    for i_cut in range(bits + 1):
        for j_cut in range(bits + 1):
            inner = {
                "i" * i_cut + "j" * j_cut,
                "j" * j_cut + "i" * i_cut,
            }
            outer_i = bits - i_cut
            outer_j = bits - j_cut
            outer = {
                "i" * outer_i + "j" * outer_j,
                "j" * outer_j + "i" * outer_i,
            }
            words.update(prefix + suffix for prefix in inner for suffix in outer)
    return tuple(sorted(words))


def grammar_words(n: int, grammar: str) -> Iterable[str]:
    if grammar == "canonical":
        return canonical_words(n)
    if grammar == "standard":
        return standard_words(n)
    raise ValueError(f"unknown grammar {grammar!r}")


def validate_matrix_size(n: int) -> None:
    if n < 2 or n & (n - 1):
        raise ValueError("matrix sizes must be powers of two greater than one")


def expand_canonical_descriptor(word: str, n: int) -> str:
    """Expand an evaluator's tiled word to its equivalent full G_C word."""

    if n < 2 or n & (n - 1):
        raise ValueError("matrix sizes must be powers of two greater than one")
    bits = n.bit_length() - 1
    i_bits = word.count("i")
    j_bits = word.count("j")
    if set(word) - {"i", "j"} or i_bits > bits or j_bits > bits:
        raise ValueError(f"{word!r} is not a canonical descriptor for N={n}")
    return word + "j" * (bits - j_bits) + "i" * (bits - i_bits)


def _deduplicated(values: Sequence[object]) -> list[object]:
    return list(dict.fromkeys(values))


def _configuration(
    args: argparse.Namespace,
    kernel_names: Sequence[str],
    sizes: Sequence[int],
) -> dict[str, object]:
    profile = get_hardware_profile(args.hardware_profile)
    grammar = GRAMMARS[args.grammar]
    layout_scope = (
        "one shared full canonical word across all target matrices"
        if args.grammar == "canonical"
        else "one shared full standard cut-point word across all target matrices"
    )
    return {
        "kernels": list(kernel_names),
        "matrix_sizes": list(sizes),
        "grammar": grammar,
        "layout_scope": layout_scope,
        "hardware_profile": args.hardware_profile,
        "hardware_profile_id": profile.profile_id,
        "samples": args.samples,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "device": args.device,
        "block_size": args.block_size,
        "block_x": args.block_x,
        "block_y": args.block_y,
        "compiler": args.compiler,
        "arch": args.arch,
        "seed": args.seed,
    }


def _configuration_id(configuration: Mapping[str, object]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def _cost_dict(cost: FrontierCost) -> dict[str, object]:
    return {
        "fine_region_count": cost.fine_region_count,
        "hardware_peak": cost.hardware_peak,
        "hardware_area": cost.hardware_area,
        "codegen_runs": cost.codegen_runs,
        "codegen_xors": cost.codegen_xors,
    }


def _member_dict(item: ScoredWord, target_names: Sequence[str]) -> dict[str, object]:
    return {
        "word": item.word,
        "layouts": {name: item.word for name in target_names},
        "score": _cost_dict(item.cost),
    }


def _problem_inputs(
    spec: KernelSpec,
    n: int,
    args: argparse.Namespace,
) -> tuple[tuple, tuple, dict[str, object], object]:
    if spec.block_style == "2d":
        block = (args.block_x, args.block_y, 1)
    else:
        block = args.block_size
    config = spec.problem.build_config(problem_size=n, block_size=block)
    matrices_tuple = tuple(spec.problem.get_matrices(config))
    events, sequences = spec.problem.get_events_and_sequences(config)
    matrices = {matrix.name: matrix for matrix in matrices_tuple}
    return matrices_tuple, tuple(events), matrices, (tuple(sequences), block)


def build_group_plan(
    spec: KernelSpec,
    n: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Enumerate and score the exact shared-word G_C family."""

    profile = get_hardware_profile(args.hardware_profile)
    matrices_tuple, event_items, matrices, sequence_and_block = _problem_inputs(
        spec, n, args
    )
    sequences, block = sequence_and_block
    components = tuple(
        build_objectives(
            (UniversalScopeObjectives(profile.byte_scales),),
            matrices,
            {event.id: event for event in event_items},
            sequences,
        )
    )
    weights = profile.component_weights(components)
    peak_tolerances = profile.peak_tolerances(components)
    raw_names = {
        profile.fine_component,
        *peak_tolerances,
        *(
            component.name
            for component in components
            if weights[component.name] > 0
        ),
    }
    context_layouts, context_scores = _context(
        matrices_tuple, components, raw_names
    )
    del context_layouts
    target_matrices = tuple(matrix for matrix in matrices_tuple if matrix.target)
    target_names = tuple(matrix.name for matrix in target_matrices)
    scorers = {
        matrix.name: _canonical_word_scorer(matrix, components, raw_names)
        for matrix in target_matrices
    }

    scored: list[ScoredWord] = []
    for word in grammar_words(n, args.grammar):
        raw_scores = dict(context_scores)
        for matrix in target_matrices:
            layout = canonical_layout_from_word(
                matrix,
                word,
                name=f"shared_{GRAMMARS[args.grammar]}_{word}.{matrix.name}",
            )
            if not isinstance(layout, CanonicalLayout):
                raise RuntimeError("canonical word produced a noncanonical layout")
            raw_scores = _add_scores(raw_scores, scorers[matrix.name](layout))
        scored.append(
            ScoredWord(
                word,
                _raw_cost(
                    raw_scores,
                    matrices_tuple,
                    components,
                    weights,
                    peak_tolerances,
                    profile.fine_component,
                ),
            )
        )

    frontier = _pareto(scored, lambda item: item.cost.values)
    fine_minimum = min(item.cost.fine_region_count for item in scored)
    fine_limit = 1.05 * fine_minimum
    fine_eligible = [
        item for item in scored if item.cost.fine_region_count <= fine_limit
    ]
    fine_gated = _pareto(
        fine_eligible,
        lambda item: item.cost.values[1:],
    )
    scalar_order = sorted(
        scored,
        key=lambda item: (item.cost.hardware_area, item.cost.codegen_runs, item.word),
    )
    minimum_area = scalar_order[0].cost.hardware_area
    minimum_area_ties = sum(
        item.cost.hardware_area == minimum_area for item in scored
    )
    lexical = min(scored, key=lambda item: (item.cost.values, item.word))
    row_major_word = "j" * (n.bit_length() - 1) + "i" * (n.bit_length() - 1)
    row_major = next(item for item in scored if item.word == row_major_word)

    mechanisms = [
        {
            "name": "lowest_hardware_area",
            "definition": (
                "minimum J_area; ties break by fewer codegen runs then canonical word"
            ),
            "score_tie_count": minimum_area_ties,
            "members": [_member_dict(scalar_order[0], target_names)],
        },
        {
            "name": "top5_hardware_area",
            "definition": (
                "five lowest J_area layouts; ties break by fewer codegen runs then "
                "canonical word"
            ),
            "members": [
                _member_dict(item, target_names) for item in scalar_order[:5]
            ],
        },
        {
            "name": "lexicographic_five_cost",
            "definition": (
                "lexicographic minimum of (Q_fine, J_peak, J_area, runs, XORs)"
            ),
            "members": [_member_dict(lexical, target_names)],
        },
        {
            "name": "fine_gated_5pct_frontier",
            "definition": (
                "Q_fine <= 1.05 Q_fine*, then Pareto over "
                "(J_peak, J_area, runs, XORs)"
            ),
            "members": [
                _member_dict(item, target_names) for item in fine_gated
            ],
        },
        {
            "name": "row_major_baseline",
            "definition": "complete row-major canonical word",
            "members": [_member_dict(row_major, target_names)],
        },
    ]
    return {
        "kernel": spec.name,
        "display_name": spec.display_name,
        "matrix_size": n,
        "grammar": GRAMMARS[args.grammar],
        "target_arrays": list(target_names),
        "block": list(block) if isinstance(block, tuple) else block,
        "total_layouts": len(scored),
        "expected_total_layouts": (
            comb(2 * (n.bit_length() - 1), n.bit_length() - 1)
            if args.grammar == "canonical"
            else len(standard_words(n))
        ),
        "frontier_objectives": [
            "fine_region_count",
            "hardware_peak",
            "hardware_area",
            "codegen_runs",
            "codegen_xors",
        ],
        "frontier": [_member_dict(item, target_names) for item in frontier],
        "fine_gated_5pct": {
            "fine_minimum": fine_minimum,
            "fine_limit": fine_limit,
            "eligible_count": len(fine_eligible),
        },
        "selection_mechanisms": mechanisms,
    }


def build_plan(
    args: argparse.Namespace,
    kernel_names: Sequence[str],
    sizes: Sequence[int],
    configuration: Mapping[str, object],
) -> dict[str, object]:
    groups = []
    for kernel_name in kernel_names:
        for n in sizes:
            spec = KERNEL_SPECS[kernel_name]
            print(
                f"Scoring exhaustive shared-word {GRAMMARS[args.grammar]}: "
                f"{spec.display_name} N={n}"
            )
            groups.append(build_group_plan(spec, n, args))
            print(
                f"  {groups[-1]['total_layouts']} layouts; "
                f"{len(groups[-1]['frontier'])} frontier layouts",
                flush=True,
            )
    return {
        "experiment": EXPERIMENT_NAME,
        "schema_version": SCHEMA_VERSION,
        "configuration_id": _configuration_id(configuration),
        "configuration": dict(configuration),
        "groups": groups,
    }


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, values: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        for value in values:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _initialize_raw(
    path: Path,
    configuration: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "record_type": "metadata",
        "experiment": EXPERIMENT_NAME,
        "schema_version": SCHEMA_VERSION,
        "configuration_id": _configuration_id(configuration),
        "configuration": dict(configuration),
    }
    with path.open("x") as stream:
        json.dump(metadata, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append_raw(path: Path, record: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        written = 0
        while written < len(encoded):
            written += os.write(descriptor, encoded[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_raw(
    path: Path,
    configuration: Mapping[str, object],
) -> dict[tuple[str, int, str], dict[str, object]]:
    """Load and validate an append-only timing checkpoint."""

    expected_id = _configuration_id(configuration)
    records: dict[tuple[str, int, str], dict[str, object]] = {}
    file_size = path.stat().st_size
    with path.open("rb+") as stream:
        line_number = 0
        while True:
            line_start = stream.tell()
            encoded_line = stream.readline()
            if not encoded_line:
                break
            line_number += 1
            try:
                record = json.loads(encoded_line)
            except json.JSONDecodeError as error:
                if stream.tell() == file_size and not encoded_line.endswith(b"\n"):
                    stream.truncate(line_start)
                    break
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if line_number == 1:
                if (
                    record.get("record_type") != "metadata"
                    or record.get("experiment") != EXPERIMENT_NAME
                    or record.get("configuration_id") != expected_id
                    or record.get("configuration") != configuration
                ):
                    raise ValueError(f"{path}: incompatible raw timing metadata")
                continue
            if record.get("record_type") != "timing":
                raise ValueError(f"{path}:{line_number}: expected a timing record")
            if record.get("configuration_id") != expected_id:
                raise ValueError(f"{path}:{line_number}: configuration ID differs")
            key = (
                str(record["kernel"]),
                int(record["matrix_size"]),
                str(record["word"]),
            )
            previous = records.get(key)
            if previous is not None and previous != record:
                raise ValueError(f"{path}:{line_number}: conflicting duplicate {key}")
            records[key] = record
    return records


def _validate_seed_configuration(
    source: Mapping[str, object],
    target: Mapping[str, object],
    path: Path,
) -> None:
    fields = {
        "samples": "samples",
        "iterations": "iterations",
        "warmup": "warmup",
        "device": "device",
        "compiler": "compiler",
        "arch": "arch",
    }
    mismatched = [
        target_name
        for source_name, target_name in fields.items()
        if source.get(source_name) != target.get(target_name)
    ]
    if source.get("one_dimensional_block_size") != target.get("block_size"):
        mismatched.append("block_size")
    if source.get("two_dimensional_block") != [
        target.get("block_x"),
        target.get("block_y"),
        1,
    ]:
        mismatched.extend(("block_x", "block_y"))
    if mismatched:
        raise ValueError(
            f"{path}: timing configuration differs in: "
            + ", ".join(dict.fromkeys(mismatched))
        )


def import_seed_timings(
    path: Path,
    raw_path: Path,
    configuration: Mapping[str, object],
    records: dict[tuple[str, int, str], dict[str, object]],
) -> int:
    """Import exact full-word evaluator cases from a ranking report.

    Short tiled descriptors can describe the same physical permutation, but
    they generate a different address expression. They are deliberately not
    reused as timing evidence for the full-word grammar.
    """

    source = json.loads(path.read_text())
    if source.get("experiment") != "multi-kernel-layout-ranking":
        raise ValueError(f"{path}: expected a multi-kernel layout-ranking report")
    source_configuration = source.get("configuration")
    if not isinstance(source_configuration, dict):
        raise ValueError(f"{path}: timing report has no configuration")
    _validate_seed_configuration(source_configuration, configuration, path)
    selected_kernels = set(configuration["kernels"])
    selected_sizes = {int(value) for value in configuration["matrix_sizes"]}
    grammar_name = (
        "canonical" if configuration["grammar"] == "G_C" else "standard"
    )
    configuration_id = _configuration_id(configuration)
    imported = 0
    for group in source.get("runs", []):
        if not isinstance(group, dict):
            continue
        kernel = str(group.get("kernel"))
        n = int(group.get("matrix_size", 0))
        if kernel not in selected_kernels or n not in selected_sizes:
            continue
        for result in group.get("results", []):
            if not isinstance(result, dict) or not isinstance(
                result.get("timing"), dict
            ):
                continue
            descriptor = str(result["word"])
            word = expand_canonical_descriptor(descriptor, n)
            if descriptor != word:
                continue
            if word not in set(grammar_words(n, grammar_name)):
                continue
            key = (kernel, n, word)
            if key in records:
                continue
            record = {
                "record_type": "timing",
                "configuration_id": configuration_id,
                "kernel": kernel,
                "matrix_size": n,
                "word": word,
                "timing": result["timing"],
                "source": {
                    "type": "seeded_layout_ranking",
                    "report": str(path.resolve()),
                    "layout": result.get("name"),
                    "descriptor": descriptor,
                },
            }
            _append_raw(raw_path, record)
            records[key] = record
            imported += 1
    return imported


def _timed_member(
    member: Mapping[str, object],
    timing_record: Mapping[str, object] | None,
) -> dict[str, object]:
    result = dict(member)
    result["timing"] = None if timing_record is None else timing_record["timing"]
    return result


def _best_timed(
    members: Sequence[Mapping[str, object]],
    records: Mapping[tuple[str, int, str], Mapping[str, object]],
    kernel: str,
    n: int,
) -> tuple[bool, float | None, list[dict[str, object]]]:
    timed = []
    complete = True
    for member in members:
        record = records.get((kernel, n, str(member["word"])))
        if record is None:
            complete = False
            continue
        timing = record["timing"]
        assert isinstance(timing, dict)
        timed.append((float(timing["median_ms"]), str(member["word"]), member, record))
    if not timed:
        return complete, None, []
    timed.sort(key=lambda item: (item[0], item[1]))
    best_ms = timed[0][0]
    best = [
        _timed_member(item[2], item[3])
        for item in timed
        if item[0] == best_ms
    ][:5]
    return complete, best_ms, best


def summary_records(
    plan: Mapping[str, object],
    raw_records: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build one compact, plot-oriented record per kernel and size."""

    configuration = plan["configuration"]
    assert isinstance(configuration, dict)
    summaries = []
    for group in plan["groups"]:
        assert isinstance(group, dict)
        kernel = str(group["kernel"])
        n = int(group["matrix_size"])
        group_records = [
            record
            for (record_kernel, record_n, _word), record in raw_records.items()
            if record_kernel == kernel and record_n == n
        ]
        group_records.sort(
            key=lambda record: (
                float(record["timing"]["median_ms"]),
                str(record["word"]),
            )
        )
        total_layouts = int(group["total_layouts"])
        oracle_complete = len(group_records) == total_layouts
        observed_top = [
            {
                "word": record["word"],
                "layouts": {
                    name: record["word"] for name in group["target_arrays"]
                },
                "timing": record["timing"],
            }
            for record in group_records[:5]
        ]
        best_observed_layouts = (
            [
                layout
                for layout, record in zip(observed_top, group_records[:5])
                if float(record["timing"]["median_ms"])
                == float(group_records[0]["timing"]["median_ms"])
            ]
            if group_records
            else []
        )
        oracle_best = (
            float(group_records[0]["timing"]["median_ms"])
            if oracle_complete and group_records
            else None
        )
        oracle_layouts = best_observed_layouts if oracle_complete else []

        frontier_members = group["frontier"]
        assert isinstance(frontier_members, list)
        frontier_complete, frontier_best, frontier_best_layouts = _best_timed(
            frontier_members, raw_records, kernel, n
        )
        frontier_layouts = [
            _timed_member(
                member,
                raw_records.get((kernel, n, str(member["word"]))),
            )
            for member in frontier_members
        ]
        selection_records = []
        for mechanism in group["selection_mechanisms"]:
            members = mechanism["members"]
            complete, best_ms, best_layouts = _best_timed(
                members, raw_records, kernel, n
            )
            selection_records.append(
                {
                    "name": mechanism["name"],
                    "definition": mechanism["definition"],
                    "selected_count": len(members),
                    "score_tie_count": mechanism.get("score_tie_count"),
                    "complete": complete,
                    "best_time_ms": best_ms if complete else None,
                    "regret": (
                        best_ms / oracle_best - 1.0
                        if complete and best_ms is not None and oracle_best is not None
                        else None
                    ),
                    "best_layouts": best_layouts if complete else [],
                    "layouts": [
                        _timed_member(
                            member,
                            raw_records.get((kernel, n, str(member["word"]))),
                        )
                        for member in members
                    ],
                }
            )
        device_names = sorted(
            {
                str(record["timing"]["device"])
                for record in group_records
                if record["timing"].get("device") is not None
            }
        )
        summaries.append(
            {
                "experiment": EXPERIMENT_NAME,
                "schema_version": SCHEMA_VERSION,
                "configuration_id": plan["configuration_id"],
                "kernel": kernel,
                "display_name": group["display_name"],
                "matrix_size": n,
                "grammar": configuration["grammar"],
                "layout_scope": configuration["layout_scope"],
                "target_arrays": group["target_arrays"],
                "device": {
                    "hardware_profile": configuration["hardware_profile"],
                    "hardware_profile_id": configuration["hardware_profile_id"],
                    "ordinal": configuration["device"],
                    "reported_names": device_names,
                },
                "timing_configuration": {
                    name: configuration[name]
                    for name in (
                        "samples",
                        "iterations",
                        "warmup",
                        "block_size",
                        "block_x",
                        "block_y",
                        "compiler",
                        "arch",
                    )
                },
                "complete": oracle_complete,
                "layout_count": total_layouts,
                "timed_layout_count": len(group_records),
                "oracle": {
                    "definition": (
                        "minimum median kernel time over every shared-word "
                        f"{configuration['grammar']} layout"
                    ),
                    "complete": oracle_complete,
                    "best_time_ms": oracle_best,
                    "best_layouts": oracle_layouts,
                    "top_layouts": observed_top if oracle_complete else [],
                    "best_observed_time_ms": (
                        float(group_records[0]["timing"]["median_ms"])
                        if group_records
                        else None
                    ),
                    "best_observed_layouts": best_observed_layouts,
                    "top_observed_layouts": observed_top,
                },
                "frontier": {
                    "definition": (
                        "exact Pareto frontier over (Q_fine, J_peak, J_area, "
                        "runs, XORs)"
                    ),
                    "objectives": group["frontier_objectives"],
                    "size": len(frontier_members),
                    "complete": frontier_complete,
                    "best_time_ms": frontier_best if frontier_complete else None,
                    "regret": (
                        frontier_best / oracle_best - 1.0
                        if frontier_complete
                        and frontier_best is not None
                        and oracle_best is not None
                        else None
                    ),
                    "best_layouts": (
                        frontier_best_layouts if frontier_complete else []
                    ),
                    "layouts": frontier_layouts,
                },
                "selection_mechanisms": selection_records,
            }
        )
    return summaries


def _write_summaries(
    path: Path,
    plan: Mapping[str, object],
    records: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> None:
    _atomic_write_jsonl(path, summary_records(plan, records))


def _priority_words(group: Mapping[str, object]) -> list[str]:
    words = [str(member["word"]) for member in group["frontier"]]
    for mechanism in group["selection_mechanisms"]:
        words.extend(str(member["word"]) for member in mechanism["members"])
    return list(dict.fromkeys(words))


def pending_jobs(
    plan: Mapping[str, object],
    records: Mapping[tuple[str, int, str], Mapping[str, object]],
    seed: int,
) -> list[tuple[str, int, str]]:
    """Return analytical selections first, then a shuffled exhaustive tail."""

    jobs: list[tuple[str, int, str]] = []
    selected: set[tuple[str, int, str]] = set()
    configuration = plan["configuration"]
    assert isinstance(configuration, dict)
    grammar_name = (
        "canonical" if configuration["grammar"] == "G_C" else "standard"
    )
    for group in plan["groups"]:
        kernel = str(group["kernel"])
        n = int(group["matrix_size"])
        for word in _priority_words(group):
            key = (kernel, n, word)
            selected.add(key)
            if key not in records:
                jobs.append(key)
    tails = []
    for group in plan["groups"]:
        kernel = str(group["kernel"])
        n = int(group["matrix_size"])
        words = [
            word
            for word in grammar_words(n, grammar_name)
            if (kernel, n, word) not in records
            and (kernel, n, word) not in selected
        ]
        random.Random(f"{seed}:{kernel}:{n}").shuffle(words)
        tails.extend((kernel, n, word) for word in words)
    jobs.extend(tails)
    return jobs


def _load_plan(path: Path, configuration: Mapping[str, object]) -> dict[str, object]:
    plan = json.loads(path.read_text())
    if (
        plan.get("experiment") != EXPERIMENT_NAME
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("configuration_id") != _configuration_id(configuration)
        or plan.get("configuration") != configuration
    ):
        raise ValueError(f"{path}: incompatible analytical plan")
    return plan


def run(argv: Sequence[str] | None = None) -> int:
    parser, args = parse_arguments(argv)
    kernel_names = [
        str(value)
        for value in _deduplicated(args.kernel or list(KERNEL_SPECS))
    ]
    sizes = [int(value) for value in _deduplicated(args.size or list(DEFAULT_SIZES))]
    try:
        for n in sizes:
            validate_matrix_size(n)
    except ValueError as error:
        parser.error(str(error))
    configuration = _configuration(args, kernel_names, sizes)
    output = args.output.expanduser().resolve()
    raw_output = args.raw_output.expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()

    try:
        if args.resume:
            missing = [path for path in (plan_path, raw_output) if not path.exists()]
            if missing:
                raise ValueError(
                    "--resume requires existing files: "
                    + ", ".join(str(path) for path in missing)
                )
            plan = _load_plan(plan_path, configuration)
            records = load_raw(raw_output, configuration)
        else:
            existing = [
                path for path in (plan_path, raw_output, output) if path.exists()
            ]
            if existing:
                raise ValueError(
                    "refusing to overwrite existing result files; use --resume: "
                    + ", ".join(str(path) for path in existing)
                )
            plan = build_plan(args, kernel_names, sizes, configuration)
            _atomic_write_json(plan_path, plan)
            _initialize_raw(raw_output, configuration)
            records = {}

        for seed_path in args.seed_timings or ():
            imported = import_seed_timings(
                seed_path.expanduser().resolve(),
                raw_output,
                configuration,
                records,
            )
            print(f"Imported {imported} timings from {seed_path}", flush=True)
        _write_summaries(output, plan, records)

        if args.prepare_only:
            print(f"Wrote analytical plan {plan_path}")
            print(f"Wrote raw checkpoint {raw_output}")
            print(f"Wrote summary {output}")
            return 0

        jobs = pending_jobs(plan, records, args.seed)
        total_missing = len(jobs)
        if args.max_benchmarks is not None:
            jobs = jobs[: args.max_benchmarks]
        configuration_id = _configuration_id(configuration)
        for index, (kernel, n, word) in enumerate(jobs, 1):
            spec = KERNEL_SPECS[kernel]
            print(
                f"[{index}/{len(jobs)}; {total_missing} missing] "
                f"Benchmarking {spec.display_name} N={n} {word}...",
                flush=True,
            )
            timing, command, _stdout, _stderr = benchmark_case(
                spec,
                n,
                LayoutCase(f"shared_G_C_{word}", word),
                args,
            )
            record = {
                "record_type": "timing",
                "configuration_id": configuration_id,
                "kernel": kernel,
                "matrix_size": n,
                "word": word,
                "timing": asdict(timing),
                "source": {
                    "type": "measured",
                    "command": command,
                },
            }
            _append_raw(raw_output, record)
            records[(kernel, n, word)] = record
            _write_summaries(output, plan, records)

        remaining = total_missing - len(jobs)
        print(f"Wrote analytical plan {plan_path}")
        print(f"Wrote raw checkpoint {raw_output}")
        print(f"Wrote summary {output}")
        print(f"Completed {len(jobs)} benchmarks; {remaining} remain")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
