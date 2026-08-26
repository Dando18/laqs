#!/usr/bin/env python3
"""Build an exhaustive shared-word G_S or G_C scoring result corpus.

The experiment applies one complete canonical word to every target matrix in
each kernel.  It computes the exact analytical frontiers on a CPU, then
correctness-checks and times every word to obtain a genuine runtime oracle.
Raw timing records are append-only JSONL checkpoints; the compact JSONL output
contains one plot-oriented summary record per kernel and matrix size.

With ``--fiber-max-xors``, each canonical flag representative is expanded by
sparse unit-upper-triangular shears. Existing identity timings can be imported
with ``--seed-raw``; the exhaustive oracle then remains explicitly scoped to
the canonical representatives rather than the enlarged fiber grammar.

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
from dataclasses import asdict, dataclass, replace
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
    ResourcePlacementScore,
    UniversalScopeObjectives,
    apply_flag_preserving_shears,
    build_resource_cohorts,
    canonical_layout_from_word,
    enumerate_flag_preserving_swizzles,
    get_hardware_profile,
    group_resource_cohorts_by_translation,
    low_address_flag,
    resource_color_destination_bits,
    score_resource_placement,
)
from relay.objectives import build_objectives
from relay.scoring import excess_footprint, normalized_excess
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
RERANK_BUDGETS = (1, 3, 5, 10)
PLACEMENT_STATISTICS = ("robust", "expected", "cvar25")


def _placement_value(
    placements: Sequence[ResourcePlacementScore], statistic: str
) -> float:
    attribute = {
        "robust": "robust_contention",
        "expected": "expected_contention",
        "cvar25": "cvar25_contention",
    }[statistic]
    return sum(
        placement.weight * float(getattr(placement, attribute))
        for placement in placements
    )


@dataclass(frozen=True)
class ScoredWord:
    """One shared flag realization and its analytical costs."""

    word: str
    cost: FrontierCost
    hardware_place: float
    flag_word: str | None = None
    shears: tuple[tuple[int, int], ...] = ()
    placements: tuple[ResourcePlacementScore, ...] = ()
    raw_scores: Mapping[str, float] | None = None

    @property
    def frontier_values(self) -> tuple[float, ...]:
        return (*self.cost.values, self.hardware_place)

    @property
    def base_word(self) -> str:
        return self.word if self.flag_word is None else self.flag_word


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


def nonnegative_float(value: str) -> float:
    parsed = float(value)
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
        "--fiber-max-xors",
        type=nonnegative_integer,
        default=0,
        metavar="COUNT",
        help=(
            "realize each retained G_S flag with unit-upper-triangular swizzles "
            "using at most COUNT XORs (default: disabled)"
        ),
    )
    parser.add_argument(
        "--locality-shell-tolerance",
        type=nonnegative_float,
        default=0.05,
        metavar="FRACTION",
        help=(
            "maximum relative locality loss admitted around each three-cost "
            "frontier member before placement reranking (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--placement-statistic",
        choices=PLACEMENT_STATISTICS,
        default="robust",
        help=(
            "global-phase placement statistic used by the reranker "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dump-oracle-components",
        action="store_true",
        help=(
            "include complete primitive locality and placement vectors for the "
            "oracle, reranker leader, and locality frontier"
        ),
    )
    parser.add_argument(
        "--check-oracle-feature-dominance",
        action="store_true",
        help=(
            "test whether any enumerated layout componentwise dominates the "
            "measured oracle in the full primitive feature vector"
        ),
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
        "--rescore",
        action="store_true",
        help=(
            "rebuild and replace the analytical plan and summary while reusing "
            "the existing raw timing checkpoint"
        ),
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
        "--seed-raw",
        type=Path,
        action="append",
        default=None,
        metavar="JSONL",
        help=(
            "import compatible identity-layout timings from another canonical "
            "scoring raw checkpoint"
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
    if args.fiber_max_xors:
        layout_scope += (
            "; one shared sparse flag-preserving swizzle is then applied to "
            "every target matrix"
        )
    configuration = {
        "kernels": list(kernel_names),
        "matrix_sizes": list(sizes),
        "grammar": grammar,
        "layout_scope": layout_scope,
        "hardware_profile": args.hardware_profile,
        "hardware_profile_id": profile.profile_id,
        "selection_policy": "locality-shell-placement-reranker-v1",
        "locality_shell_tolerance": args.locality_shell_tolerance,
        "placement_statistic": args.placement_statistic,
        "rerank_budgets": list(RERANK_BUDGETS),
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
    if args.fiber_max_xors:
        configuration["fiber_max_xors"] = args.fiber_max_xors
    return configuration


def _configuration_id(configuration: Mapping[str, object]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def _cost_dict(item: ScoredWord) -> dict[str, object]:
    return {
        "fine_region_count": item.cost.fine_region_count,
        "hardware_peak": item.cost.hardware_peak,
        "hardware_area": item.cost.hardware_area,
        "hardware_place": item.hardware_place,
        "codegen_runs": item.cost.codegen_runs,
        "codegen_xors": item.cost.codegen_xors,
    }


def _member_dict(item: ScoredWord, target_names: Sequence[str]) -> dict[str, object]:
    result = {
        "word": item.word,
        "layouts": {name: item.word for name in target_names},
        "score": _cost_dict(item),
    }
    if item.flag_word is not None:
        result["flag_word"] = item.flag_word
        result["shears"] = [list(shear) for shear in item.shears]
        result["swizzle_xors"] = len(item.shears)
    return result


def _placement_dict(
    placement: ResourcePlacementScore,
    *,
    include_phases: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": placement.name,
        "cohort_family": placement.cohort_family,
        "transaction_bytes": placement.transaction_bytes,
        "color_count": placement.color_count,
        "phase_policy": placement.phase_policy,
        "weight": placement.weight,
        "cohort_count": placement.cohort_count,
        "cohort_weight": placement.cohort_weight,
        "raw_pair_excess": placement.raw_pair_excess,
        "normalized_contention": placement.normalized_contention,
        "expected_contention": placement.expected_contention,
        "robust_contention": placement.robust_contention,
        "cvar25_contention": placement.cvar25_contention,
        "within_contention": placement.within_contention,
        "within_by_array": dict(placement.within_by_array),
        "cross_contention": placement.cross_contention,
        "weighted_contention": placement.weighted_contention,
    }
    if include_phases:
        result["phase_scores"] = [
            {
                "allocation_phases": dict(phase.allocation_phases),
                "raw_pair_excess": phase.raw_pair_excess,
                "normalized_contention": phase.normalized_contention,
                "within_contention": phase.within_contention,
                "within_by_array": dict(phase.within_by_array),
                "cross_contention": phase.cross_contention,
            }
            for phase in placement.phase_scores
        ]
    return result


def _primitive_feature_dict(
    item: ScoredWord,
    matrices: Sequence[object],
    components: Sequence[object],
    fine_component: str,
) -> dict[str, object]:
    if item.raw_scores is None:
        raise ValueError("primitive feature reporting requires raw component scores")
    locality = {}
    for component in components:
        raw = float(item.raw_scores.get(component.name, 0.0))
        bound = sum(
            component.packing_bound(matrix)
            for matrix in matrices
            if component.edges_by_array.get(matrix.name)
        )
        excess = normalized_excess(raw, bound)
        if excess == 0.0:
            continue
        locality[component.name] = {
            "raw_region_count": raw,
            "packing_lower_bound": bound,
            "normalized_excess": excess,
            "excess_footprint": excess_footprint(
                raw,
                bound,
                component.region_bytes,
                component.normalization_bytes,
            ),
        }
    return {
        "q_fine": float(item.raw_scores.get(fine_component, 0.0)),
        "locality": locality,
        "placement": [
            _placement_dict(placement, include_phases=True)
            for placement in item.placements
        ],
    }


def _in_locality_shell(
    candidate: ScoredWord,
    anchor: ScoredWord,
    tolerance: float,
) -> bool:
    return all(
        candidate_value
        <= anchor_value + tolerance * max(abs(anchor_value), 1.0)
        for candidate_value, anchor_value in zip(
            candidate.cost.values, anchor.cost.values
        )
    )


def build_placement_reranker(
    candidates: Sequence[ScoredWord],
    locality_frontier: Sequence[ScoredWord],
    tolerance: float,
) -> dict[str, object]:
    """Build a bounded locality shell and rank it by corrected placement."""

    shell: dict[str, ScoredWord] = {}
    bucket_winners: dict[str, ScoredWord] = {}
    pool: dict[str, ScoredWord] = {
        item.word: item for item in locality_frontier
    }
    for anchor in locality_frontier:
        eligible = [
            candidate
            for candidate in candidates
            if _in_locality_shell(candidate, anchor, tolerance)
        ]
        for candidate in eligible:
            shell[candidate.word] = candidate
        winner = min(
            eligible,
            key=lambda item: (
                item.hardware_place,
                item.cost.values,
                item.cost.codegen_runs,
                item.cost.codegen_xors,
                item.word,
            ),
        )
        bucket_winners[anchor.word] = winner
        pool[winner.word] = winner
    ranked = sorted(
        pool.values(),
        key=lambda item: (
            item.hardware_place,
            item.cost.values,
            item.cost.codegen_runs,
            item.cost.codegen_xors,
            item.word,
        ),
    )
    return {
        "shell": tuple(shell.values()),
        "bucket_winners": bucket_winners,
        "pool": tuple(pool.values()),
        "ranked": tuple(ranked),
    }


def _placement_signature(
    matrix: object,
    layout: object,
    resource_maps: Sequence[object],
) -> tuple[object, ...]:
    if isinstance(layout, CanonicalLayout):
        rows = layout.matrix_rows()
    else:
        rows = getattr(layout, "a_rows", ())
    starts = []
    for resource_map in resource_maps:
        if resource_map.transaction_bytes % matrix.element_bytes:
            return layout.signature()
        transaction_elements = (
            resource_map.transaction_bytes // matrix.element_bytes
        )
        if (
            transaction_elements <= 0
            or transaction_elements & (transaction_elements - 1)
        ):
            return layout.signature()
        starts.append(transaction_elements.bit_length() - 1)
    if rows and starts:
        return tuple(tuple(rows[start:]) for start in starts)
    return layout.signature()


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
    audit_features = (
        args.dump_oracle_components
        or args.check_oracle_feature_dominance
    )
    if audit_features:
        raw_names = {component.name for component in components}
    context_layouts, context_scores = _context(
        matrices_tuple, components, raw_names
    )
    resource_cohorts = build_resource_cohorts(
        matrices,
        {event.id: event for event in event_items},
        sequences,
        (resource_map.cohort_family for resource_map in profile.resource_maps),
    )
    if all(
        resource_map.phase_policy == "robust"
        for resource_map in profile.resource_maps
    ):
        resource_cohorts = {
            family: group_resource_cohorts_by_translation(matrices, cohorts)
            for family, cohorts in resource_cohorts.items()
        }
    target_matrices = tuple(matrix for matrix in matrices_tuple if matrix.target)
    target_names = tuple(matrix.name for matrix in target_matrices)
    scorers = {
        matrix.name: _canonical_word_scorer(matrix, components, raw_names)
        for matrix in target_matrices
    }

    scored: list[ScoredWord] = []
    base_scored: list[ScoredWord] = []
    placement_cache: dict[
        tuple[tuple[str, tuple[object, ...]], ...],
        tuple[ResourcePlacementScore, ...],
    ] = {}
    materialization_count = 0
    retained_materialization_count = 0
    fiber_destination_bits: tuple[int, ...] = ()
    for word in grammar_words(n, args.grammar):
        raw_scores = dict(context_scores)
        base_layouts = dict(context_layouts)
        for matrix in target_matrices:
            layout = canonical_layout_from_word(
                matrix,
                word,
                name=f"shared_{GRAMMARS[args.grammar]}_{word}.{matrix.name}",
            )
            if not isinstance(layout, CanonicalLayout):
                raise RuntimeError("canonical word produced a noncanonical layout")
            base_layouts[matrix.name] = layout
            raw_scores = _add_scores(raw_scores, scorers[matrix.name](layout))
        base_cost = _raw_cost(
            raw_scores,
            matrices_tuple,
            components,
            weights,
            peak_tolerances,
            profile.fine_component,
        )
        base_placement_key = tuple(
            (
                matrix.name,
                _placement_signature(
                    matrix,
                    base_layouts[matrix.name],
                    profile.resource_maps,
                ),
            )
            for matrix in target_matrices
        )
        base_placement = placement_cache.get(base_placement_key)
        if base_placement is None:
            base_placement = score_resource_placement(
                matrices,
                base_layouts,
                resource_cohorts,
                profile.resource_maps,
            )
            placement_cache[base_placement_key] = base_placement
        base_item = ScoredWord(
            word=word,
            cost=base_cost,
            hardware_place=_placement_value(
                base_placement, args.placement_statistic
            ),
            flag_word=word if args.fiber_max_xors else None,
            placements=base_placement,
            raw_scores=raw_scores if audit_features else None,
        )
        base_scored.append(base_item)
        if not args.fiber_max_xors:
            scored.append(base_item)
            materialization_count += 1
            retained_materialization_count += 1
            continue

        first_matrix = target_matrices[0]
        first_layout = base_layouts[first_matrix.name]
        inner_bits = sum(first_layout.tile_exponents)
        destination_bits = resource_color_destination_bits(
            profile.resource_maps,
            first_matrix.element_bytes,
            inner_bits,
        )
        fiber_destination_bits = destination_bits
        for matrix in target_matrices[1:]:
            if (
                matrix.element_bytes != first_matrix.element_bytes
                or sum(base_layouts[matrix.name].tile_exponents) != inner_bits
            ):
                raise ValueError(
                    "shared flag-fiber scoring requires equal target layout widths "
                    "and element sizes"
                )
        fiber_seeds = enumerate_flag_preserving_swizzles(
            first_matrix,
            first_layout,
            max_xors=args.fiber_max_xors,
            destination_bits=destination_bits,
        )
        expected_flag = low_address_flag(first_matrix, first_layout)
        fiber_items: list[ScoredWord] = []
        for seed in fiber_seeds:
            layouts = dict(context_layouts)
            descriptors = set()
            for matrix in target_matrices:
                swizzled = apply_flag_preserving_shears(
                    matrix,
                    base_layouts[matrix.name],
                    seed.shears,
                )
                if low_address_flag(matrix, swizzled) != low_address_flag(
                    matrix, base_layouts[matrix.name]
                ):
                    raise RuntimeError(
                        f"flag-fiber transformation changed the locality flag for "
                        f"{matrix.name}"
                    )
                layouts[matrix.name] = swizzled
                descriptors.add(swizzled.evaluator_descriptor(matrix))
            if len(descriptors) != 1:
                raise ValueError(
                    "shared flag-fiber scoring produced distinct target descriptors"
                )
            if low_address_flag(first_matrix, layouts[first_matrix.name]) != expected_flag:
                raise RuntimeError("flag-fiber enumeration changed its source flag")
            placement_key = tuple(
                (
                    matrix.name,
                    _placement_signature(
                        matrix,
                        layouts[matrix.name],
                        profile.resource_maps,
                    ),
                )
                for matrix in target_matrices
            )
            placement_scores = placement_cache.get(placement_key)
            if placement_scores is None:
                placement_scores = score_resource_placement(
                    matrices,
                    layouts,
                    resource_cohorts,
                    profile.resource_maps,
                )
                placement_cache[placement_key] = placement_scores
            fiber_items.append(
                ScoredWord(
                    word=next(iter(descriptors)),
                    cost=replace(
                        base_cost,
                        codegen_runs=sum(layout.runs for layout in layouts.values()),
                        codegen_xors=sum(
                            layout.xor_count for layout in layouts.values()
                        ),
                    ),
                    hardware_place=_placement_value(
                        placement_scores, args.placement_statistic
                    ),
                    flag_word=word,
                    shears=seed.shears,
                    placements=placement_scores,
                    raw_scores=raw_scores if audit_features else None,
                )
            )
        materialization_count += len(fiber_items)
        retained = _pareto(
            fiber_items,
            lambda item: (item.hardware_place, float(len(item.shears))),
        )
        retained_materialization_count += len(retained)
        scored.extend(retained)

    locality_frontier = _pareto(base_scored, lambda item: item.cost.values)
    reranker = build_placement_reranker(
        scored, locality_frontier, args.locality_shell_tolerance
    )
    ranked = reranker["ranked"]
    assert isinstance(ranked, tuple)
    scalar_order = sorted(
        scored,
        key=lambda item: (item.cost.hardware_area, item.cost.codegen_runs, item.word),
    )
    minimum_area = scalar_order[0].cost.hardware_area
    minimum_area_ties = sum(
        item.cost.hardware_area == minimum_area for item in scored
    )
    row_major_word = "j" * (n.bit_length() - 1) + "i" * (n.bit_length() - 1)
    row_major = next(
        item
        for item in scored
        if item.base_word == row_major_word and not item.shears
    )

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
            "name": "row_major_baseline",
            "definition": "complete row-major canonical word",
            "members": [_member_dict(row_major, target_names)],
        },
    ]
    for budget in RERANK_BUDGETS:
        mechanisms.append(
            {
                "name": f"placement_rerank_at_{budget}",
                "definition": (
                    "top corrected-J_place recommendations from the bounded "
                    f"locality shell, capped at {budget} layouts"
                ),
                "members": [
                    _member_dict(item, target_names)
                    for item in ranked[:budget]
                ],
            }
        )

    primitive_features = (
        {
            item.word: _primitive_feature_dict(
                item,
                matrices_tuple,
                components,
                profile.fine_component,
            )
            for item in scored
        }
        if audit_features
        else {}
    )
    return {
        "kernel": spec.name,
        "display_name": spec.display_name,
        "matrix_size": n,
        "grammar": GRAMMARS[args.grammar],
        "resource_maps": [
            resource_map.to_dict() for resource_map in profile.resource_maps
        ],
        "target_arrays": list(target_names),
        "block": list(block) if isinstance(block, tuple) else block,
        "total_layouts": len(base_scored),
        "expected_total_layouts": (
            comb(2 * (n.bit_length() - 1), n.bit_length() - 1)
            if args.grammar == "canonical"
            else len(standard_words(n))
        ),
        "frontier_objectives": [
            "fine_region_count",
            "hardware_peak",
            "hardware_area",
        ],
        "locality_frontier_objectives": [
            "fine_region_count",
            "hardware_peak",
            "hardware_area",
        ],
        "locality_frontier": [
            _member_dict(item, target_names) for item in locality_frontier
        ],
        "frontier": [
            _member_dict(item, target_names) for item in locality_frontier
        ],
        "placement_reranker": {
            "definition": (
                "retain the three-cost locality frontier, add the lowest "
                "corrected-J_place member of each coordinatewise locality "
                "shell bucket, then rank the bounded union by the selected "
                "global-phase J_place statistic"
            ),
            "locality_shell_tolerance": args.locality_shell_tolerance,
            "placement_statistic": args.placement_statistic,
            "shell_count": len(reranker["shell"]),
            "bucket_winner_count": len(
                {item.word for item in reranker["bucket_winners"].values()}
            ),
            "pool_count": len(reranker["pool"]),
            "bucket_winners": {
                anchor: _member_dict(winner, target_names)
                for anchor, winner in reranker["bucket_winners"].items()
            },
            "ranked": [
                _member_dict(item, target_names) for item in ranked
            ],
            "budgets": list(RERANK_BUDGETS),
        },
        "primitive_features": primitive_features,
        "oracle_audit_options": {
            "dump_components": args.dump_oracle_components,
            "check_feature_dominance": args.check_oracle_feature_dominance,
        },
        "flag_fiber": {
            "enabled": bool(args.fiber_max_xors),
            "max_xors": args.fiber_max_xors,
            "destination_bits": list(fiber_destination_bits),
            "evaluated_materializations": materialization_count,
            "retained_within_flags": retained_materialization_count,
            "locality_invariance": (
                "verified by equality of every low-address prefix subspace"
            ),
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


def import_raw_timings(
    path: Path,
    raw_path: Path,
    configuration: Mapping[str, object],
    records: dict[tuple[str, int, str], dict[str, object]],
) -> int:
    """Import compatible canonical representatives from another checkpoint."""

    with path.open() as stream:
        try:
            metadata = json.loads(stream.readline())
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid raw timing metadata") from error
    source_configuration = metadata.get("configuration")
    if (
        metadata.get("record_type") != "metadata"
        or metadata.get("experiment") != EXPERIMENT_NAME
        or not isinstance(source_configuration, dict)
    ):
        raise ValueError(f"{path}: expected a canonical scoring raw checkpoint")
    required_equal = (
        "grammar",
        "samples",
        "iterations",
        "warmup",
        "device",
        "block_size",
        "block_x",
        "block_y",
        "compiler",
        "arch",
    )
    mismatched = [
        name
        for name in required_equal
        if source_configuration.get(name) != configuration.get(name)
    ]
    if mismatched:
        raise ValueError(
            f"{path}: timing configuration differs in: "
            + ", ".join(mismatched)
        )
    selected_kernels = set(configuration["kernels"])
    selected_sizes = {int(value) for value in configuration["matrix_sizes"]}
    if not selected_kernels <= set(source_configuration.get("kernels", ())):
        raise ValueError(f"{path}: source checkpoint omits a selected kernel")
    if not selected_sizes <= {
        int(value) for value in source_configuration.get("matrix_sizes", ())
    }:
        raise ValueError(f"{path}: source checkpoint omits a selected matrix size")

    source_records = load_raw(path, source_configuration)
    grammar_name = (
        "canonical" if configuration["grammar"] == "G_C" else "standard"
    )
    configuration_id = _configuration_id(configuration)
    imported = 0
    for (kernel, n, word), source_record in source_records.items():
        if kernel not in selected_kernels or n not in selected_sizes:
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
            "timing": source_record["timing"],
            "source": {
                "type": "seeded_raw_checkpoint",
                "checkpoint": str(path.resolve()),
                "configuration_id": metadata.get("configuration_id"),
            },
        }
        _append_raw(raw_path, record)
        records[key] = record
        imported += 1
    return imported


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


def _flatten_primitive_features(
    features: Mapping[str, object],
) -> dict[str, float]:
    flattened = {"Q_fine": float(features["q_fine"])}
    locality = features["locality"]
    assert isinstance(locality, dict)
    for name, component in locality.items():
        assert isinstance(component, dict)
        flattened[f"locality.{name}.excess_footprint"] = float(
            component["excess_footprint"]
        )
    placements = features["placement"]
    assert isinstance(placements, list)
    for placement in placements:
        assert isinstance(placement, dict)
        prefix = f"placement.{placement['name']}"
        for name in (
            "raw_pair_excess",
            "expected_contention",
            "robust_contention",
            "cvar25_contention",
            "within_contention",
            "cross_contention",
        ):
            flattened[f"{prefix}.{name}"] = float(placement[name])
        within = placement["within_by_array"]
        assert isinstance(within, dict)
        for array, value in within.items():
            flattened[f"{prefix}.within.{array}"] = float(value)
        phases = placement["phase_scores"]
        assert isinstance(phases, list)
        for phase in phases:
            assert isinstance(phase, dict)
            assignment = phase["allocation_phases"]
            assert isinstance(assignment, dict)
            phase_name = ",".join(
                f"{array}={value}"
                for array, value in sorted(assignment.items())
            )
            for name in (
                "raw_pair_excess",
                "normalized_contention",
                "cross_contention",
            ):
                flattened[f"{prefix}.phase[{phase_name}].{name}"] = float(
                    phase[name]
                )
    return flattened


def _feature_dominators(
    feature_vectors: Mapping[str, Mapping[str, object]],
    oracle_word: str,
) -> tuple[list[str], list[str]]:
    flattened = {
        word: _flatten_primitive_features(features)
        for word, features in feature_vectors.items()
    }
    coordinate_names = sorted(
        {name for vector in flattened.values() for name in vector}
    )
    oracle = flattened[oracle_word]
    dominators = []
    for word, candidate in flattened.items():
        if word == oracle_word:
            continue
        comparisons = []
        strict = False
        for name in coordinate_names:
            candidate_value = candidate.get(name, 0.0)
            oracle_value = oracle.get(name, 0.0)
            tolerance = 1e-12 * max(
                abs(candidate_value), abs(oracle_value), 1.0
            )
            comparisons.append(candidate_value <= oracle_value + tolerance)
            strict = strict or candidate_value < oracle_value - tolerance
        if all(comparisons) and strict:
            dominators.append(word)
    return coordinate_names, sorted(dominators)


def _oracle_feature_audit(
    group: Mapping[str, object],
    oracle_words: Sequence[str],
) -> dict[str, object] | None:
    options = group.get("oracle_audit_options")
    features = group.get("primitive_features")
    if not isinstance(options, dict) or not isinstance(features, dict):
        return None
    if not (
        options.get("dump_components")
        or options.get("check_feature_dominance")
    ):
        return None
    reranker = group["placement_reranker"]
    assert isinstance(reranker, dict)
    ranked = reranker["ranked"]
    assert isinstance(ranked, list)
    selected_word = str(ranked[0]["word"]) if ranked else None
    frontier = group["locality_frontier"]
    assert isinstance(frontier, list)
    panel_words = {
        *(str(member["word"]) for member in frontier),
        *oracle_words,
    }
    if selected_word is not None:
        panel_words.add(selected_word)
    audits = []
    coordinate_names: list[str] = []
    for oracle_word in oracle_words:
        coordinate_names, dominators = _feature_dominators(
            features, oracle_word
        )
        audits.append(
            {
                "oracle_word": oracle_word,
                "componentwise_dominated": bool(dominators),
                "dominating_words": dominators,
            }
        )
    result: dict[str, object] = {
        "definition": (
            "componentwise minimization over Q_fine, every nonzero locality "
            "excess footprint, raw and normalized placement, within-array and "
            "cross-array placement, and every global allocation phase"
        ),
        "primitive_count": len(coordinate_names),
        "selected_word": selected_word,
        "oracles": audits,
    }
    if options.get("dump_components"):
        result["records"] = [
            {
                "word": word,
                "roles": [
                    role
                    for role, present in (
                        ("oracle", word in oracle_words),
                        ("reranker_at_1", word == selected_word),
                        (
                            "locality_frontier",
                            any(
                                str(member["word"]) == word
                                for member in frontier
                            ),
                        ),
                    )
                    if present
                ],
                "features": features[word],
            }
            for word in sorted(panel_words)
        ]
    return result


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
        grammar_name = (
            "canonical" if configuration["grammar"] == "G_C" else "standard"
        )
        base_words = set(grammar_words(n, grammar_name))
        base_records = [
            record
            for record in group_records
            if str(record["word"]) in base_words
        ]
        base_records.sort(
            key=lambda record: (
                float(record["timing"]["median_ms"]),
                str(record["word"]),
            )
        )
        oracle_complete = len(base_records) == total_layouts
        observed_top = [
            {
                "word": record["word"],
                "layouts": {
                    name: record["word"] for name in group["target_arrays"]
                },
                "timing": record["timing"],
            }
            for record in base_records[:5]
        ]
        best_observed_layouts = (
            [
                layout
                for layout, record in zip(observed_top, base_records[:5])
                if float(record["timing"]["median_ms"])
                == float(base_records[0]["timing"]["median_ms"])
            ]
            if base_records
            else []
        )
        oracle_best = (
            float(base_records[0]["timing"]["median_ms"])
            if oracle_complete and base_records
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
        locality_members = group["locality_frontier"]
        assert isinstance(locality_members, list)
        locality_complete, locality_best, locality_best_layouts = _best_timed(
            locality_members, raw_records, kernel, n
        )
        locality_layouts = [
            _timed_member(
                member,
                raw_records.get((kernel, n, str(member["word"]))),
            )
            for member in locality_members
        ]
        reranker_plan = group["placement_reranker"]
        assert isinstance(reranker_plan, dict)
        reranked_members = reranker_plan["ranked"]
        assert isinstance(reranked_members, list)
        rerank_results = {}
        for budget in reranker_plan["budgets"]:
            selected_members = reranked_members[: int(budget)]
            complete, best_ms, best_layouts = _best_timed(
                selected_members, raw_records, kernel, n
            )
            rerank_results[str(budget)] = {
                "budget": int(budget),
                "selected_count": len(selected_members),
                "complete": complete,
                "best_time_ms": best_ms if complete else None,
                "regret": (
                    best_ms / oracle_best - 1.0
                    if complete
                    and best_ms is not None
                    and oracle_best is not None
                    else None
                ),
                "best_layouts": best_layouts if complete else [],
                "layouts": [
                    _timed_member(
                        member,
                        raw_records.get((kernel, n, str(member["word"]))),
                    )
                    for member in selected_members
                ],
            }
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
                    "resource_maps": group["resource_maps"],
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
                "base_timed_layout_count": len(base_records),
                "oracle": {
                    "definition": (
                        "minimum median kernel time over every shared-word "
                        f"{configuration['grammar']} canonical flag representative; "
                        "fiber materializations are excluded"
                    ),
                    "complete": oracle_complete,
                    "best_time_ms": oracle_best,
                    "best_layouts": oracle_layouts,
                    "top_layouts": observed_top if oracle_complete else [],
                    "best_observed_time_ms": (
                        float(base_records[0]["timing"]["median_ms"])
                        if base_records
                        else None
                    ),
                    "best_observed_layouts": best_observed_layouts,
                    "top_observed_layouts": observed_top,
                },
                "flag_fiber": group["flag_fiber"],
                "frontier": {
                    "definition": (
                        "exact locality frontier over "
                        "(Q_fine, J_peak, J_area); J_place is not a "
                        "dominance coordinate"
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
                "locality_frontier": {
                    "definition": (
                        "exact Pareto frontier over "
                        "(Q_fine, J_peak, J_area)"
                    ),
                    "objectives": group["locality_frontier_objectives"],
                    "size": len(locality_members),
                    "complete": locality_complete,
                    "best_time_ms": locality_best if locality_complete else None,
                    "regret": (
                        locality_best / oracle_best - 1.0
                        if locality_complete
                        and locality_best is not None
                        and oracle_best is not None
                        else None
                    ),
                    "best_layouts": (
                        locality_best_layouts if locality_complete else []
                    ),
                    "layouts": locality_layouts,
                },
                "placement_reranker": {
                    "definition": reranker_plan["definition"],
                    "locality_shell_tolerance": reranker_plan[
                        "locality_shell_tolerance"
                    ],
                    "placement_statistic": reranker_plan[
                        "placement_statistic"
                    ],
                    "shell_count": reranker_plan["shell_count"],
                    "bucket_winner_count": reranker_plan[
                        "bucket_winner_count"
                    ],
                    "pool_count": reranker_plan["pool_count"],
                    "regret_at_k": rerank_results,
                },
                "oracle_feature_audit": (
                    _oracle_feature_audit(
                        group,
                        [str(layout["word"]) for layout in oracle_layouts],
                    )
                    if oracle_complete
                    else None
                ),
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
    words.extend(
        str(member["word"]) for member in group["locality_frontier"]
    )
    reranker = group["placement_reranker"]
    assert isinstance(reranker, dict)
    words.extend(str(member["word"]) for member in reranker["ranked"])
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
    if args.resume and args.rescore:
        parser.error("--resume and --rescore are mutually exclusive")
    kernel_names = [
        str(value)
        for value in _deduplicated(args.kernel or list(KERNEL_SPECS))
    ]
    sizes = [int(value) for value in _deduplicated(args.size or list(DEFAULT_SIZES))]
    if args.fiber_max_xors and args.grammar != "standard":
        parser.error("--fiber-max-xors currently requires --grammar standard")
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
        if args.rescore:
            if not raw_output.exists():
                raise ValueError(f"--rescore requires existing file: {raw_output}")
            records = load_raw(raw_output, configuration)
            plan = build_plan(args, kernel_names, sizes, configuration)
            _atomic_write_json(plan_path, plan)
        elif args.resume:
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

        for seed_path in args.seed_raw or ():
            imported = import_raw_timings(
                seed_path.expanduser().resolve(),
                raw_output,
                configuration,
                records,
            )
            print(
                f"Imported {imported} identity timings from {seed_path}",
                flush=True,
            )
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
                LayoutCase(f"shared_{configuration['grammar']}_{word}", word),
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
