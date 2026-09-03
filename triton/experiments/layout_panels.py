"""Random layout panels and quotient scores for final experiments 1--3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import comb
from pathlib import Path
import random
from typing import Iterable, Sequence


BYTE_SCALES = (32, 64, 128, 256)
TAU_PROFILES = Path(__file__).with_name("tau-profiles.json")


@dataclass(frozen=True)
class CounterScoreProfile:
    """Fixed, predeclared score-to-counter configuration for one platform."""

    platform: str
    profile_id: str
    native_bytes: int
    fine_component: str
    active_weights: dict[str, float]
    counter_components: dict[str, str]


def counter_score_profile(
    platform: str, component_names: set[str]
) -> CounterScoreProfile:
    """Return the tuned automatic-graph profile for one platform."""

    if platform == "tuolumne":
        native = 64
        fine = "issue.g64.stream.load.64B"
        counters = {
            "l1_cache_line_accesses": fine,
            "first_level_read_events": fine,
            "l1_to_l2_read_requests": "simd_window.t16.stream.load.128B",
            "l1_to_l2_total_requests": "simd_window.t16.stream.load.128B",
            "l2_tag_requests": "simd_window.t16.stream.load.128B",
            "second_level_read_requests": "simd_window.t16.stream.load.128B",
            "l2_hits": "simd_window.t16.stream.load.128B",
            "l2_misses": "simd_window.t16.stream.load.256B",
            "hbm_read_bytes": "simd_window.t16.stream.load.256B",
        }
    elif platform == "matrix":
        native = 32
        fine = "issue.g32.stream.load.32B"
        counters = {
            "first_level_memory_accesses": fine,
            "global_load_requests": fine,
            "l1_to_l2_read_traffic": "simd_window.t16.stream.load.128B",
            "l2_read_work": "simd_window.t16.stream.load.128B",
            "l2_read_misses": "simd_window.t16.stream.load.256B",
            "hbm_read_bytes": "simd_window.t16.stream.load.256B",
            "sectors_per_request": fine,
        }
    else:
        raise ValueError(f"unknown experiment platform {platform!r}")

    missing_fine = fine not in component_names
    if missing_fine:
        raise ValueError(f"automatic graph is missing fine component {fine!r}")
    counters = {
        counter: component if component in component_names else fine
        for counter, component in counters.items()
    }
    active = {fine: 1.0}
    profile_id = f"automatic-bootstrap-{platform}-v1"
    if TAU_PROFILES.is_file():
        document = json.loads(TAU_PROFILES.read_text(encoding="utf-8"))
        if tuple(document["byte_scales"]) != BYTE_SCALES:
            raise ValueError("tau profile byte scales do not match experiment scales")
        record = document["platforms"][platform]
        configured = {
            str(name): float(value)
            for name, value in record["active_tau"].items()
            if float(value) != 0.0
        }
        unknown = sorted(set(configured) - component_names)
        if unknown:
            raise ValueError(
                "tau profile references components absent from this graph: "
                + ", ".join(unknown)
            )
        active = configured
        profile_id = str(record["profile_id"])
        counters.update(
            {
                str(counter): str(component)
                for counter, component in record["counter_components"].items()
                if str(component) in component_names
            }
        )
    return CounterScoreProfile(
        platform,
        profile_id,
        native,
        fine,
        active,
        counters,
    )


def _stable_id(prefix: str, value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:12]}"


def _tile_exponents(matrix, tile_shape: Sequence[int]) -> tuple[int, ...]:
    shape = tuple(int(extent) for extent in tile_shape)
    if len(shape) != matrix.rank:
        raise ValueError("tile rank does not match the target tensor")
    exponents = []
    for tile_extent, matrix_extent in zip(shape, matrix.shape):
        if (
            tile_extent <= 0
            or tile_extent > matrix_extent
            or tile_extent & (tile_extent - 1)
            or matrix_extent % tile_extent
        ):
            raise ValueError("tile extents must be power-of-two tensor divisors")
        exponents.append(tile_extent.bit_length() - 1)
    return tuple(exponents)


def _canonical_word_count(counts: Sequence[int]) -> int:
    result = 1
    placed = 0
    for count in counts:
        result *= comb(placed + count, count)
        placed += count
    return result


def _canonical_words(counts: Sequence[int]) -> Iterable[tuple[int, ...]]:
    remaining = list(counts)
    word: list[int] = []

    def visit():
        if not any(remaining):
            yield tuple(word)
            return
        for mode, count in enumerate(remaining):
            if count == 0:
                continue
            remaining[mode] -= 1
            word.append(mode)
            yield from visit()
            word.pop()
            remaining[mode] += 1

    yield from visit()


def _random_canonical_word(
    counts: Sequence[int], generator: random.Random
) -> tuple[int, ...]:
    """Draw uniformly from distinct permutations of a mode multiset."""

    remaining = list(counts)
    word = []
    while sum(remaining):
        draw = generator.randrange(sum(remaining))
        cumulative = 0
        for mode, count in enumerate(remaining):
            cumulative += count
            if draw < cumulative:
                word.append(mode)
                remaining[mode] -= 1
                break
    return tuple(word)


def _random_invertible_rows(
    width: int, generator: random.Random
) -> tuple[int, ...]:
    from relay.gf2 import invert_matrix_rows

    if width <= 0:
        raise ValueError("G_OC requires a nonempty inner tile")
    while True:
        rows = tuple(generator.getrandbits(width) for _ in range(width))
        try:
            invert_matrix_rows(rows, width)
        except ValueError:
            continue
        return rows


def _score_components(analysis, target_name: str):
    from dataclasses import replace

    from relay import materialize_edge_families

    analysis.require_supported()
    available = {matrix.name: matrix for matrix in analysis.matrices}
    if target_name not in available:
        raise ValueError(
            f"automatic graph has no target allocation {target_name!r}; "
            f"available allocations are {sorted(available)}"
        )
    matrices = {target_name: replace(available[target_name], target=True)}
    families = tuple(
        replace(
            family,
            edges_by_array={target_name: family.edges_by_array[target_name]},
        )
        for family in analysis.edge_families
        if family.edges_by_array.get(target_name)
    )
    components = materialize_edge_families(
        families, matrices, BYTE_SCALES
    )
    if not components:
        raise ValueError("automatic graph produced no objective components")
    exposure = float(components[0].normalization_bytes)
    graph = {
        "construction": "automatic_post_coalescing_manifest_universal_v1",
        "manifest_schema": analysis.manifest.schema,
        "manifest_version": analysis.manifest.version,
        "grid": list(analysis.grid),
        "selected_config": dict(analysis.selected_config),
        "allocation_count": len(analysis.allocations),
        "eligible_allocations": [
            allocation.name
            for allocation in analysis.allocations
            if allocation.eligible
        ],
        "scored_allocation": target_name,
        "memory_event_count": len(analysis.events),
        "trace_class_count": len(analysis.sequences),
        "trace_class_multiplicity": sum(
            sequence.multiplicity for sequence in analysis.sequences
        ),
        "universal_edge_family_count": len(analysis.edge_families),
        "scored_edge_family_count": len(families),
        "component_count": len(components),
        "byte_scales": list(BYTE_SCALES),
    }
    return matrices[target_name], matrices, components, exposure, graph


def _layout_record(
    matrix,
    matrices,
    layout,
    components,
    profile: CounterScoreProfile,
    *,
    experiment: int,
    sample_index: int,
    sampling_attempt: int,
) -> dict[str, object]:
    from relay import (
        CanonicalLayout,
        layout_codegen_runs,
        layout_matrix_rows,
        score_layouts,
        score_to_dict,
    )

    rows = tuple(layout_matrix_rows(matrix, layout))
    component_names = {component.name for component in components}
    unknown = sorted(set(profile.active_weights) - component_names)
    if unknown:
        raise ValueError(f"score profile references missing components: {unknown}")
    weights = {component.name: 0.0 for component in components}
    weights.update(profile.active_weights)
    layouts = {matrix.name: layout}
    score = score_layouts(
        matrices,
        components,
        layouts,
        component_weights=weights,
    )
    score_record = score_to_dict(score)
    component_records = {
        component["name"]: component for component in score_record["components"]
    }
    fine_name = profile.fine_component
    fine = component_records[fine_name]
    mapping_id = _stable_id("mapping", list(rows))
    if isinstance(layout, CanonicalLayout):
        inner_word = layout.word_string(matrix)
        word = inner_word + "".join(
            matrix.mode_names[mode]
            * (matrix.mode_bits[mode] - layout.tile_exponents[mode])
            for mode in layout.outer_order
        )
        inner_rows: list[int] | None = None
    else:
        inner_word = "linear"
        word = layout.evaluator_descriptor(matrix)
        inner_rows = list(layout.a_rows)
    return {
        "candidate_id": _stable_id(
            f"experiment{experiment}_candidate",
            {"matrix": matrix.name, "a_rows": rows},
        ),
        "mapping_id": mapping_id,
        "layout": layout.name,
        "layout_descriptor": word,
        "grammar": layout.grammar,
        "word": word,
        "inner_word": inner_word,
        "inner_tile_shape": list(layout.tile_shape),
        "inner_a_rows": inner_rows,
        "fixed_outer_order": [
            matrix.mode_names[mode] for mode in layout.outer_order
        ],
        "a_rows": list(rows),
        "quotient_score": float(fine["raw_region_count"]),
        "fine_component": fine_name,
        "j_area": float(score.hardware_area),
        "peak_normalized_excess": float(score.peak_normalized_excess),
        "score": score_record,
        "address_expression_runs": int(layout_codegen_runs(matrix, layout)),
        "inner_word_runs": int(layout.runs),
        "xor_count": int(layout.xor_count),
        "sample_index": sample_index,
        "sampling_attempt": sampling_attempt,
    }


def _candidate_layouts(
    matrix,
    tile_shapes: Sequence[Sequence[int]],
    *,
    experiment: int,
    samples: int,
    seed: int,
):
    from relay import CanonicalLayout, LinearInnerLayout, layout_matrix_rows

    if samples <= 0:
        raise ValueError("sample count must be positive")
    outer_order = tuple(reversed(range(matrix.rank)))
    generator = random.Random(seed)
    if experiment == 1:
        tile_exponents = (tuple(matrix.mode_bits),)
    else:
        tile_exponents = tuple(
            dict.fromkeys(_tile_exponents(matrix, shape) for shape in tile_shapes)
        )
    if not tile_exponents:
        raise ValueError("the kernel supplies no tile hypotheses")

    grammar_size = None
    if experiment == 1:
        grammar_size = _canonical_word_count(tile_exponents[0])
    elif experiment == 2:
        grammar_size = sum(_canonical_word_count(item) for item in tile_exponents)

    layouts = []
    mappings = set()

    def add(layout, attempt: int) -> None:
        layout.validate(matrix)
        mapping = tuple(layout_matrix_rows(matrix, layout))
        if mapping in mappings:
            return
        mappings.add(mapping)
        layouts.append((layout, attempt))

    if grammar_size is not None and grammar_size <= samples * 4:
        attempt = 0
        for exponents in tile_exponents:
            for word in _canonical_words(exponents):
                attempt += 1
                add(
                    CanonicalLayout(
                        f"experiment{experiment}_canonical_{attempt}",
                        matrix.name,
                        exponents,
                        word,
                        outer_order,
                    ),
                    attempt,
                )
        generator.shuffle(layouts)
        layouts = layouts[:samples]
    else:
        max_attempts = max(10_000, samples * 1_000)
        for attempt in range(1, max_attempts + 1):
            exponents = tile_exponents[generator.randrange(len(tile_exponents))]
            if experiment in (1, 2):
                word = _random_canonical_word(exponents, generator)
                layout = CanonicalLayout(
                    f"experiment{experiment}_canonical_{attempt}",
                    matrix.name,
                    exponents,
                    word,
                    outer_order,
                )
            elif experiment == 3:
                width = sum(exponents)
                layout = LinearInnerLayout(
                    f"experiment3_goc_{attempt}",
                    matrix.name,
                    exponents,
                    _random_invertible_rows(width, generator),
                    outer_order,
                )
            else:
                raise ValueError(f"unknown final experiment {experiment}")
            add(layout, attempt)
            if len(layouts) == samples:
                break
        if not layouts:
            raise ValueError("layout sampling produced no distinct mappings")

    return layouts, tile_exponents, grammar_size


def random_experiment_panel(
    matrix,
    automatic_analysis,
    automatic_target_name: str,
    tile_shapes: Sequence[Sequence[int]],
    *,
    experiment: int,
    samples: int,
    seed: int,
    platform: str,
) -> dict[str, object]:
    """Build and score a reproducible random panel for experiment 1, 2, or 3."""

    if experiment not in (1, 2, 3):
        raise ValueError("only final experiments 1--3 define random panels")
    matrix, matrices, components, exposure, graph = _score_components(
        automatic_analysis, automatic_target_name
    )
    profile = counter_score_profile(
        platform, {component.name for component in components}
    )
    layouts, tile_exponents, grammar_size = _candidate_layouts(
        matrix,
        tile_shapes,
        experiment=experiment,
        samples=samples,
        seed=seed,
    )
    candidates = [
        _layout_record(
            matrix,
            matrices,
            layout,
            components,
            profile,
            experiment=experiment,
            sample_index=index,
            sampling_attempt=attempt,
        )
        for index, (layout, attempt) in enumerate(layouts, 1)
    ]
    levels = sorted({candidate["j_area"] for candidate in candidates})
    for candidate in candidates:
        candidate["panel_rank"] = candidate["sample_index"]
        candidate["j_area_rank"] = levels.index(candidate["j_area"]) + 1

    grammar = {
        1: "whole_tensor_canonical",
        2: "canonical_inner_tile",
        3: "outer_canonical_linear_inner",
    }[experiment]
    distribution = {
        1: {
            "layout": "uniform over distinct whole-tensor canonical words",
        },
        2: {
            "tile_shape": "uniform over kernel tile hypotheses",
            "layout_given_tile": "uniform over distinct canonical words",
        },
        3: {
            "tile_shape": "uniform over kernel tile hypotheses",
            "layout_given_tile": "uniform over GL(p,2)",
        },
    }[experiment]
    return {
        "selection": "seeded random sampling without replacement",
        "grouping": "none",
        "layout_grammar": grammar,
        "sampling_distribution": distribution,
        "random_seed": seed,
        "requested_sample_count": samples,
        "realized_sample_count": len(candidates),
        "sample_shortfall": samples - len(candidates),
        "finite_grammar_size_before_mapping_deduplication": grammar_size,
        "sampling_attempt_count": max(
            candidate["sampling_attempt"] for candidate in candidates
        ),
        "outer_layout": "row_major_tiles",
        "fixed_outer_order": [
            matrix.mode_names[mode] for mode in reversed(range(matrix.rank))
        ],
        "tile_grammars": [
            {
                "tile_shape": [1 << exponent for exponent in exponents],
                "tile_exponents": list(exponents),
                "inner_bits": sum(exponents),
            }
            for exponents in tile_exponents
        ],
        "unique_mapping_count": len(candidates),
        "representative_count": len(candidates),
        "j_area_levels": levels,
        "score_profile": {
            "profile_id": profile.profile_id,
            "platform": profile.platform,
            "native_first_level_bytes": profile.native_bytes,
            "byte_scales": list(BYTE_SCALES),
            "dynamic_useful_byte_exposure": exposure,
            "active_tau": profile.active_weights,
            "counter_components": profile.counter_components,
            "selection_score": "J_area",
            "component_model": graph,
        },
        "candidates": candidates,
    }
