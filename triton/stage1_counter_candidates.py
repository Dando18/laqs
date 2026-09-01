"""Persistent-layout panels for Stage-1 counter sweeps."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import json
from math import comb
import random


def _tile_exponents(matrix, tile_shape: Sequence[int]) -> tuple[int, ...]:
    shape = tuple(int(extent) for extent in tile_shape)
    if len(shape) != matrix.rank:
        raise ValueError("counter-panel tile rank does not match the matrix")
    exponents = []
    for tile_extent, matrix_extent in zip(shape, matrix.shape):
        if (
            tile_extent <= 0
            or tile_extent > matrix_extent
            or tile_extent & (tile_extent - 1)
        ):
            raise ValueError(
                "counter-panel tile extents must be power-of-two matrix divisors"
            )
        exponents.append(tile_extent.bit_length() - 1)
    return tuple(exponents)


def canonical_mode_words(counts: Sequence[int]) -> Iterable[tuple[int, ...]]:
    """Enumerate multiset words in stable mode-index order."""

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


def canonical_word_count(counts: Sequence[int]) -> int:
    total = sum(counts)
    result = 1
    placed = 0
    for count in counts:
        result *= comb(placed + count, count)
        placed += count
    if placed != total:
        raise AssertionError("canonical word count accounting failed")
    return result


def _word_symbols(matrix, word: Sequence[int]) -> str:
    initials = [name[0] for name in matrix.mode_names]
    if len(set(initials)) != len(initials):
        return "/".join(matrix.mode_names[mode] for mode in word)
    return "".join(initials[mode] for mode in word)


def _stable_id(prefix: str, value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:12]}"


def _canonical_metadata(layout, matrix) -> dict[str, object]:
    inner_word = "".join(matrix.mode_names[mode] for mode in layout.word)
    outer_word = "".join(
        matrix.mode_names[mode]
        * (matrix.mode_bits[mode] - layout.tile_exponents[mode])
        for mode in layout.outer_order
    )
    return {
        "word": inner_word + outer_word,
        "inner_word": inner_word,
        "inner_tile_shape": list(layout.tile_shape),
        "fixed_outer_order": [
            matrix.mode_names[mode] for mode in layout.outer_order
        ],
    }


def _candidate_record(matrix, layout, quotient_score: float):
    from relay import layout_codegen_runs, layout_matrix_rows

    rows = layout_matrix_rows(matrix, layout)
    mapping_id = _stable_id("mapping", list(rows))
    word_modes = [matrix.mode_names[mode] for mode in layout.word]
    return {
        "candidate_id": _stable_id(
            "counter_candidate",
            {
                "matrix": matrix.name,
                "tile_shape": layout.tile_shape,
                "word": word_modes,
                "a_rows": rows,
            },
        ),
        "mapping_id": mapping_id,
        "layout": layout.name,
        "grammar": layout.grammar,
        **_canonical_metadata(layout, matrix),
        "inner_word": _word_symbols(matrix, layout.word),
        "inner_mode_indices": list(layout.word),
        "inner_mode_order": word_modes,
        "a_rows": list(rows),
        "quotient_score": float(quotient_score),
        "address_expression_runs": int(layout_codegen_runs(matrix, layout)),
        "inner_word_runs": int(layout.runs),
        "xor_count": int(layout.xor_count),
    }


def _random_invertible_rows(
    width: int, generator: random.Random
) -> tuple[int, ...]:
    """Draw uniformly from the invertible binary matrices of ``width``."""

    from relay.gf2 import invert_matrix_rows

    if width <= 0:
        raise ValueError("random linear layouts require a nonempty inner tile")
    while True:
        rows = tuple(generator.getrandbits(width) for _ in range(width))
        try:
            invert_matrix_rows(rows, width)
        except ValueError:
            continue
        return rows


def _linear_candidate_record(
    matrix,
    layout,
    quotient_score: float,
    *,
    sample_index: int,
    sampling_attempt: int,
) -> dict[str, object]:
    from relay import layout_codegen_runs, layout_matrix_rows

    rows = layout_matrix_rows(matrix, layout)
    mapping_id = _stable_id("mapping", list(rows))
    return {
        "candidate_id": _stable_id(
            "random_layout_candidate",
            {"matrix": matrix.name, "a_rows": rows},
        ),
        "mapping_id": mapping_id,
        "layout": layout.name,
        "layout_descriptor": layout.descriptor(matrix),
        "grammar": layout.grammar,
        "word": None,
        "inner_word": "linear",
        "inner_mode_indices": [],
        "inner_mode_order": [],
        "inner_tile_shape": list(layout.tile_shape),
        "inner_a_rows": list(layout.a_rows),
        "fixed_outer_order": [
            matrix.mode_names[mode] for mode in layout.outer_order
        ],
        "a_rows": list(rows),
        "quotient_score": float(quotient_score),
        "address_expression_runs": int(layout_codegen_runs(matrix, layout)),
        "inner_word_runs": int(layout.runs),
        "xor_count": int(layout.xor_count),
        "sample_index": sample_index,
        "sampling_attempt": sampling_attempt,
    }


def random_linear_counter_panel(
    matrix,
    issue_component,
    tile_shapes: Sequence[Sequence[int]],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """Sample distinct linear inner layouts over uniformly drawn tile shapes."""

    from relay import LinearInnerLayout, weighted_component_region_count

    if samples <= 0:
        raise ValueError("random counter panels require at least one sample")
    tile_exponents = tuple(
        dict.fromkeys(_tile_exponents(matrix, shape) for shape in tile_shapes)
    )
    if not tile_exponents:
        raise ValueError("random counter panels require at least one tile shape")
    if any(sum(exponents) == 0 for exponents in tile_exponents):
        raise ValueError("random counter panels do not support scalar tiles")

    generator = random.Random(seed)
    outer_order = tuple(reversed(range(matrix.rank)))
    candidates = []
    mapping_ids = set()
    tile_counts = {exponents: 0 for exponents in tile_exponents}
    max_attempts = max(1_000, samples * 100)
    for attempt in range(1, max_attempts + 1):
        exponents = tile_exponents[generator.randrange(len(tile_exponents))]
        width = sum(exponents)
        inner_rows = _random_invertible_rows(width, generator)
        layout = LinearInnerLayout(
            "random_"
            + "x".join(str(1 << exponent) for exponent in exponents)
            + f"_{attempt}",
            matrix.name,
            exponents,
            inner_rows,
            outer_order,
        )
        layout.validate(matrix)
        score = float(
            weighted_component_region_count(matrix, layout, issue_component)
        )
        candidate = _linear_candidate_record(
            matrix,
            layout,
            score,
            sample_index=len(candidates) + 1,
            sampling_attempt=attempt,
        )
        if candidate["mapping_id"] in mapping_ids:
            continue
        mapping_ids.add(candidate["mapping_id"])
        candidates.append(candidate)
        tile_counts[exponents] += 1
        if len(candidates) == samples:
            break
    else:
        raise ValueError(
            f"could not draw {samples} distinct layouts in {max_attempts} attempts"
        )

    levels = sorted({candidate["quotient_score"] for candidate in candidates})
    for candidate in candidates:
        candidate["panel_rank"] = candidate["sample_index"]
        candidate["quotient_rank"] = levels.index(
            candidate["quotient_score"]
        ) + 1
    return {
        "selection": "seeded random sampling without replacement",
        "grouping": "none",
        "layout_grammar": "invertible_linear_inner",
        "sampling_distribution": {
            "tile_shape": "uniform over kernel tile hypotheses",
            "layout_given_tile": "uniform over invertible binary matrices",
        },
        "random_seed": seed,
        "requested_sample_count": samples,
        "sampling_attempt_count": max(
            candidate["sampling_attempt"] for candidate in candidates
        ),
        "outer_layout": "row_major_tiles",
        "fixed_outer_order": [
            matrix.mode_names[mode] for mode in outer_order
        ],
        "tile_grammars": [
            {
                "tile_shape": [1 << exponent for exponent in exponents],
                "tile_exponents": list(exponents),
                "sample_count": tile_counts[exponents],
            }
            for exponents in tile_exponents
        ],
        "unique_mapping_count": len(candidates),
        "representative_count": len(candidates),
        "quotient_levels": levels,
        "candidates": candidates,
    }


def canonical_counter_panel(
    matrix,
    issue_component,
    tile_shapes: Sequence[Sequence[int]],
    *,
    group_by_tile: bool,
) -> dict[str, object]:
    """Select minimum-run canonical representatives from complete grammars.

    With ``group_by_tile=False``, one mapping is retained per distinct quotient
    score over the supplied tiles.  With ``group_by_tile=True``, one mapping is
    first retained per tile and quotient pair, then identical full mappings are
    removed globally.
    """

    from relay import CanonicalLayout, weighted_component_region_count

    outer_order = tuple(reversed(range(matrix.rank)))
    scored = []
    tile_records = []
    for tile_shape in tile_shapes:
        exponents = _tile_exponents(matrix, tile_shape)
        expected_count = canonical_word_count(exponents)
        observed_count = 0
        levels: set[float] = set()
        for word in canonical_mode_words(exponents):
            observed_count += 1
            word_text = _word_symbols(matrix, word)
            layout = CanonicalLayout(
                "counter_"
                + "x".join(str(1 << exponent) for exponent in exponents)
                + "_"
                + word_text,
                matrix.name,
                exponents,
                word,
                outer_order,
            )
            layout.validate(matrix)
            score = float(
                weighted_component_region_count(
                    matrix, layout, issue_component
                )
            )
            levels.add(score)
            scored.append(_candidate_record(matrix, layout, score))
        if observed_count != expected_count:
            raise AssertionError("canonical grammar enumeration is incomplete")
        tile_records.append(
            {
                "tile_shape": [1 << exponent for exponent in exponents],
                "tile_exponents": list(exponents),
                "canonical_word_count": expected_count,
                "quotient_levels": sorted(levels),
            }
        )

    unique_mappings = {}
    for candidate in scored:
        mapping_id = candidate["mapping_id"]
        key = (
            candidate["address_expression_runs"],
            tuple(candidate["inner_mode_indices"]),
            tuple(candidate["inner_tile_shape"]),
        )
        incumbent = unique_mappings.get(mapping_id)
        if incumbent is None or key < incumbent[0]:
            unique_mappings[mapping_id] = (key, candidate)

    representatives = {}
    for _key, candidate in unique_mappings.values():
        group = (
            tuple(candidate["inner_tile_shape"]),
            candidate["quotient_score"],
        ) if group_by_tile else (candidate["quotient_score"],)
        key = (
            candidate["address_expression_runs"],
            tuple(candidate["inner_mode_indices"]),
            tuple(candidate["a_rows"]),
        )
        incumbent = representatives.get(group)
        if incumbent is None or key < incumbent[0]:
            representatives[group] = (key, candidate)

    selected_by_mapping = {}
    for _key, candidate in representatives.values():
        mapping_id = candidate["mapping_id"]
        key = (
            candidate["address_expression_runs"],
            tuple(candidate["inner_tile_shape"]),
            tuple(candidate["inner_mode_indices"]),
        )
        incumbent = selected_by_mapping.get(mapping_id)
        if incumbent is None or key < incumbent[0]:
            selected_by_mapping[mapping_id] = (key, candidate)
    candidates = [candidate for _key, candidate in selected_by_mapping.values()]
    candidates.sort(
        key=lambda candidate: (
            candidate["quotient_score"],
            candidate["address_expression_runs"],
            tuple(candidate["inner_tile_shape"]),
            tuple(candidate["inner_mode_indices"]),
        )
    )
    levels = sorted({candidate["quotient_score"] for candidate in candidates})
    for index, candidate in enumerate(candidates, 1):
        candidate["panel_rank"] = index
        candidate["quotient_rank"] = levels.index(
            candidate["quotient_score"]
        ) + 1

    return {
        "selection": (
            "minimum full address-expression runs, then stable canonical word"
        ),
        "grouping": (
            "tile_and_quotient" if group_by_tile else "quotient"
        ),
        "outer_layout": "row_major_tiles",
        "fixed_outer_order": [
            matrix.mode_names[mode] for mode in outer_order
        ],
        "tile_grammars": tile_records,
        "enumerated_word_count": len(scored),
        "unique_mapping_count": len(unique_mappings),
        "representative_count": len(candidates),
        "quotient_levels": levels,
        "candidates": candidates,
    }
