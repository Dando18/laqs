"""Shared packing, timing, and codegen utilities for Triton Stage 1."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import statistics

import torch


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def layout_rows(layout, matrix) -> tuple[int, ...]:
    from relay import layout_matrix_rows

    return layout_matrix_rows(matrix, layout)


def canonical_layout_metadata(layout, matrix) -> dict[str, object]:
    from relay import CanonicalLayout

    if not isinstance(layout, CanonicalLayout):
        raise TypeError(
            f"Stage 1 canonical search produced {type(layout).__name__}"
        )
    inner_word = layout.word_string(matrix)
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


def physical_offsets(
    shape: tuple[int, ...], rows: tuple[int, ...]
) -> torch.Tensor:
    mode_bits = tuple(extent.bit_length() - 1 for extent in shape)
    logical = torch.zeros(shape, dtype=torch.int64)
    shift = 0
    for dimension, (extent, bits) in enumerate(zip(shape, mode_bits)):
        axis_shape = [1] * len(shape)
        axis_shape[dimension] = extent
        values = torch.arange(extent, dtype=torch.int64).reshape(axis_shape)
        logical |= values << shift
        shift += bits
    physical = torch.zeros_like(logical)
    for physical_bit, row in enumerate(rows):
        selected = logical & row
        selected ^= selected >> 32
        selected ^= selected >> 16
        selected ^= selected >> 8
        selected ^= selected >> 4
        selected ^= selected >> 2
        selected ^= selected >> 1
        physical |= (selected & 1) << physical_bit
    return physical


def pack_tensor(logical: torch.Tensor, rows: tuple[int, ...]) -> torch.Tensor:
    offsets = physical_offsets(tuple(logical.shape), rows)
    packed = torch.empty_like(logical).flatten()
    packed[offsets.flatten()] = logical.flatten()
    return packed


def validate_output(
    label: str,
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-3,
) -> None:
    observed = output.cpu()
    if torch.allclose(observed, reference, rtol=rtol, atol=atol):
        return
    error = torch.max(torch.abs(observed - reference)).item()
    raise ValueError(f"{label} layout produced incorrect output: max error {error}")


def timing_summary(samples_ms: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "stdev_ms": (
            statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
        ),
        "samples_ms": samples_ms,
    }


def benchmark_layouts(
    launches,
    *,
    samples: int,
    iterations: int,
    warmup: int,
) -> dict[str, dict[str, object]]:
    labels = tuple(launches)
    for _ in range(warmup):
        for label in labels:
            launches[label]()
    torch.cuda.synchronize()

    timings = {label: [] for label in labels}
    for sample in range(samples):
        rotation = (sample // 2) % len(labels)
        base = labels if sample % 2 == 0 else tuple(reversed(labels))
        order = base[rotation:] + base[:rotation]
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                launches[label]()
            stop.record()
            stop.synchronize()
            timings[label].append(float(start.elapsed_time(stop)) / iterations)
    return {label: timing_summary(values) for label, values in timings.items()}


def benchmark_layouts_isolated(
    launches,
    *,
    before_each,
    samples: int,
    iterations: int,
    warmup: int,
) -> dict[str, dict[str, object]]:
    """Time individual launches after a cache-state preparation callback."""

    labels = tuple(launches)
    for _ in range(warmup):
        for label in labels:
            before_each()
            launches[label]()
    torch.cuda.synchronize()

    timings = {label: [] for label in labels}
    for sample in range(samples):
        rotation = (sample // 2) % len(labels)
        base = labels if sample % 2 == 0 else tuple(reversed(labels))
        order = base[rotation:] + base[:rotation]
        for label in order:
            isolated = []
            for _ in range(iterations):
                before_each()
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                launches[label]()
                stop.record()
                stop.synchronize()
                isolated.append(float(start.elapsed_time(stop)))
            timings[label].append(statistics.fmean(isolated))
    return {label: timing_summary(values) for label, values in timings.items()}


def stable_id(prefix: str, value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:12]}"


def assembly_opcode_counts(assembly: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw_line in assembly.splitlines():
        line = raw_line.split("//", 1)[0].split(";", 1)[0].strip()
        if not line or line.startswith((".", "#")) or line.endswith(":"):
            continue
        opcode = line.split(None, 1)[0]
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", opcode):
            counts[opcode] += 1
    return counts


def compiled_codegen_statistics(compiled) -> dict[str, object]:
    assembly_key = next(
        (
            key
            for key in ("amdgcn", "sass", "ptx")
            if isinstance(compiled.asm.get(key), str)
        ),
        None,
    )
    assembly = "" if assembly_key is None else compiled.asm[assembly_key]
    opcodes = assembly_opcode_counts(assembly)
    loads = sum(
        count
        for opcode, count in opcodes.items()
        if "load" in opcode or opcode.startswith(("global_atomic", "flat_atomic"))
    )
    stores = sum(count for opcode, count in opcodes.items() if "store" in opcode)
    xors = sum(count for opcode, count in opcodes.items() if "xor" in opcode)
    branches = sum(
        count
        for opcode, count in opcodes.items()
        if "branch" in opcode or "cbranch" in opcode
    )
    binary_key = next(
        (
            key
            for key, value in compiled.asm.items()
            if isinstance(value, bytes)
        ),
        None,
    )
    ir_statistics = {
        key: {
            "bytes": len(value.encode()),
            "lines": len(value.splitlines()),
        }
        for key, value in compiled.asm.items()
        if isinstance(value, str) and key != assembly_key
    }
    return {
        "n_regs": int(compiled.n_regs),
        "n_spills": int(compiled.n_spills),
        "n_max_threads": int(compiled.n_max_threads),
        "shared_bytes": int(compiled.metadata.shared),
        "binary_format": binary_key,
        "binary_bytes": (
            len(compiled.asm[binary_key]) if binary_key is not None else None
        ),
        "assembly_format": assembly_key,
        "assembly_bytes": len(assembly.encode()),
        "assembly_instruction_count": sum(opcodes.values()),
        "load_instruction_count": loads,
        "store_instruction_count": stores,
        "xor_instruction_count": xors,
        "branch_instruction_count": branches,
        "opcode_counts": dict(sorted(opcodes.items())),
        "ir": ir_statistics,
    }


def execution_layout_from_compiled(compiled, shape, names):
    from relay import TritonLinearLayout, extract_blocked_layout
    from triton.tools import LinearLayout as NativeLinearLayout

    blocked = extract_blocked_layout(compiled.asm["ttgir"])
    extracted = TritonLinearLayout.from_blocked(
        shape,
        size_per_thread=blocked.size_per_thread,
        threads_per_warp=blocked.threads_per_warp,
        warps_per_cta=blocked.warps_per_cta,
        order=blocked.order,
        output_dim_names=names,
    )
    native = NativeLinearLayout.from_bases(
        extracted.bases,
        [name for name, _ in extracted.out_dims],
        [size for _, size in extracted.out_dims],
    )
    return blocked, TritonLinearLayout.from_triton(native)


def execution_layout_record(blocked, execution) -> dict[str, object]:
    return {
        "blocked": {name: list(values) for name, values in blocked.as_dict().items()},
        "input_sizes": {
            name: execution.input_size(name) for name in execution.input_dims
        },
        "output_shape": list(execution.output_shape),
        "bases": {
            name: [list(basis) for basis in bases]
            for name, bases in execution.bases
        },
    }


def issue_events(
    execution,
    matrix,
    *,
    prefix: str,
    site: str,
    weight: float,
    coordinate_map,
):
    from relay import induce_memory_event

    events = []
    register_count = execution.input_size("register")
    warp_count = execution.input_size("warp")
    for warp in range(warp_count):
        for register in range(register_count):
            locations = execution.locations(
                fixed={"register": register, "warp": warp, "block": 0}
            )
            events.append(
                induce_memory_event(
                    execution,
                    matrix,
                    locations,
                    id=f"{prefix}.w{warp}.r{register}",
                    site=site,
                    weight=weight,
                    coordinate_map=coordinate_map,
                )
            )
    return tuple(events)


def temporal_loop_edges(
    execution,
    matrix,
    *,
    prefix: str,
    steps: int,
    window: int,
    program_instances: int,
    coordinate_map,
):
    """Compress aligned loop windows for bit-linear address mappings."""

    from relay import Hyperedge

    if steps <= 0 or window <= 0 or program_instances <= 0:
        raise ValueError("temporal loop dimensions must be positive")
    if steps % window:
        raise ValueError("temporal steps must be divisible by the window")
    translated_windows = steps // window
    locations = execution.locations(fixed={"block": 0})
    edges = []
    for location in locations:
        tensor_coord = execution.apply(location)
        points = tuple(
            tuple(coordinate_map(tensor_coord, step))
            for step in range(window)
        )
        for point in points:
            matrix.validate_coord(point)
        hardware = ".".join(
            f"{name}{value}" for name, value in location.coordinates
        )
        edges.append(
            Hyperedge.make(
                points,
                weight=program_instances * translated_windows,
                source=f"temporal:{prefix}:{hardware}:window{window}",
            )
        )
    return tuple(edges), {
        "stream": prefix,
        "fiber": "per_hardware_location",
        "window_elements": window,
        "stride_elements": window,
        "loop_steps": steps,
        "program_instances": program_instances,
        "translated_windows_per_location": translated_windows,
        "representative_edge_count": len(edges),
        "dynamic_scalar_accesses": len(locations) * steps * program_instances,
        "compression": "aligned_xor_translations",
    }


def compressed_temporal_edges(matrix, *, prefix: str, point_sets):
    """Compress exact temporal fibers that differ only by XOR translation."""

    from relay import Hyperedge

    groups = {}
    dynamic_fibers = 0
    dynamic_accesses = 0
    for points in point_sets:
        items = tuple(tuple(point) for point in points)
        unique = tuple(sorted(set(items)))
        if not unique:
            raise ValueError("a temporal fiber cannot be empty")
        for point in unique:
            matrix.validate_coord(point)
        bits = tuple(matrix.coord_to_bits(point) for point in unique)
        key = min(
            tuple(sorted(value ^ anchor for value in bits))
            for anchor in bits
        )
        dynamic_fibers += 1
        dynamic_accesses += len(items)
        group = groups.get(key)
        if group is None:
            groups[key] = (unique, 1)
        else:
            groups[key] = (group[0], group[1] + 1)

    edges = tuple(
        Hyperedge.make(
            group[0],
            weight=group[1],
            source=f"temporal:{prefix}:class{index}",
        )
        for index, (_key, group) in enumerate(sorted(groups.items()))
    )
    return edges, {
        "stream": prefix,
        "fiber": "per_hardware_location",
        "window_elements": None,
        "stride_elements": None,
        "dynamic_fibers": dynamic_fibers,
        "representative_edge_count": len(edges),
        "dynamic_scalar_accesses": dynamic_accesses,
        "compression": "exact_xor_translation_classes",
    }


def solve_layouts(
    matrices,
    events,
    args,
    name,
    *,
    inner_tile_shapes,
    temporal_edges=None,
    temporal_mode="issue",
):
    from relay import (
        ScorePolicy,
        SolverConfig,
        execution_conditioned_quotient_problem,
        solve,
    )

    if temporal_mode not in {"issue", "union", "split"}:
        raise ValueError(f"unknown temporal mode {temporal_mode!r}")
    issue_objective = f"{name}.issue.{args.transaction_bytes}B"
    temporal_objective = f"{name}.temporal.{args.transaction_bytes}B"
    if temporal_mode == "union":
        objective = f"{name}.space_time.{args.transaction_bytes}B"
        policy = ScorePolicy(
            "lexicographic", (objective, "runs", "xors")
        )
    elif temporal_mode == "split":
        objective = (issue_objective, temporal_objective)
        policy = ScorePolicy(
            kind="weighted",
            order=objective,
            weights={issue_objective: 1.0, temporal_objective: 1.0},
            tie_order=("runs", "xors"),
        )
    else:
        objective = issue_objective
        policy = ScorePolicy(
            "lexicographic", (objective, "runs", "xors")
        )
    targets = tuple(matrix for matrix in matrices if matrix.target)
    if not targets:
        raise ValueError("Stage 1 requires at least one target matrix")
    target_names = {matrix.name for matrix in targets}
    if set(inner_tile_shapes) != target_names:
        raise ValueError(
            "inner tile shapes must be supplied for every target matrix"
        )
    tile_shapes = {
        matrix.name: tuple(
            dict.fromkeys(
                tuple(shape) for shape in inner_tile_shapes[matrix.name]
            )
        )
        for matrix in targets
    }
    if any(not shapes for shapes in tile_shapes.values()):
        raise ValueError("each target matrix requires an inner tile shape")
    retained_count = max(
        args.candidates,
        1 + max(len(shapes) for shapes in tile_shapes.values()),
    )
    problem = execution_conditioned_quotient_problem(
        matrices,
        events,
        transaction_bytes=args.transaction_bytes,
        temporal_edges=temporal_edges,
        temporal_mode=temporal_mode,
        temporal_objective_name=temporal_objective,
        objective_name=(
            issue_objective if temporal_mode == "split" else objective
        ),
        config=SolverConfig(
            policy=policy,
            tile_shapes=tile_shapes,
            general_tile_shapes={matrix.name: () for matrix in targets},
            include_global_canonical=False,
            enable_linear_inner=False,
            include_column_major_control=False,
            retain_one_candidate_per_tile=True,
            canonical_candidates_per_tile=args.candidates,
            primary_tolerance=(None if temporal_mode == "split" else 0.0),
            per_array_candidates=retained_count,
            joint_candidates=retained_count,
        ),
        name=name,
    )
    return objective, problem, solve(problem)


def array_result_record(matrix, result, component, objective):
    from relay import row_major_layout, weighted_component_region_count

    default = row_major_layout(matrix)
    default_rows = layout_rows(default, matrix)
    default_score = weighted_component_region_count(matrix, default, component)
    candidates = []
    for rank, candidate in enumerate(result.arrays[matrix.name].candidates, start=1):
        layout = candidate.layout
        candidates.append(
            {
                "solver_rank": rank,
                "layout": layout.name,
                **canonical_layout_metadata(layout, matrix),
                "a_rows": list(layout_rows(layout, matrix)),
                "quotient_score": float(candidate.scores[objective]),
                "packing_bound": float(candidate.packing_bounds[objective]),
                "runs": int(candidate.scores["runs"]),
                "xor_count": layout.xor_count,
                "exact": candidate.exact,
            }
        )
    best = candidates[0]
    tile_exponents = result.arrays[matrix.name].tile_hypotheses
    inner_tile_shapes = [
        [1 << exponent for exponent in tile] for tile in tile_exponents
    ]
    fixed_outer_order = list(reversed(matrix.mode_names))
    return {
        "default": {
            "layout": default.name,
            **canonical_layout_metadata(default, matrix),
            "a_rows": list(default_rows),
            "quotient_score": default_score,
        },
        "selected": best,
        "predicted_transaction_reduction": 1.0
        - float(best["quotient_score"]) / default_score,
        "retained_candidates": candidates,
        "search_scope": {
            "grammar": "canonical_inner_tile",
            "tile_policy": "explicit_hypothesis_sweep_v1",
            "inner_tile_shapes": inner_tile_shapes,
            "outer_layout": "row_major_tiles",
            "fixed_outer_order": fixed_outer_order,
        },
        "solver": {
            "realized_candidate_count": result.arrays[
                matrix.name
            ].all_candidate_count,
            "retained_candidate_count": len(candidates),
        },
    }


def measure_variants(launches, validators, args):
    compiled = {label: launch() for label, launch in launches.items()}
    torch.cuda.synchronize()
    for validate in validators.values():
        validate()
    timings = benchmark_layouts(
        launches,
        samples=args.samples,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    return {
        label: {
            "timing": timings[label],
            "compiled_codegen": compiled_codegen_statistics(compiled[label]),
        }
        for label in launches
    }


def require_aligned(label: str, tensor: torch.Tensor, transaction_bytes: int):
    if tensor.data_ptr() % transaction_bytes:
        raise ValueError(f"{label} allocation is not transaction-aligned")


def require_power_of_two_multiple(label: str, value: int, multiple: int):
    if value < multiple or value % multiple:
        raise ValueError(f"{label} must be a multiple of {multiple}")
    if value & (value - 1):
        raise ValueError(f"{label} must be a power of two")
