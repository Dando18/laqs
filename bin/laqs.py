#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from relay import (
    HARDWARE_PROFILES,
    EventSequence,
    HardwareProfile,
    MatrixSpec,
    MemoryEvent,
    SimpleRelayProblem,
    UniversalScopeObjectives,
    get_hardware_profile,
    simple_solve,
)


def load_problem(
    problem_file: str,
    hardware_profile: HardwareProfile,
) -> tuple[
    tuple[MatrixSpec, ...],
    tuple[MemoryEvent, ...],
    tuple[EventSequence, ...],
    tuple[UniversalScopeObjectives, ...],
]:
    """Load matrix, event, and schedule facts from a kernel module."""

    import importlib.util

    spec = importlib.util.spec_from_file_location("problem_module", problem_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load problem module from {problem_file}")
    problem_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(problem_module)

    problem_config = problem_module.build_config(
        problem_size=2048,
        # TODO later also parameterize this by block size, etc.
    )

    print("Gathering matrix metadata...", end="", flush=True)
    matrices = problem_module.get_matrices(problem_config)
    print("\tdone.")

    print("Constructing memory events and sequences...", end="", flush=True)
    events, sequences = problem_module.get_events_and_sequences(problem_config)
    print("\tdone.")

    objectives = (UniversalScopeObjectives(hardware_profile.byte_scales),)

    return tuple(matrices), tuple(events), tuple(sequences), objectives


def main(args: Namespace) -> None:
    hardware_profile = get_hardware_profile(args.hardware_profile)
    matrices, events, sequences, objectives = load_problem(
        args.problem_file, hardware_profile
    )

    problem = SimpleRelayProblem(
        matrices=matrices,
        events=events,
        sequences=sequences,
        objectives=objectives,
        grammar="standard",
        hardware_profile=hardware_profile,
        fine_component=hardware_profile.fine_component,
    )

    simple_solve(problem)


if __name__ == "__main__":
    parser = ArgumentParser(description="Solve a relay problem")
    parser.add_argument("problem_file", help="Path to the relay problem file")
    parser.add_argument(
        "--hardware-profile",
        choices=tuple(HARDWARE_PROFILES),
        default="mi300a",
        help="global hardware response used for search (default: %(default)s)",
    )
    args = parser.parse_args()

    main(args)
