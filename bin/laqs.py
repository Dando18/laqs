
from argparse import ArgumentParser

from relay import (
    simple_solve,
    SimpleRelayProblem,
    MatrixSpec,
    MemoryEvent,
    EventSequence,
    ObjectiveComponent,
)

def load_problem(problem_file: str) -> list[list[MatrixSpec], list[MemoryEvent], list[EventSequence], list[ObjectiveComponent]]:
    """ problem_file is a python file that defines three functions
    - get_matrices() -> list[MatrixSpec]
    - get_events_and_sequences() -> list[MemoryEvent]
    - get_objectives() -> list[ObjectiveComponent]

    Call these functions.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("problem_module", problem_file)
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

    print("Constructing objectives...", end="", flush=True)
    objectives = problem_module.get_objectives(problem_config)
    print("\tdone.")

    return matrices, events, sequences, objectives


def main(args):
    
    matrices, events, sequences, objectives = load_problem(args.problem_file)
    
    problem = SimpleRelayProblem(
        matrices=matrices,
        events=events,
        sequences=sequences,
        objectives=objectives,
        grammar="standard"
    )

    simple_solve(problem)


if __name__ == "__main__":
    parser = ArgumentParser(description="Solve a relay problem")
    parser.add_argument("problem_file", help="Path to the relay problem file")
    args = parser.parse_args()

    main(args)