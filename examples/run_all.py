from __future__ import annotations

from examples.gesummv_multi import build_problem as gesummv_problem
from examples.jacobi_multi import build_problem as jacobi_problem
from relay import print_report, solve


def main() -> None:
    # Each builder returns only a declarative RelayProblem: logical arrays, a
    # compressed memory-event trace, access-scope objectives, and a layout search
    # space.  solve() is the point where those declarations become objective
    # hyperedges and candidate physical layouts; no GPU kernels are launched.
    for builder in (gesummv_problem, jacobi_problem):
        result = solve(builder())
        # Limit only terminal presentation here.  The builders' SolverConfig
        # limits control how many candidates are actually searched and retained.
        print_report(result, max_candidates=5, show_layouts=1, show_joint=5)
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
