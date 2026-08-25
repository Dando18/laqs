from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch

from relay import MI300A_V1, UniversalScopeObjectives


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPOSITORY_ROOT / "bin" / "laqs.py"


def load_laqs_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("relay_test_laqs_cli", CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load LAQS CLI from {CLI_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LaqsCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = load_laqs_cli()

    def test_main_attaches_selected_profile_to_simple_problem(self) -> None:
        objectives = (UniversalScopeObjectives(MI300A_V1.byte_scales),)
        with (
            patch.object(
                self.cli,
                "load_problem",
                return_value=((), (), (), objectives),
            ),
            patch.object(self.cli, "simple_solve") as solve,
        ):
            self.cli.main(
                Namespace(
                    problem_file="unused.py",
                    hardware_profile="mi300a",
                )
            )

        problem = solve.call_args.args[0]
        self.assertEqual(problem.objectives, objectives)
        self.assertIs(problem.hardware_profile, MI300A_V1)
        self.assertEqual(problem.fine_component, MI300A_V1.fine_component)


if __name__ == "__main__":
    unittest.main()
