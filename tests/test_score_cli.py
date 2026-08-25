from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
from types import ModuleType
import unittest

from relay import MI300A_V1


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPOSITORY_ROOT / "bin" / "score_layout.py"
PROBLEM_PATH = Path(__file__).parent / "fixtures" / "score_problem.py"


def load_score_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("relay_test_score_cli", CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load score CLI from {CLI_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScoreLayoutCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = load_score_cli()

    def run_json(self, *arguments: str) -> dict[str, object]:
        output = StringIO()
        with redirect_stdout(output):
            status = self.cli.run([str(PROBLEM_PATH), *arguments, "--json"])
        self.assertEqual(status, 0)
        return json.loads(output.getvalue())

    def test_human_output_labels_the_selected_score(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            status = self.cli.run(
                [
                    str(PROBLEM_PATH),
                    "--layout",
                    "A=row-major",
                    "--score-mode",
                    "hardware-peak",
                ]
            )

        self.assertEqual(status, 0)
        self.assertIn(
            "Selected score (hardware-peak): 3",
            output.getvalue(),
        )
        self.assertIn("Address codegen costs", output.getvalue())
        self.assertIn("total                                             2     0", output.getvalue())

    def test_json_reports_selected_mode_value_and_layout(self) -> None:
        report = self.run_json(
            "--layout",
            "A=column-major",
            "--score-mode",
            "weighted-region-count",
        )

        self.assertEqual(report["selected_score_mode"], "weighted-region-count")
        self.assertEqual(
            report["selected_score"],
            report["aggregates"]["weighted_region_count"],
        )
        self.assertEqual(report["layouts"], {"A": "iijj"})
        self.assertEqual(
            report["hardware_profile"]["profile_id"],
            MI300A_V1.profile_id,
        )
        self.assertIn(
            report["hardware_profile"]["fine_component"],
            report["component_weights"],
        )
        self.assertEqual(
            {
                name
                for name, weight in report["component_weights"].items()
                if weight > 0
            },
            set(MI300A_V1.tau),
        )
        self.assertEqual(report["component_weight_overrides"], {})
        self.assertEqual(
            report["codegen"],
            {
                "runs": 2,
                "xors": 0,
                "arrays": [
                    {
                        "name": "A",
                        "grammar": "canonical",
                        "runs": 2,
                        "xors": 0,
                    }
                ],
            },
        )

    def test_row_column_and_word_layout_specs(self) -> None:
        cases = (
            ("row-major", "jjii"),
            ("column-major", "iijj"),
            ("word:jiji", "jiji"),
        )
        for specification, expected_word in cases:
            with self.subTest(specification=specification):
                report = self.run_json(
                    "--layout",
                    f"A={specification}",
                    "--score-mode",
                    "weighted-region-count",
                )
                self.assertEqual(report["layouts"], {"A": expected_word})

    def test_problem_options_are_forwarded_to_build_config(self) -> None:
        report = self.run_json(
            "--problem-option",
            "problem_size=8",
            "--layout",
            "A=jjjiii",
            "--score-mode",
            "weighted-region-count",
        )

        # Three bits per mode are only valid if problem_size=8 reached the
        # fixture's build_config function; its default problem size is four.
        self.assertEqual(report["layouts"], {"A": "jjjiii"})
        issue = next(
            component
            for component in report["components"]
            if component["name"] == "issue.g8.stream.load.16B"
        )
        self.assertEqual(issue["arrays"][0]["raw_region_count"], 8.0)

    def test_command_line_weight_overrides_profile_default(self) -> None:
        report = self.run_json(
            "--layout",
            "A=column-major",
            "--score-mode",
            "weighted-region-count",
            "--component-weight",
            "issue.g8.stream.load.16B=2",
        )

        self.assertEqual(
            report["component_weights"]["issue.g8.stream.load.16B"],
            2.0,
        )
        self.assertEqual(
            report["component_weight_overrides"],
            {"issue.g8.stream.load.16B": 2.0},
        )
        self.assertEqual(
            report["hardware_profile"]["tau"]["issue.g8.stream.load.16B"],
            2.0,
        )
        self.assertTrue(
            report["hardware_profile"]["profile_id"].endswith(
                "+cli-tau-overrides"
            )
        )

    def test_missing_layout_is_an_argparse_error_without_stdout(self) -> None:
        output = StringIO()
        errors = StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaises(SystemExit) as caught:
                self.cli.run([str(PROBLEM_PATH)])

        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn(
            "the following arguments are required: --layout",
            errors.getvalue(),
        )

    def test_invalid_layout_is_an_argparse_error_without_stdout(self) -> None:
        output = StringIO()
        errors = StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaises(SystemExit) as caught:
                self.cli.run([str(PROBLEM_PATH), "--layout", "A=ix"])

        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("unknown mode 'x'", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
