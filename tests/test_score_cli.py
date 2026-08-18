from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
from types import ModuleType
import unittest


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
                    "peak-normalized-excess",
                ]
            )

        self.assertEqual(status, 0)
        self.assertIn(
            "Selected score (peak-normalized-excess): 3",
            output.getvalue(),
        )

    def test_json_reports_selected_mode_value_and_layout(self) -> None:
        report = self.run_json(
            "--layout",
            "A=column-major",
            "--score-mode",
            "weighted-region-count",
        )

        self.assertEqual(report["selected_score_mode"], "weighted-region-count")
        self.assertEqual(report["selected_score"], 1.0)
        self.assertEqual(report["layouts"], {"A": "iijj"})

    def test_row_column_and_word_layout_specs(self) -> None:
        cases = (
            ("row-major", "jjii", 4.0),
            ("column-major", "iijj", 1.0),
            ("word:jiji", "jiji", 2.0),
        )
        for specification, expected_word, expected_score in cases:
            with self.subTest(specification=specification):
                report = self.run_json(
                    "--layout",
                    f"A={specification}",
                    "--score-mode",
                    "weighted-region-count",
                )
                self.assertEqual(report["layouts"], {"A": expected_word})
                self.assertEqual(report["selected_score"], expected_score)

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
        self.assertEqual(report["selected_score"], 8.0)

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
