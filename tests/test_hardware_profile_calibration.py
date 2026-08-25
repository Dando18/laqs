from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.hardware_profile_calibration import (
    CalibrationConfig,
    calibrate_report,
    evaluate_weights,
    main,
    prepare_corpus,
)


FINE = "issue.g64.stream.load.64B"
FEATURE_A = "issue.g8.stream.load.64B"
FEATURE_A_ALIAS = "lane_window.t4.stream.load.128B"
FEATURE_B = "simd_window.t4.stream.load.128B"
CONSTANT = "phase.simd.stream.load.128B"


def _region(name: str) -> int:
    return int(name.rsplit(".", 1)[1][:-1])


def _profile() -> dict[str, object]:
    return {
        "profile_id": "synthetic-universal-v1",
        "device": {"target": "test"},
        "byte_scales": [64, 128],
        "fine_component": FINE,
        "tau": {FEATURE_A: 0.25, FEATURE_B: 0.5},
        "kappa": {FEATURE_A: 1.0},
    }


def _group(
    kernel: str,
    size: int,
    layouts: list[tuple[str, float, dict[str, float]]],
    *,
    feature_names: tuple[str, ...] = (
        FINE,
        FEATURE_A,
        FEATURE_A_ALIAS,
        FEATURE_B,
        CONSTANT,
    ),
) -> dict[str, object]:
    objectives = []
    for name in feature_names:
        region = _region(name)
        objectives.append(
            {
                "name": name,
                "region_bytes": region,
                "edge_family": name.rsplit(".", 1)[0],
                "normalization_bytes": 64.0,
                "provenance": "universal-v1",
            }
        )

    results = []
    for layout_name, runtime, values in layouts:
        components = []
        for name in feature_names:
            value = values.get(name, 0.0)
            region = _region(name)
            components.append(
                {
                    "name": name,
                    "region_bytes": region,
                    "raw_region_count": 1.0 + value * 64.0 / region,
                    "packing_lower_bound": 1.0,
                    "normalized_excess": value,
                    "normalization_bytes": 64.0,
                    "excess_footprint": value,
                }
            )
        results.append(
            {
                "name": layout_name,
                "timing": {"median_ms": runtime},
                "score": {
                    "components": components,
                    "aggregates": {"hardware_peak": 0.0},
                    "codegen": {"runs": 1, "xors": 0},
                },
            }
        )
    return {
        "kernel": kernel,
        "display_name": kernel.upper(),
        "matrix_size": size,
        "block": [64, 1, 1],
        "hardware_profile": _profile(),
        "fine_component": FINE,
        "peak_tolerances": {FEATURE_A: 1.0},
        "objectives": objectives,
        "results": results,
    }


def _report(*groups: dict[str, object]) -> dict[str, object]:
    return {
        "experiment": "multi-kernel-layout-ranking",
        "complete": True,
        "runs": list(groups),
    }


def _calibration_group(kernel: str = "atax", size: int = 256) -> dict[str, object]:
    return _group(
        kernel,
        size,
        [
            (
                "fast",
                1.0,
                {
                    FEATURE_A: 0.0,
                    FEATURE_A_ALIAS: 0.0,
                    FEATURE_B: 1.0,
                    CONSTANT: 5.0,
                },
            ),
            (
                "slow",
                1.5,
                {
                    FEATURE_A: 1.0,
                    FEATURE_A_ALIAS: 3.0,
                    FEATURE_B: 0.0,
                    CONSTANT: 5.0,
                },
            ),
            (
                "junk",
                2.0,
                {
                    FEATURE_A: 2.0,
                    FEATURE_A_ALIAS: 6.0,
                    FEATURE_B: 2.0,
                    CONSTANT: 5.0,
                },
            ),
        ],
    )


class FeaturePreparationTests(unittest.TestCase):
    def test_columns_are_normalized_deduplicated_and_zero_filled(self) -> None:
        second = _group(
            "gesummv",
            512,
            [
                ("row", 1.0, {FEATURE_B: 0.0}),
                ("column", 1.1, {FEATURE_B: 2.0}),
            ],
            feature_names=(FINE, FEATURE_B),
        )
        corpus = prepare_corpus(_report(_calibration_group(), second))

        self.assertEqual(corpus.input_feature_count, 5)
        self.assertEqual(corpus.informative_feature_count, 3)
        self.assertEqual(
            set(corpus.dropped_groupwise_constant), {FINE, CONSTANT}
        )
        self.assertEqual(
            [column.name for column in corpus.columns], [FEATURE_A, FEATURE_B]
        )
        first = corpus.columns[0]
        self.assertEqual(first.scale, 2.0)
        self.assertEqual(
            first.aliases,
            ((FEATURE_A, 1.0), (FEATURE_A_ALIAS, 3.0)),
        )
        self.assertEqual(corpus.groups[1].layouts[0].features[0], 0.0)

    def test_feature_can_be_reconstructed_from_universal_q_and_lb(self) -> None:
        report = _report(_calibration_group())
        component = report["runs"][0]["results"][1]["score"]["components"][1]
        del component["excess_footprint"]
        component["normalization_bytes"] = None

        corpus = prepare_corpus(report)

        self.assertEqual(corpus.columns[0].name, FEATURE_A)
        self.assertEqual(corpus.columns[0].scale, 2.0)

    def test_global_tau_scaling_does_not_change_the_frontier(self) -> None:
        corpus = prepare_corpus(_report(_calibration_group()))

        first = evaluate_weights(corpus, (1.0, 0.0))
        second = evaluate_weights(corpus, (17.0, 0.0))

        self.assertEqual(first, second)
        self.assertEqual(first.instances[0].frontier_names, ("fast",))

    def test_filtering_uses_whole_kernel_size_groups(self) -> None:
        report = _report(
            _calibration_group(),
            _calibration_group("gesummv", 512),
        )

        corpus = prepare_corpus(report, kernels=("gesummv",), sizes=(512,))

        self.assertEqual(len(corpus.groups), 1)
        self.assertEqual(corpus.groups[0].kernel, "gesummv")


class CalibrationTests(unittest.TestCase):
    def test_search_is_reproducible_and_greedily_sparse(self) -> None:
        config = CalibrationConfig(
            seed=19,
            iterations=20,
            min_candidates=1,
            max_candidates=1,
            max_regret=0.0,
            search_support=2,
        )
        report = _report(_calibration_group())

        first = calibrate_report(report, config=config)
        second = calibrate_report(report, config=config)

        self.assertEqual(first, second)
        self.assertEqual(
            first["recommendation"]["tau"],
            {FEATURE_A: 0.25, FEATURE_A_ALIAS: 1.0 / 12.0},
        )
        self.assertEqual(first["recommendation"]["independent_support_size"], 1)
        self.assertEqual(first["recommendation"]["cell_support_size"], 2)
        self.assertEqual(first["recommendation"]["status"], "passing")
        self.assertTrue(first["metrics"]["constraints_met"])
        self.assertEqual(first["metrics"]["instances"][0]["frontier_names"], ["fast"])
        self.assertTrue(first["full_basis_diagnostic"]["all_instances_covered"])

    def test_cli_filters_and_writes_json(self) -> None:
        report = _report(
            _calibration_group(),
            _calibration_group("gesummv", 512),
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "ranking.json"
            output_path = Path(directory) / "tau.json"
            input_path.write_text(json.dumps(report))

            status = main(
                [
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--kernel",
                    "atax",
                    "--size",
                    "256",
                    "--iterations",
                    "0",
                    "--min-candidates",
                    "1",
                    "--max-candidates",
                    "1",
                    "--max-regret",
                    "0",
                ]
            )

            self.assertEqual(status, 0)
            result = json.loads(output_path.read_text())
            self.assertEqual(result["corpus"]["instance_count"], 1)
            self.assertEqual(
                result["recommendation"]["tau"],
                {FEATURE_A: 0.25, FEATURE_A_ALIAS: 1.0 / 12.0},
            )

    def test_dominated_winner_is_reported_as_a_basis_failure(self) -> None:
        group = _group(
            "atax",
            256,
            [
                ("measured-winner", 1.0, {FEATURE_A: 2.0, FEATURE_B: 2.0}),
                ("analytical-winner", 2.0, {FEATURE_A: 0.0, FEATURE_B: 0.0}),
            ],
            feature_names=(FINE, FEATURE_A, FEATURE_B),
        )

        result = calibrate_report(
            _report(group),
            config=CalibrationConfig(
                iterations=0,
                min_candidates=1,
                max_candidates=1,
                max_regret=0.0,
            ),
        )

        diagnostic = result["full_basis_diagnostic"]
        self.assertFalse(diagnostic["all_instances_covered"])
        self.assertEqual(
            diagnostic["instances"][0]["dominated_winner_certificates"],
            [{"winner": "measured-winner", "dominator": "analytical-winner"}],
        )
        self.assertEqual(result["recommendation"]["status"], "best-effort")
        self.assertTrue(result["warnings"])

    def test_search_support_is_a_hard_independent_column_limit(self) -> None:
        first = _group(
            "atax",
            256,
            [
                ("fast", 1.0, {FEATURE_A: 0.0, FEATURE_B: 1.0}),
                ("slow", 2.0, {FEATURE_A: 1.0, FEATURE_B: 0.0}),
            ],
            feature_names=(FINE, FEATURE_A, FEATURE_B),
        )
        second = _group(
            "gesummv",
            256,
            [
                ("fast", 1.0, {FEATURE_A: 1.0, FEATURE_B: 0.0}),
                ("slow", 2.0, {FEATURE_A: 0.0, FEATURE_B: 1.0}),
            ],
            feature_names=(FINE, FEATURE_A, FEATURE_B),
        )

        result = calibrate_report(
            _report(first, second),
            config=CalibrationConfig(
                iterations=10,
                min_candidates=1,
                max_candidates=2,
                max_regret=0.0,
                search_support=1,
            ),
        )

        self.assertEqual(result["recommendation"]["independent_support_size"], 1)
        self.assertEqual(result["recommendation"]["status"], "best-effort")


class ValidationTests(unittest.TestCase):
    def test_incomplete_legacy_and_corrupt_reports_are_rejected(self) -> None:
        incomplete = _report(_calibration_group())
        incomplete["complete"] = False
        with self.assertRaisesRegex(ValueError, "completed"):
            prepare_corpus(incomplete)

        legacy = _report(_calibration_group())
        legacy["runs"][0]["objectives"][0]["provenance"] = "grounded"
        with self.assertRaisesRegex(ValueError, "reuse-timings"):
            prepare_corpus(legacy)

        corrupt = _report(_calibration_group())
        corrupt["runs"][0]["results"][0]["score"]["components"].pop()
        with self.assertRaisesRegex(ValueError, "component sets|every declared"):
            prepare_corpus(corrupt)

        invalid_runtime = _report(_calibration_group())
        invalid_runtime["runs"][0]["results"][0]["timing"]["median_ms"] = 0.0
        with self.assertRaisesRegex(ValueError, "positive"):
            prepare_corpus(invalid_runtime)

        stale = _report(_calibration_group())
        stale["runs"][0]["results"][0]["score"]["components"][1][
            "normalization_bytes"
        ] = 32.0
        with self.assertRaisesRegex(ValueError, "disagrees"):
            prepare_corpus(stale)

    def test_inputs_are_not_mutated(self) -> None:
        report = _report(_calibration_group())
        original = copy.deepcopy(report)

        calibrate_report(
            report,
            config=CalibrationConfig(
                iterations=0,
                min_candidates=1,
                max_candidates=2,
                max_regret=0.5,
            ),
        )

        self.assertEqual(report, original)


if __name__ == "__main__":
    unittest.main()
