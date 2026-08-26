from __future__ import annotations

import argparse
from math import comb
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.layout_ranking import KERNEL_SPECS
from experiments.scoring_results import (
    _configuration,
    _configuration_id,
    _feature_dominators,
    _append_raw,
    _initialize_raw,
    _placement_signature,
    build_group_plan,
    canonical_words,
    expand_canonical_descriptor,
    import_seed_timings,
    import_raw_timings,
    load_raw,
    parse_arguments,
    run,
    standard_words,
    summary_records,
)
from relay import MatrixSpec, canonical_layout_from_word, get_hardware_profile


def fake_timing(median_ms: float) -> dict[str, object]:
    return {
        "device": "AMD Radeon Graphics",
        "median_ms": median_ms,
        "mean_ms": median_ms,
        "min_ms": median_ms,
        "sd_ms": 0.0,
        "gflops": 1.0,
        "samples_ms": [median_ms] * 5,
    }


class CanonicalWordTests(unittest.TestCase):
    def test_full_word_enumeration_has_exact_grammar_size(self) -> None:
        words = tuple(canonical_words(8))

        self.assertEqual(len(words), comb(6, 3))
        self.assertEqual(len(set(words)), len(words))
        self.assertTrue(
            all(word.count("i") == word.count("j") == 3 for word in words)
        )

    def test_tiled_descriptor_expands_to_the_same_full_canonical_layout(self) -> None:
        self.assertEqual(expand_canonical_descriptor("ji", 4), "jiji")
        self.assertEqual(expand_canonical_descriptor("jjii", 4), "jjii")
        with self.assertRaisesRegex(ValueError, "canonical descriptor"):
            expand_canonical_descriptor("jjj", 4)

    def test_standard_words_match_the_four_form_grammar(self) -> None:
        self.assertEqual(len(standard_words(512)), 146)
        self.assertEqual(len(standard_words(1024)), 182)
        self.assertTrue(set(standard_words(512)) < set(canonical_words(512)))


class ScoringResultPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        _, self.args = parse_arguments(
            ["--kernel", "atax", "--size", "4", "--prepare-only"]
        )

    def test_group_plan_records_exact_frontier_and_selection_sets(self) -> None:
        group = build_group_plan(KERNEL_SPECS["atax"], 4, self.args)

        self.assertEqual(group["total_layouts"], 6)
        self.assertEqual(group["expected_total_layouts"], 6)
        self.assertTrue(group["frontier"])
        self.assertEqual(
            group["locality_frontier_objectives"],
            ["fine_region_count", "hardware_peak", "hardware_area"],
        )
        self.assertEqual(len(group["locality_frontier"]), group["total_layouts"])
        self.assertEqual(
            set(group["frontier_objectives"]),
            {
                "fine_region_count",
                "hardware_peak",
                "hardware_area",
            },
        )
        self.assertEqual(group["frontier"], group["locality_frontier"])
        self.assertEqual(group["placement_reranker"]["budgets"], [1, 3, 5, 10])
        self.assertLessEqual(
            group["placement_reranker"]["pool_count"],
            2 * len(group["locality_frontier"]),
        )
        mechanisms = {
            mechanism["name"]: mechanism
            for mechanism in group["selection_mechanisms"]
        }
        self.assertEqual(len(mechanisms["lowest_hardware_area"]["members"]), 1)
        self.assertEqual(len(mechanisms["top5_hardware_area"]["members"]), 5)
        self.assertIn("placement_rerank_at_5", mechanisms)
        for member in group["frontier"]:
            self.assertEqual(member["layouts"], {"A": member["word"]})
            self.assertIn("codegen_runs", member["score"])
            self.assertNotIn("codegen_runs", group["frontier_objectives"])

    def test_complete_summary_uses_exhaustive_oracle_for_regret(self) -> None:
        group = build_group_plan(KERNEL_SPECS["atax"], 4, self.args)
        configuration = _configuration(self.args, ["atax"], [4])
        plan = {
            "configuration_id": _configuration_id(configuration),
            "configuration": configuration,
            "groups": [group],
        }
        records = {}
        for index, word in enumerate(canonical_words(4), 1):
            records[("atax", 4, word)] = {
                "word": word,
                "timing": fake_timing(float(index)),
            }

        summary = summary_records(plan, records)[0]

        self.assertTrue(summary["complete"])
        self.assertEqual(summary["layout_count"], 6)
        self.assertEqual(summary["timed_layout_count"], 6)
        self.assertEqual(summary["oracle"]["best_time_ms"], 1.0)
        self.assertEqual(len(summary["oracle"]["best_layouts"]), 1)
        self.assertEqual(len(summary["oracle"]["top_layouts"]), 5)
        frontier = summary["frontier"]
        self.assertTrue(frontier["complete"])
        self.assertAlmostEqual(
            frontier["regret"], frontier["best_time_ms"] - 1.0
        )
        self.assertIn("5", summary["placement_reranker"]["regret_at_k"])

    def test_prepare_cli_writes_plan_checkpoint_and_summary(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            status = run(
                [
                    "--kernel",
                    "atax",
                    "--size",
                    "4",
                    "--prepare-only",
                    "--plan",
                    str(root / "plan.json"),
                    "--raw-output",
                    str(root / "raw.jsonl"),
                    "--output",
                    str(root / "summary.jsonl"),
                ]
            )

            self.assertEqual(status, 0)
            self.assertTrue((root / "plan.json").is_file())
            self.assertTrue((root / "raw.jsonl").is_file())
            self.assertTrue((root / "summary.jsonl").is_file())

    def test_standard_prepare_cli_records_the_selected_grammar(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            status = run(
                [
                    "--kernel",
                    "atax",
                    "--size",
                    "8",
                    "--grammar",
                    "standard",
                    "--prepare-only",
                    "--plan",
                    str(root / "plan.json"),
                    "--raw-output",
                    str(root / "raw.jsonl"),
                    "--output",
                    str(root / "summary.jsonl"),
                ]
            )

            self.assertEqual(status, 0)
            import json

            summary = json.loads((root / "summary.jsonl").read_text())
            self.assertEqual(summary["grammar"], "G_S")
            self.assertEqual(summary["layout_count"], len(standard_words(8)))

    def test_oracle_audit_dumps_primitives_and_checks_dominance(self) -> None:
        _, args = parse_arguments(
            [
                "--kernel",
                "atax",
                "--size",
                "4",
                "--dump-oracle-components",
                "--check-oracle-feature-dominance",
                "--prepare-only",
            ]
        )
        group = build_group_plan(KERNEL_SPECS["atax"], 4, args)
        configuration = _configuration(args, ["atax"], [4])
        records = {
            ("atax", 4, word): {
                "word": word,
                "timing": fake_timing(float(index)),
            }
            for index, word in enumerate(canonical_words(4), 1)
        }

        summary = summary_records(
            {
                "configuration_id": _configuration_id(configuration),
                "configuration": configuration,
                "groups": [group],
            },
            records,
        )[0]

        audit = summary["oracle_feature_audit"]
        self.assertGreater(audit["primitive_count"], 0)
        self.assertTrue(audit["records"])
        self.assertIn("componentwise_dominated", audit["oracles"][0])

    def test_feature_dominance_ignores_roundoff_only_differences(self) -> None:
        features = {
            "oracle": {
                "q_fine": 2.0,
                "locality": {
                    "x": {"excess_footprint": 2.0},
                },
                "placement": [],
            },
            "candidate": {
                "q_fine": 1.0,
                "locality": {
                    "x": {"excess_footprint": 2.0 + 1e-13},
                },
                "placement": [],
            },
        }

        _coordinates, dominators = _feature_dominators(features, "oracle")

        self.assertEqual(dominators, ["candidate"])

    def test_placement_cache_includes_the_transaction_index_projection(self) -> None:
        matrix = MatrixSpec("A", (1024, 1024), 8, ("i", "j"))
        resource_maps = get_hardware_profile("mi300a").resource_maps
        color_only_groups = {}
        for word in standard_words(1024):
            layout = canonical_layout_from_word(matrix, word)
            color_only = tuple(
                layout.matrix_rows()[bit] for bit in (9, 10, 11)
            )
            color_only_groups.setdefault(color_only, set()).add(
                _placement_signature(matrix, layout, resource_maps)
            )

        self.assertTrue(
            any(
                len(transaction_signatures) > 1
                for transaction_signatures in color_only_groups.values()
            )
        )

    def test_flag_fiber_plan_records_identity_realizations_at_small_width(self) -> None:
        _, args = parse_arguments(
            [
                "--kernel",
                "atax",
                "--size",
                "8",
                "--grammar",
                "standard",
                "--fiber-max-xors",
                "1",
                "--prepare-only",
            ]
        )

        group = build_group_plan(KERNEL_SPECS["atax"], 8, args)

        self.assertTrue(group["flag_fiber"]["enabled"])
        self.assertEqual(group["flag_fiber"]["max_xors"], 1)
        self.assertEqual(group["flag_fiber"]["destination_bits"], [])
        self.assertEqual(
            group["flag_fiber"]["evaluated_materializations"],
            len(standard_words(8)),
        )
        self.assertTrue(
            all("flag_word" in member for member in group["frontier"])
        )


class SeedTimingTests(unittest.TestCase):
    def test_identity_timings_can_seed_a_fiber_checkpoint(self) -> None:
        _, source_args = parse_arguments(
            ["--kernel", "atax", "--size", "8", "--grammar", "standard"]
        )
        _, target_args = parse_arguments(
            [
                "--kernel",
                "atax",
                "--size",
                "8",
                "--grammar",
                "standard",
                "--fiber-max-xors",
                "1",
            ]
        )
        source_configuration = _configuration(source_args, ["atax"], [8])
        target_configuration = _configuration(target_args, ["atax"], [8])
        word = standard_words(8)[0]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.raw.jsonl"
            target_path = root / "target.raw.jsonl"
            _initialize_raw(source_path, source_configuration)
            _append_raw(
                source_path,
                {
                    "record_type": "timing",
                    "configuration_id": _configuration_id(source_configuration),
                    "kernel": "atax",
                    "matrix_size": 8,
                    "word": word,
                    "timing": fake_timing(1.0),
                },
            )
            _initialize_raw(target_path, target_configuration)
            records = {}

            imported = import_raw_timings(
                source_path,
                target_path,
                target_configuration,
                records,
            )

            self.assertEqual(imported, 1)
            loaded = load_raw(target_path, target_configuration)
            self.assertEqual(loaded[("atax", 8, word)]["timing"], fake_timing(1.0))

    def test_only_exact_full_word_layout_ranking_timings_are_seeded(self) -> None:
        _, args = parse_arguments(
            ["--kernel", "atax", "--size", "4", "--prepare-only"]
        )
        configuration = _configuration(args, ["atax"], [4])
        source = {
            "experiment": "multi-kernel-layout-ranking",
            "configuration": {
                "samples": 5,
                "iterations": 3,
                "warmup": 2,
                "device": 0,
                "compiler": "/opt/rocm-7.0.2/bin/hipcc",
                "arch": "gfx942",
                "one_dimensional_block_size": 128,
                "two_dimensional_block": [32, 32, 1],
            },
            "runs": [
                {
                    "kernel": "atax",
                    "matrix_size": 4,
                    "results": [
                        {
                            "name": "short_tile",
                            "word": "ji",
                            "timing": fake_timing(2.0),
                        },
                        {
                            "name": "full_word",
                            "word": "jiji",
                            "timing": fake_timing(3.0),
                        }
                    ],
                }
            ],
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            raw_path = root / "raw.jsonl"
            import json

            source_path.write_text(json.dumps(source))
            _initialize_raw(raw_path, configuration)
            records = {}

            imported = import_seed_timings(
                source_path, raw_path, configuration, records
            )
            loaded = load_raw(raw_path, configuration)

        self.assertEqual(imported, 1)
        self.assertIn(("atax", 4, "jiji"), loaded)
        self.assertEqual(
            loaded[("atax", 4, "jiji")]["source"]["descriptor"], "jiji"
        )

    def test_incomplete_final_checkpoint_record_is_recovered(self) -> None:
        _, args = parse_arguments(
            ["--kernel", "atax", "--size", "4", "--prepare-only"]
        )
        configuration = _configuration(args, ["atax"], [4])
        with TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.jsonl"
            _initialize_raw(raw_path, configuration)
            valid_size = raw_path.stat().st_size
            with raw_path.open("ab") as stream:
                stream.write(b'{"record_type":"tim')

            records = load_raw(raw_path, configuration)

            self.assertEqual(records, {})
            self.assertEqual(raw_path.stat().st_size, valid_size)


if __name__ == "__main__":
    unittest.main()
