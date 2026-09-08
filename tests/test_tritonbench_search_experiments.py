from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "triton" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from tritonbench_cases import CASES, OPERATORS
from search_algorithms import _word_count, load_tau_profile, natural_tile_exponents
from relay import Access, MatrixSpec, MemoryEvent


class TritonBenchSearchExperimentTests(unittest.TestCase):
    def test_all_six_named_tau_profiles_load(self):
        path = EXPERIMENTS / "tau-profiles.json"
        profiles = {
            (platform, name): load_tau_profile(platform, path, name)
            for platform in ("tuolumne", "matrix")
            for name in ("expert", "l1_to_l2", "speedup")
        }

        self.assertEqual(len(profiles), 6)
        self.assertTrue(all(profile.tau for profile in profiles.values()))
        self.assertEqual(
            profiles[("matrix", "expert")].device["tau_name"], "expert"
        )
        self.assertEqual(
            profiles[("tuolumne", "l1_to_l2")].fine_component,
            "issue.g64.stream.load.64B",
        )

    def test_broad_portable_panel_has_declared_breadth(self):
        self.assertEqual(len(CASES), 29)
        self.assertEqual(len(OPERATORS), 15)
        self.assertEqual(len({case.case_id for case in CASES}), 29)
        self.assertEqual(
            {case.operator for case in CASES},
            {
                "vector_add", "vector_exp", "low_mem_dropout", "softmax",
                "sum", "layer_norm", "gemm", "bf16xint16_gemm", "int4_gemm",
                "fp8_gemm", "gather_gemv", "template_attention", "jagged_sum",
                "jagged_mean", "jagged_softmax",
            },
        )

    def test_natural_tile_is_power_of_two_bounding_footprint(self):
        matrix = MatrixSpec("x", (64, 128), 4, ("i", "j"), role="read")
        event = MemoryEvent.make(
            "load", "load",
            [Access("x", (i, j)) for i in range(3, 8) for j in range(9, 42)],
        )
        self.assertEqual(natural_tile_exponents(matrix, (event,)), (3, 6))
        self.assertEqual(_word_count((3, 6)), 84)

    def test_legacy_entry_point_remains_limited_to_experiments_one_to_three(self):
        source = (EXPERIMENTS / "run.py").read_text(encoding="utf-8")
        self.assertIn("choices=(1, 2, 3)", source)
        for experiment in (4, 5, 6):
            for platform in ("tuolumne", "matrix"):
                self.assertTrue(
                    (EXPERIMENTS / f"submit-experiment-{experiment}-{platform}.bash").is_file()
                )


class SearchRepairTests(unittest.TestCase):
    def _problem(self, points=((0, 0),), *, shape=(8, 16), region=16):
        from relay import HardwareProfile, SimpleRelayProblem
        from relay.objectives import EdgeFamily, Hyperedge, ScopeKey
        matrix = MatrixSpec("x", shape, 4, ("i", "j"), role="read")
        component = EdgeFamily(ScopeKey("issue", 32, "stream", "load"),
            {"x": (Hyperedge.make(points),)}, normalization_bytes=4).at_scale(region)
        event = MemoryEvent.make("load", "load", [Access("x", (i, j)) for i in range(2) for j in range(2)])
        profile = HardwareProfile(profile_id="test", device={}, byte_scales=(region,),
            tau={component.name: 1.}, fine_component=component.name)
        problem = SimpleRelayProblem((matrix,), (event,), (), (), "canonical")
        return matrix, component, profile, problem

    def test_every_grammar_retains_actual_ordinary_mapping_on_zero_score(self):
        from relay import layout_matrix_rows, row_major_layout
        from search_algorithms import _search
        matrix, component, profile, problem = self._problem()
        for experiment in (4, 5, 6):
            with self.subTest(experiment=experiment):
                layouts, record = _search(problem, profile, experiment=experiment, components=(component,))
                self.assertEqual(layout_matrix_rows(matrix, layouts["x"]),
                                 layout_matrix_rows(matrix, row_major_layout(matrix)))
                self.assertEqual(record["score"], record["baseline_score"])
                self.assertTrue(record["array_searches"][0]["baseline_fallback"])

    def test_tile_search_cannot_deploy_a_worse_tiled_mapping(self):
        from relay import layout_matrix_rows, row_major_layout
        from search_algorithms import _search
        matrix, component, profile, problem = self._problem(points=((0, 0), (0, 1), (0, 2), (0, 3)))
        layouts, record = _search(problem, profile, experiment=5, components=(component,))
        self.assertEqual(layout_matrix_rows(matrix, layouts["x"]), layout_matrix_rows(matrix, row_major_layout(matrix)))
        self.assertLessEqual(record["score"]["hardware_area"], record["baseline_score"]["hardware_area"])

    def test_scalar_inner_search_matches_exhaustive_small_gl(self):
        from itertools import permutations
        from relay import LinearInnerLayout, score_layouts
        from relay.gf2 import rref_basis
        from search_algorithms import _bounded_goc_candidates, _weights
        matrix, component, profile, _ = self._problem(points=((0, 0), (1, 1)), shape=(4, 2), region=8)
        candidates = list(_bounded_goc_candidates(matrix, (component,), _weights((component,), profile), 0))
        def score(layout):
            return score_layouts({"x": matrix}, (component,), {"x": layout}, hardware_profile=profile).hardware_area
        exhaustive = [LinearInnerLayout("exhaustive", "x", matrix.mode_bits, rows, (1, 0))
                      for rows in permutations(range(1, 8), 3) if len(rref_basis(rows)) == 3]
        self.assertAlmostEqual(min(map(score, candidates)), min(map(score, exhaustive)))

    def test_score_invisible_inner_space_is_skipped(self):
        from search_algorithms import _bounded_goc_candidates, _weights
        matrix, component, profile, _ = self._problem(region=64)
        self.assertEqual(list(_bounded_goc_candidates(matrix, (component,), _weights((component,), profile), 0)), [])

    def test_native_vector_bits_cannot_be_swapped_or_mixed_into_high_bits(self):
        from layout_contract import preserves_vector_bits
        native = (1, 2, 4, 8)
        self.assertTrue(preserves_vector_bits((1, 2, 8, 4), native, 2))
        self.assertFalse(preserves_vector_bits((1, 4, 8, 2), native, 2))
        self.assertFalse(preserves_vector_bits((1, 2, 5, 8), native, 2))

    def test_natural_tile_unions_register_slices_and_waves(self):
        matrix = MatrixSpec("x", (64, 64), 4, ("i", "j"))
        events = [MemoryEvent.make(str(wave), "load", [Access("x", (i, j)) for i in range(wave * 16, (wave + 1) * 16) for j in range(32)],
                  metadata={"workgroup": "0", "wave": str(wave), "operation_instance": "0"}) for wave in range(4)]
        self.assertEqual(natural_tile_exponents(matrix, events), (6, 5))

    def test_natural_tile_accounts_for_crossed_alignment_boundaries(self):
        matrix = MatrixSpec("x", (64, 64), 4, ("i", "j"))
        event = MemoryEvent.make("load", "load", [Access("x", (i, j)) for i in range(3, 7) for j in range(20, 52)])
        self.assertEqual(natural_tile_exponents(matrix, (event,)), (3, 6))

    def test_process_statistics_do_not_treat_samples_as_independent_runs(self):
        from experiment_support import process_summary
        records = [{"timings": {label: {"samples_ms": [value] * 21, "median_ms": value}
                    for label, value in (("baseline", base), ("selected", 1), ("identity", base))}}
                   for base in (1., 2., 3.)]
        result = process_summary(records)
        self.assertAlmostEqual(result["speedup"], 6 ** (1 / 3))
        self.assertEqual(result["identity_speedups"], [1., 1., 1.])
        self.assertLess(result["speedup_ci95"][0], 1.)
        self.assertGreater(result["speedup_ci95"][1], 3.)


class NativeBaselineAndProvenanceTests(unittest.TestCase):
    def test_aligned_nonpower_pitch_preserves_region_partition(self):
        from types import SimpleNamespace
        from relay.triton_frontend import _native_row_major_score_equivalent
        allocation = SimpleNamespace(envelope_shape=(4, 16), strides=(12, 1),
                                     element_bytes=4, dense_status="dense")
        self.assertTrue(_native_row_major_score_equivalent(allocation, (4, 8, 16)))
        self.assertFalse(_native_row_major_score_equivalent(allocation, (32,)))
        for region in (4, 8, 16):
            points = [(i, j) for i in range(4) for j in range(12)]
            for left in points:
                for right in points:
                    native_equal = (left[0] * 12 + left[1]) * 4 // region == (right[0] * 12 + right[1]) * 4 // region
                    padded_equal = (left[0] * 16 + left[1]) * 4 // region == (right[0] * 16 + right[1]) * 4 // region
                    self.assertEqual(native_equal, padded_equal)

    def test_misaligned_native_base_is_not_scored_as_an_aligned_envelope(self):
        from types import SimpleNamespace
        from relay.triton_frontend import _native_row_major_score_equivalent
        allocation = SimpleNamespace(envelope_shape=(4, 16), strides=(16, 1),
            element_bytes=4, dense_status="dense", base_pointer=4)
        self.assertFalse(_native_row_major_score_equivalent(allocation, (32, 64)))

    def test_internal_graph_cache_rejects_changed_content(self):
        import tempfile
        from types import MappingProxyType
        from experiment_support import load_graph, save_graph
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.pkl.gz"
            expected = {"config": MappingProxyType({"num_warps": 4}), "events": ()}
            signature = save_graph(path, expected)
            self.assertEqual(load_graph(path, signature), expected)
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_graph(path, signature)

    def test_resume_identity_ignores_elapsed_time_but_includes_physical_choice(self):
        from experiment_support import proposal_key
        proposal = dict.fromkeys(("schema", "experiment", "platform", "operator", "config", "tau_name",
            "source_identity", "capture_identity", "hardware_profile", "tau_profile_hash",
            "selected_config", "output_arguments", "baseline_layouts", "kernel_name"), "fixture")
        proposal.update(runtime_layouts=[], search={"algorithm": "test", "elapsed_seconds": 1.})
        first = proposal_key(proposal)
        proposal["search"]["elapsed_seconds"] = 20.
        self.assertEqual(proposal_key(proposal), first)
        proposal["runtime_layouts"] = [{"argument": 0, "rows": [2, 1]}]
        self.assertNotEqual(proposal_key(proposal), first)

    def test_speedup_fitter_rejects_profiler_duration(self):
        import json
        import tempfile
        from unittest.mock import patch
        from tune_tau import _speedup_groups
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"final_experiment": 1, "case": "gemv", "candidates": [
                {"complete": True, "sampling_origin": "row_major_anchor", "mapping_id": "ordinary",
                 "counters": {"steady_state": {"duration_ns": 100}}}]}))
            with patch("tune_tau._reports", return_value=[path]):
                with self.assertRaisesRegex(ValueError, "profiler duration is not a timing oracle"):
                    _speedup_groups(Path(directory), "tuolumne")

    def test_realization_contract_fails_closed_without_final_assembly(self):
        from layout_contract import realization_rejections
        stats = {"n_regs": 4, "n_spills": 0, "shared_bytes": 0, "load_instruction_count": 1}
        self.assertIn("final device assembly unavailable", realization_rejections(stats, stats))


class StagedWorkflowTests(unittest.TestCase):
    def test_measurement_requires_finished_preflight_and_graph_resume_keeps_complete_report(self):
        import importlib.util
        import json
        import tempfile
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location("suite_test", EXPERIMENTS / "run-search-suite.py")
        suite = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(suite)
        with tempfile.TemporaryDirectory() as directory:
            args = suite.arguments(["--stage", "measure", "--platform", "tuolumne",
                "--case", "vector_add--small", "--suite-root", directory,
                "--experiments", "4", "--tau-names", "expert"])
            path = suite.cell_root(args, 4, "expert") / "report.json"
            suite.write_json(path, {**suite.base_report(args, 4, "expert"), "status": "prepared"})
            with patch.object(suite.RUNNER, "_activate_triton_source"), patch.object(suite.RUNNER, "orchestrate") as measure:
                suite.measure(args)
                measure.assert_not_called()
            self.assertEqual(json.loads(path.read_text())["status"], "incomplete")
            suite.write_json(path, {"status": "complete", "run_hash": "unchanged"})
            before = path.read_bytes()
            suite.stage_status(args, "running", "graph")
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
