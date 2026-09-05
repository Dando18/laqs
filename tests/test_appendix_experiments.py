from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "triton" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from appendix_common import perturbed_profiles, timed_repetitions
from relay import HardwareProfile


class AppendixExperimentTests(unittest.TestCase):
    def profile(self, tau=None):
        return HardwareProfile(
            profile_id="test-profile",
            device={"platform": "test"},
            byte_scales=(32, 64),
            fine_component="issue.g32.stream.load.32B",
            tau=tau
            or {
                "issue.g32.stream.load.32B": 0.75,
                "lane_window.t4.array.load.64B": 0.25,
            },
        )

    def test_tau_perturbations_are_bounded_deterministic_and_global(self):
        first = perturbed_profiles(self.profile(), (0.1, 0.5), 3, 17)
        second = perturbed_profiles(self.profile(), (0.1, 0.5), 3, 17)

        self.assertEqual(len(first), 7)
        self.assertEqual(
            [metadata for metadata, _ in first],
            [metadata for metadata, _ in second],
        )
        self.assertEqual(first[0][0]["trial_id"], "nominal")
        for metadata, profile in first[1:]:
            magnitude = metadata["magnitude"]
            for name, factor in metadata["factors"].items():
                self.assertGreaterEqual(factor, 1.0 - magnitude)
                self.assertLessEqual(factor, 1.0 + magnitude)
                self.assertAlmostEqual(
                    profile.tau[name], self.profile().tau[name] * factor
                )

    def test_single_active_tau_still_produces_valid_trials(self):
        profile = self.profile({"issue.g32.stream.load.32B": 1.0})
        trials = perturbed_profiles(profile, (0.25,), 2, 0)

        self.assertEqual(len(trials), 3)
        self.assertTrue(
            all(
                tuple(perturbed.tau) == ("issue.g32.stream.load.32B",)
                for _, perturbed in trials
            )
        )

    def test_phase_timing_retains_every_sample_and_result(self):
        calls = iter((3, 4, 5))
        result, timing = timed_repetitions(lambda: next(calls), 3)

        self.assertEqual(result, 5)
        self.assertEqual(timing["repeats"], 3)
        self.assertEqual(len(timing["samples_seconds"]), 3)
        self.assertGreaterEqual(timing["median_seconds"], 0.0)

    def test_appendix_entry_points_do_not_replace_active_experiment_runners(self):
        legacy = (EXPERIMENTS / "run.py").read_text(encoding="utf-8")
        search = (EXPERIMENTS / "run-search.py").read_text(encoding="utf-8")

        self.assertIn("choices=(1, 2, 3)", legacy)
        self.assertIn("choices=(4, 5, 6)", search)
        for experiment in (10, 12):
            for platform in ("tuolumne", "matrix"):
                self.assertTrue(
                    (
                        EXPERIMENTS
                        / f"submit-experiment-{experiment}-{platform}.bash"
                    ).is_file()
                )


if __name__ == "__main__":
    unittest.main()
