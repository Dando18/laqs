from __future__ import annotations

import json
import math
import unittest

from relay.hardware import HardwareProfile
from relay.objectives import ObjectiveComponent


def component(family: str, scale: int) -> ObjectiveComponent:
    return ObjectiveComponent(
        name=f"{family}.{scale}B",
        region_bytes=scale,
        edges_by_array={},
        edge_family=family,
        normalization_bytes=1.0,
    )


class HardwareProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.issue64 = "issue.g64.stream.load.64B"
        self.issue128 = "issue.g64.stream.load.128B"
        self.window128 = "lane_window.t8.stream.load.128B"
        self.profile = HardwareProfile(
            profile_id="test-gpu-v1",
            device={
                "vendor": "Example",
                "model": "GPU",
                "execution": {"simd_width": 64},
                "tags": ["lab", "prototype"],
            },
            byte_scales=(64, 128, 256),
            fine_component=self.issue64,
            tau={self.window128: 2.5, self.issue64: 1.0},
            kappa={self.issue128: 0.25, self.window128: 1.5},
        )

    def test_sparse_responses_expand_for_one_kernel(self) -> None:
        components = (
            component("issue.g64.stream.load", 64),
            component("issue.g64.stream.load", 128),
            component("lane_window.t8.stream.load", 128),
            component("phase.simd.array_joint.load", 256),
        )

        self.assertEqual(
            self.profile.component_weights(components),
            {
                self.issue64: 1.0,
                self.issue128: 0.0,
                self.window128: 2.5,
                "phase.simd.array_joint.load.256B": 0.0,
            },
        )
        self.assertEqual(
            self.profile.peak_tolerances(components),
            {self.issue128: 0.25, self.window128: 1.5},
        )

    def test_area_and_peak_support_are_independent(self) -> None:
        components = (
            component("issue.g64.stream.load", 64),
            component("issue.g64.stream.load", 128),
        )

        weights = self.profile.component_weights(components)
        tolerances = self.profile.peak_tolerances(components)
        self.assertGreater(weights[self.issue64], 0.0)
        self.assertNotIn(self.issue64, tolerances)
        self.assertEqual(weights[self.issue128], 0.0)
        self.assertGreater(tolerances[self.issue128], 0.0)

    def test_description_is_stable_and_json_compatible(self) -> None:
        description = self.profile.to_dict()

        self.assertEqual(
            list(description),
            [
                "profile_id",
                "device",
                "byte_scales",
                "fine_component",
                "tau",
                "kappa",
            ],
        )
        self.assertEqual(
            list(description["tau"]),
            sorted((self.issue64, self.window128)),
        )
        self.assertEqual(description["byte_scales"], [64, 128, 256])
        self.assertEqual(
            json.loads(json.dumps(description, allow_nan=False)), description
        )
        self.assertEqual(self.profile.to_dict(), description)

    def test_profile_copies_input_mappings(self) -> None:
        device = {"model": "original"}
        tau = {self.issue64: 1.0}
        profile = HardwareProfile(
            "immutable-inputs",
            device,
            (64,),
            self.issue64,
            tau,
        )

        device["model"] = "changed"
        tau[self.issue64] = 4.0
        self.assertEqual(profile.to_dict()["device"], {"model": "original"})
        self.assertEqual(profile.tau[self.issue64], 1.0)

    def test_invalid_profile_identity_and_scale_ladder_are_rejected(self) -> None:
        cases = (
            ({"profile_id": ""}, "profile_id"),
            ({"byte_scales": ()}, "nonempty"),
            ({"byte_scales": (128, 64)}, "increasing"),
            ({"byte_scales": (64, 64)}, "increasing"),
            ({"byte_scales": (64, 96)}, "powers of two"),
        )
        base = {
            "profile_id": "test",
            "device": {},
            "byte_scales": (64, 128),
            "fine_component": self.issue64,
        }
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    HardwareProfile(**(base | overrides))

    def test_component_keys_must_use_a_profile_scale(self) -> None:
        base = {
            "profile_id": "test",
            "device": {},
            "byte_scales": (64, 128),
            "fine_component": self.issue64,
        }
        with self.assertRaisesRegex(ValueError, "standardized"):
            HardwareProfile(**base, tau={"issue": 1.0})
        with self.assertRaisesRegex(ValueError, "byte-scale ladder"):
            HardwareProfile(**base, tau={"issue.g64.stream.load.256B": 1.0})
        with self.assertRaisesRegex(ValueError, "fine component"):
            HardwareProfile(**(base | {"fine_component": "issue.g64.stream.load.256B"}))

    def test_area_tau_must_be_finite_and_nonnegative(self) -> None:
        for value in (-1.0, math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "tau"):
                    HardwareProfile(
                        "test",
                        {},
                        (64,),
                        self.issue64,
                        tau={self.issue64: value},
                    )

    def test_peak_kappa_must_be_finite_and_positive(self) -> None:
        for value in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "kappa"):
                    HardwareProfile(
                        "test",
                        {},
                        (64,),
                        self.issue64,
                        kappa={self.issue64: value},
                    )

    def test_component_identity_and_scale_are_checked_at_lookup(self) -> None:
        missing_family = ObjectiveComponent(self.issue64, 64, {})
        mismatched_name = ObjectiveComponent(
            self.issue128,
            64,
            {},
            edge_family="issue.g64.stream.load",
        )
        outside_ladder = component("issue.g64.stream.load", 512)

        with self.assertRaisesRegex(ValueError, "edge_family"):
            self.profile.component_weights((missing_family,))
        with self.assertRaisesRegex(ValueError, "incoherent"):
            self.profile.component_weights((mismatched_name,))
        with self.assertRaisesRegex(ValueError, "outside profile"):
            self.profile.component_weights((outside_ladder,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.profile.component_weights(
                (
                    component("issue.g64.stream.load", 64),
                    component("issue.g64.stream.load", 64),
                )
            )

    def test_device_metadata_must_be_strict_json(self) -> None:
        with self.assertRaisesRegex(TypeError, "JSON-compatible"):
            HardwareProfile(
                "test",
                {"invalid": {1, 2}},
                (64,),
                self.issue64,
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            HardwareProfile(
                "test",
                {"clock": math.nan},
                (64,),
                self.issue64,
            )


if __name__ == "__main__":
    unittest.main()
