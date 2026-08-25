"""Versioned hardware-response profiles for RELAY's universal scope basis."""

from __future__ import annotations

from .hardware import HardwareProfile


MI300A_V1 = HardwareProfile(
    profile_id="mi300a-gfx942-universal-v1-poc",
    device={
        "vendor": "AMD",
        "architecture": "CDNA 3",
        "target": "gfx942",
        "simd_width": 64,
        "status": "proof-of-concept",
        "calibration": {
            "corpus": "layout_ranking.json",
            "basis": "universal-v1",
            "objective": "small transferable candidate frontier",
            "training_instances": [
                "atax:N256",
                "atax:N512",
                "gesummv:N256",
                "gesummv:N512",
            ],
            "search_seed": 7,
        },
    },
    byte_scales=(
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    ),
    fine_component="issue.g64.stream.load.64B",
    tau={
        # Hand-reviewed, solver-compatible feasible variant from seeded sparse
        # fits on the four training instances. Other sparse supports satisfy
        # the same small-frontier/regret constraints. Proportional stream/array
        # cells split each response equally; common rescaling is immaterial.
        "lane_window.t4.array.load.64B": 0.00924,
        "lane_window.t4.stream.load.64B": 0.00924,
        "lane_window.t16.array.load.256B": 0.00157,
        "lane_window.t16.stream.load.256B": 0.00157,
        "simd_window.t4.array.load.128B": 0.018,
        "simd_window.t4.stream.load.128B": 0.018,
        "simd_window.t4.array.load.512B": 1.0,
        "simd_window.t4.stream.load.512B": 1.0,
        "simd_window.t4.array.load.2048B": 0.249,
        "simd_window.t4.stream.load.2048B": 0.249,
    },
    kappa={
        "issue.g8.stream.load.16B": 1.0,
        "issue.g16.stream.load.512B": 15.0,
        "issue.g64.stream.load.4096B": 63.0,
        "lane_window.t4.stream.load.64B": 3.0,
        "lane_window.t16.stream.load.256B": 15.0,
        "simd_window.t4.stream.load.128B": 3.0,
    },
)


HARDWARE_PROFILES = {"mi300a": MI300A_V1}


def get_hardware_profile(name: str) -> HardwareProfile:
    try:
        return HARDWARE_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown hardware profile {name!r}") from error
