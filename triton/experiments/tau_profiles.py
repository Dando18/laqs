"""Load the named device profiles used by the final Triton experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


TAU_NAMES = ("expert", "l1_to_l2", "speedup")


def load_tau_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "relay.triton.tau_profiles" or document.get(
        "version"
    ) != 2:
        raise ValueError(f"{path} is not a version-2 named tau profile file")
    if tuple(document.get("profile_order", ())) != TAU_NAMES:
        raise ValueError(
            f"tau profile order must be {TAU_NAMES}, got "
            f"{tuple(document.get('profile_order', ()))!r}"
        )
    return document


def tau_profile_record(
    document: Mapping[str, Any], platform: str, name: str | None = None
) -> tuple[str, Mapping[str, Any]]:
    platform_record = document["platforms"][platform]
    selected = str(name or platform_record["default_profile"])
    if selected not in TAU_NAMES:
        raise ValueError(f"unknown tau profile {selected!r}; choose from {TAU_NAMES}")
    try:
        record = platform_record["profiles"][selected]
    except KeyError as error:
        raise ValueError(
            f"tau profile {selected!r} is not defined for {platform}"
        ) from error
    return selected, record


def load_hardware_profile(platform: str, path: Path, name: str | None = None):
    from relay import HardwareProfile

    document = load_tau_document(path)
    selected, record = tau_profile_record(document, platform, name)
    tau = {
        str(component): float(weight)
        for component, weight in record["active_tau"].items()
        if float(weight) != 0.0
    }
    if not tau:
        raise ValueError(f"{platform}/{selected} has no positive tau components")
    if any(weight < 0.0 for weight in tau.values()):
        raise ValueError(f"{platform}/{selected} has a negative tau weight")
    return HardwareProfile(
        profile_id=str(record["profile_id"]),
        device={"platform": platform, "tau_name": selected,
                "tau_semantics": "analytical expert weights" if selected == "expert" else
                    "frozen pilot weight ablation; inspect training protocol and source before transferring",
                "training_protocol": record.get("fit", {}).get("measurement_protocol", "legacy_v1"),
                "training_source_hash": record.get("fit", {}).get("source_hash")},
        byte_scales=tuple(map(int, document["byte_scales"])),
        fine_component=str(record["fine_component"]),
        tau=tau,
    )
