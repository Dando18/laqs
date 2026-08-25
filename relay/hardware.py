"""Device-specific responses over the universal objective-component basis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any

from .objectives import ObjectiveComponent


def _component_scale(name: str) -> int:
    """Return the byte scale encoded in a standardized component name."""

    if not isinstance(name, str) or "." not in name:
        raise ValueError(
            f"component key {name!r} must end in a standardized '.<bytes>B' scale"
        )
    family, scale_label = name.rsplit(".", 1)
    if not family or not scale_label.endswith("B"):
        raise ValueError(
            f"component key {name!r} must end in a standardized '.<bytes>B' scale"
        )
    digits = scale_label[:-1]
    if not digits.isascii() or not digits.isdigit() or digits.startswith("0"):
        raise ValueError(
            f"component key {name!r} must end in a standardized '.<bytes>B' scale"
        )
    scale = int(digits)
    if scale <= 0:
        raise ValueError(f"component key {name!r} has a nonpositive byte scale")
    return scale


def _freeze_json(value: Any, path: str) -> Any:
    """Copy JSON-compatible metadata into a stable, read-only form."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} object keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_json(value[key], f"{path}.{key}")
                for key in sorted(value)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} contains {type(value).__name__}, which is not JSON-compatible"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _response_map(
    values: Mapping[str, float],
    *,
    name: str,
    scales: frozenset[int],
    positive: bool,
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping from component names to responses")

    normalized: dict[str, float] = {}
    for component_name, value in values.items():
        scale = _component_scale(component_name)
        if scale not in scales:
            raise ValueError(
                f"{name} component {component_name!r} uses {scale} B, which is not "
                "in the profile byte-scale ladder"
            )
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name}[{component_name!r}] must be a real number")
        response = float(value)
        if not isfinite(response):
            raise ValueError(f"{name}[{component_name!r}] must be finite")
        if positive and response <= 0:
            raise ValueError(f"{name} peak tolerances must be positive")
        if not positive and response < 0:
            raise ValueError(f"{name} area responses must be nonnegative")
        normalized[component_name] = 0.0 if response == 0 else response
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class HardwareProfile:
    """A reusable device response over standardized scope-scale components.

    ``tau`` controls the exposure-preserving area score. ``kappa`` separately
    declares acceptable normalized-excess tolerances for the peak score. Both
    mappings may contain global scope cells that a particular kernel does not
    instantiate.
    """

    profile_id: str
    device: Mapping[str, Any]
    byte_scales: tuple[int, ...]
    fine_component: str
    tau: Mapping[str, float] = field(default_factory=dict)
    kappa: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("hardware profile_id must be a nonempty string")

        try:
            scales = tuple(self.byte_scales)
        except TypeError as error:
            raise TypeError("hardware byte_scales must be an iterable of integers") from error
        if not scales:
            raise ValueError("hardware byte_scales must be nonempty")
        if any(isinstance(scale, bool) or not isinstance(scale, int) for scale in scales):
            raise TypeError("hardware byte_scales must contain only integers")
        if any(scale <= 0 for scale in scales):
            raise ValueError("hardware byte_scales must be positive")
        if any(scale & (scale - 1) for scale in scales):
            raise ValueError("hardware byte_scales must be powers of two")
        if tuple(sorted(set(scales))) != scales:
            raise ValueError(
                "hardware byte_scales must be strictly increasing and unique"
            )

        scale_set = frozenset(scales)
        fine_scale = _component_scale(self.fine_component)
        if fine_scale not in scale_set:
            raise ValueError(
                f"fine component {self.fine_component!r} uses {fine_scale} B, which "
                "is not in the profile byte-scale ladder"
            )
        if not isinstance(self.device, Mapping):
            raise TypeError("hardware device metadata must be a mapping")

        object.__setattr__(self, "byte_scales", scales)
        object.__setattr__(self, "device", _freeze_json(self.device, "device"))
        object.__setattr__(
            self,
            "tau",
            _response_map(
                self.tau,
                name="tau",
                scales=scale_set,
                positive=False,
            ),
        )
        object.__setattr__(
            self,
            "kappa",
            _response_map(
                self.kappa,
                name="kappa",
                scales=scale_set,
                positive=True,
            ),
        )

    def _validated_components(
        self, components: Sequence[ObjectiveComponent]
    ) -> tuple[ObjectiveComponent, ...]:
        items = tuple(components)
        names: set[str] = set()
        scale_set = frozenset(self.byte_scales)
        for component in items:
            if component.name in names:
                raise ValueError(
                    f"duplicate objective component name {component.name!r}"
                )
            names.add(component.name)
            if component.edge_family is None:
                raise ValueError(
                    f"objective {component.name!r} has no standardized edge_family"
                )
            expected_name = f"{component.edge_family}.{component.region_bytes}B"
            if component.name != expected_name:
                raise ValueError(
                    f"objective {component.name!r} is incoherent with edge family "
                    f"and scale; expected {expected_name!r}"
                )
            if _component_scale(component.name) != component.region_bytes:
                raise ValueError(
                    f"objective {component.name!r} is incoherent with its region_bytes"
                )
            if component.region_bytes not in scale_set:
                raise ValueError(
                    f"objective {component.name!r} uses a scale outside profile "
                    f"{self.profile_id!r}"
                )
        return items

    def component_weights(
        self, components: Sequence[ObjectiveComponent]
    ) -> dict[str, float]:
        """Return a complete tau map, assigning zero to unsupported cells."""

        return {
            component.name: self.tau.get(component.name, 0.0)
            for component in self._validated_components(components)
        }

    def peak_tolerances(
        self, components: Sequence[ObjectiveComponent]
    ) -> dict[str, float]:
        """Return active kappa values for components present in one kernel."""

        return {
            component.name: self.kappa[component.name]
            for component in self._validated_components(components)
            if component.name in self.kappa
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible profile description."""

        return {
            "profile_id": self.profile_id,
            "device": _thaw_json(self.device),
            "byte_scales": list(self.byte_scales),
            "fine_component": self.fine_component,
            "tau": dict(sorted(self.tau.items())),
            "kappa": dict(sorted(self.kappa.items())),
        }
