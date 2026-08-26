"""Device-specific responses over the universal objective-component basis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal

from .gf2 import rank
from .objectives import ObjectiveComponent


PhasePolicy = Literal["controlled", "observed", "robust"]


@dataclass(frozen=True)
class ResourceMap:
    """A small address-to-resource color sketch.

    Each mask defines one output bit as the parity of selected byte-address
    bits. Transactions are deduplicated before colors are counted. ``robust``
    phase policy maximizes contention over every relative allocation color;
    the other policies use caller-supplied allocation base addresses.
    """

    name: str
    transaction_bytes: int
    xor_masks: tuple[int, ...]
    cohort_family: str
    phase_policy: PhasePolicy = "robust"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("resource map name must be a nonempty string")
        if (
            isinstance(self.transaction_bytes, bool)
            or not isinstance(self.transaction_bytes, int)
            or self.transaction_bytes <= 0
            or self.transaction_bytes & (self.transaction_bytes - 1)
        ):
            raise ValueError(
                "resource transaction_bytes must be a positive power of two"
            )
        try:
            masks = tuple(self.xor_masks)
        except TypeError as error:
            raise TypeError(
                "resource xor_masks must be an iterable of integers"
            ) from error
        if not masks:
            raise ValueError("resource xor_masks must be nonempty")
        if any(isinstance(mask, bool) or not isinstance(mask, int) for mask in masks):
            raise TypeError("resource xor_masks must contain only integers")
        if any(mask <= 0 for mask in masks):
            raise ValueError("resource xor_masks must be positive")
        if len(set(masks)) != len(masks) or rank(masks) != len(masks):
            raise ValueError("resource xor_masks must be linearly independent")
        transaction_bits = self.transaction_bytes.bit_length() - 1
        if any(mask & ((1 << transaction_bits) - 1) for mask in masks):
            raise ValueError(
                "resource xor_masks cannot select bits within a transaction"
            )
        if not isinstance(self.cohort_family, str) or not self.cohort_family.strip():
            raise ValueError("resource cohort_family must be a nonempty string")
        if self.phase_policy not in ("controlled", "observed", "robust"):
            raise ValueError(f"unknown resource phase policy {self.phase_policy!r}")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, Real)
            or not isfinite(float(self.weight))
            or self.weight < 0
        ):
            raise ValueError("resource map weight must be finite and nonnegative")
        object.__setattr__(self, "xor_masks", masks)
        object.__setattr__(self, "weight", float(self.weight))

    @property
    def color_count(self) -> int:
        return 1 << len(self.xor_masks)

    def color(self, byte_address: int) -> int:
        """Return the resource color of one aligned byte address."""

        if isinstance(byte_address, bool) or not isinstance(byte_address, int):
            raise TypeError("byte address must be an integer")
        if byte_address < 0:
            raise ValueError("byte address must be nonnegative")
        return sum(
            ((byte_address & mask).bit_count() & 1) << bit
            for bit, mask in enumerate(self.xor_masks)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transaction_bytes": self.transaction_bytes,
            "xor_masks": list(self.xor_masks),
            "cohort_family": self.cohort_family,
            "phase_policy": self.phase_policy,
            "weight": self.weight,
            "color_count": self.color_count,
        }


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
    resource_maps: tuple[ResourceMap, ...] = ()

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
        try:
            resource_maps = tuple(self.resource_maps)
        except TypeError as error:
            raise TypeError("hardware resource_maps must be an iterable") from error
        if any(
            not isinstance(resource_map, ResourceMap)
            for resource_map in resource_maps
        ):
            raise TypeError("hardware resource_maps must contain ResourceMap values")
        resource_names = [resource_map.name for resource_map in resource_maps]
        if len(resource_names) != len(set(resource_names)):
            raise ValueError("hardware resource map names must be unique")
        object.__setattr__(self, "resource_maps", resource_maps)

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
            "resource_maps": [
                resource_map.to_dict() for resource_map in self.resource_maps
            ],
        }
