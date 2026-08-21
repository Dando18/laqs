"""Shared scalar-type descriptions for generated HIP evaluators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluatorDType:
    """Storage and accumulation choices for one generated evaluator type."""

    name: str
    label: str
    element_bytes: int
    scalar_type: str
    accumulator_type: str
    load_expression: str
    store_expression: str
    tolerance: float


EVALUATOR_DTYPES = {
    "fp64": EvaluatorDType(
        "fp64",
        "FP64",
        8,
        "double",
        "double",
        "static_cast<accum_t>(value)",
        "static_cast<scalar_t>(value)",
        1.0e-10,
    ),
    "fp32": EvaluatorDType(
        "fp32",
        "FP32",
        4,
        "float",
        "float",
        "static_cast<accum_t>(value)",
        "static_cast<scalar_t>(value)",
        2.0e-4,
    ),
    "fp16": EvaluatorDType(
        "fp16",
        "FP16-storage/FP32-accumulation",
        2,
        "__half",
        "float",
        "__half2float(value)",
        "__float2half(value)",
        5.0e-3,
    ),
}


def get_evaluator_dtype(name: str) -> EvaluatorDType:
    try:
        return EVALUATOR_DTYPES[name]
    except KeyError as error:
        raise ValueError(f"unknown evaluator dtype {name!r}") from error
