"""Types the rough Bergomi plugin returns.

A host such as volsurface duck-types these. Nothing in this package loads
that library. The fields match the structural protocols: ``Paths`` has
``times``, ``Z`` and ``state``; ``PriceResult`` has ``value``, ``stderr``,
``n_evals`` and ``wall_time``; ``ParamSpec`` has ``name``, ``lower``,
``upper``, ``default`` and ``transform``.

``transform`` is ``"log"``, ``"logit"`` or ``"identity"``, the same three
maps a fitter in volsurface applies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

State = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class Paths:
    """Unit-mean martingale on a grid that includes t = 0.

    ``times`` has shape ``(n + 1,)``. ``Z`` has shape ``(m, n + 1)``.
    ``state["v"]`` has shape ``(m, n + 1)``. ``state["xi"]``, when the
    model was asked for a forward curve, has shape ``(m, n + 1, d)``.
    """

    times: np.ndarray
    Z: np.ndarray
    state: dict[str, np.ndarray]


@dataclass(frozen=True)
class PriceResult:
    """A price and the standard error of the estimator that produced it."""

    value: np.ndarray
    stderr: np.ndarray | None
    n_evals: int
    wall_time: float


@dataclass(frozen=True)
class ParamSpec:
    """One fitter coordinate. See the module note for ``transform``."""

    name: str
    lower: float
    upper: float
    default: float
    transform: str


def to_unconstrained_value(value: float, spec: ParamSpec) -> float:
    """Map a parameter into an unconstrained real."""
    if spec.transform == "identity":
        return float(value)
    if spec.transform == "log":
        if value <= 0.0:
            raise ValueError(f"{spec.name} must be positive for a log transform, got {value}")
        return float(np.log(value))
    if spec.transform == "logit":
        span = spec.upper - spec.lower
        if span <= 0.0:
            raise ValueError(f"{spec.name} needs upper > lower")
        unit = (float(value) - spec.lower) / span
        unit = float(np.clip(unit, 1e-12, 1.0 - 1e-12))
        return float(np.log(unit / (1.0 - unit)))
    raise ValueError(f"unknown transform {spec.transform!r} on {spec.name}")


def from_unconstrained_value(x: float, spec: ParamSpec) -> float:
    """Map an unconstrained real back into the parameter range."""
    if spec.transform == "identity":
        return float(np.clip(x, spec.lower, spec.upper))
    if spec.transform == "log":
        return float(np.exp(x))
    if spec.transform == "logit":
        unit = 1.0 / (1.0 + np.exp(-float(x)))
        return float(spec.lower + (spec.upper - spec.lower) * unit)
    raise ValueError(f"unknown transform {spec.transform!r} on {spec.name}")
