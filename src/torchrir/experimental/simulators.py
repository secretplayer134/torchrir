"""Experimental simulation backends (placeholders)."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

from ..models import RIRResult, SceneLike
from ..config import SimulationConfig


@dataclass(frozen=True)
class RayTracingSimulator:
    """Work in progress placeholder for ray tracing simulation.

    Goal:
        Provide a geometric acoustics backend that traces specular/diffuse
        reflection paths, supports frequency-dependent absorption/scattering,
        and returns a RIRResult compatible with the ISM path. The intent is to
        reuse Scene/SimulationConfig for inputs and keep output shape parity.
    """

    def __post_init__(self) -> None:
        warnings.warn(
            "RayTracingSimulator is experimental and not implemented.",
            RuntimeWarning,
            stacklevel=2,
        )

    def simulate(
        self, scene: SceneLike, config: SimulationConfig | None = None
    ) -> RIRResult:
        raise NotImplementedError("RayTracingSimulator is not implemented yet")


@dataclass(frozen=True)
class FDTDSimulator:
    """Work in progress placeholder for FDTD simulation.

    Goal:
        Provide a wave-based solver (finite-difference time-domain) with
        configurable grid resolution, boundary conditions, and stability
        constraints. The solver should target CPU/GPU execution and return
        RIRResult with the same metadata contract as ISM.
    """

    def __post_init__(self) -> None:
        warnings.warn(
            "FDTDSimulator is experimental and not implemented.",
            RuntimeWarning,
            stacklevel=2,
        )

    def simulate(
        self, scene: SceneLike, config: SimulationConfig | None = None
    ) -> RIRResult:
        raise NotImplementedError("FDTDSimulator is not implemented yet")
