from __future__ import annotations

"""Simulation configuration for torchrir."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration values for RIR simulation and convolution."""

    use_lut: bool = True
    mixed_precision: bool = False
    frac_delay_length: int = 81
    sinc_lut_granularity: int = 20
    image_chunk_size: int = 2048
    accumulate_chunk_size: int = 4096
    use_compile: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        if self.frac_delay_length <= 0 or self.frac_delay_length % 2 == 0:
            raise ValueError("frac_delay_length must be a positive odd integer")
        if self.sinc_lut_granularity <= 0:
            raise ValueError("sinc_lut_granularity must be positive")
        if self.image_chunk_size <= 0:
            raise ValueError("image_chunk_size must be positive")
        if self.accumulate_chunk_size <= 0:
            raise ValueError("accumulate_chunk_size must be positive")

    def replace(self, **kwargs) -> "SimulationConfig":
        """Return a new config with updated fields."""
        new_cfg = replace(self, **kwargs)
        new_cfg.validate()
        return new_cfg


def default_config() -> SimulationConfig:
    """Return the default simulation configuration."""
    cfg = SimulationConfig()
    cfg.validate()
    return cfg
