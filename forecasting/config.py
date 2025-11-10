"""Configuration dataclasses for the nitrate forecasting system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


def _default_layers() -> List[int]:
    return [256, 256]


@dataclass
class MetadataEncoderConfig:
    """Configuration for the textual metadata encoder."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 128
    trainable: bool = False
    pooling: str = "mean"


@dataclass
class TimeSeriesEncoderConfig:
    """Configuration for the time-series encoder."""

    input_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    use_transformer: bool = True
    num_heads: int = 4


@dataclass
class FusionConfig:
    """Configuration for the multimodal fusion module."""

    fused_dim: int = 256
    cross_attention_heads: int = 4
    gating: bool = True
    alignment_temperature: float = 0.05
    alignment_loss_weight: float = 0.1


@dataclass
class DiffusionConfig:
    """Configuration for the latent diffusion forecaster."""

    steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    latent_dim: int = 64
    decoder_layers: List[int] = field(default_factory=_default_layers)


@dataclass
class DriftCorrectionConfig:
    """Configuration for the drift correction module."""

    forgetting_factor: float = 0.98
    slope_window: int = 120
    enable_high_pass: bool = True


@dataclass
class ForecastConfig:
    """Top-level configuration for the nitrate forecasting pipeline."""

    metadata: MetadataEncoderConfig = field(default_factory=MetadataEncoderConfig)
    time_series: TimeSeriesEncoderConfig = field(default_factory=TimeSeriesEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    drift: DriftCorrectionConfig = field(default_factory=DriftCorrectionConfig)
    forecast_horizon: int = 30
    sampling_interval_minutes: int = 1
    metadata_use_cache: bool = True
    device: str = "cpu"

