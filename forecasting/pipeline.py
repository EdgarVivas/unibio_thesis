"""End-to-end pipeline orchestrating nitrate forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .config import ForecastConfig
from .drift_correction import DriftCorrector
from .fusion import FusionModule
from .history_encoder import TimeSeriesEncoder
from .metadata import MetadataEncoder
from .probabilistic_forecast import LatentDiffusionForecaster


@dataclass
class ForecastingArtifacts:
    fused_state: torch.Tensor
    latent_samples: torch.Tensor
    corrected_history: np.ndarray


class ForecastingPipeline(nn.Module):
    """High-level interface that ties all subsystems together."""

    def __init__(self, config: ForecastConfig) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.metadata_encoder = MetadataEncoder(config.metadata, device=config.device)
        self.time_series_encoder = TimeSeriesEncoder(config.time_series).to(self.device)
        self.fusion_module = FusionModule(
            metadata_dim=self.metadata_encoder.embedding_dim,
            history_dim=config.time_series.hidden_dim,
            config=config.fusion,
        ).to(self.device)
        self.drift_corrector = DriftCorrector(config.drift)
        self.forecaster = LatentDiffusionForecaster(
            fused_dim=config.fusion.fused_dim,
            horizon=config.forecast_horizon,
            config=config.diffusion,
            device=self.device,
        )
        self.to(self.device)

    def preprocess_history(self, readings: np.ndarray) -> np.ndarray:
        return self.drift_corrector.correct(readings)

    def encode_metadata(self, metadata_texts: Iterable[str], batch: bool = False) -> torch.Tensor:
        with torch.no_grad():
            if batch:
                embeddings = self.metadata_encoder.encode(metadata_texts).to(self.device)
            else:
                combined = " ".join(metadata_texts)
                embeddings = self.metadata_encoder.encode([combined]).to(self.device)
        return embeddings

    def encode_history(self, history: torch.Tensor) -> Dict[str, torch.Tensor]:
        summary, sequence = self.time_series_encoder(history)
        return {"summary": summary, "sequence": sequence}

    def fuse(self, metadata_embedding: torch.Tensor, history_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused, _ = self.fusion_module(metadata_embedding, history_embeddings["summary"], history_embeddings["sequence"])
        return fused

    def forecast(
        self,
        history: np.ndarray,
        metadata_texts: Sequence[str],
        num_samples: int = 50,
    ) -> ForecastingArtifacts:
        """Generate nitrate trajectory samples for the forecast horizon."""

        corrected = self.preprocess_history(history)
        metadata_embedding = self.encode_metadata(metadata_texts)
        history_tensor = torch.tensor(corrected, dtype=torch.float32, device=self.device)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)
        elif history_tensor.ndim == 1:
            history_tensor = history_tensor.unsqueeze(0).unsqueeze(-1)
        self.eval()
        with torch.no_grad():
            history_embeddings = self.encode_history(history_tensor)
            fused_state = self.fuse(metadata_embedding, history_embeddings)
            samples = self.forecaster.sample(fused_state, num_samples=num_samples)
        return ForecastingArtifacts(
            fused_state=fused_state.detach().cpu(),
            latent_samples=samples.detach().cpu(),
            corrected_history=corrected,
        )

    def forward(
        self,
        history: np.ndarray,
        metadata_texts: Sequence[str],
        num_samples: int = 50,
    ) -> ForecastingArtifacts:
        return self.forecast(history, metadata_texts, num_samples=num_samples)

