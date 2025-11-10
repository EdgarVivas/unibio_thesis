"""Time-series encoder for multivariate reactor data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .config import TimeSeriesEncoderConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeSeriesEncoder(nn.Module):
    """Encodes multivariate sensor history into a latent representation."""

    def __init__(self, config: TimeSeriesEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_transformer:
            self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.positional_encoding = PositionalEncoding(config.hidden_dim)
        else:
            self.model = nn.LSTM(
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.positional_encoding = None

        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode time-series input.

        Args:
            x: Tensor of shape (batch, time, features).

        Returns:
            pooled embedding and sequence of hidden states.
        """

        if self.config.use_transformer:
            projected = self.input_projection(x)
            encoded = self.model(self.positional_encoding(projected))
        else:
            encoded, _ = self.model(x)

        normalized = self.layer_norm(encoded)
        pooled = normalized[:, -1]
        return pooled, normalized

