"""Multimodal fusion components."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .config import FusionConfig


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(query, context, context)
        out = self.norm(attn_output + query)
        out = self.norm(self.ff(out) + out)
        return out


class FusionModule(nn.Module):
    """Fuse metadata and time-series embeddings."""

    def __init__(self, metadata_dim: int, history_dim: int, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.metadata_projection = nn.Linear(metadata_dim, config.fused_dim)
        self.history_projection = nn.Linear(history_dim, config.fused_dim)
        self.cross_attention = CrossAttentionBlock(config.fused_dim, config.cross_attention_heads)
        self.gate = (
            nn.Sequential(
                nn.Linear(config.fused_dim * 2, config.fused_dim),
                nn.GELU(),
                nn.Linear(config.fused_dim, config.fused_dim),
                nn.Sigmoid(),
            )
            if config.gating
            else None
        )
        self.output_projection = nn.Linear(config.fused_dim * 2, config.fused_dim)

    def forward(
        self,
        metadata: torch.Tensor,
        history_summary: torch.Tensor,
        history_sequence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse metadata and history embeddings."""

        meta_proj = self.metadata_projection(metadata)
        hist_proj = self.history_projection(history_summary)
        if history_sequence is None:
            history_sequence = hist_proj.unsqueeze(1)
        else:
            history_sequence = self.history_projection(history_sequence)

        meta_expanded = meta_proj.unsqueeze(1)
        attn_output = self.cross_attention(meta_expanded, history_sequence)
        attn_output = attn_output.squeeze(1)

        if self.gate is not None:
            gate_input = torch.cat([attn_output, hist_proj], dim=-1)
            gates = self.gate(gate_input)
            hist_proj = hist_proj * gates

        fused = torch.cat([attn_output, hist_proj], dim=-1)
        return self.output_projection(fused), attn_output

