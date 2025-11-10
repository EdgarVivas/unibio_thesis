"""Textual metadata encoding utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from .config import MetadataEncoderConfig


@dataclass
class MetadataEncoder:
    """Encodes textual metadata into semantic embeddings using a pretrained LLM."""

    config: MetadataEncoderConfig
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self._device = torch.device(self.device or "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModel.from_pretrained(self.config.model_name)
        self.embedding_dim = self._model.config.hidden_size
        if not self.config.trainable:
            for param in self._model.parameters():
                param.requires_grad = False
        self._model.to(self._device)

    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        """Return embeddings for a batch of texts."""

        encodings = self._tokenizer(  # type: ignore[call-arg]
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self._device) for k, v in encodings.items()}
        outputs = self._model(**encodings)

        if self.config.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            mask = encodings.get("attention_mask")
            if mask is None:
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                mask = mask.unsqueeze(-1)
                embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        return embeddings

