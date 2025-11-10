"""Utilities for training the nitrate forecasting pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from .pipeline import ForecastingPipeline


class TimeSeriesDataset(Dataset):
    def __init__(self, histories: np.ndarray, futures: np.ndarray, metadata_texts: Sequence[Sequence[str]]) -> None:
        assert len(histories) == len(futures) == len(metadata_texts)
        self.histories = torch.tensor(histories, dtype=torch.float32)
        self.futures = torch.tensor(futures, dtype=torch.float32)
        self.metadata_texts = metadata_texts

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, idx: int):
        return self.histories[idx], self.futures[idx], self.metadata_texts[idx]


@dataclass
class TrainingState:
    epoch: int
    loss: float


def train_epoch(
    pipeline: ForecastingPipeline,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> TrainingState:
    pipeline.forecaster.train()
    total_loss = 0.0
    for histories, futures, metadata_texts in dataloader:
        histories = histories.to(device)
        futures = futures.to(device)
        flat_texts = [" ".join(texts) for texts in metadata_texts]
        metadata_embedding = pipeline.encode_metadata(flat_texts, batch=True)
        history_embeddings = pipeline.encode_history(histories)
        fused_state = pipeline.fuse(metadata_embedding, history_embeddings)
        loss_dict = pipeline.forecaster(fused_state, futures)
        loss = loss_dict["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * histories.size(0)
    return TrainingState(epoch=epoch, loss=total_loss / len(dataloader.dataset))


def fit(
    pipeline: ForecastingPipeline,
    dataset: Dataset,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> Iterable[TrainingState]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pipeline.parameters()), lr=lr)
    device = torch.device(pipeline.config.device)
    for epoch in range(epochs):
        state = train_epoch(pipeline, dataloader, optimizer, device, epoch)
        yield state

