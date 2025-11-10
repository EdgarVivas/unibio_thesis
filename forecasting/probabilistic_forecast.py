"""Probabilistic forecasting via latent diffusion."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from .config import DiffusionConfig


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, device=timesteps.device) / half_dim)
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class TemporalDenoiser(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256, num_layers: int = 3) -> None:
        super().__init__()
        layers = []
        input_dim = latent_dim * 2
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, noisy_latent: torch.Tensor, timestep_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noisy_latent, timestep_embed], dim=-1)
        hidden = self.mlp(x)
        return self.output(hidden)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_cumprod: torch.Tensor
    alpha_cumprod_prev: torch.Tensor

    @classmethod
    def from_config(cls, config: DiffusionConfig, device: torch.device) -> "DiffusionSchedule":
        betas = torch.linspace(config.beta_start, config.beta_end, config.steps, device=device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alpha_cumprod[:-1]])
        return cls(betas, alphas, alpha_cumprod, alpha_cumprod_prev)


class LatentDiffusionForecaster(nn.Module):
    """Conditional diffusion model for nitrate trajectory forecasting."""

    def __init__(self, fused_dim: int, horizon: int, config: DiffusionConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.horizon = horizon
        self.device = device
        self.latent_dim = config.latent_dim
        self.condition_projection = nn.Linear(fused_dim, self.latent_dim)
        self.future_projection = nn.Linear(horizon, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, horizon),
        )
        self.denoiser = TemporalDenoiser(self.latent_dim)
        self.timestep_embed = nn.Linear(self.latent_dim, self.latent_dim)
        self.schedule = DiffusionSchedule.from_config(config, device)
        self.to(device)

    def _get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        embedding = sinusoidal_time_embedding(timesteps, self.latent_dim)
        return self.timestep_embed(embedding)

    def forward(self, fused_state: torch.Tensor, nitrate_future: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = fused_state.size(0)
        noise = torch.randn(batch, self.latent_dim, device=self.device)
        timesteps = torch.randint(0, self.config.steps, (batch,), device=self.device)
        alpha_bar = self.schedule.alpha_cumprod[timesteps].unsqueeze(-1)

        conditional = self.condition_projection(fused_state)
        latent_future = self.future_projection(nitrate_future)
        noisy_latent = (
            torch.sqrt(alpha_bar) * latent_future + torch.sqrt(1 - alpha_bar) * noise
        )
        timestep_embed = self._get_timestep_embedding(timesteps.float())
        pred_noise = self.denoiser(noisy_latent + conditional, timestep_embed)
        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        return {"loss": loss}

    @torch.no_grad()
    def sample(self, fused_state: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        batch = fused_state.size(0)
        conditional = self.condition_projection(fused_state)
        x = torch.randn(batch * num_samples, self.latent_dim, device=self.device)
        conditional = conditional.repeat_interleave(num_samples, dim=0)

        for t in reversed(range(self.config.steps)):
            beta_t = self.schedule.betas[t]
            alpha_t = self.schedule.alphas[t]
            alpha_bar_t = self.schedule.alpha_cumprod[t]
            alpha_bar_prev = self.schedule.alpha_cumprod_prev[t]

            timestep = torch.full((x.size(0),), t, device=self.device, dtype=torch.float32)
            timestep_embed = self._get_timestep_embedding(timestep)
            eps = self.denoiser(x + conditional, timestep_embed)
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            x = coef1 * (x - coef2 * eps)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t)
                x = x + sigma * noise
        decoded = self.decoder(x)
        return decoded.view(batch, num_samples, -1)

