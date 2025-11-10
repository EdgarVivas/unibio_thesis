# Probabilistic Forecast Module

## Goal
Generate calibrated distributions over nitrate (and ammonium) trajectories 30 minutes into the future, conditioned on fused multimodal context.

## Inputs
- Fused latent representation `z_fused_global` and sequence embeddings (optional).
- Historical nitrate observations for teacher-forced training.
- Stochastic noise seeds for sampling future trajectories.

## Candidate Architectures

### 1. Latent Diffusion Model
- **Forward Process:** Adds Gaussian noise across diffusion timesteps to latent nitrate trajectories.
- **Reverse Process:** U-Net or Transformer-based denoiser conditioned on `z_fused` reconstructs clean trajectories.
- **Conditioning:** Cross-attention between diffusion states and `z_fused_sequence` to inject temporal context.
- **Outputs:** Sampled trajectories obtained via ancestral sampling; supports quantile and interval computation.

### 2. Neural Stochastic Differential Equation (SDE)
- **Dynamics:** Learn drift `f_θ` and diffusion `g_θ` functions governing latent state evolution.
- **Integration:** Use stochastic Runge–Kutta or Euler–Maruyama solvers during training and inference.
- **Inference:** Sample multiple paths by drawing distinct Wiener process realizations.

### 3. Mixture Density / Quantile Regression Baseline
- **Decoder:** Feed-forward network predicting parameters of a Gaussian mixture or specific quantiles.
- **Use Case:** Baseline benchmarking or low-resource deployments.

## Training Objectives
- **Diffusion:** Mean squared error between predicted and true noise; optional variational lower bound on log-likelihood.
- **SDE:** Score matching or flow-matching loss aligning sampled paths with empirical futures.
- **Baselines:** Negative log-likelihood or pinball loss for specified quantiles.
- **Calibration:** Continuous Ranked Probability Score (CRPS) for validation and hyperparameter tuning.

## Decoding to Physical Units
- Linear or non-linear decoder mapping latent states to nitrate concentration.
- Apply inverse normalization and reintroduce drift correction offsets if removed upstream.
- Produce per-timestep statistics (mean, variance, quantiles) for operator dashboards.

## Uncertainty Quantification
- Generate ensembles (e.g., 50–200 samples) to estimate prediction intervals.
- Compute exceedance probabilities for regulatory thresholds.
- Flag multi-modal distributions via clustering on sampled trajectories.

## Deployment Considerations
- Cache warm-up samples to reduce latency.
- Support adaptive sampling depth (fewer diffusion steps under latency constraints).
- Provide hooks for scenario analysis (e.g., "simulate reduced methane feed" by perturbing `z_meta`).

