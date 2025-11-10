# Training and Evaluation Pipeline

## Overview
Defines staged training, validation, and monitoring procedures for the multimodal nitrate forecasting system.

## Data Preparation
- **Windowing:** Extract overlapping sequences with 30-minute forecast horizons and 60–120 minute history windows.
- **Splitting:** Use chronological splits (train/validation/test) respecting batch boundaries to avoid leakage.
- **Feature Engineering:**
  - Normalize sensor data using training split statistics.
  - Encode categorical covariates (batch ID, phase) via embeddings.
  - Align metadata snapshots with sensor windows; store embedding version hashes.
- **Augmentation:** Simulate sensor dropouts, drift perturbations, and synthetic interventions to broaden coverage.

## Stage 1: Encoder Alignment Pre-training
1. Freeze LLM encoder; train history encoder projection heads with contrastive loss.
2. Include prototype vocabulary and hard-negative mining.
3. Monitor alignment via cosine similarity and retrieval accuracy benchmarks.

## Stage 2: Fusion Warm-up
- Train cross-attention and gating modules using supervised auxiliary tasks (e.g., predicting intervention tags) if available.
- Regularize with dropout on metadata channels to ensure graceful degradation when text is missing.

## Stage 3: Probabilistic Forecast Training
- Select primary generative backbone (diffusion or SDE) and corresponding loss.
- Employ teacher forcing with scheduled sampling for decoder stability.
- Apply gradient clipping and EMA of model weights for consistent sampling quality.

## Calibration & Validation
- Metrics: RMSE (point forecast), CRPS (distribution), energy score, prediction interval coverage probability (PICP).
- Backtesting: Rolling-origin evaluation replicating real-time deployment.
- Calibration: Use isotonic regression or temperature scaling on predictive quantiles if systematic bias observed.

## Drift-Aware Fine-Tuning
- Periodically retrain drift correction parameters with latest lab assays.
- Fine-tune fusion gates to reflect updated sensor reliability scores.
- Log calibration deltas and compare forecast performance pre/post adjustment.

## Deployment Readiness Checklist
- Latency budgets met (inference < 1 minute per cycle).
- Robustness validated on synthetic edge cases (intervention storms, sensor failures).
- Monitoring hooks implemented (embedding drift, forecast residual control charts).
- Documentation completed (model cards, data lineage, audit logs).

