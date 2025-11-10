# Unibio Nitrate Forecasting Architecture

This repository documents the proposed multimodal, probabilistic forecasting system for nitrate accumulation in Unibio's U-loop bioreactors. It combines Large Language Model (LLM) metadata embeddings with high-frequency sensor data to deliver 30-minute horizon forecasts and calibrated uncertainty estimates.

## Repository Structure

- `docs/architecture_overview.md` – End-to-end architecture summary and pipeline description.
- `docs/components/` – Detailed design notes for each subsystem:
  - `metadata_encoding.md`
  - `history_encoder.md`
  - `fusion_module.md`
  - `probabilistic_forecast.md`
  - `drift_correction.md`
  - `training_pipeline.md`
- `docs/diagrams/architecture.mmd` – Mermaid diagram illustrating the data and model flow.

## Usage

Use these documents as guidance for implementation, experimentation, and integration with Unibio's bioreactor operations. The design emphasizes:

- Semantic grounding via LLM-aligned metadata embeddings.
- Transformer-based temporal encoding with prototype alignment.
- Cross-attention fusion and gating to blend knowledge with data.
- Latent diffusion or neural SDE-based probabilistic forecasting.
- Sensor drift correction and calibration feedback loops.
- Rigorous training, validation, and deployment readiness workflows.

