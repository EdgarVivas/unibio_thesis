# Nitrate Forecasting Architecture for U-Loop Bioreactors

This document outlines the end-to-end architecture for forecasting nitrate (NO₃⁻) accumulation over a 30-minute horizon in Unibio's U-loop bioreactor. The system integrates rich semantic context from sensor metadata, high-frequency multivariate telemetry, and probabilistic sequence modeling to deliver calibrated uncertainty estimates that are robust to manual interventions and sensor drift.

## System Goals

- **30-minute probabilistic forecasts** of nitrate and ammonium trends with full trajectory sampling.
- **Semantic grounding** of sensor time-series using Large Language Model (LLM) embeddings of textual metadata and operator notes.
- **Multimodal alignment and fusion** that couples contextual knowledge with recent dynamics.
- **Robustness to sensor drift** via optional calibration and bias-correction pipelines.
- **Operational readiness** for integration with Unibio's soft-sensor and control toolchain.

## High-Level Pipeline

1. **Data Ingestion**
   - Acquire multivariate sensor streams sampled at 1-minute cadence (flows, pressures, temperatures, gas composition, pump states, etc.).
   - Retrieve textual metadata per sensor and global process notes from documentation and operator logs.
   - Optionally capture periodic offline laboratory assays for nitrate and ammonium to support drift correction.

2. **Metadata Encoding**
   - Preprocess textual descriptions (tokenization, normalization, deduplication).
   - Encode each text span using a frozen scientific-domain LLM encoder (e.g., SciBERT, BioGPT) to generate sensor-level embeddings \(z_{\text{meta}}\).
   - Aggregate embeddings across sensors and contextual notes to derive global context vectors.

3. **Time-Series History Encoding**
   - Normalize each sensor stream with rolling statistics and add time/covariate features.
   - Pass the past \(N=60\)–\(120\) minutes of aligned data through a Transformer or stacked bidirectional GRU encoder to obtain latent history representation \(z_{\text{history}}\).
   - Maintain variable-level hidden states for cross-attention fusion and prototype alignment.

4. **Multimodal Alignment & Fusion**
   - Apply contrastive alignment losses to place \(z_{\text{history}}\) in the same semantic geometry as \(z_{\text{meta}}\).
   - Fuse modalities through gated cross-attention layers and feature-wise linear modulation, producing \(z_{\text{fused}}\).
   - Optionally perform retrieval-augmented refinement using nearest-neighbour lookups in the LLM embedding space.

5. **Probabilistic Forecasting**
   - Condition a latent diffusion model or stochastic differential equation (SDE) simulator on \(z_{\text{fused}}\).
   - Sample multiple future latent trajectories and decode them into nitrate/ammonium predictions to obtain predictive distributions and intervals.

6. **Sensor Drift Correction (Optional)**
   - Update per-sensor bias and slope estimates using lab assays or long-term residual tracking.
   - Adjust incoming measurements and propagate calibration metadata back into \(z_{\text{meta}}\) and \(z_{\text{history}}\).

7. **Outputs**
   - Generate ensemble trajectories, prediction intervals, and risk indicators for nitrate excursions.
   - Provide interpretable diagnostics (attention weights, retrieved concepts) for operator situational awareness.

Refer to the component-specific documents in `docs/components/` for detailed design and implementation guidance.

