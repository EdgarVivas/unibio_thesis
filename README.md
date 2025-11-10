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

The `forecasting/` package contains an executable prototype of the architecture described in the design documents. It ships with:

- `config.py` – dataclasses configuring each subsystem.
- `metadata.py` – wrapper around Hugging Face encoders for semantic metadata embeddings.
- `history_encoder.py` – Transformer/LSTM time-series encoder implementation.
- `fusion.py` – cross-attention and gating fusion module.
- `drift_correction.py` – online bias/slope drift correction utilities.
- `probabilistic_forecast.py` – latent diffusion forecaster for trajectory sampling.
- `pipeline.py` – high-level orchestration object tying the subsystems together.
- `training.py` – minimal training loop and dataset helpers.

### Quickstart

Install dependencies (PyTorch, Transformers, NumPy):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers numpy
```

```python
import numpy as np
from forecasting import ForecastConfig, ForecastingPipeline

config = ForecastConfig()
pipeline = ForecastingPipeline(config)

# Fake data: 120 minutes of 8-sensor readings sampled once per minute.
history = np.random.randn(120, config.time_series.input_dim).astype("float32")
metadata = [
    "Methane feed rate sensor, carbon source",
    "Dissolved oxygen reading, proxy for oxidation",
]

artifacts = pipeline.forecast(history, metadata, num_samples=20)
print(artifacts.latent_samples.shape)  # -> (1, 20, 30)
```

To train the diffusion forecaster on historical runs, prepare a dataset of history/future windows and metadata descriptions, then:

```python
from forecasting.training import TimeSeriesDataset, fit

dataset = TimeSeriesDataset(histories, futures, metadata_sequences)
for state in fit(pipeline, dataset, epochs=5):
    print(f"epoch={state.epoch} loss={state.loss:.4f}")
```

Use the documents as guidance for experimentation and integration with Unibio's operations. The design emphasizes:

- Semantic grounding via LLM-aligned metadata embeddings.
- Transformer-based temporal encoding with prototype alignment.
- Cross-attention fusion and gating to blend knowledge with data.
- Latent diffusion or neural SDE-based probabilistic forecasting.
- Sensor drift correction and calibration feedback loops.
- Rigorous training, validation, and deployment readiness workflows.

