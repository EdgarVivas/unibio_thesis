# Time-Series History Encoder

## Purpose
Transform high-frequency multivariate telemetry into latent state representations that summarize the bioreactor's recent dynamics and provide semantically aligned features for fusion with metadata embeddings.

## Inputs
- Drift-corrected sensor readings over a sliding window (typically 60–120 minutes at 1-minute resolution).
- Auxiliary covariates: timestamps, batch phase labels, control setpoints, recent intervention indicators.
- Metadata-derived modulation vectors for feature-wise conditioning.

## Architecture Options
1. **Transformer Encoder**
   - Multi-head self-attention over time with relative positional encodings.
   - Variable grouping through learnable channel embeddings.
   - Static enrichment sub-layer to inject `z_meta_global` (similar to Temporal Fusion Transformer).
2. **Stacked Bidirectional GRUs / LSTMs**
   - Suitable for lower-latency deployment.
   - Incorporate peephole connections and layer normalization.
3. **Temporal Convolutional Networks (TCNs)**
   - Dilated causal convolutions for long receptive fields.
   - Lightweight alternative when compute is constrained.

## Alignment Strategy
- Project hidden states via linear heads into the LLM embedding dimension.
- Optimize InfoNCE contrastive loss against sensor descriptions and concept prototypes.
- Employ temperature parameter tuning to balance positive/negative pair weighting.

## Outputs
- `z_history_global`: aggregated latent summarizing overall system state.
- `z_history_tokens[t, i]`: per-time-step, per-variable embeddings for cross-attention fusion.
- Attention weights or saliency maps for interpretability dashboards.

## Training Considerations
- Mask future data to maintain causal consistency.
- Use curriculum sampling: short windows during early training, extend to full horizon as stability improves.
- Apply stochastic feature dropout to encourage robustness to missing sensors.

## Deployment Notes
- Maintain rolling normalization statistics with exponential decay.
- Support warm starts by persisting final hidden states between inference steps.
- Monitor embedding drift; realign using metadata prototypes if cosine similarity degrades.

