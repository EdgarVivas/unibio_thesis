# Multimodal Alignment and Fusion Module

## Objectives
- Harmonize textual metadata embeddings with latent time-series features.
- Emphasize causal relationships implied by domain knowledge while preserving data-driven dynamics.
- Produce a fused representation `z_fused` suitable for probabilistic forecasting.

## Subsystems

### 1. Contrastive Alignment
- **Loss Function:** InfoNCE or NT-Xent between `z_history_global` and `z_meta_global`, supplemented with sensor-level pairs.
- **Negative Sampling:** Cross-batch negatives augmented with hard negatives retrieved by nearest-neighbour search.
- **Prototype Anchoring:** Optional learnable concept tokens to stabilize alignment and improve interpretability.

### 2. Cross-Attention Fusion Transformer
- **Inputs:**
  - Query sequence: time-series tokens (`z_history_tokens`).
  - Key/Value memory: metadata tokens (sensor embeddings, prototypes, global context).
- **Mechanism:** Multi-head cross-attention followed by feed-forward refinement and residual connections.
- **Output:** Contextualized temporal embeddings reflecting relevant textual cues.

### 3. Gating & Feature-wise Modulation
- **Gating Network:** Small MLP that maps `z_meta_global` to sigmoid gates for each latent dimension of `z_history_global`.
- **FiLM Layers:** Apply affine transformations conditioned on metadata to intermediate activations within the history encoder and forecast decoder.
- **Purpose:** Dynamically modulate the influence of sensors based on reliability, maintenance state, or domain constraints encoded in text.

### 4. Retrieval-Augmented Context (Optional)
- Use metadata vectors to query a vector store of annotated historical events.
- Concatenate retrieved exemplars to the attention memory, enabling analogy-based reasoning.

## Outputs
- `z_fused_global`: final pooled embedding driving the probabilistic forecast module.
- `z_fused_sequence`: optionally the fused temporal sequence for trajectory-level conditioning.
- Attention weights and gate values for interpretability dashboards.

## Training Signals
- Downstream probabilistic forecasting loss.
- Auxiliary supervised signals (if available) such as intervention classifications or anomaly labels.
- Alignment regularizers encouraging low-entropy gate distributions when metadata indicates high confidence.

## Implementation Notes
- Normalize latent spaces before fusion (LayerNorm).
- Apply dropout to metadata inputs to prevent over-reliance on text when unavailable.
- Support metadata-free fallback by learning a null token representing missing descriptions.

