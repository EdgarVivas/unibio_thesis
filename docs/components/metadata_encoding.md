# Metadata Encoding Module

## Purpose
The metadata encoder injects domain knowledge and operator context into the forecasting stack by transforming textual descriptions into dense embeddings. These embeddings shape downstream temporal modeling, enabling semantic differentiation between sensors and operating regimes.

## Inputs
- Sensor-level descriptors (name, units, physical role, maintenance notes).
- Operator annotations and intervention logs.
- High-level process context (e.g., "oxygen-limited fermentation", "post-maintenance startup").

## Processing Steps
1. **Text Normalization**
   - Lowercasing, punctuation handling, and domain-specific token preservation (chemical formulas, subscripts).
   - Optional entity linking for chemicals, equipment, and actions using domain ontologies.
2. **LLM Embedding Extraction**
   - Utilize a frozen encoder such as SciBERT, BioClinicalBERT, or Instructor-XL.
   - Produce embeddings at the span, sentence, and document level.
   - Apply mean pooling with attention masking; optionally fine-tune projection heads while keeping the backbone frozen.
3. **Aggregation Strategy**
   - Concatenate sensor embeddings with positional tags and pass through a transformer to model relationships across sensors.
   - Generate:
     - `z_meta_sensor[i]` for each sensor *i*.
     - `z_meta_global` summarizing system-level context.
4. **Prototype Library (Optional)**
   - Maintain a set of concept prototypes (e.g., {"flow increase", "pressure drop", "maintenance"}).
   - Align sensor embeddings to prototypes for improved interpretability and contrastive training.

## Outputs
- Sensor-specific metadata embeddings `z_meta_sensor[i] \in \mathbb{R}^d`.
- Global metadata context vector `z_meta_global \in \mathbb{R}^d`.
- Prototype similarity scores for diagnostics.

## Interfaces
- Supplies conditioning vectors to the history encoder (for feature-wise modulation) and to the fusion transformer.
- Writes calibration status flags to drift-correction metadata feeds.

## Operational Considerations
- Recompute embeddings whenever metadata changes (new sensor, updated SOP, maintenance log).
- Cache embeddings with checksum-based versioning to avoid redundant LLM inference.
- Provide explainability via nearest-neighbour phrases retrieved from the embedding space.

