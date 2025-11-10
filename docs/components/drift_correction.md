# Sensor Drift Correction Module

## Purpose
Mitigate bias and drift in online nitrate proxy sensors to maintain long-term forecasting accuracy and prevent misinterpretation of gradual fouling as process dynamics.

## Inputs
- Raw online sensor data (nitrate, ammonium, dissolved oxygen, etc.).
- Periodic laboratory assays providing ground-truth nitrate measurements.
- Maintenance logs indicating cleaning, recalibration, or sensor replacement events.

## Workflow
1. **Bias Estimation**
   - Maintain exponential moving averages of residuals between online sensors and lab assays.
   - Fit a Bayesian linear model or Kalman filter that tracks offset and slope drift parameters.
2. **Correction Application**
   - Adjust incoming sensor readings by subtracting estimated bias and applying slope correction.
   - Propagate uncertainty of correction (variance of bias estimate) as additional metadata for fusion.
3. **Quality Monitoring**
   - Compute drift metrics (e.g., magnitude, rate of change) and raise alerts when thresholds are exceeded.
   - Trigger re-embedding of metadata to reflect sensor health status ("calibration overdue").
4. **Data Imputation**
   - When sensors are offline, use surrogate models (e.g., regression on correlated sensors) to fill gaps.
   - Flag imputed intervals for the forecast module to inflate predictive uncertainty.

## Outputs
- Drift-corrected sensor streams for the history encoder.
- Metadata annotations summarizing correction parameters and confidence.
- Audit logs for compliance and traceability.

## Integration Points
- Supplies correction statistics to metadata encoder for context embedding.
- Provides reliability scores for fusion gating and probabilistic forecast variance scaling.
- Interfaces with operator dashboards to visualize calibration status.

## Operational Guidelines
- Store calibration history in a persistent database for reproducibility.
- Support manual overrides when operators apply ad-hoc calibrations.
- Periodically validate correction efficacy by comparing forecast residuals pre- and post-adjustment.

