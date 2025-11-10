"""Sensor drift correction utilities."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional

import numpy as np

from .config import DriftCorrectionConfig


@dataclass
class DriftState:
    bias: float = 0.0
    slope: float = 0.0
    last_calibration_value: Optional[float] = None


class DriftCorrector:
    """Applies online drift correction to sensor readings."""

    def __init__(self, config: DriftCorrectionConfig) -> None:
        self.config = config
        self._slope_windows: List[Deque[float]] = []
        self._bias: Optional[np.ndarray] = None

    def _ensure_state(self, num_features: int) -> None:
        if self._bias is None:
            self._bias = np.zeros(num_features, dtype=np.float32)
            self._slope_windows = [deque(maxlen=self.config.slope_window) for _ in range(num_features)]

    def update_with_calibration(self, sensor_value: Iterable[float], lab_value: Iterable[float]) -> DriftState:
        """Update the bias term based on an offline lab calibration."""

        sensor_arr = np.atleast_1d(np.array(sensor_value, dtype=np.float32))
        lab_arr = np.atleast_1d(np.array(lab_value, dtype=np.float32))
        self._ensure_state(sensor_arr.size)
        assert self._bias is not None
        self._bias = (
            self.config.forgetting_factor * self._bias
            + (1 - self.config.forgetting_factor) * (lab_arr - sensor_arr)
        )
        slopes = [self._estimate_slope(i) for i in range(sensor_arr.size)]
        return DriftState(
            bias=float(self._bias.mean()),
            slope=float(np.mean(slopes)) if slopes else 0.0,
            last_calibration_value=float(lab_arr.mean()),
        )

    def _estimate_slope(self, feature_idx: int) -> float:
        window = self._slope_windows[feature_idx]
        if len(window) < 2:
            return 0.0
        y = np.fromiter(window, dtype=np.float32)
        x = np.arange(len(y), dtype=np.float32)
        slope, _ = np.polyfit(x, y, deg=1)
        return float(slope)

    def correct(self, readings: Iterable[Iterable[float]]) -> np.ndarray:
        """Apply drift correction to a sequence of readings."""

        readings_arr = np.asarray(readings, dtype=np.float32)
        if readings_arr.ndim == 1:
            readings_arr = readings_arr[:, None]
        self._ensure_state(readings_arr.shape[1])
        assert self._bias is not None

        corrected = np.empty_like(readings_arr)
        for t, row in enumerate(readings_arr):
            for feature_idx, value in enumerate(row):
                self._slope_windows[feature_idx].append(float(value))
                slope = self._estimate_slope(feature_idx) if self.config.enable_high_pass else 0.0
                corrected[t, feature_idx] = value + self._bias[feature_idx] - slope
        return corrected.squeeze()  # type: ignore[no-any-return]

