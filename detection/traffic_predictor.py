"""Traffic density prediction using trained ML models per lane.

This module provides traffic density forecasting using joblib-trained ML models
for each direction (N/S/E/W). Falls back to heuristic prediction if model loading
fails.

Input: Historical lane densities + horizon (seconds).
Output: Predicted densities for the specified horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore


@dataclass(frozen=True)
class DensityPrediction:
    """Immutable container for traffic density prediction output.

    Attributes:
        predicted_densities: Dict with lanes (N/S/E/W) -> predicted count.
        confidence_scores: Dict with lanes -> confidence (0-1).
        prediction_horizon: Seconds ahead that this prediction covers.
        timestamp: When the prediction was made.
    """

    predicted_densities: dict[str, float]
    confidence_scores: dict[str, float]
    prediction_horizon: int
    timestamp: str


class TrafficDensityPredictor:
    """Predicts traffic density using trained ML models per direction.

    Loads joblib-trained models for each lane (N/S/E/W) from config.
    Falls back to heuristic-based prediction if model loading fails.

    Strategy:
    - Primary: Use trained ML models (joblib) for each direction
    - Fallback: Linear trend extrapolation if models unavailable
    - Input: Current densities + historical pattern + horizon
    - Features: 5-value history + normalized prediction horizon
    """

    def __init__(self, model_paths: dict[str, Path] | None = None) -> None:
        """Initialize the predictor and load trained models.

        Args:
            model_paths: Dict with lanes -> Path to joblib .ubj files.
                        If None, imports from config.
        """
        self._history: dict[str, list[float]] = {
            "N": [],
            "S": [],
            "E": [],
            "W": [],
        }
        self._max_history: int = 20  # Keep last 20 measurements
        self._models: dict[str, Any] = {}
        self._use_ml_models: bool = False

        # Load model paths from config if not provided
        if model_paths is None:
            try:
                from config import DENSITY_PREDICTOR_MODELS
                model_paths = DENSITY_PREDICTOR_MODELS
            except ImportError:
                model_paths = {}

        # Load trained models for each lane
        if joblib is not None and model_paths:
            for lane in ["N", "S", "E", "W"]:
                try:
                    model_path = model_paths.get(lane)
                    if model_path and Path(model_path).exists():
                        self._models[lane] = joblib.load(model_path)
                    else:
                        print(
                            f"Warning: Model for lane {lane} not found "
                            f"at {model_path}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Failed to load model for lane {lane}: {e}")

            # Enable ML models only if all 4 lanes loaded successfully
            if len(self._models) == 4:
                self._use_ml_models = True
                print("Using trained ML models for density prediction")
            else:
                print(
                    f"Falling back to heuristic prediction "
                    f"({len(self._models)}/4 models loaded)"
                )

    def update_history(
        self, current_densities: dict[str, float]
    ) -> None:
        """Update the historical record with current densities.

        Args:
            current_densities: Dict with lanes (N/S/E/W) -> vehicle count.
        """
        for lane in ["N", "S", "E", "W"]:
            count = current_densities.get(lane, 0)
            if lane in self._history:
                self._history[lane].append(float(count))
                # Keep only recent history
                if len(self._history[lane]) > self._max_history:
                    self._history[lane].pop(0)

    def _prepare_features(
        self,
        history: list[float],
        current: float,
        prediction_seconds: int,
    ) -> np.ndarray:
        """Prepare feature vector for ML model prediction.

        Args:
            history: Historical density values (last 5).
            current: Current density value.
            prediction_seconds: Horizon for prediction.

        Returns:
            Feature array for model input.
        """
        # Use last 5 values from history, or pad with current value
        if len(history) >= 5:
            recent_5 = history[-5:]
        else:
            recent_5 = history + [current] * (5 - len(history))

        # Normalize prediction horizon (0-1 scale, assuming max 120s)
        normalized_horizon = min(prediction_seconds / 120.0, 1.0)

        # Stack features: [h1, h2, h3, h4, h5, horizon]
        features = np.array(recent_5 + [normalized_horizon], dtype=np.float32)
        return features.reshape(1, -1)  # Reshape for sklearn model

    def predict(
        self,
        current_densities: dict[str, float],
        prediction_seconds: int = 30,
    ) -> DensityPrediction:
        """Predict traffic density ahead using ML models or heuristics.

        Args:
            current_densities: Current vehicle counts {N/S/E/W -> count}.
            prediction_seconds: How far ahead to predict (10-90s).

        Returns:
            DensityPrediction with predicted densities and confidence scores.
        """
        predicted = {}
        confidence = {}
        now = datetime.now().isoformat()

        for lane in ["N", "S", "E", "W"]:
            current = current_densities.get(lane, 0)
            hist = self._history.get(lane, [])

            # Try ML model prediction first
            if (
                self._use_ml_models
                and lane in self._models
                and len(hist) > 0
            ):
                try:
                    features = self._prepare_features(
                        hist, current, prediction_seconds
                    )
                    pred_value = float(
                        self._models[lane].predict(features)[0]
                    )
                    # Clamp to reasonable bounds
                    pred_value = max(0, min(50, pred_value))
                    # Higher confidence for ML predictions
                    conf = max(
                        0.6,
                        1.0 - (prediction_seconds / 150.0),
                    )
                except Exception as e:
                    print(f"ML prediction failed for {lane}: {e}")
                    # Fall back to heuristic
                    pred_value, conf = self._heuristic_predict(
                        hist, current, prediction_seconds
                    )
            else:
                # Use heuristic prediction
                pred_value, conf = self._heuristic_predict(
                    hist, current, prediction_seconds
                )

            predicted[lane] = round(pred_value, 1)
            confidence[lane] = conf

        return DensityPrediction(
            predicted_densities=predicted,
            confidence_scores=confidence,
            prediction_horizon=prediction_seconds,
            timestamp=now,
        )

    def _heuristic_predict(
        self,
        history: list[float],
        current: float,
        prediction_seconds: int,
    ) -> tuple[float, float]:
        """Fallback heuristic prediction via linear trend extrapolation.

        Args:
            history: Historical density values.
            current: Current density value.
            prediction_seconds: Prediction horizon.

        Returns:
            Tuple of (predicted_value, confidence_score).
        """
        if len(history) >= 2:
            # Simple linear trend from last 5 measurements
            recent = history[-5:] if len(history) >= 5 else history
            trend = (recent[-1] - recent[0]) / (len(recent) - 1)
            pred_value = current + trend * (prediction_seconds / 5.0)
        else:
            # Not enough history: use current value
            pred_value = current

        # Clamp to reasonable bounds (0-50 vehicles per lane)
        pred_value = max(0, min(50, pred_value))

        # Confidence decreases with prediction horizon
        conf = max(0.5, 1.0 - (prediction_seconds / 120.0))

        return pred_value, conf

    def get_history(self, lane: str) -> list[float]:
        """Return historical density values for a lane.

        Args:
            lane: One of 'N', 'S', 'E', 'W'.

        Returns:
            List of recent density measurements.
        """
        return self._history.get(lane, [])

    def clear_history(self) -> None:
        """Reset historical data (for testing)."""
        for lane in self._history:
            self._history[lane] = []
