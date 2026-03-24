"""Traffic density prediction using trained ML models per lane.

This module provides traffic density forecasting using joblib-trained ML models
for each direction (N/S/E/W). Requires all models to be present - raises
exceptions if model loading fails.

Input: Historical lane densities + horizon (seconds).
Output: Predicted densities for the specified horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import joblib  # Required dependency for loading trained models


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
    Raises exceptions if models cannot be loaded.

    Features:
    - Uses trained ML models (scikit-learn) for each direction
    - Input: Current densities + historical pattern + horizon
    - Features: 5-value history + normalized prediction horizon
    - Clamps predictions to reasonable bounds (0-50 vehicles)
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

        # Load model paths from config if not provided
        if model_paths is None:
            from config import DENSITY_PREDICTOR_MODELS

            model_paths = DENSITY_PREDICTOR_MODELS

        if not model_paths:
            raise ValueError("No model paths provided. Cannot initialize TrafficDensityPredictor.")

        # Load trained models for each lane (strict validation)
        print("Loading density predictor models...")
        for lane in ["N", "S", "E", "W"]:
            model_path = model_paths.get(lane)

            if not model_path:
                raise ValueError(f"Model path for lane {lane} is missing from config")

            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Model file for lane {lane} not found at {model_path}. "
                    f"Ensure all trained models are present in the models/ directory."
                )

            try:
                self._models[lane] = joblib.load(model_path)
                print(f"  ✓ Loaded model for lane {lane}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model for lane {lane} from {model_path}: {e}"
                ) from e

        print(f"✓ Successfully loaded all {len(self._models)} ML models for density prediction")

    def update_history(self, current_densities: dict[str, float]) -> None:
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
        """Predict traffic density ahead using trained ML models.

        Args:
            current_densities: Current vehicle counts {N/S/E/W -> count}.
            prediction_seconds: How far ahead to predict (10-90s).

        Returns:
            DensityPrediction with predicted densities and confidence scores.

        Raises:
            KeyError: If a lane model is missing.
            ValueError: If feature preparation fails.
            RuntimeError: If model prediction fails.
        """
        predicted = {}
        confidence = {}
        now = datetime.now().isoformat()

        for lane in ["N", "S", "E", "W"]:
            current = current_densities.get(lane, 0)
            hist = self._history.get(lane, [])

            # Prepare features and predict using ML model
            features = self._prepare_features(hist, current, prediction_seconds)
            pred_value = float(self._models[lane].predict(features)[0])

            # Clamp to reasonable bounds
            pred_value = max(0, min(50, pred_value))

            # Confidence decreases with prediction horizon
            conf = max(
                0.6,
                1.0 - (prediction_seconds / 150.0),
            )

            predicted[lane] = round(pred_value, 1)
            confidence[lane] = conf

        return DensityPrediction(
            predicted_densities=predicted,
            confidence_scores=confidence,
            prediction_horizon=prediction_seconds,
            timestamp=now,
        )

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
