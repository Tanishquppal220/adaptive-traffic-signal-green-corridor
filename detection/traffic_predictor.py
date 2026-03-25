"""Traffic density prediction using trained XGBoost models per lane.

This module provides traffic density forecasting using XGBoost models trained
for each direction (N/S/E/W). Requires all models to be present - raises
exceptions if model loading fails.

Input: Historical lane densities (100 timesteps x 4 directions).
Output: Predicted densities for the next 60 seconds.

Feature Schema (404 features, matching Colab training):
- 400 lag features: 100 timesteps x 4 directions (N/S/E/W), flattened
- 4 cyclic time features: sin/cos hour, sin/cos day-of-week
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

# Constants matching training notebook (traffic-Density.ipynb)
_DIRECTIONS = ["N", "S", "E", "W"]
_WINDOW = 100  # 100 timesteps of history (matches training WINDOW)
_PREDICTION_HORIZON_SEC = 60  # Fixed 60s horizon (matches training HORIZON x 3s)


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
    """Predicts traffic density using trained XGBoost models per direction.

    Loads XGBoost models for each lane (N/S/E/W) from config.
    Raises exceptions if models cannot be loaded.

    Features:
    - Uses XGBoost models trained on 404-feature schema
    - Input: 100 timesteps x 4 directions + cyclic time features
    - Output: Predicted mean density over next 60 seconds
    - Clamps predictions to reasonable bounds (0-50 vehicles)
    """

    def __init__(self, model_paths: dict[str, Path] | None = None) -> None:
        """Initialize the predictor and load trained models.

        Args:
            model_paths: Dict with lanes -> Path to XGBoost .ubj files.
                        If None, imports from config.
        """
        # Unified history: list of snapshots, each snapshot is {N: x, S: y, E: z, W: w}
        self._history: list[dict[str, float]] = []
        self._max_history: int = _WINDOW  # Keep last 100 measurements
        self._models: dict[str, Any] = {}

        # Load model paths from config if not provided
        if model_paths is None:
            from config import DENSITY_PREDICTOR_MODELS

            model_paths = DENSITY_PREDICTOR_MODELS

        if not model_paths:
            raise ValueError("No model paths provided. Cannot initialize TrafficDensityPredictor.")

        # Load trained models for each lane (strict validation)
        print("Loading density predictor models...")
        for lane in _DIRECTIONS:
            model_path = model_paths.get(lane)

            if not model_path:
                raise ValueError(f"Model path for lane {lane} is missing from config")

            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Model file for lane {lane} not found at {model_path}. "
                    f"Ensure all trained models are present in the models/ directory."
                )

            try:
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                self._models[lane] = model
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
        # Create snapshot with all 4 directions
        snapshot = {d: float(current_densities.get(d, 0)) for d in _DIRECTIONS}
        self._history.append(snapshot)

        # Keep only recent history
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def _prepare_features(self) -> np.ndarray:
        """Prepare 404-feature vector matching training schema.

        Returns:
            Feature array (1, 404) for XGBoost model input.
            - 400 lag features: 100 timesteps x 4 directions, flattened
            - 4 cyclic time features: sin/cos hour, sin/cos day-of-week
        """
        # Build lag block: 100 timesteps × 4 directions = 400 features
        lag_block: list[float] = []
        for snapshot in self._history[-_WINDOW:]:
            for d in _DIRECTIONS:
                lag_block.append(snapshot.get(d, 0.0))

        # Zero-pad if insufficient history (same as training behavior)
        while len(lag_block) < _WINDOW * len(_DIRECTIONS):
            lag_block.insert(0, 0.0)

        # 4 cyclic time features (matching training)
        now = datetime.now()
        hour = now.hour + now.minute / 60
        dow = now.weekday()
        time_feats = [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]

        features = np.array(lag_block + time_feats, dtype=np.float32)
        return features.reshape(1, -1)

    def predict(
        self,
        current_densities: dict[str, float],
        prediction_seconds: int = _PREDICTION_HORIZON_SEC,
    ) -> DensityPrediction:
        """Predict traffic density ahead using trained XGBoost models.

        Args:
            current_densities: Current vehicle counts {N/S/E/W -> count}.
            prediction_seconds: Kept for API compatibility (model uses fixed 60s horizon).

        Returns:
            DensityPrediction with predicted densities and confidence scores.

        Raises:
            KeyError: If a lane model is missing.
            RuntimeError: If model prediction fails.
        """
        predicted = {}
        confidence = {}
        now = datetime.now().isoformat()

        # Prepare shared features (same for all direction models)
        features = self._prepare_features()

        # Confidence based on history completeness
        history_ratio = len(self._history) / _WINDOW
        base_conf = 0.6 + 0.35 * history_ratio  # 0.6 to 0.95

        for lane in _DIRECTIONS:
            pred_value = float(self._models[lane].predict(features)[0])

            # Clamp to reasonable bounds
            pred_value = max(0, min(50, pred_value))

            predicted[lane] = round(pred_value, 1)
            confidence[lane] = round(base_conf, 2)

        return DensityPrediction(
            predicted_densities=predicted,
            confidence_scores=confidence,
            prediction_horizon=_PREDICTION_HORIZON_SEC,
            timestamp=now,
        )

    def get_history(self, lane: str) -> list[float]:
        """Return historical density values for a lane.

        Args:
            lane: One of 'N', 'S', 'E', 'W'.

        Returns:
            List of recent density measurements for that lane.
        """
        return [snapshot.get(lane, 0.0) for snapshot in self._history]

    def history_ready(self) -> bool:
        """Check if sufficient history is collected for reliable predictions.

        Returns:
            True if history has at least WINDOW (100) samples.
        """
        return len(self._history) >= _WINDOW

    def clear_history(self) -> None:
        """Reset historical data (for testing)."""
        self._history = []
