from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    DENSITY_HISTORY_WINDOW,
    DENSITY_MAX_CLIP,
    DENSITY_PREDICTION_HORIZON_SEC,
    DENSITY_PREDICTOR_MODELS,
)
from control.schema import lane_counts_to_direction_counts

try:
    import xgboost as xgb
except Exception:
    xgb = None


class DensityPredictor:
    def __init__(
        self,
        model_paths: dict[str, Path] = DENSITY_PREDICTOR_MODELS,
        history_window: int = DENSITY_HISTORY_WINDOW,
    ) -> None:
        self._model_paths = {k: Path(v) for k, v in model_paths.items()}
        self._window = int(history_window)
        self._history: deque[dict[str, float]] = deque(maxlen=self._window)
        self._models: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._load_models()

    def _load_models(self) -> None:
        if xgb is None:
            self._errors = {
                direction: "xgboost is not installed" for direction in self._model_paths
            }
            return

        for direction, model_path in self._model_paths.items():
            if not model_path.exists():
                self._errors[direction] = f"model not found: {model_path}"
                continue
            try:
                booster = xgb.Booster()
                booster.load_model(str(model_path))
                self._models[direction] = booster
            except Exception as exc:
                self._errors[direction] = str(exc)

    @property
    def is_loaded(self) -> bool:
        return len(self._models) == 4

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "loaded_models": sorted(self._models.keys()),
            "errors": self._errors,
            "history_size": len(self._history),
            "window": self._window,
        }

    def update_history(self, lane_counts: dict[str, int]) -> None:
        direction_counts = lane_counts_to_direction_counts(lane_counts)
        self._history.append({k: float(v)
                             for k, v in direction_counts.items()})

    def predict(self, lane_counts: dict[str, int] | None = None) -> dict[str, Any]:
        if lane_counts is not None:
            self.update_history(lane_counts)

        if not self._history:
            result = {
                "predictions": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
                "mode": "empty-history",
                "horizon_sec": DENSITY_PREDICTION_HORIZON_SEC,
            }
            print("DensityPredictor.predict:", result)
            return result

        if not self._models:
            latest = self._history[-1]
            result = {
                "predictions": {d: float(latest.get(d, 0.0)) for d in ("N", "S", "E", "W")},
                "mode": "heuristic",
                "horizon_sec": DENSITY_PREDICTION_HORIZON_SEC,
            }
            print("DensityPredictor.predict:", result)
            return result

        features = self._prepare_features()
        dmatrix = xgb.DMatrix(features)

        predictions: dict[str, float] = {}
        for direction in ("N", "S", "E", "W"):
            model = self._models.get(direction)
            if model is None:
                predictions[direction] = float(
                    self._history[-1].get(direction, 0.0))
                continue
            pred = float(model.predict(dmatrix)[0])
            predictions[direction] = float(
                np.clip(pred, 0.0, DENSITY_MAX_CLIP))

        result = {
            "predictions": predictions,
            "mode": "xgboost",
            "horizon_sec": DENSITY_PREDICTION_HORIZON_SEC,
        }
        print("DensityPredictor.predict:", result)
        return result

    def _prepare_features(self) -> np.ndarray:
        lag_block: list[float] = []
        for snapshot in self._history:
            lag_block.extend(
                [
                    float(snapshot.get("N", 0.0)),
                    float(snapshot.get("S", 0.0)),
                    float(snapshot.get("E", 0.0)),
                    float(snapshot.get("W", 0.0)),
                ]
            )

        needed = self._window * 4
        if len(lag_block) < needed:
            lag_block = [0.0] * (needed - len(lag_block)) + lag_block
        elif len(lag_block) > needed:
            lag_block = lag_block[-needed:]

        now = datetime.now()
        hour = now.hour + now.minute / 60.0
        dow = now.weekday()
        temporal = [
            float(np.sin(2 * np.pi * hour / 24.0)),
            float(np.cos(2 * np.pi * hour / 24.0)),
            float(np.sin(2 * np.pi * dow / 7.0)),
            float(np.cos(2 * np.pi * dow / 7.0)),
        ]

        features = np.asarray(lag_block + temporal, dtype=np.float32)
        return features.reshape(1, -1)
