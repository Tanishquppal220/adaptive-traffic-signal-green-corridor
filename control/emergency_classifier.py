from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from config import (
    EMERGENCY_CONFIDENCE_THRESHOLD,
    EMERGENCY_LABEL_KEYWORDS,
    EMERGENCY_VEHICLE_MODEL_PATH,
)
from control.schema import resolve_direction_from_point

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class EmergencyClassifier:
    def __init__(
        self,
        model_path: str | Path = EMERGENCY_VEHICLE_MODEL_PATH,
        confidence_threshold: float = EMERGENCY_CONFIDENCE_THRESHOLD,
        emergency_keywords: tuple[str, ...] = EMERGENCY_LABEL_KEYWORDS,
    ) -> None:
        self._model_path = Path(model_path)
        self._confidence_threshold = float(confidence_threshold)
        self._emergency_keywords = tuple(k.lower() for k in emergency_keywords)
        self._model: Any = None
        self._error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        if YOLO is None:
            self._error = "ultralytics is not installed"
            return
        if not self._model_path.exists():
            self._error = f"model not found: {self._model_path}"
            return
        try:
            self._model = YOLO(str(self._model_path))
            self._error = None
        except Exception as exc:
            self._error = str(exc)
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_path": str(self._model_path),
            "error": self._error,
        }

    def classify(
        self,
        frame: np.ndarray,
        boxes: list[dict[str, float | str]] | None = None,
    ) -> dict[str, Any]:
        if frame is None or frame.size == 0:
            return self._empty_result(mode="invalid-frame")
        if not self.is_loaded:
            return self._empty_result(mode="unavailable")

        height, width = frame.shape[:2]
        candidates = boxes or []

        if not candidates:
            full_box = {
                "x1": 0.0,
                "y1": 0.0,
                "x2": float(width),
                "y2": float(height),
                "direction": resolve_direction_from_point(
                    width / 2.0, height / 2.0, width, height
                ),
            }
            candidates = [full_box]

        best: dict[str, Any] | None = None
        predictions: list[dict[str, Any]] = []

        for candidate in candidates:
            x1 = int(max(0, min(width - 1, float(candidate["x1"]))))
            y1 = int(max(0, min(height - 1, float(candidate["y1"]))))
            x2 = int(max(1, min(width, float(candidate["x2"]))))
            y2 = int(max(1, min(height, float(candidate["y2"]))))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            result = self._model.predict(crop, verbose=False)[0]
            probs = getattr(result, "probs", None)
            if probs is None:
                continue

            if hasattr(probs, "top1"):
                top_idx = int(probs.top1)
                confidence = (
                    float(probs.top1conf.item())
                    if hasattr(probs.top1conf, "item")
                    else float(probs.top1conf)
                )
            else:
                data = (
                    probs.data.cpu().numpy()
                    if hasattr(probs.data, "cpu")
                    else np.asarray(probs.data)
                )
                top_idx = int(np.argmax(data))
                confidence = float(data[top_idx])

            names = result.names or getattr(self._model, "names", {})
            label = (
                str(names[top_idx])
                if isinstance(names, dict) and top_idx in names
                else str(top_idx)
            )
            direction = str(candidate.get("direction", "N"))

            item = {
                "label": label,
                "confidence": confidence,
                "direction": direction,
                "is_emergency": (
                    self._is_emergency(label)
                    and confidence >= self._confidence_threshold
                ),
            }
            predictions.append(item)

            if item["is_emergency"] and (best is None or confidence > best["confidence"]):
                best = item

        if best is None:
            return {
                "detected": False,
                "label": None,
                "confidence": 0.0,
                "direction": None,
                "predictions": predictions,
                "mode": "classification",
            }

        return {
            "detected": True,
            "label": best["label"],
            "confidence": float(best["confidence"]),
            "direction": best["direction"],
            "predictions": predictions,
            "mode": "classification",
        }

    def _is_emergency(self, label: str) -> bool:
        label_lower = label.lower()
        return any(keyword in label_lower for keyword in self._emergency_keywords)

    def _empty_result(self, mode: str) -> dict[str, Any]:
        return {
            "detected": False,
            "label": None,
            "confidence": 0.0,
            "direction": None,
            "predictions": [],
            "mode": mode,
        }
