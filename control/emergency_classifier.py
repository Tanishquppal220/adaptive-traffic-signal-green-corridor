from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    EMERGENCY_CONFIDENCE_THRESHOLD,
    EMERGENCY_TARGET_CLASS_IDS,
    EMERGENCY_THRESHOLD_PATH,
    EMERGENCY_VEHICLE_MODEL_PATH,
)
from control.schema import resolve_direction_from_point


class EmergencyClassifier:
    def __init__(
        self,
        model_path: str | Path = EMERGENCY_VEHICLE_MODEL_PATH,
        confidence_threshold: float = EMERGENCY_CONFIDENCE_THRESHOLD,
        target_class_ids: tuple[int, ...] = EMERGENCY_TARGET_CLASS_IDS,
        threshold_path: str | Path = EMERGENCY_THRESHOLD_PATH,
    ) -> None:
        self._model_path = Path(model_path)
        self._threshold_path = Path(threshold_path)
        self._confidence_threshold = float(confidence_threshold)
        self._target_class_ids = tuple(
            sorted({int(v) for v in target_class_ids}))
        self._threshold_source = "default"
        self._model: Any = None
        self._error: str | None = None
        self._load_calibrated_threshold()
        self._load_model()

    def _load_calibrated_threshold(self) -> None:
        if not self._threshold_path.exists():
            return

        try:
            with self._threshold_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return

        if not isinstance(payload, dict):
            return

        threshold_keys = (
            "threshold",
            "selected_threshold",
            "best_threshold",
            "f1_optimal_threshold",
        )
        candidate: float | None = None
        for key in threshold_keys:
            raw_value = payload.get(key)
            if raw_value is None:
                continue
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= parsed <= 1.0:
                candidate = parsed
                break

        if candidate is None:
            return

        self._confidence_threshold = candidate
        self._threshold_source = str(self._threshold_path)

    def _load_model(self) -> None:
        if not self._model_path.exists():
            self._error = f"model not found: {self._model_path}"
            return

        try:
            from ultralytics import YOLO
        except Exception:
            self._error = "ultralytics is not installed"
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
            "confidence_threshold": self._confidence_threshold,
            "target_class_ids": list(self._target_class_ids),
            "threshold_source": self._threshold_source,
        }

    def classify(
        self,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        if frame is None or frame.size == 0:
            return self._empty_result(mode="invalid-frame")
        if not self.is_loaded:
            return self._empty_result(mode="unavailable")

        result = self._model.predict(
            frame,
            conf=self._confidence_threshold,
            verbose=False,
        )[0]
        return self._classify_from_boxes(frame, result)

    def _classify_from_boxes(
        self,
        frame: np.ndarray,
        result: Any,
    ) -> dict[str, Any]:
        height, width = frame.shape[:2]
        predictions: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        names = self._resolve_names(result)
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return {
                "detected": False,
                "label": None,
                "confidence": 0.0,
                "direction": None,
                "predictions": predictions,
                "mode": "detection",
            }

        for idx in range(len(boxes)):
            cls_idx = int(boxes.cls[idx].item()
                          ) if boxes.cls is not None else -1
            confidence = (
                float(boxes.conf[idx].item()
                      ) if boxes.conf is not None else 0.0
            )
            xyxy = [float(v) for v in boxes.xyxy[idx].tolist()]
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            direction = resolve_direction_from_point(cx, cy, width, height)
            label = self._label_for_class(names, cls_idx)
            is_emergency = (
                self._is_target_class(cls_idx)
                and confidence >= self._confidence_threshold
            )

            item = {
                "class_id": cls_idx,
                "label": label,
                "confidence": confidence,
                "direction": direction,
                "bbox_xyxy": xyxy,
                "is_emergency": is_emergency,
            }
            predictions.append(item)

            if is_emergency and (best is None or confidence > best["confidence"]):
                best = item

        if best is None:
            return {
                "detected": False,
                "label": None,
                "confidence": 0.0,
                "direction": None,
                "predictions": predictions,
                "mode": "detection",
            }

        return {
            "detected": True,
            "label": best["label"],
            "confidence": float(best["confidence"]),
            "direction": best["direction"],
            "predictions": predictions,
            "mode": "detection",
        }

    def _resolve_names(self, result: Any) -> dict[int, str]:
        names = getattr(result, "names", None) or getattr(
            self._model, "names", {})
        if isinstance(names, dict):
            resolved: dict[int, str] = {}
            for key, value in names.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                resolved[idx] = str(value)
            return resolved
        if isinstance(names, (list, tuple)):
            return {idx: str(value) for idx, value in enumerate(names)}
        return {}

    def _label_for_class(self, names: dict[int, str], class_id: int) -> str:
        if class_id in names:
            return str(names[class_id])
        return str(class_id)

    def _is_target_class(self, class_id: int) -> bool:
        return int(class_id) in self._target_class_ids

    def _empty_result(self, mode: str) -> dict[str, Any]:
        return {
            "detected": False,
            "label": None,
            "confidence": 0.0,
            "direction": None,
            "predictions": [],
            "mode": mode,
        }
