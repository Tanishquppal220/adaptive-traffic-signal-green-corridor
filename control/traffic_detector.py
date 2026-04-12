from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from config import CONFIDENCE_THRESHOLD, TRAFFIC_DETECTION_MODEL_PATH
from control.schema import direction_counts_to_lane_counts, resolve_direction_from_point

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class TrafficDetector:
    def __init__(
        self,
        model_path: str | Path = TRAFFIC_DETECTION_MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._model_path = Path(model_path)
        self._confidence_threshold = float(confidence_threshold)
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

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        if frame is None or frame.size == 0:
            result = {
                "lane_counts": direction_counts_to_lane_counts({"N": 0, "S": 0, "E": 0, "W": 0}),
                "direction_counts": {"N": 0, "S": 0, "E": 0, "W": 0},
                "boxes": [],
                "total": 0,
                "mode": "invalid-frame",
            }
            print("TrafficDetector.detect:", result)
            return result

        if not self.is_loaded:
            result = {
                "lane_counts": direction_counts_to_lane_counts({"N": 0, "S": 0, "E": 0, "W": 0}),
                "direction_counts": {"N": 0, "S": 0, "E": 0, "W": 0},
                "boxes": [],
                "total": 0,
                "mode": "unavailable",
            }
            print("TrafficDetector.detect:", result)
            return result

        height, width = frame.shape[:2]
        direction_counts = {"N": 0, "S": 0, "E": 0, "W": 0}
        decoded_boxes: list[dict[str, float | str]] = []

        results = self._model.predict(
            frame, conf=self._confidence_threshold, verbose=False)
        boxes = results[0].boxes if results else None
        if boxes is not None:
            for idx in range(len(boxes)):
                conf = float(boxes.conf[idx].item()
                             ) if boxes.conf is not None else 0.0
                xyxy = boxes.xyxy[idx].tolist()
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                direction = resolve_direction_from_point(cx, cy, width, height)
                direction_counts[direction] += 1
                decoded_boxes.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": conf,
                        "direction": direction,
                    }
                )

        result = {
            "lane_counts": direction_counts_to_lane_counts(direction_counts),
            "direction_counts": direction_counts,
            # "boxes": decoded_boxes,
            "total": int(sum(direction_counts.values())),
            "mode": "yolo",
        }
        print("TrafficDetector.detect:", result)
        return result
