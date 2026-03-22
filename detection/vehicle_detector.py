"""YOLO-based vehicle detector with a clean, framework-agnostic interface."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CONFIDENCE_THRESHOLD,
    EMERGENCY_VEHICLE_MODEL_PATH,
    TRAFFIC_DETECTION_MODEL_PATH,
)


@dataclass(frozen=True)
class DetectionResult:
    """Immutable container for a single frame's detection output.

    Attributes:
        annotated_frame: The original frame with bounding-box overlays drawn.
        vehicle_count:   Number of vehicles detected in this frame.
        boxes:           Raw ultralytics Boxes object for downstream consumers
                         (density predictor, Q-learning agent, etc.).
    """

    annotated_frame: np.ndarray
    vehicle_count: int
    boxes: Any = field(repr=False)


class VehicleDetector:
    """Wraps a YOLO model and provides a simple *detect → result* interface.

    Parameters:
        model_path: Path to a ``.pt`` weights file.  Defaults to the
                    centralized ``TRAFFIC_DETECTION_MODEL_PATH`` from config.
        confidence: Minimum confidence threshold for detections.
        load_emergency: Whether to load the emergency vehicle classifier model.
    """

    def __init__(
        self,
        model_path: pathlib.Path = TRAFFIC_DETECTION_MODEL_PATH,
        confidence: float = CONFIDENCE_THRESHOLD,
        load_emergency: bool = False,
    ) -> None:
        self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._emergency_model: YOLO | None = None
        if load_emergency:
            self._emergency_model = YOLO(str(EMERGENCY_VEHICLE_MODEL_PATH))

    def detect(self, frame: np.ndarray, *, confidence: float | None = None) -> DetectionResult:
        """Run inference on a single BGR frame.

        Args:
            frame:      A BGR ``numpy`` image (e.g. from OpenCV or a decoded upload).
            confidence: Override the default confidence threshold for this call.

        Returns:
            A ``DetectionResult`` containing the annotated frame, vehicle count,
            and raw bounding-box data.
        """
        conf = confidence if confidence is not None else self._confidence
        results = self._model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        count = len(results[0].boxes)
        return DetectionResult(
            annotated_frame=annotated,
            vehicle_count=count,
            boxes=results[0].boxes,
        )

    @staticmethod
    def draw_vehicle_count(frame: np.ndarray, count: int) -> np.ndarray:
        """Overlay a vehicle-count label on the top-left corner of *frame*.

        Modifies *frame* in-place and also returns it for convenience.
        """
        cv2.putText(
            frame,
            f"Vehicles: {count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        return frame

    def classify_emergency_vehicle(
        self, frame: np.ndarray
    ) -> dict[str, Any]:
        """Classify a vehicle image using the emergency vehicle classifier.

        Args:
            frame: A BGR vehicle image (e.g. cropped vehicle or full image).

        Returns:
            A dictionary with classification results:
            - 'class_name': Predicted class (ambulance/fire-truck/police/normal)
            - 'confidence': Confidence score (0.0 to 1.0)
            - 'annotated_frame': Frame with label overlay
            - 'all_classes': Dict of all class confidences
        """
        if self._emergency_model is None:
            return {
                "error": "Emergency model not loaded",
                "class_name": None,
                "confidence": 0.0,
                "annotated_frame": frame,
            }

        results = self._emergency_model.predict(frame, verbose=False)
        result = results[0]
        probs = result.probs

        class_names = probs.top5c if hasattr(probs, "top5c") else []
        class_confidences = (
            probs.top5conf if hasattr(probs, "top5conf") else []
        )
        predicted_class_id = probs.top1 if hasattr(probs, "top1") else 0
        confidence = float(probs.top1conf if hasattr(
            probs, "top1conf") else 0.0)
        class_name = (
            result.names.get(predicted_class_id, "unknown")
            if result.names
            else "unknown"
        )

        # Overlay prediction on frame
        annotated = frame.copy()
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if class_name == "normal" else (0, 0, 255),
            2,
        )

        return {
            "class_name": class_name,
            "confidence": confidence,
            "annotated_frame": annotated,
            "all_classes": {
                result.names.get(cid, "unknown"): float(conf)
                for cid, conf in zip(class_names, class_confidences)
            },
        }
