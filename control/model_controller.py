from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from config import DIRECTION_TO_LANE, DIRECTIONS, LANE_KEYS
from control.density_predictor import DensityPredictor
from control.emergency_classifier import EmergencyClassifier
from control.schema import (
    lane_counts_to_direction_counts,
    normalize_lane_counts,
    top_direction,
)
from control.signal_controller import SignalController
from control.traffic_detector import TrafficDetector


class ModelController:
    """End-to-end controller that orchestrates all runtime models.

    Pipeline:
            frame -> traffic detection -> emergency classification ->
            density prediction -> DQN/proportional decision
    """

    def __init__(
        self,
        dqn_weights_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self._traffic_detector = TrafficDetector()
        self._emergency_classifier = EmergencyClassifier()
        self._density_predictor = DensityPredictor()
        self._signal_controller = SignalController(
            weights_path=dqn_weights_path or Path(
                "models") / "dqn_signal_optimizer.pt",
            device=device,
        )
        self._last_lane_counts = {k: 0 for k in LANE_KEYS}

    @property
    def mode(self) -> str:
        return self._signal_controller.mode

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "traffic_detector": self._traffic_detector.status(),
            "emergency_classifier": self._emergency_classifier.status(),
            "density_predictor": self._density_predictor.status(),
            "signal_controller": self._signal_controller.status(),
        }

    def decide_from_frame(self, frame: np.ndarray) -> dict[str, Any]:
        detection = self._traffic_detector.detect(frame)
        lane_counts = normalize_lane_counts(detection["lane_counts"])
        return self.decide_from_lane_counts(
            lane_counts=lane_counts,
            frame=frame,
            detection=detection,
        )

    def decide_from_lane_frames(
        self,
        lane_frames: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Run one merged decision cycle from 4 per-lane uploaded images."""
        lane_counts = {lane: 0 for lane in LANE_KEYS}
        per_lane_detection: dict[str, Any] = {}
        emergency_candidates: list[dict[str, Any]] = []

        for lane in LANE_KEYS:
            frame = lane_frames.get(lane)
            if frame is None:
                continue

            detection = self._traffic_detector.detect(frame)
            lane_count = int(detection.get("total", 0))
            lane_counts[lane] = max(0, lane_count)
            per_lane_detection[lane] = detection

            emergency = self._emergency_classifier.classify(
                frame,
                boxes=detection.get("boxes", []),
            )
            if emergency.get("detected"):
                candidate = {
                    **emergency,
                    "lane": lane,
                    "direction": lane.replace("lane", ""),
                }
                emergency_candidates.append(candidate)

        best_emergency: dict[str, Any]
        if emergency_candidates:
            best_emergency = max(
                emergency_candidates,
                key=lambda item: float(item.get("confidence", 0.0)),
            )
        else:
            best_emergency = {
                "detected": False,
                "label": None,
                "confidence": 0.0,
                "direction": None,
                "predictions": [],
                "mode": "lane-uploads",
            }

        merged_detection = {
            "lane_counts": lane_counts,
            "direction_counts": lane_counts_to_direction_counts(lane_counts),
            "boxes": [],
            "total": int(sum(lane_counts.values())),
            "mode": "lane-uploads",
            "per_lane": per_lane_detection,
        }

        return self.decide_from_lane_counts(
            lane_counts=lane_counts,
            frame=None,
            detection=merged_detection,
            emergency_override=best_emergency,
        )

    def decide_from_lane_counts(
        self,
        lane_counts: dict[str, int],
        frame: np.ndarray | None = None,
        detection: dict[str, Any] | None = None,
        emergency_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lane_counts = normalize_lane_counts(lane_counts)
        self._last_lane_counts = lane_counts

        detection_payload = detection or {
            "lane_counts": lane_counts,
            "direction_counts": lane_counts_to_direction_counts(lane_counts),
            "boxes": [],
            "total": int(sum(lane_counts.values())),
            "mode": "external",
        }

        if emergency_override is not None:
            emergency = emergency_override
        elif frame is not None:
            emergency = self._emergency_classifier.classify(
                frame,
                boxes=detection_payload.get("boxes", []),
            )
        else:
            emergency = {
                "detected": False,
                "label": None,
                "confidence": 0.0,
                "direction": None,
                "predictions": [],
                "mode": "no-frame",
            }

        density = self._density_predictor.predict(lane_counts)
        decision = self._signal_controller.decide(lane_counts)

        if emergency.get("detected"):
            emergency_direction = str(emergency.get("direction") or top_direction(
                lane_counts_to_direction_counts(lane_counts)
            ))
            emergency_lane = DIRECTION_TO_LANE.get(
                emergency_direction, "laneN")
            emergency_duration = max(decision["duration"], 45)
            decision = {
                **{lane: 0 for lane in LANE_KEYS},
                emergency_lane: emergency_duration,
                "direction": emergency_direction,
                "duration": emergency_duration,
                "action": decision.get("action", 0),
                "mode": "emergency-override",
            }

        return {
            **decision,
            "lane_counts": lane_counts,
            "detection": detection_payload,
            "density": density,
            "emergency": emergency,
            "diagnostics": {
                "active_models": {
                    "traffic_detector": self._traffic_detector.is_loaded,
                    "emergency_classifier": self._emergency_classifier.is_loaded,
                    "density_predictor": self._density_predictor.is_loaded,
                    "dqn_controller": self._signal_controller.mode == "dqn",
                },
                "controller_mode": decision.get("mode", self.mode),
            },
        }

    def online_update(
        self,
        action_taken: int,
        reward: float,
        next_lane_counts: dict[str, int],
        done: bool = False,
    ) -> None:
        self._signal_controller.online_update(
            prev_lane_counts=self._last_lane_counts,
            action_taken=action_taken,
            reward=reward,
            next_lane_counts=normalize_lane_counts(next_lane_counts),
            done=done,
        )


__all__ = ["ModelController", "DIRECTIONS", "LANE_KEYS"]
