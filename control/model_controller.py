from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import config as cfg

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


FAIRNESS_MODES = {"off", "soft", "hard"}


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
        self._last_emergency_active = False
        self._fairness_state = {
            lane: {
                "wait_seconds": 0.0,
                "missed_turns": 0,
            }
            for lane in LANE_KEYS
        }

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
        fairness_mode: str | None = None,
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
            fairness_mode=fairness_mode,
        )

    def decide_from_lane_counts(
        self,
        lane_counts: dict[str, int],
        frame: np.ndarray | None = None,
        detection: dict[str, Any] | None = None,
        emergency_override: dict[str, Any] | None = None,
        fairness_mode: str | None = None,
        locked_control: dict[str, Any] | None = None,
        update_fairness_state: bool = True,
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

        if locked_control is not None:
            baseline_decision = {
                **locked_control.get("baseline_decision", {}),
            }
            if not baseline_decision:
                baseline_decision = {
                    **locked_control.get("decision", {}),
                }
        else:
            baseline_decision = self._signal_controller.decide(lane_counts)

        decision = {
            **locked_control.get("decision", {})
        } if locked_control is not None else {**baseline_decision}
        fairness_mode_resolved = self._normalize_fairness_mode(fairness_mode)
        fairness_info = {
            "mode": fairness_mode_resolved,
            "applied": False,
            "reason": None,
            "selected_lane": self._lane_from_direction(
                str(baseline_decision.get("direction", "N"))
            ),
            "selected_duration": int(baseline_decision.get("duration", 0)),
            "baseline_lane": self._lane_from_direction(
                str(baseline_decision.get("direction", "N"))
            ),
            "baseline_duration": int(baseline_decision.get("duration", 0)),
            "lane_state": self._fairness_snapshot(),
        }

        emergency_detected = bool(emergency.get("detected", False))
        was_emergency_active = self._last_emergency_active

        if emergency_detected:
            emergency_direction = str(emergency.get("direction") or top_direction(
                lane_counts_to_direction_counts(lane_counts)
            ))
            emergency_lane = DIRECTION_TO_LANE.get(
                emergency_direction, "laneN")
            base_duration = int(baseline_decision.get("duration", 0))
            queue_in_emergency_lane = int(lane_counts.get(emergency_lane, 0))

            emergency_duration = min(
                base_duration + int(cfg.EMERGENCY_DURATION_BUFFER_SEC),
                int(cfg.EMERGENCY_DURATION_CAP_SEC),
            )

            if queue_in_emergency_lane <= int(cfg.EMERGENCY_LOW_QUEUE_THRESHOLD):
                emergency_duration = min(
                    emergency_duration,
                    int(cfg.EMERGENCY_LOW_QUEUE_MAX_SEC),
                )

            emergency_duration = max(1, int(emergency_duration))
            decision = {
                **{lane: 0 for lane in LANE_KEYS},
                emergency_lane: emergency_duration,
                "direction": emergency_direction,
                "duration": emergency_duration,
                "action": decision.get("action", 0),
                "mode": "emergency-override",
            }

            emergency = {
                **emergency,
                "status": "active",
                "release_reason": None,
                "baseline_duration": base_duration,
                "adjusted_duration": emergency_duration,
                "emergency_lane": emergency_lane,
            }
            fairness_info["reason"] = "bypassed_during_emergency"
            fairness_info["selected_lane"] = emergency_lane
            fairness_info["selected_duration"] = emergency_duration
        else:
            if locked_control is not None:
                fairness_info = {
                    **locked_control.get("fairness", {}),
                    "mode": fairness_mode_resolved,
                    "applied": False,
                    "reason": "cycle_lock_reuse",
                    "selected_lane": locked_control.get("selected_lane")
                    or self._lane_from_direction(
                        str(locked_control.get(
                            "decision", {}).get("direction", "N"))
                    ),
                    "selected_duration": int(locked_control.get("decision", {}).get("duration", 0)),
                    "baseline_lane": self._lane_from_direction(str(baseline_decision.get("direction", "N"))),
                    "baseline_duration": int(baseline_decision.get("duration", 0)),
                }
            else:
                decision, fairness_info = self._apply_fairness_policy(
                    lane_counts=lane_counts,
                    baseline_decision=baseline_decision,
                    fairness_mode=fairness_mode_resolved,
                )
            emergency = {
                **emergency,
                "status": "cleared" if was_emergency_active else "inactive",
                "release_reason": "ambulance_not_detected" if was_emergency_active else None,
                "baseline_duration": int(baseline_decision.get("duration", 0)),
                "adjusted_duration": None,
                "emergency_lane": None,
            }

        if update_fairness_state:
            self._update_fairness_state(
                lane_counts=lane_counts,
                served_lane=fairness_info.get("selected_lane"),
            )

        self._last_emergency_active = emergency_detected

        return {
            **decision,
            "lane_counts": lane_counts,
            "detection": detection_payload,
            "density": density,
            "emergency": emergency,
            "baseline_decision": baseline_decision,
            "fairness": {
                **fairness_info,
                "lane_state": self._fairness_snapshot(),
            },
            "diagnostics": {
                "active_models": {
                    "traffic_detector": self._traffic_detector.is_loaded,
                    "emergency_classifier": self._emergency_classifier.is_loaded,
                    "density_predictor": self._density_predictor.is_loaded,
                    "dqn_controller": self._signal_controller.mode == "dqn",
                },
                "controller_mode": decision.get("mode", self.mode),
                "fairness_mode": fairness_mode_resolved,
                "fairness_applied": bool(fairness_info.get("applied", False)),
            },
        }

    def _normalize_fairness_mode(self, fairness_mode: str | None) -> str:
        mode = str(fairness_mode or cfg.FAIRNESS_DEFAULT_MODE).strip().lower()
        if mode not in FAIRNESS_MODES:
            return str(cfg.FAIRNESS_DEFAULT_MODE)
        return mode

    def _lane_from_direction(self, direction: str) -> str:
        return DIRECTION_TO_LANE.get(direction, "laneN")

    def _fairness_snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            lane: {
                "wait_seconds": float(self._fairness_state[lane]["wait_seconds"]),
                "missed_turns": int(self._fairness_state[lane]["missed_turns"]),
            }
            for lane in LANE_KEYS
        }

    def _apply_fairness_policy(
        self,
        lane_counts: dict[str, int],
        baseline_decision: dict[str, Any],
        fairness_mode: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        baseline_lane = self._lane_from_direction(
            str(baseline_decision.get("direction", "N"))
        )
        baseline_duration = int(baseline_decision.get("duration", 0))

        info = {
            "mode": fairness_mode,
            "applied": False,
            "reason": None,
            "selected_lane": baseline_lane,
            "selected_duration": baseline_duration,
            "baseline_lane": baseline_lane,
            "baseline_duration": baseline_duration,
            "lane_state": self._fairness_snapshot(),
        }

        if fairness_mode == "off":
            info["reason"] = "fairness_disabled"
            return {**baseline_decision}, info

        eligible = [lane for lane in LANE_KEYS if lane_counts.get(lane, 0) > 0]
        if not eligible:
            info["reason"] = "no_queued_vehicles"
            return {**baseline_decision}, info

        wait_threshold = float(cfg.FAIRNESS_WAIT_THRESHOLD_SEC)
        missed_threshold = int(cfg.FAIRNESS_MISSED_TURNS_THRESHOLD)

        hard_candidates = [
            lane for lane in eligible
            if self._fairness_state[lane]["wait_seconds"] >= wait_threshold
            and self._fairness_state[lane]["missed_turns"] >= missed_threshold
        ]

        if fairness_mode == "hard" and hard_candidates:
            forced_lane = max(
                hard_candidates,
                key=lambda lane: (
                    self._fairness_state[lane]["wait_seconds"],
                    self._fairness_state[lane]["missed_turns"],
                    lane_counts.get(lane, 0),
                ),
            )
            if forced_lane != baseline_lane:
                direction = forced_lane.replace("lane", "")
                decision = {
                    **{lane: 0 for lane in LANE_KEYS},
                    forced_lane: baseline_duration,
                    "direction": direction,
                    "duration": baseline_duration,
                    "action": baseline_decision.get("action", 0),
                    "mode": "fairness-hard-override",
                }
                info.update({
                    "applied": True,
                    "reason": "hard_threshold_breach",
                    "selected_lane": forced_lane,
                    "selected_duration": baseline_duration,
                })
                return decision, info

            info["reason"] = "hard_threshold_breach_baseline_already_serves"
            return {**baseline_decision}, info

        scores: dict[str, float] = {}
        for lane in eligible:
            wait_ratio = self._fairness_state[lane]["wait_seconds"] / \
                max(wait_threshold, 1.0)
            missed_ratio = self._fairness_state[lane]["missed_turns"] / max(
                float(missed_threshold), 1.0)
            queue_score = float(lane_counts.get(lane, 0))
            scores[lane] = (
                queue_score
                + wait_ratio * float(cfg.FAIRNESS_SOFT_WAIT_WEIGHT)
                + missed_ratio * float(cfg.FAIRNESS_SOFT_MISSED_WEIGHT)
            )

        best_lane = max(scores, key=scores.get)
        baseline_score = scores.get(baseline_lane, -1.0)
        best_score = scores.get(best_lane, -1.0)
        margin = float(cfg.FAIRNESS_SOFT_OVERRIDE_MARGIN)

        if fairness_mode == "soft" and best_lane != baseline_lane and (best_score - baseline_score) >= margin:
            direction = best_lane.replace("lane", "")
            decision = {
                **{lane: 0 for lane in LANE_KEYS},
                best_lane: baseline_duration,
                "direction": direction,
                "duration": baseline_duration,
                "action": baseline_decision.get("action", 0),
                "mode": "fairness-soft-override",
            }
            info.update({
                "applied": True,
                "reason": "soft_priority_margin",
                "selected_lane": best_lane,
                "selected_duration": baseline_duration,
            })
            return decision, info

        info["reason"] = "baseline_retained"
        return {**baseline_decision}, info

    def _update_fairness_state(
        self,
        lane_counts: dict[str, int],
        served_lane: str | None,
    ) -> None:
        cycle_increment = float(cfg.INFERENCE_INTERVAL)
        for lane in LANE_KEYS:
            queue_count = int(lane_counts.get(lane, 0))
            if lane == served_lane:
                self._fairness_state[lane]["wait_seconds"] = 0.0
                self._fairness_state[lane]["missed_turns"] = 0
            elif queue_count > 0:
                self._fairness_state[lane]["wait_seconds"] += cycle_increment
                self._fairness_state[lane]["missed_turns"] += 1
            else:
                self._fairness_state[lane]["wait_seconds"] = 0.0
                self._fairness_state[lane]["missed_turns"] = 0

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

    def reset_runtime_state(self) -> None:
        self._last_lane_counts = {k: 0 for k in LANE_KEYS}
        self._last_emergency_active = False
        self._fairness_state = {
            lane: {
                "wait_seconds": 0.0,
                "missed_turns": 0,
            }
            for lane in LANE_KEYS
        }


__all__ = ["ModelController", "DIRECTIONS", "LANE_KEYS"]
