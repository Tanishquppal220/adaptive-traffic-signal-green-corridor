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
from control.siren_detector import SirenDetector
from control.traffic_detector import TrafficDetector
from training.DQN.environment import encode_action

FAIRNESS_MODES = {"off", "soft", "hard"}


class ModelController:
    """End-to-end controller that orchestrates all runtime models.

    Pipeline:
            frame(s) -> traffic detection -> emergency classification ->
            siren detection -> density prediction -> DQN/proportional decision
    """

    def __init__(
        self,
        dqn_weights_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self._traffic_detector = TrafficDetector()
        self._emergency_classifier = EmergencyClassifier()
        self._siren_detector = SirenDetector()
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
        self._predictive_ema_counts = {lane: 0.0 for lane in LANE_KEYS}
        self._predictive_last_selected_lane: str | None = None
        self._predictive_hold_cycles = 0
        self._cached_emergency_override: dict[str, Any] | None = None
        self._cached_siren_detection: dict[str, Any] | None = None

    @property
    def mode(self) -> str:
        return self._signal_controller.mode

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "traffic_detector": self._traffic_detector.status(),
            "emergency_classifier": self._emergency_classifier.status(),
            "siren_detector": self._siren_detector.status(),
            "density_predictor": self._density_predictor.status(),
            "signal_controller": self._signal_controller.status(),
            "cached_cycle_context": {
                "has_emergency": self._cached_emergency_override is not None,
                "has_siren": self._cached_siren_detection is not None,
            },
            "predictive_control": {
                "enabled": bool(cfg.PREDICTIVE_CONTROL_ENABLED),
                "last_selected_lane": self._predictive_last_selected_lane,
                "hold_cycles": int(self._predictive_hold_cycles),
                "ema_lane_counts": {
                    lane: round(float(self._predictive_ema_counts[lane]), 3) for lane in LANE_KEYS
                },
            },
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
        audio_bytes: bytes | None = None,
        fairness_mode: str | None = None,
        current_active_lane: str | None = None,
        cache_cycle_context: bool = True,
    ) -> dict[str, Any]:
        """Run one merged decision cycle from 4 per-lane uploaded images."""
        lane_counts = {lane: 0 for lane in LANE_KEYS}
        per_lane_detection: dict[str, Any] = {}
        emergency_candidates: list[dict[str, Any]] = []
        emergency_lane_counts = {lane: 0 for lane in LANE_KEYS}

        for lane in LANE_KEYS:
            frame = lane_frames.get(lane)
            if frame is None:
                continue

            detection = self._traffic_detector.detect(frame)
            lane_count = int(detection.get("total", 0))
            lane_counts[lane] = max(0, lane_count)
            per_lane_detection[lane] = detection

            emergency = self._emergency_classifier.classify(frame)
            emergency_count = int(
                sum(1 for item in emergency.get("predictions", []) if item.get("is_emergency"))
            )
            emergency_lane_counts[lane] = max(0, emergency_count)
            if emergency.get("detected"):
                candidate = {
                    **emergency,
                    "lane": lane,
                    "emergency_lane": lane,
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

        best_emergency["lane_emergency_counts"] = {
            lane: int(emergency_lane_counts.get(lane, 0)) for lane in LANE_KEYS
        }

        siren = self._siren_detector.detect(audio_bytes)

        merged_detection = {
            "lane_counts": lane_counts,
            "direction_counts": lane_counts_to_direction_counts(lane_counts),
            "boxes": [],
            "total": int(sum(lane_counts.values())),
            "mode": "lane-uploads",
            "per_lane": per_lane_detection,
            "emergency_lane_counts": {
                lane: int(emergency_lane_counts.get(lane, 0)) for lane in LANE_KEYS
            },
        }

        return self.decide_from_lane_counts(
            lane_counts=lane_counts,
            frame=None,
            detection=merged_detection,
            emergency_override=best_emergency,
            siren_override=siren,
            fairness_mode=fairness_mode,
            current_active_lane=current_active_lane,
            cache_cycle_context=cache_cycle_context,
        )

    def has_cached_cycle_context(self) -> bool:
        return (
            self._cached_emergency_override is not None
            and self._cached_siren_detection is not None
        )

    def decide_next_cycle_from_lane_counts(
        self,
        lane_counts: dict[str, int],
        current_active_lane: str | None = None,
        fairness_mode: str | None = None,
    ) -> dict[str, Any]:
        if not self.has_cached_cycle_context():
            raise ValueError("No cached cycle context. Run /api/run_cycle first.")

        detection_payload = {
            "lane_counts": normalize_lane_counts(lane_counts),
            "direction_counts": lane_counts_to_direction_counts(lane_counts),
            "boxes": [],
            "total": int(sum(max(0, int(v)) for v in lane_counts.values())),
            "mode": "iterative-cycle",
        }

        return self.decide_from_lane_counts(
            lane_counts=lane_counts,
            frame=None,
            detection=detection_payload,
            emergency_override={**(self._cached_emergency_override or {})},
            siren_override={**(self._cached_siren_detection or {})},
            fairness_mode=fairness_mode,
            current_active_lane=current_active_lane,
            cache_cycle_context=False,
        )

    def decide_from_lane_counts(
        self,
        lane_counts: dict[str, int],
        frame: np.ndarray | None = None,
        detection: dict[str, Any] | None = None,
        emergency_override: dict[str, Any] | None = None,
        siren_override: dict[str, Any] | None = None,
        fairness_mode: str | None = None,
        current_active_lane: str | None = None,
        locked_control: dict[str, Any] | None = None,
        update_fairness_state: bool = True,
        cache_cycle_context: bool = True,
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
            emergency = self._emergency_classifier.classify(frame)
        else:
            emergency = {
                "detected": False,
                "label": None,
                "confidence": 0.0,
                "direction": None,
                "predictions": [],
                "mode": "no-frame",
            }

        if siren_override is not None:
            siren = siren_override
        else:
            siren = self._siren_detector.detect(None)

        if cache_cycle_context:
            self._cached_emergency_override = {**emergency}
            self._cached_siren_detection = {**siren}

        density = self._density_predictor.predict(lane_counts)
        emergency_visual_detected = bool(emergency.get("detected", False))
        siren_detected = bool(siren.get("detected", False))
        emergency_detected = emergency_visual_detected and siren_detected
        was_emergency_active = self._last_emergency_active

        predictive_control = self._build_predictive_control_inputs(
            lane_counts=lane_counts,
            density=density,
            emergency_detected=emergency_detected,
            cycle_locked=locked_control is not None,
        )

        if locked_control is not None:
            baseline_decision = {
                **locked_control.get("baseline_decision", {}),
            }
            if not baseline_decision:
                baseline_decision = {
                    **locked_control.get("decision", {}),
                }
        else:
            baseline_decision = self._signal_controller.decide(
                predictive_control["effective_lane_counts"]
            )
            baseline_decision, predictive_control = self._apply_predictive_stability_guard(
                baseline_decision=baseline_decision,
                predictive_control=predictive_control,
            )
            self._update_predictive_selection(
                self._lane_from_direction(
                    str(baseline_decision.get("direction", "N"))),
                emergency_detected=emergency_detected,
            )

        decision = (
            {**locked_control.get("decision", {})}
            if locked_control is not None
            else {**baseline_decision}
        )
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

        if emergency_detected:
            emergency_lane = str(emergency.get("emergency_lane") or "")
            if emergency_lane not in DIRECTION_TO_LANE.values():
                emergency_direction = str(
                    emergency.get("direction")
                    or top_direction(lane_counts_to_direction_counts(lane_counts))
                )
                emergency_lane = DIRECTION_TO_LANE.get(
                    emergency_direction, "laneN")
            emergency_direction = emergency_lane.replace("lane", "")
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
            current_lane = str(current_active_lane or "")
            preemption_buffer_sec = 0
            if current_lane in LANE_KEYS and current_lane != emergency_lane:
                preemption_buffer_sec = int(cfg.EMERGENCY_PREEMPTION_BUFFER_SEC)

            decision = {
                **{lane: 0 for lane in LANE_KEYS},
                emergency_lane: emergency_duration,
                "direction": emergency_direction,
                "duration": emergency_duration,
                "action": self._encode_decision_action(emergency_direction, emergency_duration),
                "mode": "emergency-override",
            }

            emergency = {
                **emergency,
                "status": "active",
                "release_reason": None,
                "baseline_duration": base_duration,
                "adjusted_duration": emergency_duration,
                "emergency_lane": emergency_lane,
                "gated_by_siren": True,
                "visual_detected": emergency_visual_detected,
                "siren_detected": siren_detected,
                "preemption_buffer_sec": preemption_buffer_sec,
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
                    "baseline_lane": self._lane_from_direction(
                        str(baseline_decision.get("direction", "N"))
                    ),
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
                "release_reason": (
                    "siren_not_detected"
                    if emergency_visual_detected and not siren_detected
                    else ("ambulance_not_detected" if was_emergency_active else None)
                ),
                "baseline_duration": int(baseline_decision.get("duration", 0)),
                "adjusted_duration": None,
                "emergency_lane": None,
                "gated_by_siren": emergency_visual_detected and not siren_detected,
                "visual_detected": emergency_visual_detected,
                "siren_detected": siren_detected,
                "preemption_buffer_sec": 0,
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
            "siren": siren,
            "baseline_decision": baseline_decision,
            "fairness": {
                **fairness_info,
                "lane_state": self._fairness_snapshot(),
            },
            "diagnostics": {
                "active_models": {
                    "traffic_detector": self._traffic_detector.is_loaded,
                    "emergency_classifier": self._emergency_classifier.is_loaded,
                    "siren_detector": self._siren_detector.is_loaded,
                    "density_predictor": self._density_predictor.is_loaded,
                    "dqn_controller": self._signal_controller.mode == "dqn",
                },
                "controller_mode": decision.get("mode", self.mode),
                "fairness_mode": fairness_mode_resolved,
                "fairness_applied": bool(fairness_info.get("applied", False)),
                "emergency_visual_detected": emergency_visual_detected,
                "siren_detected": siren_detected,
                "predictive_control": predictive_control,
            },
        }

    def _build_predictive_control_inputs(
        self,
        lane_counts: dict[str, int],
        density: dict[str, Any],
        emergency_detected: bool,
        cycle_locked: bool,
    ) -> dict[str, Any]:
        raw_counts = {lane: int(lane_counts.get(lane, 0))
                      for lane in LANE_KEYS}
        payload: dict[str, Any] = {
            "enabled": bool(cfg.PREDICTIVE_CONTROL_ENABLED),
            "applied": False,
            "reason": "predictive_disabled",
            "raw_lane_counts": {lane: float(raw_counts[lane]) for lane in LANE_KEYS},
            "smoothed_lane_counts": {lane: float(raw_counts[lane]) for lane in LANE_KEYS},
            "forecast_lane_counts": {lane: float(raw_counts[lane]) for lane in LANE_KEYS},
            "effective_scores": {lane: float(raw_counts[lane]) for lane in LANE_KEYS},
            "effective_lane_counts": raw_counts,
            "surge_detected_by_lane": {lane: False for lane in LANE_KEYS},
            "switch_penalty_applied": False,
            "switch_blocked": False,
            "selected_lane": None,
            "selected_lane_gain": 0.0,
            "hold_cycles": int(self._predictive_hold_cycles),
            "last_selected_lane": self._predictive_last_selected_lane,
        }

        if not bool(cfg.PREDICTIVE_CONTROL_ENABLED):
            return payload
        if cycle_locked:
            payload["reason"] = "cycle_lock_reuse"
            return payload
        if emergency_detected:
            payload["reason"] = "bypassed_during_emergency"
            return payload

        predictions = density.get(
            "predictions", {}) if isinstance(density, dict) else {}
        alpha_current = float(cfg.PREDICTIVE_ALPHA_CURRENT)
        beta_forecast = float(cfg.PREDICTIVE_BETA_FORECAST)
        ema_alpha = float(cfg.PREDICTIVE_EMA_ALPHA)
        surge_threshold = float(cfg.PREDICTIVE_SURGE_THRESHOLD)
        surge_bonus_cap = float(cfg.PREDICTIVE_SURGE_BONUS_CAP)
        switch_penalty = float(cfg.PREDICTIVE_SWITCH_PENALTY)

        scores: dict[str, float] = {}
        effective_lane_counts: dict[str, int] = {}
        smoothed_lane_counts: dict[str, float] = {}
        forecast_lane_counts: dict[str, float] = {}
        surge_detected_by_lane: dict[str, bool] = {}

        for lane in LANE_KEYS:
            direction = lane.replace("lane", "")
            current_value = float(raw_counts[lane])
            forecast_value = float(predictions.get(direction, current_value))

            prev_ema = float(
                self._predictive_ema_counts.get(lane, current_value))
            smoothed_value = ema_alpha * current_value + \
                (1.0 - ema_alpha) * prev_ema
            self._predictive_ema_counts[lane] = smoothed_value

            surge_delta = max(0.0, forecast_value - current_value)
            surge_detected = surge_delta >= surge_threshold
            surge_bonus = min(
                surge_bonus_cap, surge_delta) if surge_detected else 0.0

            score = (
                alpha_current * current_value
                + (1.0 - alpha_current) * smoothed_value
                + beta_forecast * surge_delta
                + surge_bonus
            )

            if (
                self._predictive_last_selected_lane is not None
                and lane != self._predictive_last_selected_lane
            ):
                score -= switch_penalty
                payload["switch_penalty_applied"] = True

            scores[lane] = max(0.0, score)
            smoothed_lane_counts[lane] = float(smoothed_value)
            forecast_lane_counts[lane] = float(forecast_value)
            surge_detected_by_lane[lane] = surge_detected
            effective_lane_counts[lane] = max(0, int(round(scores[lane])))

        payload.update(
            {
                "applied": True,
                "reason": "predictive_fusion_applied",
                "smoothed_lane_counts": smoothed_lane_counts,
                "forecast_lane_counts": forecast_lane_counts,
                "effective_scores": scores,
                "effective_lane_counts": effective_lane_counts,
                "surge_detected_by_lane": surge_detected_by_lane,
            }
        )
        return payload

    def _apply_predictive_stability_guard(
        self,
        baseline_decision: dict[str, Any],
        predictive_control: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not bool(predictive_control.get("applied", False)):
            return baseline_decision, predictive_control

        selected_lane = self._lane_from_direction(
            str(baseline_decision.get("direction", "N")))
        predictive_control["selected_lane"] = selected_lane

        last_lane = self._predictive_last_selected_lane
        if not last_lane or selected_lane == last_lane:
            predictive_control["reason"] = "predictive_baseline_selected"
            return baseline_decision, predictive_control

        scores = predictive_control.get("effective_scores", {})
        selected_score = float(scores.get(selected_lane, 0.0))
        last_score = float(scores.get(last_lane, 0.0))
        score_gain = selected_score - last_score
        predictive_control["selected_lane_gain"] = float(score_gain)

        min_hold_cycles = int(cfg.PREDICTIVE_MIN_HOLD_CYCLES)
        hard_margin = float(cfg.PREDICTIVE_HARD_OVERRIDE_MARGIN)
        surge_detected_by_lane = predictive_control.get(
            "surge_detected_by_lane", {})
        selected_surge = bool(surge_detected_by_lane.get(selected_lane, False))

        should_hold = (
            self._predictive_hold_cycles < min_hold_cycles
            and not selected_surge
            and score_gain < hard_margin
        )
        if not should_hold:
            predictive_control["reason"] = (
                "surge_override"
                if selected_surge and score_gain >= hard_margin
                else "predictive_switch_allowed"
            )
            return baseline_decision, predictive_control

        duration = int(baseline_decision.get("duration", int(cfg.MIN_GREEN)))
        duration = max(int(cfg.MIN_GREEN), min(int(cfg.MAX_GREEN), duration))
        hold_direction = last_lane.replace("lane", "")
        hold_index = DIRECTIONS.index(hold_direction)
        held_decision = {
            **{lane: 0 for lane in LANE_KEYS},
            last_lane: duration,
            "direction": hold_direction,
            "duration": duration,
            "action": encode_action(hold_index, duration),
            "mode": "predictive-hold",
        }

        predictive_control["switch_blocked"] = True
        predictive_control["reason"] = "predictive_hold"
        predictive_control["selected_lane"] = last_lane
        return held_decision, predictive_control

    def _update_predictive_selection(
        self,
        selected_lane: str,
        emergency_detected: bool,
    ) -> None:
        if emergency_detected:
            return

        if self._predictive_last_selected_lane == selected_lane:
            self._predictive_hold_cycles += 1
        else:
            self._predictive_last_selected_lane = selected_lane
            self._predictive_hold_cycles = 1

    def _normalize_fairness_mode(self, fairness_mode: str | None) -> str:
        mode = str(fairness_mode or cfg.FAIRNESS_DEFAULT_MODE).strip().lower()
        if mode not in FAIRNESS_MODES:
            return str(cfg.FAIRNESS_DEFAULT_MODE)
        return mode

    def _encode_decision_action(self, direction: str, duration: int) -> int:
        resolved_direction = str(direction or "N").strip().upper()
        if resolved_direction not in DIRECTIONS:
            resolved_direction = "N"

        clamped_duration = max(
            int(cfg.MIN_GREEN),
            min(int(cfg.MAX_GREEN), int(duration)),
        )
        direction_index = DIRECTIONS.index(resolved_direction)
        return encode_action(direction_index, clamped_duration)

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
            str(baseline_decision.get("direction", "N")))
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
            lane
            for lane in eligible
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
                override_duration = max(
                    int(cfg.MIN_GREEN),
                    min(int(cfg.MAX_GREEN), baseline_duration),
                )
                decision = {
                    **{lane: 0 for lane in LANE_KEYS},
                    forced_lane: override_duration,
                    "direction": direction,
                    "duration": override_duration,
                    "action": self._encode_decision_action(direction, override_duration),
                    "mode": "fairness-hard-override",
                }
                info.update(
                    {
                        "applied": True,
                        "reason": "hard_threshold_breach",
                        "selected_lane": forced_lane,
                        "selected_duration": override_duration,
                    }
                )
                return decision, info

            info["reason"] = "hard_threshold_breach_baseline_already_serves"
            return {**baseline_decision}, info

        scores: dict[str, float] = {}
        for lane in eligible:
            wait_ratio = self._fairness_state[lane]["wait_seconds"] / \
                max(wait_threshold, 1.0)
            missed_ratio = self._fairness_state[lane]["missed_turns"] / max(
                float(missed_threshold), 1.0
            )
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

        if (
            fairness_mode == "soft"
            and best_lane != baseline_lane
            and (best_score - baseline_score) >= margin
        ):
            direction = best_lane.replace("lane", "")
            override_duration = max(
                int(cfg.MIN_GREEN),
                min(int(cfg.MAX_GREEN), baseline_duration),
            )
            decision = {
                **{lane: 0 for lane in LANE_KEYS},
                best_lane: override_duration,
                "direction": direction,
                "duration": override_duration,
                "action": self._encode_decision_action(direction, override_duration),
                "mode": "fairness-soft-override",
            }
            info.update(
                {
                    "applied": True,
                    "reason": "soft_priority_margin",
                    "selected_lane": best_lane,
                    "selected_duration": override_duration,
                }
            )
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
        self._predictive_ema_counts = {lane: 0.0 for lane in LANE_KEYS}
        self._predictive_last_selected_lane = None
        self._predictive_hold_cycles = 0
        self._cached_emergency_override = None
        self._cached_siren_detection = None


__all__ = ["ModelController", "DIRECTIONS", "LANE_KEYS"]
