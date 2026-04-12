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
        print("🚀 Model loading started...")

        self._traffic_detector = TrafficDetector()
        traffic_status = "loaded" if self._traffic_detector.is_loaded else "not loaded"
        print(f"🚘 TrafficDetector {traffic_status}")

        self._emergency_classifier = EmergencyClassifier()
        emergency_status = "loaded" if self._emergency_classifier.is_loaded else "not loaded"
        print(f"🚑 EmergencyClassifier {emergency_status}")

        self._siren_detector = SirenDetector()
        siren_status = "loaded" if self._siren_detector.is_loaded else "not loaded"
        print(f"🔊 SirenDetector {siren_status}")

        self._density_predictor = DensityPredictor()
        density_status = "loaded" if self._density_predictor.is_loaded else "not loaded"
        print(f"📈 DensityPredictor {density_status}")

        self._signal_controller = SignalController(
            weights_path=dqn_weights_path or Path(
                "models") / "dqn_signal_optimizer.pt",
            device=device,
        )
        print(
            f"🧠 SignalController initialized in {self._signal_controller.mode} mode")

        print("✅ Model loading completed.")
        self._last_lane_counts = {k: 0 for k in LANE_KEYS}
        self._last_emergency_active = False
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
                sum(1 for item in emergency.get("predictions", [])
                    if item.get("is_emergency"))
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
    ) -> dict[str, Any]:
        if not self.has_cached_cycle_context():
            raise ValueError(
                "No cached cycle context. Run /api/run_cycle first.")

        lane_counts = normalize_lane_counts(lane_counts)
        emergency_override = {**(self._cached_emergency_override or {})}
        siren_override = {**(self._cached_siren_detection or {})}

        cached_emergency_lane = str(
            emergency_override.get("emergency_lane") or ""
        ).strip()
        if cached_emergency_lane not in LANE_KEYS:
            cached_direction = str(
                emergency_override.get("direction") or ""
            ).strip().upper()
            cached_emergency_lane = DIRECTION_TO_LANE.get(cached_direction, "")

        cached_override_active = bool(
            emergency_override.get("detected", False)
        ) and bool(siren_override.get("detected", False))
        emergency_lane_cleared = (
            cached_override_active
            and cached_emergency_lane in LANE_KEYS
            and int(lane_counts.get(cached_emergency_lane, 0)) <= 0
        )

        if emergency_lane_cleared:
            emergency_override = {
                **emergency_override,
                "detected": False,
                "status": "cleared",
                "release_reason": "emergency_lane_cleared",
                "direction": None,
                "emergency_lane": None,
                "visual_detected": False,
                "siren_detected": False,
            }
            siren_override = {
                **siren_override,
                "detected": False,
                "confidence": 0.0,
            }
            self._cached_emergency_override = {**emergency_override}
            self._cached_siren_detection = {**siren_override}

        detection_payload = {
            "lane_counts": lane_counts,
            "direction_counts": lane_counts_to_direction_counts(lane_counts),
            "boxes": [],
            "total": int(sum(max(0, int(v)) for v in lane_counts.values())),
            "mode": "iterative-cycle",
        }

        return self.decide_from_lane_counts(
            lane_counts=lane_counts,
            frame=None,
            detection=detection_payload,
            emergency_override=emergency_override,
            siren_override=siren_override,
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
        current_active_lane: str | None = None,
        locked_control: dict[str, Any] | None = None,
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
                preemption_buffer_sec = int(
                    cfg.EMERGENCY_PREEMPTION_BUFFER_SEC)

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
        else:
            if locked_control is not None:
                decision = {**locked_control.get("decision", {})}
            else:
                decision = {**baseline_decision}
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

        decision, empty_lane_guard = self._apply_empty_lane_guard(
            decision=decision,
            lane_counts=lane_counts,
            emergency_detected=emergency_detected,
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
            "diagnostics": {
                "active_models": {
                    "traffic_detector": self._traffic_detector.is_loaded,
                    "emergency_classifier": self._emergency_classifier.is_loaded,
                    "siren_detector": self._siren_detector.is_loaded,
                    "density_predictor": self._density_predictor.is_loaded,
                    "dqn_controller": self._signal_controller.mode == "dqn",
                },
                "controller_mode": decision.get("mode", self.mode),
                "emergency_visual_detected": emergency_visual_detected,
                "siren_detected": siren_detected,
                "predictive_control": predictive_control,
                "empty_lane_guard": empty_lane_guard,
            },
        }

    def _apply_empty_lane_guard(
        self,
        decision: dict[str, Any],
        lane_counts: dict[str, int],
        emergency_detected: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        selected_direction = str(decision.get(
            "direction", "N")).strip().upper()
        selected_lane = self._lane_from_direction(selected_direction)
        selected_queue = int(lane_counts.get(selected_lane, 0))
        max_queue = max(int(lane_counts.get(lane, 0)) for lane in LANE_KEYS)

        diagnostics = {
            "applied": False,
            "reason": None,
            "selected_lane_before": selected_lane,
            "selected_lane_after": selected_lane,
            "selected_queue_before": selected_queue,
            "max_queue": max_queue,
        }

        if emergency_detected:
            diagnostics["reason"] = "emergency_override_active"
            return decision, diagnostics

        if max_queue <= 0:
            diagnostics["reason"] = "all_queues_empty"
            return decision, diagnostics

        if selected_queue > 0:
            diagnostics["reason"] = "selected_lane_has_queue"
            return decision, diagnostics

        candidates = [
            lane for lane in LANE_KEYS if int(lane_counts.get(lane, 0)) == max_queue
        ]
        fallback_lane = candidates[0] if candidates else selected_lane

        if (
            self._predictive_last_selected_lane in candidates
            and len(candidates) > 1
        ):
            idx = candidates.index(self._predictive_last_selected_lane)
            fallback_lane = candidates[(idx + 1) % len(candidates)]

        fallback_direction = fallback_lane.replace("lane", "")
        fallback_duration = max(
            int(cfg.MIN_GREEN),
            min(int(cfg.MAX_GREEN), int(decision.get("duration", cfg.MIN_GREEN))),
        )
        mode_prefix = str(decision.get("mode", self.mode))
        guarded_decision = {
            **{lane: 0 for lane in LANE_KEYS},
            fallback_lane: fallback_duration,
            "direction": fallback_direction,
            "duration": fallback_duration,
            "action": self._encode_decision_action(fallback_direction, fallback_duration),
            "mode": f"{mode_prefix}-empty-lane-guard",
        }

        diagnostics.update(
            {
                "applied": True,
                "reason": "selected_lane_empty_with_nonempty_alternative",
                "selected_lane_after": fallback_lane,
            }
        )
        return guarded_decision, diagnostics

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
        self._predictive_ema_counts = {lane: 0.0 for lane in LANE_KEYS}
        self._predictive_last_selected_lane = None
        self._predictive_hold_cycles = 0
        self._cached_emergency_override = None
        self._cached_siren_detection = None


__all__ = ["ModelController", "DIRECTIONS", "LANE_KEYS"]
