from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request

import config as cfg
from config import DIRECTION_TO_LANE, DIRECTIONS, FAIRNESS_DEFAULT_MODE, LANE_KEYS
from control.model_controller import ModelController
from control.schema import normalize_lane_counts
from training.DQN.environment import encode_action

bp = Blueprint("gui", __name__)
controller = ModelController()

TRAFFIC_PROFILES = {
    "morning_peak": {"laneN": 22, "laneS": 19, "laneE": 11, "laneW": 9},
    "mid_day": {"laneN": 12, "laneS": 11, "laneE": 12, "laneW": 11},
    "evening_peak": {"laneN": 10, "laneS": 12, "laneE": 21, "laneW": 20},
    "night": {"laneN": 4, "laneS": 4, "laneE": 3, "laneW": 3},
}

FAIRNESS_MODES = {"off", "soft", "hard"}
CONTROLLER_MODES = {"adaptive_predictive", "fixed_rr_60", "compare"}


def _new_synthetic_runtime_state() -> dict[str, Any]:
    return {
        "locked_control": None,
        "remaining_sec": 0.0,
        "last_tick_monotonic": None,
        "last_counts": {lane: 0 for lane in LANE_KEYS},
        "fairness_mode": str(FAIRNESS_DEFAULT_MODE),
        "last_served_lane": None,
        "cycle_index": 0,
        "controller_mode": "compare",
    }


SYNTHETIC_RUNTIME = _new_synthetic_runtime_state()
SYNTHETIC_RUNTIME_LOCK = threading.Lock()


# Synthetic runtime state is process-global and mutable.
# Always access it through lock-guarded helpers to avoid cross-request races.
def _read_synthetic_runtime_snapshot() -> dict[str, Any]:
    with SYNTHETIC_RUNTIME_LOCK:
        snapshot = dict(SYNTHETIC_RUNTIME)

    snapshot["locked_control"] = deepcopy(snapshot.get("locked_control"))
    snapshot["last_counts"] = dict(snapshot.get("last_counts") or {
                                   lane: 0 for lane in LANE_KEYS})
    return snapshot


def _update_synthetic_runtime(fields: dict[str, Any]) -> None:
    with SYNTHETIC_RUNTIME_LOCK:
        SYNTHETIC_RUNTIME.update(fields)


def _reset_synthetic_runtime_state() -> None:
    with SYNTHETIC_RUNTIME_LOCK:
        SYNTHETIC_RUNTIME.clear()
        SYNTHETIC_RUNTIME.update(_new_synthetic_runtime_state())


def _normalize_fairness_mode(mode: Any) -> str:
    resolved = str(mode or FAIRNESS_DEFAULT_MODE).strip().lower()
    if resolved not in FAIRNESS_MODES:
        return str(FAIRNESS_DEFAULT_MODE)
    return resolved


def _normalize_controller_mode(mode: Any) -> str:
    resolved = str(mode or "compare").strip().lower()
    if resolved not in CONTROLLER_MODES:
        return "compare"
    return resolved


def _parse_float_field(
    body: dict[str, Any],
    field_name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = body.get(field_name, default)
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid '{field_name}': expected a number") from exc

    if parsed < minimum or parsed > maximum:
        raise ValueError(
            f"Invalid '{field_name}': expected value in [{minimum}, {maximum}]"
        )
    return parsed


def _parse_int_field(
    body: dict[str, Any],
    field_name: str,
    default: int,
    minimum: int,
) -> int:
    raw_value = body.get(field_name, default)
    if isinstance(raw_value, bool):
        raise ValueError(f"Invalid '{field_name}': boolean is not allowed")

    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid '{field_name}': expected an integer") from exc

    if parsed < minimum:
        raise ValueError(
            f"Invalid '{field_name}': expected value >= {minimum}")
    return parsed


def _encode_route_decision_action(direction: str, duration: int) -> int:
    resolved_direction = str(direction or "N").strip().upper()
    if resolved_direction not in DIRECTIONS:
        resolved_direction = "N"

    clamped_duration = max(
        int(cfg.MIN_GREEN),
        min(int(cfg.MAX_GREEN), int(duration)),
    )
    direction_index = DIRECTIONS.index(resolved_direction)
    return encode_action(direction_index, clamped_duration)


def _parse_lane_counts(raw_counts: Any) -> dict[str, int]:
    if raw_counts is not None and not isinstance(raw_counts, dict):
        raise ValueError(
            "Invalid 'current_counts': expected an object mapping lane counts")

    try:
        return normalize_lane_counts(raw_counts)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid 'current_counts': lane values must be integers") from exc


def _decode_image(field_name: str) -> np.ndarray:
    if field_name not in request.files:
        raise ValueError(f"Missing file: {field_name}")

    upload = request.files[field_name]
    if not upload or not upload.filename:
        raise ValueError(f"No image uploaded for {field_name}")

    raw = upload.read()
    if not raw:
        raise ValueError(f"Empty file for {field_name}")

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Invalid image format for {field_name}")
    return frame


def _build_simulation_payload(result: dict, cycle_meta: dict[str, Any] | None = None) -> dict:
    initial_counts = {
        lane: int(result["lane_counts"].get(lane, 0)) for lane in LANE_KEYS}
    selected_direction = str(result.get("direction", "N"))
    selected_lane = DIRECTION_TO_LANE.get(selected_direction, "laneN")
    selected_duration = int(result.get("duration", 10))

    remaining = [lane for lane in LANE_KEYS if lane != selected_lane]
    remaining_sorted = sorted(
        remaining, key=lambda lane: initial_counts.get(lane, 0), reverse=True)
    sequence = [selected_lane, *remaining_sorted]
    lane_timings = {lane: (selected_duration if lane ==
                           selected_lane else 0) for lane in LANE_KEYS}

    emergency = result.get("emergency", {})
    fairness = result.get("fairness", {})
    emergency_detected = bool(emergency.get("detected", False))
    emergency_status = str(
        emergency.get("status") or (
            "active" if emergency_detected else "inactive")
    )
    emergency_release_reason = emergency.get("release_reason")
    emergency_message = ""
    if emergency_detected:
        label = emergency.get("label") or "emergency vehicle"
        confidence = float(emergency.get("confidence", 0.0))
        direction = emergency.get("direction") or selected_direction
        emergency_message = (
            f"Emergency detected ({label}, {confidence:.2f}) in lane {direction}. "
            f"That lane was prioritized."
        )
    elif emergency_status == "cleared":
        emergency_message = "Emergency corridor cleared. Returning to normal DQN control."

    payload = {
        "initial_counts": initial_counts,
        "selected_lane": selected_lane,
        "selected_direction": selected_direction,
        "selected_duration": selected_duration,
        "sequence": sequence,
        "lane_timings": lane_timings,
        "decision_scope": "single_lane",
        "active_lane_only_duration_applied": True,
        "mode": result.get("mode", "unknown"),
        "emergency_detected": emergency_detected,
        "emergency_status": emergency_status,
        "emergency_release_reason": emergency_release_reason,
        "emergency_message": emergency_message,
        "clear_rate_min": 1,
        "clear_rate_max": 2,
        "yellow_ms": 900,
        "seed": int(time.time() * 1000),
        "tick_interval_sec": 3,
        "fairness": fairness,
    }
    if cycle_meta:
        payload.update(cycle_meta)
    return payload


def _synthesize_lane_counts(
    profile_name: str,
    intensity: float,
    current_counts: dict[str, int],
    seed: int,
    tick: int,
) -> dict[str, int]:
    profile = TRAFFIC_PROFILES.get(profile_name, TRAFFIC_PROFILES["mid_day"])
    rng = np.random.default_rng(seed + tick)

    next_counts: dict[str, int] = {}
    for lane in LANE_KEYS:
        baseline = max(0.0, profile[lane] * intensity)
        arrivals = int(rng.poisson(max(0.1, baseline)))
        carry = int(round(current_counts.get(lane, 0) * 0.55))
        next_counts[lane] = int(np.clip(carry + arrivals, 0, 120))

    return next_counts


def _apply_low_traffic_duration_policy(
    result: dict[str, Any],
    lane_counts: dict[str, int],
    profile_name: str,
) -> tuple[dict[str, Any], bool]:
    total_queue = int(sum(lane_counts.values()))
    low_traffic_profile = profile_name in set(cfg.LOW_TRAFFIC_PROFILES)

    if not low_traffic_profile and total_queue > int(cfg.LOW_TRAFFIC_QUEUE_THRESHOLD):
        return result, False

    selected_lane = DIRECTION_TO_LANE.get(
        str(result.get("direction", "N")), "laneN")
    selected_lane_count = int(lane_counts.get(selected_lane, 0))
    current_duration = int(result.get("duration", cfg.MIN_GREEN))
    target_floor = min(
        int(cfg.LOW_TRAFFIC_MAX_GREEN_FLOOR),
        int(cfg.LOW_TRAFFIC_MIN_GREEN_FLOOR)
        + selected_lane_count * int(cfg.LOW_TRAFFIC_PER_VEHICLE_BONUS),
    )
    adjusted_duration = max(current_duration, target_floor)

    if adjusted_duration == current_duration:
        return result, False

    selected_direction = str(result.get("direction", "N"))

    adjusted = {
        **result,
        **{lane: 0 for lane in LANE_KEYS},
        selected_lane: adjusted_duration,
        "duration": adjusted_duration,
        "action": _encode_route_decision_action(selected_direction, adjusted_duration),
    }

    fairness = dict(adjusted.get("fairness", {}))
    fairness["selected_duration"] = adjusted_duration
    if fairness.get("reason") in (None, "baseline_retained"):
        fairness["reason"] = "low_traffic_duration_floor"
    adjusted["fairness"] = fairness
    return adjusted, True


def _compute_congestion_metrics(
    lane_counts: dict[str, int],
    result: dict[str, Any],
    controller_mode: str = "compare",
) -> dict[str, float]:
    total_queue = int(sum(lane_counts.values()))
    max_lane_queue = int(max(lane_counts.values())) if lane_counts else 0
    horizon_sec = 120

    if total_queue <= 0:
        return {
            "total_queue": 0,
            "baseline_clearance_seconds": 0.0,
            "dqn_clearance_seconds": 0.0,
            "improvement_pct": 0.0,
            "comparison": {
                "horizon_sec": horizon_sec,
                "method_note": "proxy_estimate_from_queue_and_control",
                "baseline": {
                    "clearance_seconds": 0.0,
                    "avg_wait_proxy_seconds": 0.0,
                    "max_lane_queue_proxy": 0,
                    "vehicles_cleared_in_horizon": 0,
                },
                "adaptive": {
                    "clearance_seconds": 0.0,
                    "avg_wait_proxy_seconds": 0.0,
                    "max_lane_queue_proxy": 0,
                    "vehicles_cleared_in_horizon": 0,
                },
                "improvements": {
                    "clearance_seconds_pct": 0.0,
                    "avg_wait_proxy_pct": 0.0,
                    "max_lane_queue_proxy_pct": 0.0,
                    "vehicles_cleared_in_horizon_pct": 0.0,
                },
            },
        }

    selected_lane = DIRECTION_TO_LANE.get(
        str(result.get("direction", "N")), "laneN")
    selected_lane_queue = float(lane_counts.get(selected_lane, 0))
    selected_share = selected_lane_queue / max(1.0, float(total_queue))
    selected_duration = float(
        max(1, int(result.get("duration", cfg.MIN_GREEN))))
    fairness = result.get("fairness", {})

    baseline_rate = 1.15
    baseline_clearance_seconds = float(total_queue) / baseline_rate

    use_adaptive_proxy = controller_mode in {"adaptive_predictive", "compare"}
    if use_adaptive_proxy:
        duration_gain = min(selected_duration / 30.0, 1.5) * 0.25
        concentration_gain = max(0.0, selected_share - 0.25) * 1.2
        fairness_gain = 0.08 if bool(fairness.get("applied", False)) else 0.0
        dqn_rate = baseline_rate + duration_gain + concentration_gain + fairness_gain
    else:
        dqn_rate = baseline_rate
    dqn_clearance_seconds = float(total_queue) / max(0.4, dqn_rate)

    improvement_pct = round(
        max(
            0.0,
            (baseline_clearance_seconds - dqn_clearance_seconds)
            / max(1e-6, baseline_clearance_seconds)
            * 100.0,
        ),
        2,
    )

    baseline_avg_wait = baseline_clearance_seconds * 0.5
    adaptive_avg_wait = dqn_clearance_seconds * 0.46
    baseline_max_lane_proxy = max_lane_queue
    adaptive_max_lane_proxy = max(
        0, int(round(max_lane_queue * (1.0 - improvement_pct / 180.0))))
    baseline_cleared = min(total_queue, int(
        round(baseline_rate * horizon_sec)))
    adaptive_cleared = min(total_queue, int(round(dqn_rate * horizon_sec)))

    def pct_gain(base: float, improved: float, lower_is_better: bool) -> float:
        if base <= 0:
            return 0.0
        if lower_is_better:
            return round(max(0.0, (base - improved) / base * 100.0), 2)
        return round(max(0.0, (improved - base) / base * 100.0), 2)

    return {
        "total_queue": total_queue,
        "baseline_clearance_seconds": round(baseline_clearance_seconds, 2),
        "dqn_clearance_seconds": round(dqn_clearance_seconds, 2),
        "improvement_pct": improvement_pct,
        "comparison": {
            "horizon_sec": horizon_sec,
            "method_note": "proxy_estimate_from_queue_and_control",
            "baseline": {
                "clearance_seconds": round(baseline_clearance_seconds, 2),
                "avg_wait_proxy_seconds": round(baseline_avg_wait, 2),
                "max_lane_queue_proxy": int(baseline_max_lane_proxy),
                "vehicles_cleared_in_horizon": int(baseline_cleared),
            },
            "adaptive": {
                "clearance_seconds": round(dqn_clearance_seconds, 2),
                "avg_wait_proxy_seconds": round(adaptive_avg_wait, 2),
                "max_lane_queue_proxy": int(adaptive_max_lane_proxy),
                "vehicles_cleared_in_horizon": int(adaptive_cleared),
            },
            "improvements": {
                "clearance_seconds_pct": pct_gain(
                    baseline_clearance_seconds,
                    dqn_clearance_seconds,
                    lower_is_better=True,
                ),
                "avg_wait_proxy_pct": pct_gain(
                    baseline_avg_wait,
                    adaptive_avg_wait,
                    lower_is_better=True,
                ),
                "max_lane_queue_proxy_pct": pct_gain(
                    float(baseline_max_lane_proxy),
                    float(adaptive_max_lane_proxy),
                    lower_is_better=True,
                ),
                "vehicles_cleared_in_horizon_pct": pct_gain(
                    float(baseline_cleared),
                    float(adaptive_cleared),
                    lower_is_better=False,
                ),
            },
        },
    }


def _apply_fixed_round_robin_override(
    result: dict[str, Any],
    lane_counts: dict[str, int],
    last_served_lane: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lanes_with_queue = [lane for lane in LANE_KEYS if int(
        lane_counts.get(lane, 0)) > 0]
    if not lanes_with_queue:
        chosen_lane = "laneN"
        reason = "rr_idle_default"
    else:
        if last_served_lane in LANE_KEYS:
            start_idx = (LANE_KEYS.index(
                last_served_lane) + 1) % len(LANE_KEYS)
        else:
            start_idx = 0
        chosen_lane = lanes_with_queue[0]
        for offset in range(len(LANE_KEYS)):
            candidate = LANE_KEYS[(start_idx + offset) % len(LANE_KEYS)]
            if candidate in lanes_with_queue:
                chosen_lane = candidate
                break
        reason = "rr_60_rotation"

    direction = chosen_lane.replace("lane", "")
    duration = 60
    adjusted = {
        **result,
        **{lane: 0 for lane in LANE_KEYS},
        chosen_lane: duration,
        "direction": direction,
        "duration": duration,
        "action": _encode_route_decision_action(direction, duration),
        "mode": "fixed-rr-60",
    }
    fairness = dict(adjusted.get("fairness", {}))
    fairness.update(
        {
            "applied": False,
            "reason": "fixed_rr_60",
            "selected_lane": chosen_lane,
            "selected_duration": duration,
            "baseline_lane": chosen_lane,
            "baseline_duration": duration,
        }
    )
    adjusted["fairness"] = fairness
    meta = {
        "selected_by_scheduler": False,
        "lane_repeat_blocked": False,
        "scheduler_reason": reason,
        "previous_lane": last_served_lane,
    }
    return adjusted, meta


def _apply_non_repeat_scheduler(
    result: dict[str, Any],
    lane_counts: dict[str, int],
    last_served_lane: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_direction = str(result.get("direction", "N"))
    selected_lane = DIRECTION_TO_LANE.get(selected_direction, "laneN")

    meta = {
        "selected_by_scheduler": False,
        "lane_repeat_blocked": False,
        "scheduler_reason": "baseline_boundary_selection",
        "previous_lane": last_served_lane,
    }

    if not last_served_lane or selected_lane != last_served_lane:
        return result, meta

    alternatives = [
        lane for lane in LANE_KEYS if lane != last_served_lane and int(lane_counts.get(lane, 0)) > 0
    ]
    if not alternatives:
        meta["scheduler_reason"] = "same_lane_allowed_only_lane_has_queue"
        return result, meta

    next_lane = max(
        alternatives,
        key=lambda lane: (int(lane_counts.get(lane, 0)), -
                          LANE_KEYS.index(lane)),
    )
    next_direction = next_lane.replace("lane", "")
    duration = int(result.get("duration", 0))

    adjusted = {
        **result,
        **{lane: 0 for lane in LANE_KEYS},
        next_lane: duration,
        "direction": next_direction,
        "action": _encode_route_decision_action(next_direction, duration),
        "mode": "scheduler-no-repeat",
    }

    fairness = dict(adjusted.get("fairness", {}))
    fairness["selected_lane"] = next_lane
    fairness["selected_duration"] = duration
    fairness["reason"] = "scheduler_no_repeat"
    adjusted["fairness"] = fairness

    meta.update(
        {
            "selected_by_scheduler": True,
            "lane_repeat_blocked": True,
            "scheduler_reason": "no_consecutive_same_lane",
        }
    )
    return adjusted, meta


def _build_response(result: dict, payload: dict, extra: dict | None = None) -> dict:
    detection = result.get("detection", {})
    density = result.get("density", {})
    emergency = result.get("emergency", {})
    fairness = result.get("fairness", {})
    baseline_decision = result.get("baseline_decision", {})
    diagnostics = result.get("diagnostics", {})
    predictive = diagnostics.get("predictive_control", {})

    response = {
        "result": result,
        "simulation": payload,
        "lane_counts": {lane: int(result["lane_counts"].get(lane, 0)) for lane in LANE_KEYS},
        "decision": {
            "direction": result.get("direction"),
            "duration": int(result.get("duration", 0)),
            "mode": result.get("mode"),
            "action": result.get("action"),
        },
        "emergency": emergency,
        "fairness": fairness,
        "model_outputs": {
            "detector": {
                "mode": detection.get("mode", "unknown"),
                "total": int(detection.get("total", 0)),
                "lane_counts": {
                    lane: int(result["lane_counts"].get(lane, 0)) for lane in LANE_KEYS
                },
            },
            "density": {
                "mode": density.get("mode", "unknown"),
                "horizon_sec": density.get("horizon_sec"),
                "predictions": density.get("predictions", {}),
            },
            "emergency": {
                "detected": bool(emergency.get("detected", False)),
                "status": emergency.get("status"),
                "label": emergency.get("label"),
                "confidence": float(emergency.get("confidence", 0.0)),
                "direction": emergency.get("direction"),
                "release_reason": emergency.get("release_reason"),
                "baseline_duration": emergency.get("baseline_duration"),
                "adjusted_duration": emergency.get("adjusted_duration"),
                "message": payload.get("emergency_message", ""),
            },
            "dqn": {
                "mode": result.get("mode"),
                "action": result.get("action"),
                "direction": result.get("direction"),
                "duration": int(result.get("duration", 0)),
                "baseline_mode": baseline_decision.get("mode"),
                "baseline_action": baseline_decision.get("action"),
                "baseline_direction": baseline_decision.get("direction"),
                "baseline_duration": baseline_decision.get("duration"),
            },
            "fairness": {
                "mode": fairness.get("mode"),
                "applied": bool(fairness.get("applied", False)),
                "reason": fairness.get("reason"),
                "selected_lane": fairness.get("selected_lane"),
                "baseline_lane": fairness.get("baseline_lane"),
                "lane_state": fairness.get("lane_state", {}),
            },
            "predictive_control": {
                "enabled": bool(predictive.get("enabled", False)),
                "applied": bool(predictive.get("applied", False)),
                "reason": predictive.get("reason"),
                "raw_lane_counts": predictive.get("raw_lane_counts", {}),
                "smoothed_lane_counts": predictive.get("smoothed_lane_counts", {}),
                "forecast_lane_counts": predictive.get("forecast_lane_counts", {}),
                "effective_scores": predictive.get("effective_scores", {}),
                "effective_lane_counts": predictive.get("effective_lane_counts", {}),
                "surge_detected_by_lane": predictive.get("surge_detected_by_lane", {}),
                "switch_penalty_applied": bool(predictive.get("switch_penalty_applied", False)),
                "switch_blocked": bool(predictive.get("switch_blocked", False)),
                "selected_lane": predictive.get("selected_lane"),
                "selected_lane_gain": float(predictive.get("selected_lane_gain", 0.0)),
                "last_selected_lane": predictive.get("last_selected_lane"),
                "hold_cycles": int(predictive.get("hold_cycles", 0)),
            },
        },
    }

    if extra:
        response.update(extra)

    return response


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/demo/upload")
def upload_demo():
    return render_template("upload_demo.html", lane_keys=LANE_KEYS)


@bp.get("/demo/synthetic")
def synthetic_demo():
    return render_template("synthetic_demo.html", lane_keys=LANE_KEYS)


@bp.get("/api/status")
def status():
    return jsonify(controller.status())


@bp.post("/api/run_cycle")
def run_cycle():
    try:
        lane_frames = {lane: _decode_image(lane) for lane in LANE_KEYS}
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    fairness_mode = _normalize_fairness_mode(request.form.get("fairness_mode"))
    result = controller.decide_from_lane_frames(
        lane_frames,
        fairness_mode=fairness_mode,
    )
    payload = _build_simulation_payload(result)
    return jsonify(_build_response(result, payload))


@bp.post("/api/synthetic_cycle")
def synthetic_cycle():
    body = request.get_json(silent=True) or {}
    try:
        profile = str(body.get("time_of_day", "mid_day"))
        intensity = _parse_float_field(
            body=body,
            field_name="intensity",
            default=1.0,
            minimum=float(cfg.SYNTHETIC_INTENSITY_MIN),
            maximum=float(cfg.SYNTHETIC_INTENSITY_MAX),
        )
        seed = _parse_int_field(
            body=body,
            field_name="seed",
            default=42,
            minimum=int(cfg.SYNTHETIC_SEED_MIN),
        )
        tick = _parse_int_field(
            body=body,
            field_name="tick",
            default=int(time.time()),
            minimum=int(cfg.SYNTHETIC_TICK_MIN),
        )
        current_counts = _parse_lane_counts(body.get("current_counts"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    fairness_mode = _normalize_fairness_mode(body.get("fairness_mode"))
    runtime_snapshot = _read_synthetic_runtime_snapshot()
    controller_mode = _normalize_controller_mode(
        body.get("controller_mode", runtime_snapshot.get(
            "controller_mode", "compare"))
    )
    if controller_mode != runtime_snapshot.get("controller_mode"):
        _update_synthetic_runtime(
            {
                "locked_control": None,
                "remaining_sec": 0.0,
                "last_tick_monotonic": None,
            }
        )
        runtime_snapshot["locked_control"] = None
        runtime_snapshot["remaining_sec"] = 0.0
        runtime_snapshot["last_tick_monotonic"] = None

    synthetic_counts = _synthesize_lane_counts(
        profile_name=profile,
        intensity=intensity,
        current_counts=current_counts,
        seed=seed,
        tick=tick,
    )

    now = time.monotonic()
    last_tick = runtime_snapshot.get("last_tick_monotonic")
    remaining_sec = float(runtime_snapshot.get("remaining_sec", 0.0))

    if last_tick is not None:
        elapsed = max(0.0, now - float(last_tick))
    else:
        elapsed = float(cfg.INFERENCE_INTERVAL)

    remaining_sec = max(0.0, remaining_sec - elapsed)
    locked_control = runtime_snapshot.get("locked_control")
    rerun_control = locked_control is None or remaining_sec <= 0.0
    previous_lane = runtime_snapshot.get("last_served_lane")
    next_last_served_lane = previous_lane
    next_cycle_index = int(runtime_snapshot.get("cycle_index", 0))

    control_source = "cycle_lock_reuse"
    dqn_reran_this_tick = False
    scheduler_meta = {
        "selected_by_scheduler": False,
        "lane_repeat_blocked": False,
        "scheduler_reason": "cycle_lock_reuse",
        "previous_lane": previous_lane,
    }

    if rerun_control:
        result = controller.decide_from_lane_counts(
            synthetic_counts,
            fairness_mode=fairness_mode,
            locked_control=None,
            update_fairness_state=True,
        )
        if controller_mode == "fixed_rr_60":
            result, scheduler_meta = _apply_fixed_round_robin_override(
                result=result,
                lane_counts=synthetic_counts,
                last_served_lane=previous_lane,
            )
            control_source = "fixed_rr_cycle"
        else:
            result, scheduler_meta = _apply_non_repeat_scheduler(
                result=result,
                lane_counts=synthetic_counts,
                last_served_lane=previous_lane,
            )
            result, low_traffic_adjusted = _apply_low_traffic_duration_policy(
                result=result,
                lane_counts=synthetic_counts,
                profile_name=profile,
            )
            if low_traffic_adjusted:
                scheduler_meta["scheduler_reason"] = "low_traffic_duration_floor"
            control_source = "dqn_cycle_boundary"

        selected_lane = DIRECTION_TO_LANE.get(
            str(result.get("direction", "N")), "laneN")
        locked_control = {
            "decision": {lane: int(result.get(lane, 0)) for lane in LANE_KEYS}
            | {
                "direction": result.get("direction"),
                "duration": int(result.get("duration", 0)),
                "action": result.get("action"),
                "mode": result.get("mode"),
            },
            "baseline_decision": dict(result.get("baseline_decision", {})),
            "fairness": dict(result.get("fairness", {})),
            "selected_lane": selected_lane,
        }
        remaining_sec = float(max(0, int(result.get("duration", 0))))
        dqn_reran_this_tick = controller_mode != "fixed_rr_60"
        next_last_served_lane = selected_lane
        next_cycle_index += 1
    else:
        result = controller.decide_from_lane_counts(
            synthetic_counts,
            fairness_mode=fairness_mode,
            locked_control=locked_control,
            update_fairness_state=False,
        )

    locked_control_for_state = locked_control if remaining_sec > 0 else None
    _update_synthetic_runtime(
        {
            "locked_control": locked_control_for_state,
            "remaining_sec": remaining_sec,
            "last_tick_monotonic": now,
            "last_counts": dict(synthetic_counts),
            "fairness_mode": fairness_mode,
            "controller_mode": controller_mode,
            "last_served_lane": next_last_served_lane,
            "cycle_index": next_cycle_index,
        }
    )

    cycle_meta = {
        "cycle_locked": remaining_sec > 0,
        "cycle_remaining_sec": round(remaining_sec, 2),
        "dqn_reran_this_tick": dqn_reran_this_tick,
        "control_source": control_source,
        "model_refresh_sec": float(cfg.INFERENCE_INTERVAL),
        "cycle_index": next_cycle_index,
        **scheduler_meta,
    }

    payload = _build_simulation_payload(result, cycle_meta=cycle_meta)
    payload["source"] = "synthetic"
    payload["traffic_profile"] = profile

    congestion_metrics = _compute_congestion_metrics(
        synthetic_counts,
        result,
        controller_mode=controller_mode,
    )

    extra = {
        "scenario": {
            "mode": "synthetic",
            "time_of_day": profile,
            "intensity": intensity,
            "seed": seed,
            "tick": tick,
            "fairness_mode": fairness_mode,
            "controller_mode": controller_mode,
            "dqn_reran_this_tick": dqn_reran_this_tick,
            "control_source": control_source,
            "scheduler_reason": scheduler_meta.get("scheduler_reason"),
        },
        "congestion_metrics": {
            **congestion_metrics,
        },
    }
    return jsonify(_build_response(result, payload, extra=extra))


@bp.post("/api/spawn_ambulance")
def spawn_ambulance():
    body = request.get_json(silent=True) or {}
    try:
        current_counts = _parse_lane_counts(body.get("current_counts"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    fairness_mode = _normalize_fairness_mode(body.get("fairness_mode"))
    lane = str(body.get("lane", "auto"))

    if lane not in LANE_KEYS:
        lane = max(LANE_KEYS, key=lambda key: current_counts.get(key, 0))

    direction = lane.replace("lane", "")
    emergency_override = {
        "detected": True,
        "label": "ambulance-manual-spawn",
        "confidence": 1.0,
        "direction": direction,
        "predictions": [],
        "mode": "synthetic-spawn",
    }

    result = controller.decide_from_lane_counts(
        lane_counts=current_counts,
        emergency_override=emergency_override,
        fairness_mode=fairness_mode,
        locked_control=None,
        update_fairness_state=True,
    )

    emergency_duration = float(max(0, int(result.get("duration", 0))))
    runtime_snapshot = _read_synthetic_runtime_snapshot()
    previous_lane = runtime_snapshot.get("last_served_lane")
    next_cycle_index = int(runtime_snapshot.get("cycle_index", 0)) + 1
    selected_lane = DIRECTION_TO_LANE.get(
        str(result.get("direction", "N")), "laneN")
    _update_synthetic_runtime(
        {
            "locked_control": {
                "decision": {lane: int(result.get(lane, 0)) for lane in LANE_KEYS}
                | {
                    "direction": result.get("direction"),
                    "duration": int(result.get("duration", 0)),
                    "action": result.get("action"),
                    "mode": result.get("mode"),
                },
                "baseline_decision": dict(result.get("baseline_decision", {})),
                "fairness": dict(result.get("fairness", {})),
                "selected_lane": selected_lane,
            },
            "remaining_sec": emergency_duration,
            "last_tick_monotonic": time.monotonic(),
            "last_counts": dict(current_counts),
            "fairness_mode": fairness_mode,
            "last_served_lane": selected_lane,
            "cycle_index": next_cycle_index,
        }
    )

    cycle_meta = {
        "cycle_locked": emergency_duration > 0,
        "cycle_remaining_sec": round(emergency_duration, 2),
        "dqn_reran_this_tick": False,
        "control_source": "emergency_override",
        "emergency_preempted_cycle": True,
        "model_refresh_sec": float(cfg.INFERENCE_INTERVAL),
        "cycle_index": next_cycle_index,
        "selected_by_scheduler": False,
        "lane_repeat_blocked": False,
        "scheduler_reason": "emergency_override",
        "previous_lane": previous_lane,
    }

    payload = _build_simulation_payload(result, cycle_meta=cycle_meta)
    payload["source"] = "synthetic"
    payload["traffic_profile"] = "ambulance_spawn"

    extra = {
        "scenario": {
            "mode": "synthetic",
            "event": "ambulance_spawn",
            "lane": lane,
            "fairness_mode": fairness_mode,
        }
    }
    return jsonify(_build_response(result, payload, extra=extra))


@bp.post("/api/synthetic_reset")
def synthetic_reset():
    controller.reset_runtime_state()
    _reset_synthetic_runtime_state()
    return jsonify({"ok": True})
