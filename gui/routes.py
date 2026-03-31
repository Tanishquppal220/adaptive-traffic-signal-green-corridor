from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request

import config as cfg
from config import DIRECTION_TO_LANE, FAIRNESS_DEFAULT_MODE, LANE_KEYS
from control.model_controller import ModelController

bp = Blueprint("gui", __name__)
controller = ModelController()

TRAFFIC_PROFILES = {
    "morning_peak": {"laneN": 22, "laneS": 19, "laneE": 11, "laneW": 9},
    "mid_day": {"laneN": 12, "laneS": 11, "laneE": 12, "laneW": 11},
    "evening_peak": {"laneN": 10, "laneS": 12, "laneE": 21, "laneW": 20},
    "night": {"laneN": 4, "laneS": 4, "laneE": 3, "laneW": 3},
}

FAIRNESS_MODES = {"off", "soft", "hard"}


def _new_synthetic_runtime_state() -> dict[str, Any]:
    return {
        "locked_control": None,
        "remaining_sec": 0.0,
        "last_tick_monotonic": None,
        "last_counts": {lane: 0 for lane in LANE_KEYS},
        "fairness_mode": str(FAIRNESS_DEFAULT_MODE),
        "last_served_lane": None,
        "cycle_index": 0,
    }


SYNTHETIC_RUNTIME = _new_synthetic_runtime_state()


def _normalize_fairness_mode(mode: Any) -> str:
    resolved = str(mode or FAIRNESS_DEFAULT_MODE).strip().lower()
    if resolved not in FAIRNESS_MODES:
        return str(FAIRNESS_DEFAULT_MODE)
    return resolved


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
    lane_timings = {lane: selected_duration for lane in sequence}

    emergency = result.get("emergency", {})
    fairness = result.get("fairness", {})
    emergency_detected = bool(emergency.get("detected", False))
    emergency_status = str(emergency.get("status") or (
        "active" if emergency_detected else "inactive"))
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


def _normalize_lane_counts(raw_counts: dict[str, Any] | None) -> dict[str, int]:
    counts = {lane: 0 for lane in LANE_KEYS}
    if not raw_counts:
        return counts

    for lane in LANE_KEYS:
        counts[lane] = max(0, int(raw_counts.get(lane, 0)))
    return counts


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

    adjusted = {
        **result,
        **{lane: 0 for lane in LANE_KEYS},
        selected_lane: adjusted_duration,
        "duration": adjusted_duration,
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
) -> dict[str, float]:
    total_queue = int(sum(lane_counts.values()))
    if total_queue <= 0:
        return {
            "total_queue": 0,
            "baseline_clearance_seconds": 0.0,
            "dqn_clearance_seconds": 0.0,
            "improvement_pct": 0.0,
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

    duration_gain = min(selected_duration / 30.0, 1.5) * 0.25
    concentration_gain = max(0.0, selected_share - 0.25) * 1.2
    fairness_gain = 0.08 if bool(fairness.get("applied", False)) else 0.0
    dqn_rate = baseline_rate + duration_gain + concentration_gain + fairness_gain
    dqn_clearance_seconds = float(total_queue) / max(0.4, dqn_rate)

    improvement_pct = round(
        max(0.0, (baseline_clearance_seconds - dqn_clearance_seconds)
            / max(1e-6, baseline_clearance_seconds) * 100.0),
        2,
    )

    return {
        "total_queue": total_queue,
        "baseline_clearance_seconds": round(baseline_clearance_seconds, 2),
        "dqn_clearance_seconds": round(dqn_clearance_seconds, 2),
        "improvement_pct": improvement_pct,
    }


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
        lane for lane in LANE_KEYS
        if lane != last_served_lane and int(lane_counts.get(lane, 0)) > 0
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
        "mode": "scheduler-no-repeat",
    }

    fairness = dict(adjusted.get("fairness", {}))
    fairness["selected_lane"] = next_lane
    fairness["selected_duration"] = duration
    fairness["reason"] = "scheduler_no_repeat"
    adjusted["fairness"] = fairness

    meta.update({
        "selected_by_scheduler": True,
        "lane_repeat_blocked": True,
        "scheduler_reason": "no_consecutive_same_lane",
    })
    return adjusted, meta


def _build_response(result: dict, payload: dict, extra: dict | None = None) -> dict:
    detection = result.get("detection", {})
    density = result.get("density", {})
    emergency = result.get("emergency", {})
    fairness = result.get("fairness", {})
    baseline_decision = result.get("baseline_decision", {})

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
                "lane_counts": {lane: int(result["lane_counts"].get(lane, 0)) for lane in LANE_KEYS},
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
    profile = str(body.get("time_of_day", "mid_day"))
    intensity = float(body.get("intensity", 1.0))
    seed = int(body.get("seed", 42))
    tick = int(body.get("tick", int(time.time())))
    fairness_mode = _normalize_fairness_mode(body.get("fairness_mode"))
    current_counts = _normalize_lane_counts(body.get("current_counts"))

    synthetic_counts = _synthesize_lane_counts(
        profile_name=profile,
        intensity=max(0.2, min(3.0, intensity)),
        current_counts=current_counts,
        seed=seed,
        tick=tick,
    )

    now = time.monotonic()
    last_tick = SYNTHETIC_RUNTIME.get("last_tick_monotonic")
    remaining_sec = float(SYNTHETIC_RUNTIME.get("remaining_sec", 0.0))

    if last_tick is not None:
        elapsed = max(0.0, now - float(last_tick))
    else:
        elapsed = float(cfg.INFERENCE_INTERVAL)

    remaining_sec = max(0.0, remaining_sec - elapsed)
    locked_control = SYNTHETIC_RUNTIME.get("locked_control")
    rerun_control = locked_control is None or remaining_sec <= 0.0

    control_source = "cycle_lock_reuse"
    dqn_reran_this_tick = False
    scheduler_meta = {
        "selected_by_scheduler": False,
        "lane_repeat_blocked": False,
        "scheduler_reason": "cycle_lock_reuse",
        "previous_lane": SYNTHETIC_RUNTIME.get("last_served_lane"),
    }

    if rerun_control:
        result = controller.decide_from_lane_counts(
            synthetic_counts,
            fairness_mode=fairness_mode,
            locked_control=None,
            update_fairness_state=True,
        )
        result, scheduler_meta = _apply_non_repeat_scheduler(
            result=result,
            lane_counts=synthetic_counts,
            last_served_lane=SYNTHETIC_RUNTIME.get("last_served_lane"),
        )
        result, low_traffic_adjusted = _apply_low_traffic_duration_policy(
            result=result,
            lane_counts=synthetic_counts,
            profile_name=profile,
        )
        if low_traffic_adjusted:
            scheduler_meta["scheduler_reason"] = "low_traffic_duration_floor"
        selected_lane = DIRECTION_TO_LANE.get(
            str(result.get("direction", "N")), "laneN")
        locked_control = {
            "decision": {
                lane: int(result.get(lane, 0)) for lane in LANE_KEYS
            } | {
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
        control_source = "dqn_cycle_boundary"
        dqn_reran_this_tick = True
        SYNTHETIC_RUNTIME["last_served_lane"] = selected_lane
        SYNTHETIC_RUNTIME["cycle_index"] = int(
            SYNTHETIC_RUNTIME.get("cycle_index", 0)) + 1
    else:
        result = controller.decide_from_lane_counts(
            synthetic_counts,
            fairness_mode=fairness_mode,
            locked_control=locked_control,
            update_fairness_state=False,
        )

    SYNTHETIC_RUNTIME["locked_control"] = locked_control if remaining_sec > 0 else None
    SYNTHETIC_RUNTIME["remaining_sec"] = remaining_sec
    SYNTHETIC_RUNTIME["last_tick_monotonic"] = now
    SYNTHETIC_RUNTIME["last_counts"] = dict(synthetic_counts)
    SYNTHETIC_RUNTIME["fairness_mode"] = fairness_mode

    cycle_meta = {
        "cycle_locked": remaining_sec > 0,
        "cycle_remaining_sec": round(remaining_sec, 2),
        "dqn_reran_this_tick": dqn_reran_this_tick,
        "control_source": control_source,
        "model_refresh_sec": float(cfg.INFERENCE_INTERVAL),
        "cycle_index": int(SYNTHETIC_RUNTIME.get("cycle_index", 0)),
        **scheduler_meta,
    }

    payload = _build_simulation_payload(result, cycle_meta=cycle_meta)
    payload["source"] = "synthetic"
    payload["traffic_profile"] = profile

    congestion_metrics = _compute_congestion_metrics(synthetic_counts, result)

    extra = {
        "scenario": {
            "mode": "synthetic",
            "time_of_day": profile,
            "intensity": intensity,
            "seed": seed,
            "tick": tick,
            "fairness_mode": fairness_mode,
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
    current_counts = _normalize_lane_counts(body.get("current_counts"))
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
    previous_lane = SYNTHETIC_RUNTIME.get("last_served_lane")
    selected_lane = DIRECTION_TO_LANE.get(
        str(result.get("direction", "N")), "laneN")
    SYNTHETIC_RUNTIME["locked_control"] = {
        "decision": {
            lane: int(result.get(lane, 0)) for lane in LANE_KEYS
        } | {
            "direction": result.get("direction"),
            "duration": int(result.get("duration", 0)),
            "action": result.get("action"),
            "mode": result.get("mode"),
        },
        "baseline_decision": dict(result.get("baseline_decision", {})),
        "fairness": dict(result.get("fairness", {})),
        "selected_lane": selected_lane,
    }
    SYNTHETIC_RUNTIME["remaining_sec"] = emergency_duration
    SYNTHETIC_RUNTIME["last_tick_monotonic"] = time.monotonic()
    SYNTHETIC_RUNTIME["last_counts"] = dict(current_counts)
    SYNTHETIC_RUNTIME["fairness_mode"] = fairness_mode
    SYNTHETIC_RUNTIME["last_served_lane"] = selected_lane
    SYNTHETIC_RUNTIME["cycle_index"] = int(
        SYNTHETIC_RUNTIME.get("cycle_index", 0)) + 1

    cycle_meta = {
        "cycle_locked": emergency_duration > 0,
        "cycle_remaining_sec": round(emergency_duration, 2),
        "dqn_reran_this_tick": False,
        "control_source": "emergency_override",
        "emergency_preempted_cycle": True,
        "model_refresh_sec": float(cfg.INFERENCE_INTERVAL),
        "cycle_index": int(SYNTHETIC_RUNTIME.get("cycle_index", 0)),
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
    SYNTHETIC_RUNTIME.clear()
    SYNTHETIC_RUNTIME.update(_new_synthetic_runtime_state())
    return jsonify({"ok": True})
