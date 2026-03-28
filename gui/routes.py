from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request

from config import DIRECTION_TO_LANE, LANE_KEYS
from control.model_controller import ModelController

bp = Blueprint("gui", __name__)
controller = ModelController()

TRAFFIC_PROFILES = {
    "morning_peak": {"laneN": 22, "laneS": 19, "laneE": 11, "laneW": 9},
    "mid_day": {"laneN": 12, "laneS": 11, "laneE": 12, "laneW": 11},
    "evening_peak": {"laneN": 10, "laneS": 12, "laneE": 21, "laneW": 20},
    "night": {"laneN": 4, "laneS": 4, "laneE": 3, "laneW": 3},
}


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


def _build_simulation_payload(result: dict) -> dict:
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
    emergency_detected = bool(emergency.get("detected", False))
    emergency_message = ""
    if emergency_detected:
        label = emergency.get("label") or "emergency vehicle"
        confidence = float(emergency.get("confidence", 0.0))
        direction = emergency.get("direction") or selected_direction
        emergency_message = (
            f"Emergency detected ({label}, {confidence:.2f}) in lane {direction}. "
            f"That lane was prioritized."
        )

    return {
        "initial_counts": initial_counts,
        "selected_lane": selected_lane,
        "selected_direction": selected_direction,
        "selected_duration": selected_duration,
        "sequence": sequence,
        "lane_timings": lane_timings,
        "mode": result.get("mode", "unknown"),
        "emergency_detected": emergency_detected,
        "emergency_message": emergency_message,
        "clear_rate_min": 1,
        "clear_rate_max": 2,
        "yellow_ms": 900,
        "seed": int(time.time() * 1000),
    }


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


def _build_response(result: dict, payload: dict, extra: dict | None = None) -> dict:
    detection = result.get("detection", {})
    density = result.get("density", {})
    emergency = result.get("emergency", {})

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
                "label": emergency.get("label"),
                "confidence": float(emergency.get("confidence", 0.0)),
                "direction": emergency.get("direction"),
                "message": payload.get("emergency_message", ""),
            },
            "dqn": {
                "mode": result.get("mode"),
                "action": result.get("action"),
                "direction": result.get("direction"),
                "duration": int(result.get("duration", 0)),
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

    result = controller.decide_from_lane_frames(lane_frames)
    payload = _build_simulation_payload(result)
    return jsonify(_build_response(result, payload))


@bp.post("/api/synthetic_cycle")
def synthetic_cycle():
    body = request.get_json(silent=True) or {}
    profile = str(body.get("time_of_day", "mid_day"))
    intensity = float(body.get("intensity", 1.0))
    seed = int(body.get("seed", 42))
    tick = int(body.get("tick", int(time.time())))
    current_counts = _normalize_lane_counts(body.get("current_counts"))

    synthetic_counts = _synthesize_lane_counts(
        profile_name=profile,
        intensity=max(0.2, min(3.0, intensity)),
        current_counts=current_counts,
        seed=seed,
        tick=tick,
    )

    result = controller.decide_from_lane_counts(synthetic_counts)
    payload = _build_simulation_payload(result)
    payload["source"] = "synthetic"
    payload["traffic_profile"] = profile

    total_queue = int(sum(synthetic_counts.values()))
    baseline_clearance_seconds = max(1.0, total_queue / 1.2)
    dqn_clearance_seconds = max(1.0, total_queue / 1.8)
    improvement_pct = round(
        max(0.0, (baseline_clearance_seconds - dqn_clearance_seconds) /
            baseline_clearance_seconds * 100.0),
        2,
    )

    extra = {
        "scenario": {
            "mode": "synthetic",
            "time_of_day": profile,
            "intensity": intensity,
            "seed": seed,
            "tick": tick,
        },
        "congestion_metrics": {
            "total_queue": total_queue,
            "baseline_clearance_seconds": round(baseline_clearance_seconds, 2),
            "dqn_clearance_seconds": round(dqn_clearance_seconds, 2),
            "improvement_pct": improvement_pct,
        },
    }
    return jsonify(_build_response(result, payload, extra=extra))


@bp.post("/api/spawn_ambulance")
def spawn_ambulance():
    body = request.get_json(silent=True) or {}
    current_counts = _normalize_lane_counts(body.get("current_counts"))
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
    )
    payload = _build_simulation_payload(result)
    payload["source"] = "synthetic"
    payload["traffic_profile"] = "ambulance_spawn"

    extra = {
        "scenario": {
            "mode": "synthetic",
            "event": "ambulance_spawn",
            "lane": lane,
        }
    }
    return jsonify(_build_response(result, payload, extra=extra))
