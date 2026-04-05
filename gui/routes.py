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

FAIRNESS_MODES = {"off", "soft", "hard"}


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


def _build_response(
    result: dict,
    payload: dict,
    extra: dict | None = None,
    *,
    include_fairness_telemetry: bool = True,
) -> dict:
    detection = result.get("detection", {})
    density = result.get("density", {})
    emergency = result.get("emergency", {})
    fairness = result.get("fairness", {})
    baseline_decision = result.get("baseline_decision", {})

    model_outputs = {
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
    }
    if include_fairness_telemetry:
        model_outputs["fairness"] = {
            "mode": fairness.get("mode"),
            "applied": bool(fairness.get("applied", False)),
            "reason": fairness.get("reason"),
            "selected_lane": fairness.get("selected_lane"),
            "baseline_lane": fairness.get("baseline_lane"),
            "lane_state": fairness.get("lane_state", {}),
        }

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
        "model_outputs": model_outputs,
    }
    if include_fairness_telemetry:
        response["fairness"] = fairness

    if extra:
        response.update(extra)

    return response


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/demo/upload")
def upload_demo():
    return render_template("upload_demo.html", lane_keys=LANE_KEYS)


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
