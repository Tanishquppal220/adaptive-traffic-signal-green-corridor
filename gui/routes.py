from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request

from config import DIRECTION_TO_LANE, FAIRNESS_DEFAULT_MODE, LANE_KEYS
from control.model_controller import ModelController
from control.schema import normalize_lane_counts

bp = Blueprint("gui", __name__)
controller = ModelController()

FAIRNESS_MODES = {"off", "soft", "hard"}

# Initialize demo cache on controller (persists across requests)
controller._demo_cache = {
    "lane_frames": {},
    "audio_bytes": None,
    "lane_counts": {},
    "emergency": {},
    "siren": {},
    "smoothed_counts": {},
}


# ════════════════════════════════════════════════════════════
# Pipeline Demo Cache (persists across HTTP requests)
# ════════════════════════════════════════════════════════════
def _get_demo_cache() -> dict[str, Any]:
    """Get demo cache from controller (persists across requests)."""
    return controller._demo_cache


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


def _decode_audio(field_name: str) -> bytes:
    if field_name not in request.files:
        raise ValueError(f"Missing file: {field_name}")

    upload = request.files[field_name]
    if not upload or not upload.filename:
        raise ValueError(f"No audio uploaded for {field_name}")

    raw = upload.read()
    if not raw:
        raise ValueError(f"Empty file for {field_name}")
    return raw


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
        "preemption_buffer_sec": int(emergency.get("preemption_buffer_sec", 0) or 0),
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
    siren = result.get("siren", {})
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
            "lane_counts": detection.get(
                "emergency_lane_counts",
                {lane: 0 for lane in LANE_KEYS},
            ),
            "release_reason": emergency.get("release_reason"),
            "baseline_duration": emergency.get("baseline_duration"),
            "adjusted_duration": emergency.get("adjusted_duration"),
            "message": payload.get("emergency_message", ""),
        },
        "siren": {
            "detected": bool(siren.get("detected", False)),
            "confidence": float(siren.get("confidence", 0.0)),
            "mode": siren.get("mode"),
            "sample_rate": siren.get("sample_rate"),
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


# ════════════════════════════════════════════════════════════
# Pipeline Demo: 5-Step Orchestration (for interactive UI)
# ════════════════════════════════════════════════════════════

@bp.post("/api/step1/detect-traffic")
def step1_detect_traffic():
    """Step 1: Traffic Detection (YOLOv8) on 4 lane images."""
    try:
        cache = _get_demo_cache()
        t0 = time.perf_counter()

        # Decode all 4 lane images
        lane_frames = {lane: _decode_image(lane) for lane in LANE_KEYS}
        cache["lane_frames"] = lane_frames

        # Run traffic detection on each lane
        lane_counts = {}
        for lane in LANE_KEYS:
            detection = controller._traffic_detector.detect(lane_frames[lane])
            lane_counts[lane] = int(detection.get("total", 0))

        cache["lane_counts"] = lane_counts
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        return jsonify({
            "counts": lane_counts,
            "inference_ms": elapsed_ms,
            "mode": "yolov8s",
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Traffic detection failed: {str(exc)}"}), 500


@bp.post("/api/step2/detect-emergency")
def step2_detect_emergency():
    """Step 2: Emergency Vehicle Detection (YOLOv8) on 4 lane images."""
    try:
        cache = _get_demo_cache()
        lane_frames = cache.get("lane_frames", {})
        t0 = time.perf_counter()

        if not lane_frames:
            return jsonify({"error": "No lane frames in cache. Run step 1 first."}), 400

        # Run emergency detection on each lane
        emergency_lane_counts = {}
        best_emergency = None
        best_confidence = 0.0

        for lane in LANE_KEYS:
            emergency = controller._emergency_classifier.classify(
                lane_frames.get(lane))
            emergency_count = int(
                sum(1 for item in emergency.get("predictions", [])
                    if item.get("is_emergency"))
            )
            emergency_lane_counts[lane] = emergency_count

            if emergency.get("detected"):
                confidence = float(emergency.get("confidence", 0.0))
                if confidence > best_confidence:
                    best_emergency = {
                        **emergency,
                        "lane": lane,
                        "direction": lane.replace("lane", ""),
                        "confidence": confidence,
                    }
                    best_confidence = confidence

        if best_emergency is None:
            best_emergency = {
                "detected": False,
                "label": None,
                "confidence": 0.0,
                "direction": None,
                "lane": None,
            }

        cache["emergency"] = best_emergency
        cache["emergency_lane_counts"] = emergency_lane_counts
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        return jsonify({
            "emergency_detected": best_emergency.get("detected", False),
            "lane": best_emergency.get("direction"),
            "label": best_emergency.get("label"),
            "confidence": float(best_emergency.get("confidence", 0.0)),
            "inference_ms": elapsed_ms,
            "mode": "yolov8s",
        })
    except Exception as exc:
        return jsonify({"error": f"Emergency detection failed: {str(exc)}"}), 500


@bp.post("/api/step3/detect-siren")
def step3_detect_siren():
    """Step 3: Siren Detection (TFLite) on audio."""
    try:
        cache = _get_demo_cache()
        t0 = time.perf_counter()

        # Decode audio if not already cached
        if "audio_bytes" not in cache or cache["audio_bytes"] is None:
            cache["audio_bytes"] = _decode_audio("sirenAudio")

        # Run siren detection
        siren = controller._siren_detector.detect(cache["audio_bytes"])
        cache["siren"] = siren
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        return jsonify({
            "siren_detected": bool(siren.get("detected", False)),
            "confidence": float(siren.get("confidence", 0.0)),
            "inference_ms": elapsed_ms,
            "mode": siren.get("mode", "tflite"),
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Siren detection failed: {str(exc)}"}), 500


@bp.post("/api/step4/predict-density")
def step4_predict_density():
    """Step 4: Density Prediction (XGBoost) for smoothing."""
    try:
        cache = _get_demo_cache()
        lane_counts = cache.get("lane_counts", {})
        t0 = time.perf_counter()

        if not lane_counts:
            return jsonify({"error": "No lane counts in cache. Run step 1 first."}), 400

        # Normalize counts
        normalized_counts = normalize_lane_counts(lane_counts)

        # Run density prediction
        density = controller._density_predictor.predict(normalized_counts)
        cache["density"] = density

        # Extract smoothed counts (typically direction-based)
        predictions = density.get("predictions", {})
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        return jsonify({
            "smoothed_counts": predictions,
            "horizon_sec": density.get("horizon_sec"),
            "inference_ms": elapsed_ms,
            "mode": density.get("mode", "xgboost"),
        })
    except Exception as exc:
        return jsonify({"error": f"Density prediction failed: {str(exc)}"}), 500


@bp.post("/api/step5/optimize-signal")
def step5_optimize_signal():
    """Step 5: DQN Signal Optimization."""
    try:
        cache = _get_demo_cache()
        lane_counts = cache.get("lane_counts", {})
        emergency = cache.get("emergency", {})
        siren = cache.get("siren", {})
        t0 = time.perf_counter()

        if not lane_counts:
            return jsonify({"error": "No lane counts in cache. Run steps 1-4 first."}), 400

        # Normalize counts
        normalized_counts = normalize_lane_counts(lane_counts)

        # Check for emergency+siren condition
        emergency_visual = bool(emergency.get("detected", False))
        siren_detected = bool(siren.get("detected", False))
        emergency_active = emergency_visual and siren_detected

        # Run full decision pipeline
        result = controller.decide_from_lane_counts(
            lane_counts=normalized_counts,
            frame=None,
            emergency_override=emergency if emergency_visual else None,
            siren_override=siren,
            cache_cycle_context=False,
        )

        cache["result"] = result
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        return jsonify({
            "direction": result.get("direction", "N"),
            "duration_s": int(result.get("duration", 10)),
            "mode": result.get("mode", "dqn"),
            "action": int(result.get("action", 0)),
            "q_value": float(result.get("q_value", 0.0)) if "q_value" in result else 0.5,
            "state_vector": result.get("state_vector", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "emergency_detected": emergency_active,
            "inference_ms": elapsed_ms,
        })
    except Exception as exc:
        return jsonify({"error": f"Signal optimization failed: {str(exc)}"}), 500


# ════════════════════════════════════════════════════════════
# Original Routes
# ════════════════════════════════════════════════════════════

@bp.get("/demo/orchestrator")
def demo_orchestrator():
    """Interactive 5-step demo with live pipeline visualization."""
    return render_template("demo_orchestrator.html")


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
        audio_bytes = _decode_audio("sirenAudio")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    fairness_mode = _normalize_fairness_mode(request.form.get("fairness_mode"))
    result = controller.decide_from_lane_frames(
        lane_frames,
        audio_bytes=audio_bytes,
        fairness_mode=fairness_mode,
    )
    payload = _build_simulation_payload(result)
    return jsonify(_build_response(result, payload))


@bp.post("/api/next_cycle")
def next_cycle():
    body = request.get_json(silent=True) or {}
    lane_counts_raw = body.get("lane_counts")
    if not isinstance(lane_counts_raw, dict):
        return jsonify({"error": "lane_counts must be an object"}), 400

    lane_counts = {lane: int(lane_counts_raw.get(lane, 0))
                   for lane in LANE_KEYS}
    fairness_mode = _normalize_fairness_mode(body.get("fairness_mode"))
    current_active_lane = body.get("current_active_lane")

    try:
        result = controller.decide_next_cycle_from_lane_counts(
            lane_counts=lane_counts,
            current_active_lane=str(current_active_lane or "").strip() or None,
            fairness_mode=fairness_mode,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    payload = _build_simulation_payload(result)
    return jsonify(_build_response(result, payload))
