"""Flask routes for the adaptive traffic signal GUI.

All detection / inference logic lives in the ``detection`` package.
This module is a thin controller that handles HTTP requests and
renders templates — nothing more.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from detection import VehicleDetector
from detection.camera import CameraStream, generate_annotated_stream
from detection.traffic_predictor import TrafficDensityPredictor
from control.rl_agent import RLAgent


def create_app() -> Flask:
    """Application factory — builds and returns a configured Flask app."""
    app = Flask(__name__)

    # Shared detector instance (loaded once, reused across requests)
    # Load emergency model in detector for classification endpoint
    detector = VehicleDetector(load_emergency=True)

    # Traffic density predictor for adaptive signal timing
    predictor = TrafficDensityPredictor()

    # RL signal controller — loads signal_policy.zip once at startup
    agent = RLAgent()

    # ── Signal state tracker (shared across requests) ──────────────────
    # Tracks the current phase, how long we've been in it, and the last
    # vehicle counts so /signal_decision can work without camera input.
    _signal_state: dict[str, Any] = {
        "phase": 0,               # 0 = NS green, 1 = EW green
        "phase_start_time": time.time(),
        "last_counts": {"N": 0, "S": 0, "E": 0, "W": 0},
        "last_predictions": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        "ambulance_detected": False,
        "ambulance_direction": -1,
        "current_green_duration": 30,   # seconds chosen by RL
        "green_expires_at": time.time() + 30,
    }

    # ── Routes ───────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        camera = CameraStream()
        try:
            camera.open()
        except RuntimeError:
            camera = None  # generate_annotated_stream will serve placeholder frames
        return Response(
            generate_annotated_stream(detector, camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/predict_frame", methods=["POST"])
    def predict_frame():
        """Accept a single JPEG frame from the browser camera and return
        the annotated frame + vehicle count as JSON."""
        file = request.files.get("frame")
        if not file:
            return jsonify({"error": "No frame uploaded"}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        result = detector.detect(img)  # type: ignore[arg-type]
        annotated = detector.draw_vehicle_count(
            result.annotated_frame, result.vehicle_count
        )
        _, buf = cv2.imencode(".jpg", annotated)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return jsonify({"annotated": b64, "vehicle_count": result.vehicle_count})

    # Lane identifiers for the four-way intersection
    LANES = {
        "laneN": "North",
        "laneS": "South",
        "laneE": "East",
        "laneW": "West",
    }

    @app.route("/test", methods=["GET", "POST"])
    def test():
        lane_counts: dict[str, int] = {}
        lane_images: dict[str, str] = {}
        signal_result: dict[str, Any] = {}

        if request.method == "POST":
            for lane_key in LANES:
                file = request.files.get(lane_key)
                if file and file.filename:
                    # Decode the uploaded image
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    # Run YOLO detection
                    result = detector.detect(img)  # type: ignore[arg-type]
                    annotated = detector.draw_vehicle_count(
                        result.annotated_frame, result.vehicle_count
                    )
                    lane_counts[lane_key] = result.vehicle_count

                    # Encode to base64 for embedding in the template
                    _, buf = cv2.imencode(".jpg", annotated)
                    lane_images[lane_key] = base64.b64encode(
                        buf.tobytes()).decode("utf-8")

            if lane_counts:
                # Convert laneN/S/E/W keys → N/S/E/W for the models
                dir_counts = {
                    "N": lane_counts.get("laneN", 0),
                    "S": lane_counts.get("laneS", 0),
                    "E": lane_counts.get("laneE", 0),
                    "W": lane_counts.get("laneW", 0),
                }

                # XGBoost: predict density 60 s ahead
                predictor.update_history(dir_counts)
                pred = predictor.predict(dir_counts, prediction_seconds=60)
                predicted_densities = pred.predicted_densities

                # Try both phases and pick the one the RL agent prefers
                # (phase 0 = NS green, phase 1 = EW green)
                results_by_phase: dict[int, int] = {}
                for phase in (0, 1):
                    detections = {
                        "vehicle_counts":      dir_counts,
                        "predicted_densities": predicted_densities,
                        "current_phase":       phase,
                        "time_in_phase":       0.0,
                        "ambulance_detected":  False,
                        "ambulance_direction": -1,
                    }
                    results_by_phase[phase] = agent.get_action(detections)

                # Heuristic: pick the phase that serves the heavier pair
                ns_load = dir_counts["N"] + dir_counts["S"]
                ew_load = dir_counts["E"] + dir_counts["W"]
                recommended_phase = 0 if ns_load >= ew_load else 1
                green_duration = results_by_phase[recommended_phase]

                # Build per-direction signal colour map
                if recommended_phase == 0:
                    signal_states = {"N": "GREEN", "S": "GREEN", "E": "RED", "W": "RED"}
                else:
                    signal_states = {"N": "RED", "S": "RED", "E": "GREEN", "W": "GREEN"}

                signal_result = {
                    "phase":               recommended_phase,
                    "phase_label":         "NS GREEN" if recommended_phase == 0 else "EW GREEN",
                    "green_duration":      green_duration,
                    "signal_states":       signal_states,
                    "vehicle_counts":      dir_counts,
                    "predicted_densities": predicted_densities,
                    "rl_model_loaded":     agent.is_loaded,
                }

        return render_template(
            "test.html",
            lane_counts=lane_counts,
            lane_images=lane_images,
            lanes=LANES,
            signal_result=signal_result,
        )

    @app.route("/test_emergency", methods=["GET", "POST"])
    def test_emergency():
        """Test route for emergency vehicle classifier (Model 3).

        GET: Renders upload form
        POST: Accepts a vehicle image and returns classification results
        """
        result: dict[str, Any] = {}

        if request.method == "POST":
            file = request.files.get("vehicle_image")
            if not file or not file.filename:
                result = {"error": "No image uploaded"}
            else:
                # Decode the uploaded image
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    result = {"error": "Could not decode image"}
                else:
                    # Classify with emergency vehicle model
                    classification = detector.classify_emergency_vehicle(img)

                    # Encode annotated frame to base64
                    _, buf = cv2.imencode(
                        ".jpg", classification["annotated_frame"]
                    )
                    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

                    result = {
                        "class_name": classification.get("class_name"),
                        "confidence": classification.get("confidence"),
                        "annotated": b64,
                        "all_classes": classification.get("all_classes", {}),
                    }

        return render_template("test_emergency.html", result=result)

    @app.route("/test_traffic_prediction", methods=["GET", "POST"])
    def test_traffic_prediction():
        """Test route for traffic density predictor (Model 3).

        GET: Renders input form for current lane densities
        POST: Accepts current densities and returns predictions
        """
        prediction_result: dict[str, Any] = {}

        if request.method == "POST":
            try:
                # Get current densities from form
                current_densities = {
                    "N": float(request.form.get("density_N", 0)),
                    "S": float(request.form.get("density_S", 0)),
                    "E": float(request.form.get("density_E", 0)),
                    "W": float(request.form.get("density_W", 0)),
                }

                # Get prediction horizon (default 30 seconds)
                pred_horizon = int(request.form.get("horizon", 30))
                pred_horizon = max(10, min(90, pred_horizon))  # Clamp 10-90s

                # Update predictor history and get prediction
                predictor.update_history(current_densities)
                prediction = predictor.predict(
                    current_densities, prediction_seconds=pred_horizon
                )

                # Pre-compute history visualization data in Python
                history_data = {}
                for lane in ["N", "S", "E", "W"]:
                    hist = predictor.get_history(lane)
                    if hist:
                        max_val = max(hist) if hist else 1
                        max_val = max(max_val, 1)  # Avoid division by zero

                        bars = []
                        for idx, value in enumerate(hist):
                            is_very_recent = idx == len(hist) - 1
                            is_recent = idx >= len(
                                hist) - 3 and idx < len(hist)
                            opacity = 0.5 + (idx / len(hist)) * 0.5

                            bars.append({
                                "value": int(value),
                                "opacity": round(opacity, 2),
                                "is_very_recent": is_very_recent,
                                "is_recent": is_recent,
                            })
                        history_data[lane] = bars
                    else:
                        history_data[lane] = []

                # Pre-compute change indicators
                changes = {}
                for lane in ["N", "S", "E", "W"]:
                    current = current_densities[lane]
                    predicted = prediction.predicted_densities[lane]
                    change = predicted - current

                    if change > 0.5:
                        changes[lane] = {
                            "direction": "up",
                            "text": f"↑ +{int(change)} vehicles (Increasing)",
                        }
                    elif change < -0.5:
                        changes[lane] = {
                            "direction": "down",
                            "text": f"↓ {int(change)} vehicles (Decreasing)",
                        }
                    else:
                        changes[lane] = {
                            "direction": "stable",
                            "text": "～ No change (Stable)",
                        }

                # Format for template
                prediction_result = {
                    "current_densities": current_densities,
                    "predicted_densities": prediction.predicted_densities,
                    "confidence_scores": {
                        lane: round(conf * 100)
                        for lane, conf in prediction.confidence_scores.items()
                    },
                    "prediction_horizon": prediction.prediction_horizon,
                    "timestamp": prediction.timestamp,
                    "history": history_data,
                    "changes": changes,
                }
            except (ValueError, TypeError) as e:
                prediction_result = {"error": f"Invalid input: {str(e)}"}

        return render_template(
            "test_traffic_prediction.html", result=prediction_result
        )

    # ── RL Signal Decision endpoint ───────────────────────────────────────

    @app.route("/signal_decision", methods=["GET", "POST"])
    def signal_decision():
        """Return the RL agent's recommended signal state.

        GET  — returns current cached signal state (no inference).
        POST — accepts new detection data, runs RL inference, updates state.

        POST body (JSON), all fields optional:
        {
            "vehicle_counts":      {"N": 7, "S": 3, "E": 12, "W": 1},
            "ambulance_detected": true,
            "ambulance_direction": 2
        }

        Response JSON:
        {
            "phase":              0,           // 0=NS green, 1=EW green
            "green_duration":     30,          // seconds RL chose
            "time_in_phase":      12.4,        // seconds elapsed
            "time_remaining":     17.6,        // seconds left
            "signal_states":  {"N":"GREEN", "S":"GREEN", "E":"RED", "W":"RED"},
            "ambulance_detected": false,
            "ambulance_direction": -1,
            "rl_model_loaded":    true,
            "predicted_densities": {"N":8.0, "S":3.5, "E":13.0, "W":1.2}
        }
        """
        now = time.time()

        if request.method == "POST":
            data = request.get_json(silent=True) or {}

            # Update counts if provided
            if "vehicle_counts" in data:
                counts = data["vehicle_counts"]
                _signal_state["last_counts"] = {
                    k: int(counts.get(k, 0)) for k in ["N", "S", "E", "W"]
                }
                # Update predictor history
                predictor.update_history(_signal_state["last_counts"])

            # Update ambulance state if provided
            if "ambulance_detected" in data:
                _signal_state["ambulance_detected"] = bool(data["ambulance_detected"])
                _signal_state["ambulance_direction"] = int(
                    data.get("ambulance_direction", -1)
                )

            # Update XGBoost predictions
            pred = predictor.predict(_signal_state["last_counts"], prediction_seconds=60)
            _signal_state["last_predictions"] = pred.predicted_densities

            # Check if current green phase has expired → let RL decide next duration
            if now >= _signal_state["green_expires_at"]:
                # Flip phase
                _signal_state["phase"] = 1 - _signal_state["phase"]
                _signal_state["phase_start_time"] = now

                # Ask RL agent for new green duration
                detections = {
                    "vehicle_counts":      _signal_state["last_counts"],
                    "predicted_densities": _signal_state["last_predictions"],
                    "current_phase":       _signal_state["phase"],
                    "time_in_phase":       0.0,
                    "ambulance_detected":  _signal_state["ambulance_detected"],
                    "ambulance_direction": _signal_state["ambulance_direction"],
                }
                green_dur = agent.get_action(detections)
                _signal_state["current_green_duration"] = green_dur
                _signal_state["green_expires_at"] = now + green_dur

        # ── Build response ────────────────────────────────────────────────
        phase = _signal_state["phase"]
        phase_start = _signal_state["phase_start_time"]
        green_dur = _signal_state["current_green_duration"]
        time_in_phase = now - phase_start
        time_remaining = max(0.0, _signal_state["green_expires_at"] - now)

        # Map phase → signal colours for all 4 directions
        if phase == 0:  # NS green
            signal_states = {"N": "GREEN", "S": "GREEN", "E": "RED", "W": "RED"}
        else:           # EW green
            signal_states = {"N": "RED", "S": "RED", "E": "GREEN", "W": "GREEN"}

        # Yellow transition: last 3 seconds of a phase
        if time_remaining <= 3.0:
            for lane in (["N", "S"] if phase == 0 else ["E", "W"]):
                signal_states[lane] = "YELLOW"

        return jsonify({
            "phase":               phase,
            "green_duration":      green_dur,
            "time_in_phase":       round(time_in_phase, 1),
            "time_remaining":      round(time_remaining, 1),
            "signal_states":       signal_states,
            "ambulance_detected":  _signal_state["ambulance_detected"],
            "ambulance_direction": _signal_state["ambulance_direction"],
            "rl_model_loaded":     agent.is_loaded,
            "predicted_densities": _signal_state["last_predictions"],
            "vehicle_counts":      _signal_state["last_counts"],
        })

    return app


# Allow running directly with `python gui/app.py` for convenience
if __name__ == "__main__":
    create_app().run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
