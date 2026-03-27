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
from control.rl_agent import RLAgent
from detection import VehicleDetector
from detection.camera import CameraStream, generate_annotated_stream
from detection.traffic_predictor import TrafficDensityPredictor


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
        "phase": 0,  # 0 = NS green, 1 = EW green
        "phase_start_time": time.time(),
        "last_counts": {"N": 0, "S": 0, "E": 0, "W": 0},
        "last_predictions": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        "ambulance_detected": False,
        "ambulance_direction": -1,
        "current_green_duration": 30,  # seconds chosen by RL
        "green_expires_at": time.time() + 30,
    }

    # ── Simulation state (for /simulation route) ───────────────────────────
    # Stores uploaded images and running simulation state for multi-cycle demo
    _simulation_state: dict[str, Any] = {
        "initialized": False,
        "cycle": 0,
        "phase": 0,  # 0 = NS green, 1 = EW green
        "green_duration": 30,
        "phase_steps_completed": 0,
        "single_cycle_complete": False,
        "input_mode": "upload",  # "upload" or "manual"
        "vehicle_counts": {"N": 0, "S": 0, "E": 0, "W": 0},
        "predicted_densities": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        "emergency_detected": False,
        "emergency_lane": None,  # "N", "S", "E", or "W"
        "emergency_class": None,  # "ambulance", "fire-truck", "police"
        "lane_images_b64": {},  # Base64-encoded annotated images
        "detection_results": {},  # Per-lane detection details
        "confidence": 0.0,  # Predictor confidence (0-1)
        "vehicles_passed_total": 0,
        "wait_time_sum": 0.0,
        "started_at": 0.0,
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
        annotated = detector.draw_vehicle_count(result.annotated_frame, result.vehicle_count)
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
                    lane_images[lane_key] = base64.b64encode(buf.tobytes()).decode("utf-8")

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
                        "vehicle_counts": dir_counts,
                        "predicted_densities": predicted_densities,
                        "current_phase": phase,
                        "time_in_phase": 0.0,
                        "ambulance_detected": False,
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
                    "phase": recommended_phase,
                    "phase_label": "NS GREEN" if recommended_phase == 0 else "EW GREEN",
                    "green_duration": green_duration,
                    "signal_states": signal_states,
                    "vehicle_counts": dir_counts,
                    "predicted_densities": predicted_densities,
                    "rl_model_loaded": agent.is_loaded,
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
                    _, buf = cv2.imencode(".jpg", classification["annotated_frame"])
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
                prediction = predictor.predict(current_densities, prediction_seconds=pred_horizon)

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
                            is_recent = idx >= len(hist) - 3 and idx < len(hist)
                            opacity = 0.5 + (idx / len(hist)) * 0.5

                            bars.append(
                                {
                                    "value": int(value),
                                    "opacity": round(opacity, 2),
                                    "is_very_recent": is_very_recent,
                                    "is_recent": is_recent,
                                }
                            )
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
            except FileNotFoundError as e:
                prediction_result = {"error": f"Model file missing: {str(e)}"}
            except RuntimeError as e:
                prediction_result = {"error": f"Model loading error: {str(e)}"}
            except Exception as e:
                prediction_result = {"error": f"Prediction failed: {str(e)}"}

        return render_template("test_traffic_prediction.html", result=prediction_result)

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
                _signal_state["ambulance_direction"] = int(data.get("ambulance_direction", -1))

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
                    "vehicle_counts": _signal_state["last_counts"],
                    "predicted_densities": _signal_state["last_predictions"],
                    "current_phase": _signal_state["phase"],
                    "time_in_phase": 0.0,
                    "ambulance_detected": _signal_state["ambulance_detected"],
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
        else:  # EW green
            signal_states = {"N": "RED", "S": "RED", "E": "GREEN", "W": "GREEN"}

        # Yellow transition: last 3 seconds of a phase
        if time_remaining <= 3.0:
            for lane in ["N", "S"] if phase == 0 else ["E", "W"]:
                signal_states[lane] = "YELLOW"

        return jsonify(
            {
                "phase": phase,
                "green_duration": green_dur,
                "time_in_phase": round(time_in_phase, 1),
                "time_remaining": round(time_remaining, 1),
                "signal_states": signal_states,
                "ambulance_detected": _signal_state["ambulance_detected"],
                "ambulance_direction": _signal_state["ambulance_direction"],
                "rl_model_loaded": agent.is_loaded,
                "predicted_densities": _signal_state["last_predictions"],
                "vehicle_counts": _signal_state["last_counts"],
            }
        )

    # ── Full Pipeline Simulation routes ───────────────────────────────────

    # Lane to direction index mapping for RL agent
    LANE_TO_DIRECTION = {"N": 0, "S": 1, "E": 2, "W": 3}
    # Phase to lane mapping (4-phase system: one direction at a time)
    PHASE_TO_LANE = {0: "N", 1: "S", 2: "E", 3: "W"}
    PHASE_LABELS = {0: "NORTH GREEN", 1: "SOUTH GREEN", 2: "EAST GREEN", 3: "WEST GREEN"}
    EMERGENCY_CLASSES = {"ambulance", "fire-truck", "police"}

    def _build_signal_states(phase: int, yellow: bool = False) -> dict[str, str]:
        """Build signal states dict for given phase (4-phase system).

        Phase 0: N green, all others red
        Phase 1: S green, all others red
        Phase 2: E green, all others red
        Phase 3: W green, all others red
        """
        green_lane = PHASE_TO_LANE[phase]
        states = {"N": "RED", "S": "RED", "E": "RED", "W": "RED"}
        states[green_lane] = "YELLOW" if yellow else "GREEN"
        return states

    def _get_rl_decision_for_phase(
        phase: int,
        vehicle_counts: dict[str, int],
        predicted_densities: dict[str, float],
        emergency_detected: bool,
        emergency_lane: str | None,
    ) -> int:
        """Get RL agent decision adapted for 4-phase system.

        The RL model was trained on 2-phase (NS vs EW), so we need to:
        1. Map current 4-phase to 2-phase for RL inference
        2. Return green duration for the active lane
        """
        # Map 4-phase to 2-phase for RL model
        # Phase 0 (N) or 1 (S) → RL phase 0 (NS)
        # Phase 2 (E) or 3 (W) → RL phase 1 (EW)
        rl_phase = 0 if phase in [0, 1] else 1

        amb_dir = LANE_TO_DIRECTION.get(emergency_lane, -1) if emergency_lane else -1
        detections = {
            "vehicle_counts": vehicle_counts,
            "predicted_densities": predicted_densities,
            "current_phase": rl_phase,  # Use 2-phase for RL model
            "time_in_phase": 0.0,
            "ambulance_detected": emergency_detected,
            "ambulance_direction": amb_dir,
        }
        return agent.get_action(detections)

    def _calculate_stats() -> dict[str, float]:
        started = float(_simulation_state["started_at"]) or time.time()
        elapsed_minutes = max((time.time() - started) / 60.0, 1 / 60.0)
        vehicles_passed = int(_simulation_state["vehicles_passed_total"])
        throughput = vehicles_passed / elapsed_minutes
        avg_wait = (
            float(_simulation_state["wait_time_sum"]) / vehicles_passed
            if vehicles_passed > 0
            else 0.0
        )
        return {
            "vehicles_passed": vehicles_passed,
            "avg_wait_seconds": round(avg_wait, 1),
            "throughput_per_min": round(throughput, 1),
        }

    def _build_model_outputs(
        *,
        vehicle_counts: dict[str, int],
        predicted_densities: dict[str, float],
        confidence: float,
        green_duration: int,
        phase: int,
        emergency_detected: bool,
        emergency_lane: str | None,
        detection_results: dict[str, dict[str, Any]],
        input_mode: str,
    ) -> dict[str, Any]:
        next_phase = (phase + 1) % 4
        return {
            "detection": {
                "model": "YOLOv8s",
                "status": "complete" if input_mode == "upload" else "skipped-manual",
                "vehicle_counts": vehicle_counts,
                "per_lane": detection_results,
            },
            "emergency": {
                "model": "YOLOv8-cls",
                "status": "active" if emergency_detected else "clear",
                "lane": emergency_lane,
            },
            "prediction": {
                "model": "XGBoost",
                "horizon_seconds": 60,
                "confidence": round(confidence, 2),
                "current": vehicle_counts,
                "predicted": predicted_densities,
            },
            "rl": {
                "model": "DQN",
                "model_loaded": agent.is_loaded,
                "decision_seconds": green_duration,
                "current_phase": PHASE_LABELS[phase],
                "next_phase": PHASE_LABELS[next_phase],
                "reason": (
                    f"Emergency priority for lane {emergency_lane}"
                    if emergency_detected and emergency_lane
                    else "Balanced decision from traffic state"
                ),
            },
        }

    def _drain_active_lane_counts(
        vehicle_counts: dict[str, int], active_lane: str, green_duration: int
    ) -> tuple[dict[str, int], int]:
        updated = dict(vehicle_counts)
        current = int(updated.get(active_lane, 0))
        if current <= 0:
            return updated, 0
        max_drain = max(1, green_duration // 6)
        drained = min(current, max_drain)
        updated[active_lane] = current - drained
        return updated, drained

    @app.route("/simulation", methods=["GET", "POST"])
    def simulation():
        """Full pipeline simulation with 4-way intersection visualization.

        GET: Render upload form
        POST: Process 4 lane images, run full model pipeline, initialize simulation
        """
        if request.method == "GET":
            return render_template("simulation.html")

        lane_keys = ["laneN", "laneS", "laneE", "laneW"]
        lane_map = {"laneN": "N", "laneS": "S", "laneE": "E", "laneW": "W"}
        vehicle_counts: dict[str, int] = {}
        lane_images_b64: dict[str, str] = {}
        detection_results: dict[str, dict[str, Any]] = {}
        emergency_detected = False
        emergency_lane: str | None = None
        emergency_class: str | None = None
        input_mode = request.form.get("input_mode", "").strip().lower()
        manual_counts_supplied = all(
            request.form.get(f"count{lane}") not in (None, "") for lane in ["N", "S", "E", "W"]
        )
        uploaded_all = all(k in request.files and request.files[k].filename for k in lane_keys)

        if input_mode not in {"manual", "upload"}:
            input_mode = "manual" if manual_counts_supplied and not uploaded_all else "upload"

        if input_mode == "manual":
            try:
                vehicle_counts = {
                    lane: max(0, int(request.form.get(f"count{lane}", "0")))
                    for lane in ["N", "S", "E", "W"]
                }
            except ValueError:
                return jsonify({"error": "Manual counts must be valid non-negative integers."}), 400
            emergency_lane_input = request.form.get("emergency_lane", "").strip().upper()
            if emergency_lane_input in {"N", "S", "E", "W"}:
                emergency_detected = True
                emergency_lane = emergency_lane_input
                emergency_class = "manual-priority"
            for lane in ["N", "S", "E", "W"]:
                detection_results[lane] = {
                    "vehicle_count": vehicle_counts[lane],
                    "emergency_class": "manual-input",
                    "emergency_confidence": 1.0 if emergency_lane == lane else 0.0,
                }
        else:
            missing = [
                k for k in lane_keys if k not in request.files or not request.files[k].filename
            ]
            if missing:
                missing_lanes = [lane_map[k] for k in missing]
                return jsonify(
                    {
                        "error": (
                            "All 4 lanes required for upload mode. "
                            f"Missing: {', '.join(missing_lanes)}"
                        )
                    }
                ), 400

            for lane_key in lane_keys:
                lane = lane_map[lane_key]
                file = request.files[lane_key]

                try:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if img is None:
                        detection_results[lane] = {"error": "Invalid image format"}
                        vehicle_counts[lane] = 0
                        continue

                    result = detector.detect(img)
                    vehicle_counts[lane] = result.vehicle_count

                    emergency_result = detector.classify_emergency_vehicle(img)
                    emg_class = emergency_result.get("class_name", "normal")
                    emg_conf = emergency_result.get("confidence", 0.0)

                    if emg_class in EMERGENCY_CLASSES and emg_conf > 0.5:
                        emergency_detected = True
                        emergency_lane = lane
                        emergency_class = emg_class

                    annotated = detector.draw_vehicle_count(
                        result.annotated_frame, result.vehicle_count
                    )
                    _, buf = cv2.imencode(".jpg", annotated)
                    lane_images_b64[lane] = base64.b64encode(buf.tobytes()).decode("utf-8")

                    detection_results[lane] = {
                        "vehicle_count": result.vehicle_count,
                        "emergency_class": emg_class,
                        "emergency_confidence": round(emg_conf, 2),
                    }

                except Exception as e:
                    detection_results[lane] = {"error": str(e)}
                    vehicle_counts[lane] = 0

        # Update predictor history
        predictor.update_history(vehicle_counts)
        prediction = predictor.predict(vehicle_counts, prediction_seconds=60)

        # Determine initial phase (0=N, 1=S, 2=E, 3=W) based on traffic load
        # If emergency, start with that lane
        if emergency_detected and emergency_lane:
            initial_phase = LANE_TO_DIRECTION[emergency_lane]  # 0/1/2/3
        else:
            # Choose lane with highest density
            max_lane = max(vehicle_counts.items(), key=lambda x: x[1])[0]
            initial_phase = LANE_TO_DIRECTION[max_lane]

        # Get RL agent decision for this phase
        green_duration = _get_rl_decision_for_phase(
            initial_phase,
            vehicle_counts,
            prediction.predicted_densities,
            emergency_detected,
            emergency_lane,
        )

        # Calculate confidence
        confidence = sum(prediction.confidence_scores.values()) / 4

        # Update simulation state
        _simulation_state.update(
            {
                "initialized": True,
                "cycle": 1,
                "phase": initial_phase,
                "green_duration": green_duration,
                "phase_steps_completed": 1,
                "single_cycle_complete": False,
                "input_mode": input_mode,
                "vehicle_counts": vehicle_counts,
                "predicted_densities": prediction.predicted_densities,
                "emergency_detected": emergency_detected,
                "emergency_lane": emergency_lane,
                "emergency_class": emergency_class,
                "lane_images_b64": lane_images_b64,
                "detection_results": detection_results,
                "confidence": round(confidence, 2),
                "vehicles_passed_total": 0,
                "wait_time_sum": 0.0,
                "started_at": time.time(),
            }
        )
        model_outputs = _build_model_outputs(
            vehicle_counts=vehicle_counts,
            predicted_densities=prediction.predicted_densities,
            confidence=confidence,
            green_duration=green_duration,
            phase=initial_phase,
            emergency_detected=emergency_detected,
            emergency_lane=emergency_lane,
            detection_results=detection_results,
            input_mode=input_mode,
        )

        # Build response
        return jsonify(
            {
                "cycle": 1,
                "phase": initial_phase,
                "phase_label": PHASE_LABELS[initial_phase],
                "green_duration": green_duration,
                "signal_states": _build_signal_states(initial_phase),
                "single_cycle_complete": False,
                "steps_completed": 1,
                "vehicle_counts": vehicle_counts,
                "predicted_densities": prediction.predicted_densities,
                "emergency_detected": emergency_detected,
                "emergency_lane": emergency_lane,
                "emergency_class": emergency_class,
                "rl_model_loaded": agent.is_loaded,
                "confidence": round(confidence, 2),
                "lane_images": {
                    k: f"data:image/jpeg;base64,{v}" for k, v in lane_images_b64.items()
                },
                "detection_results": detection_results,
                "model_outputs": model_outputs,
                "stats": _calculate_stats(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    @app.route("/simulation/next_cycle", methods=["POST"])
    def simulation_next_cycle():
        """Advance simulation to next phase/cycle (4-phase system).

        Returns updated signal state for the next phase.
        """
        if not _simulation_state["initialized"]:
            return (
                jsonify({"error": "Simulation not initialized. POST to /simulation first."}),
                400,
            )

        # Advance to next phase (0→1→2→3→0)
        current_phase = _simulation_state["phase"]
        emergency_detected = _simulation_state["emergency_detected"]
        emergency_lane = _simulation_state["emergency_lane"]
        steps_completed = int(_simulation_state["phase_steps_completed"])
        if _simulation_state["single_cycle_complete"]:
            return jsonify(
                {
                    "cycle": _simulation_state["cycle"],
                    "phase": current_phase,
                    "phase_label": PHASE_LABELS[current_phase],
                    "green_duration": _simulation_state["green_duration"],
                    "signal_states": _build_signal_states(current_phase),
                    "single_cycle_complete": True,
                    "steps_completed": steps_completed,
                    "vehicle_counts": _simulation_state["vehicle_counts"],
                    "predicted_densities": _simulation_state["predicted_densities"],
                    "emergency_detected": emergency_detected,
                    "emergency_lane": emergency_lane,
                    "emergency_class": _simulation_state["emergency_class"],
                    "rl_model_loaded": agent.is_loaded,
                    "confidence": _simulation_state["confidence"],
                    "model_outputs": _build_model_outputs(
                        vehicle_counts=_simulation_state["vehicle_counts"],
                        predicted_densities=_simulation_state["predicted_densities"],
                        confidence=_simulation_state["confidence"],
                        green_duration=_simulation_state["green_duration"],
                        phase=current_phase,
                        emergency_detected=emergency_detected,
                        emergency_lane=emergency_lane,
                        detection_results=_simulation_state["detection_results"],
                        input_mode=_simulation_state["input_mode"],
                    ),
                    "stats": _calculate_stats(),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

        active_lane = PHASE_TO_LANE[current_phase]
        updated_counts, drained = _drain_active_lane_counts(
            _simulation_state["vehicle_counts"],
            active_lane,
            int(_simulation_state["green_duration"]),
        )
        _simulation_state["vehicle_counts"] = updated_counts
        _simulation_state["vehicles_passed_total"] += drained
        _simulation_state["wait_time_sum"] += sum(updated_counts.values()) * 1.0

        # Emergency override: stay in phase that serves emergency lane
        if emergency_detected and emergency_lane:
            required_phase = LANE_TO_DIRECTION[emergency_lane]
            new_phase = required_phase  # Keep green for emergency lane
        else:
            # Normal operation: advance to next phase (4-phase cycle)
            new_phase = (current_phase + 1) % 4

        # Increment cycle if we wrapped back to phase 0
        new_cycle = _simulation_state["cycle"]
        if new_phase == 0 and current_phase == 3:
            new_cycle += 1

        steps_completed += 1
        single_cycle_complete = steps_completed >= 4

        # Update predictor and get new prediction
        predictor.update_history(_simulation_state["vehicle_counts"])
        prediction = predictor.predict(_simulation_state["vehicle_counts"], prediction_seconds=60)

        # Get RL decision for new phase
        green_duration = _get_rl_decision_for_phase(
            new_phase,
            _simulation_state["vehicle_counts"],
            prediction.predicted_densities,
            emergency_detected,
            emergency_lane,
        )
        confidence = sum(prediction.confidence_scores.values()) / 4

        # Update state
        _simulation_state.update(
            {
                "cycle": new_cycle,
                "phase": new_phase,
                "green_duration": green_duration,
                "predicted_densities": prediction.predicted_densities,
                "confidence": round(confidence, 2),
                "phase_steps_completed": steps_completed,
                "single_cycle_complete": single_cycle_complete,
            }
        )
        model_outputs = _build_model_outputs(
            vehicle_counts=_simulation_state["vehicle_counts"],
            predicted_densities=prediction.predicted_densities,
            confidence=confidence,
            green_duration=green_duration,
            phase=new_phase,
            emergency_detected=emergency_detected,
            emergency_lane=emergency_lane,
            detection_results=_simulation_state["detection_results"],
            input_mode=_simulation_state["input_mode"],
        )

        return jsonify(
            {
                "cycle": new_cycle,
                "phase": new_phase,
                "phase_label": PHASE_LABELS[new_phase],
                "green_duration": green_duration,
                "signal_states": _build_signal_states(new_phase),
                "single_cycle_complete": single_cycle_complete,
                "steps_completed": steps_completed,
                "vehicle_counts": _simulation_state["vehicle_counts"],
                "predicted_densities": prediction.predicted_densities,
                "emergency_detected": emergency_detected,
                "emergency_lane": emergency_lane,
                "emergency_class": _simulation_state["emergency_class"],
                "rl_model_loaded": agent.is_loaded,
                "confidence": round(confidence, 2),
                "model_outputs": model_outputs,
                "stats": _calculate_stats(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    @app.route("/simulation/ambulance", methods=["POST"])
    def simulation_ambulance():
        """Set or clear emergency lane during simulation."""
        if not _simulation_state["initialized"]:
            return jsonify({"error": "Simulation not initialized."}), 400
        payload = request.get_json(silent=True) or {}
        lane = str(payload.get("lane", "")).upper()
        if lane == "CLEAR":
            _simulation_state["emergency_detected"] = False
            _simulation_state["emergency_lane"] = None
            _simulation_state["emergency_class"] = None
        elif lane in {"N", "S", "E", "W"}:
            _simulation_state["emergency_detected"] = True
            _simulation_state["emergency_lane"] = lane
            _simulation_state["emergency_class"] = "manual-priority"
        else:
            return jsonify({"error": "lane must be one of N/S/E/W or CLEAR"}), 400
        return jsonify(
            {
                "emergency_detected": _simulation_state["emergency_detected"],
                "emergency_lane": _simulation_state["emergency_lane"],
                "emergency_class": _simulation_state["emergency_class"],
            }
        )

    @app.route("/simulation/reset", methods=["POST"])
    def simulation_reset():
        """Reset simulation state."""
        _simulation_state.update(
            {
                "initialized": False,
                "cycle": 0,
                "phase": 0,
                "green_duration": 30,
                "phase_steps_completed": 0,
                "single_cycle_complete": False,
                "input_mode": "upload",
                "vehicle_counts": {"N": 0, "S": 0, "E": 0, "W": 0},
                "predicted_densities": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
                "emergency_detected": False,
                "emergency_lane": None,
                "emergency_class": None,
                "lane_images_b64": {},
                "detection_results": {},
                "confidence": 0.0,
                "vehicles_passed_total": 0,
                "wait_time_sum": 0.0,
                "started_at": 0.0,
            }
        )
        return jsonify({"status": "reset", "message": "Simulation state cleared"})

    return app


# Allow running directly with `python gui/app.py` for convenience
if __name__ == "__main__":
    create_app().run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
