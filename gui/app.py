"""Flask routes for the adaptive traffic signal GUI.

All detection / inference logic lives in the ``detection`` package.
This module is a thin controller that handles HTTP requests and
renders templates — nothing more.
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
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

        if request.method == "POST":
            for lane_key in LANES:
                file = request.files.get(lane_key)
                if file and file.filename:
                    # Decode the uploaded image
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    # Run detection
                    result = detector.detect(img)  # type: ignore[arg-type]
                    annotated = detector.draw_vehicle_count(
                        result.annotated_frame, result.vehicle_count
                    )
                    lane_counts[lane_key] = result.vehicle_count

                    # Encode to base64 for embedding in the template
                    _, buf = cv2.imencode(".jpg", annotated)
                    lane_images[lane_key] = base64.b64encode(
                        buf.tobytes()).decode("utf-8")

        return render_template(
            "test.html",
            lane_counts=lane_counts,
            lane_images=lane_images,
            lanes=LANES,
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

    return app


# Allow running directly with `python gui/app.py` for convenience
if __name__ == "__main__":
    create_app().run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
