"""
Flask application for the traffic simulation GUI.

Provides SSE streaming of simulation state and control endpoints.
Runs on port 5001 to avoid conflict with main app on port 5000.
"""

from __future__ import annotations

import json
import time

from flask import Flask, Response, jsonify, render_template, request

from simulation_gui.engine import SCENARIOS, SimConfig, SimulationEngine


def create_app() -> Flask:
    """Application factory for the simulation GUI."""
    app = Flask(__name__)

    # Single engine instance, started when app starts
    engine = SimulationEngine(SimConfig())
    engine.start()

    @app.route("/")
    def index():
        """Serve the main simulation page."""
        return render_template("simulation.html", scenarios=SCENARIOS)

    @app.route("/api/state")
    def state_stream():
        """
        Server-Sent Events endpoint. Frontend connects once and receives
        a snapshot every second.
        """

        def generate():
            while True:
                snapshot = engine.get_snapshot()
                yield f"data: {json.dumps(snapshot)}\n\n"
                time.sleep(1.0)  # Fixed 1Hz updates regardless of sim speed

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/set_lane_counts", methods=["POST"])
    def set_lane_counts():
        """Set vehicle counts for each lane manually."""
        data = request.get_json()
        counts = {
            "N": int(data.get("N", 0)),
            "S": int(data.get("S", 0)),
            "E": int(data.get("E", 0)),
            "W": int(data.get("W", 0)),
        }
        engine.set_lane_counts(counts)
        return jsonify({"status": "ok", "counts": counts})

    @app.route("/api/trigger_ambulance", methods=["POST"])
    def trigger_ambulance():
        """Spawn an ambulance from the specified direction."""
        data = request.get_json()
        entry = data.get("entry", "N")
        exit_dir = data.get("exit", None)
        engine.trigger_ambulance(entry_direction=entry, exit_direction=exit_dir)
        return jsonify({"status": "ok", "entry": entry, "exit": exit_dir})

    @app.route("/api/set_scenario", methods=["POST"])
    def set_scenario():
        """Load a preset traffic scenario."""
        data = request.get_json()
        scenario = data.get("scenario", "normal")
        engine.set_scenario(scenario)
        return jsonify({"status": "ok", "scenario": scenario})

    @app.route("/api/set_speed", methods=["POST"])
    def set_speed():
        """Set simulation speed multiplier."""
        data = request.get_json()
        speed = float(data.get("speed", 1.0))
        engine.set_sim_speed(speed)
        return jsonify({"status": "ok", "speed": speed})

    @app.route("/api/toggle_auto_gen", methods=["POST"])
    def toggle_auto_gen():
        """Enable or disable automatic vehicle generation."""
        data = request.get_json()
        enabled = bool(data.get("enabled", True))
        engine.toggle_auto_generate(enabled)
        return jsonify({"status": "ok", "auto_generate": enabled})

    @app.route("/api/reset", methods=["POST"])
    def reset():
        """Reset simulation to initial state."""
        engine.reset()
        engine.start()
        return jsonify({"status": "ok"})

    return app


# Allow running directly with `python simulation_gui/app.py`
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
