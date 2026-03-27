"""RL agent for live traffic signal control.

Loads the trained DQN model (signal_policy.zip) and exposes a simple
``get_action(detections)`` interface for use in the Flask app.

Usage in flask/app.py (after training is done):
    from control.rl_agent import RLAgent

    agent = RLAgent()  # loads model once on startup

    # Every time you get new detections from YOLO / XGBoost:
    green_duration = agent.get_action(detections_dict)  # returns e.g. 30

    # Then use green_duration to control the signals / send to Arduino.
"""

from __future__ import annotations

from pathlib import Path

from config import GREEN_DURATIONS
from control.state_encoder import StateEncoder

# ── Lazy imports (avoid importing SB3 until needed) ───────────────────────────
_sb3_available = False
try:
    from stable_baselines3 import DQN as _DQN

    _sb3_available = True
except ImportError:
    pass

# ── Default model path ────────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "training" / "checkpoints" / "signal_policy.zip"
)


class RLAgent:
    """Wraps the trained DQN model for live inference.

    Parameters:
        model_path: Path to the ``signal_policy.zip`` file produced by
                    ``training/train_rl.py``.  Defaults to the standard
                    checkpoint location.

    Example:
        agent = RLAgent()
        green_secs = agent.get_action({
            "vehicle_counts": {"N": 7, "S": 3, "E": 12, "W": 1},
            "predicted_densities": {"N": 8.0, "S": 3.5, "E": 13.0, "W": 1.2},
            "current_phase": 0,
            "time_in_phase": 20.0,
            "ambulance_detected": False,
            "ambulance_direction": -1,
        })
        print(f"Green for {green_secs} seconds")
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._encoder = StateEncoder()
        self._model = None
        self._loaded = False

        path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH

        if not _sb3_available:
            print(
                "⚠️  stable-baselines3 not installed. "
                "RLAgent will use fallback fixed-30s policy."
            )
            return

        if not path.exists():
            print(
                f"⚠️  Model not found at: {path}\n"
                "    Run  python training/train_rl.py  first.\n"
                "    Falling back to fixed-30s policy."
            )
            return

        try:
            self._model = _DQN.load(str(path))
            self._loaded = True
            print(f"✅  RLAgent: model loaded from {path}")
        except Exception as exc:
            print(f"⚠️  Failed to load model ({exc}). Using fixed-30s fallback.")

    @property
    def is_loaded(self) -> bool:
        """True if the DQN model was loaded successfully."""
        return self._loaded

    def get_action(self, detections: dict) -> int:
        """Return the recommended green duration (seconds) for the current state.

        Args:
            detections: Dict with keys:
                vehicle_counts      (dict[str, int])
                predicted_densities (dict[str, float])
                current_phase       (int: 0 or 1)
                time_in_phase       (float: seconds elapsed)
                ambulance_detected  (bool)
                ambulance_direction (int: 0-3 or -1)

        Returns:
            Green duration in seconds.  One of [10, 20, 30, 40, 50, 60].
            Falls back to 30 s if the model is not loaded.
        """
        if not self._loaded or self._model is None:
            return 30  # safe default

        state = self._encoder.encode_from_dict(detections)
        action, _ = self._model.predict(state, deterministic=True)
        return GREEN_DURATIONS[int(action)]

    def get_action_label(self, detections: dict) -> str:
        """Same as ``get_action()`` but returns a human-readable string.

        Returns:
            e.g. "30s"  or  "20s  (fallback)"
        """
        if not self._loaded:
            return "30s (fallback)"
        secs = self.get_action(detections)
        return f"{secs}s"

    def get_action_with_info(self, detections: dict) -> dict:
        """Return the action plus extra debugging information.

        Returns a dict:
            green_duration  : int   — recommended seconds
            action_index    : int   — DQN action index (0-5)
            state           : list  — the 12-float state that was fed to the model
            model_loaded    : bool  — whether real model or fallback was used
        """
        state = self._encoder.encode_from_dict(detections)

        if not self._loaded or self._model is None:
            return {
                "green_duration": 30,
                "action_index": 2,
                "state": state.tolist(),
                "model_loaded": False,
            }

        action, _ = self._model.predict(state, deterministic=True)
        action = int(action)
        return {
            "green_duration": GREEN_DURATIONS[action],
            "action_index": action,
            "state": state.tolist(),
            "model_loaded": True,
        }


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("RLAgent Smoke Test")
    print("=" * 65)

    agent = RLAgent()
    print(f"\nModel loaded: {agent.is_loaded}\n")

    test_scenarios = [
        {
            "name": "Heavy East traffic, EW should get longer green",
            "detections": {
                "vehicle_counts": {"N": 1, "S": 2, "E": 18, "W": 15},
                "predicted_densities": {"N": 1.5, "S": 2.0, "E": 19.0, "W": 14.5},
                "current_phase": 1,   # EW green
                "time_in_phase": 10.0,
                "ambulance_detected": False,
                "ambulance_direction": -1,
            },
        },
        {
            "name": "Ambulance in North lane, NS should stay green longer",
            "detections": {
                "vehicle_counts": {"N": 3, "S": 3, "E": 5, "W": 2},
                "predicted_densities": {"N": 4.0, "S": 3.5, "E": 5.0, "W": 2.0},
                "current_phase": 0,   # NS green
                "time_in_phase": 5.0,
                "ambulance_detected": True,
                "ambulance_direction": 0,  # North
            },
        },
        {
            "name": "Balanced light traffic, switch phase quickly",
            "detections": {
                "vehicle_counts": {"N": 2, "S": 1, "E": 2, "W": 1},
                "predicted_densities": {"N": 2.0, "S": 1.5, "E": 2.0, "W": 1.5},
                "current_phase": 0,
                "time_in_phase": 30.0,
                "ambulance_detected": False,
                "ambulance_direction": -1,
            },
        },
        {
            "name": "All lanes empty",
            "detections": {
                "vehicle_counts": {"N": 0, "S": 0, "E": 0, "W": 0},
                "predicted_densities": {"N": 0, "S": 0, "E": 0, "W": 0},
                "current_phase": 0,
                "time_in_phase": 0.0,
                "ambulance_detected": False,
                "ambulance_direction": -1,
            },
        },
        {
            "name": "Missing keys (edge case)",
            "detections": {},
        },
    ]

    for scenario in test_scenarios:
        info = agent.get_action_with_info(scenario["detections"])
        print(f"  {scenario['name']}")
        print(f"    → Green for: {info['green_duration']}s  "
              f"(action index {info['action_index']})  "
              f"[model_loaded={info['model_loaded']}]")
        print()

    print("✅  Smoke test complete!")
