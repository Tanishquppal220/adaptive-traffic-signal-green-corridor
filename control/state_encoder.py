"""State encoder — converts live model outputs into the RL agent's state vector.

This is the bridge between your existing pipeline and the RL agent.

Your existing models produce:
  • VehicleDetector    → vehicle_counts {N/S/E/W: int}
  • TrafficPredictor   → predicted_densities {N/S/E/W: float}
  • EmergencyClassify  → ambulance_detected bool, ambulance_direction int (0-3 or -1)
  • SignalController   → current_phase int (0=NS, 1=EW), time_in_phase float (seconds)

This module collects all of the above and builds the normalized 12-float
numpy array that the DQN agent expects.

State layout (matches TrafficEnv._get_obs exactly):
  Index  Feature                      Source              Normalization
  ─────  ──────────────────────────── ─────────────────── ─────────────────
  0      count_N                      YOLO                ÷ MAX_QUEUE (20)
  1      count_S                      YOLO                ÷ MAX_QUEUE
  2      count_E                      YOLO                ÷ MAX_QUEUE
  3      count_W                      YOLO                ÷ MAX_QUEUE
  4      pred_N                       XGBoost             ÷ MAX_QUEUE
  5      pred_S                       XGBoost             ÷ MAX_QUEUE
  6      pred_E                       XGBoost             ÷ MAX_QUEUE
  7      pred_W                       XGBoost             ÷ MAX_QUEUE
  8      current_phase                SignalController    0.0 or 1.0
  9      time_in_phase                SignalController    ÷ 60  (capped at 1)
  10     ambulance_detected           Emergency model     0.0 or 1.0
  11     ambulance_direction          Emergency model     lane÷3 (or 0 if none)

Usage:
    encoder = StateEncoder()
    state = encoder.encode(
        vehicle_counts      = {"N": 7, "S": 3, "E": 12, "W": 1},
        predicted_densities = {"N": 9.2, "S": 3.5, "E": 13.1, "W": 1.2},
        current_phase       = 0,
        time_in_phase       = 18.5,
        ambulance_detected  = True,
        ambulance_direction = 2,   # East (0=N,1=S,2=E,3=W), -1=none
    )
    # state is np.ndarray shape (12,) dtype float32, all values in [0, 1]
"""

from __future__ import annotations

import numpy as np

# Must match TrafficEnv.MAX_QUEUE
MAX_QUEUE: int = 20

# Direction strings → lane index for normalization
_DIR_STR_TO_IDX: dict[str, int] = {"N": 0, "S": 1, "E": 2, "W": 3}


class StateEncoder:
    """Converts raw model outputs to the 12-feature normalized state vector.

    This class is stateless — every call to ``encode()`` is independent.
    No history or memory is kept here (that lives in TrafficPredictor).
    """

    def encode(
        self,
        vehicle_counts: dict[str, int | float],
        predicted_densities: dict[str, float],
        current_phase: int,
        time_in_phase: float,
        ambulance_detected: bool,
        ambulance_direction: int,
    ) -> np.ndarray:
        """Build the 12-float normalized state vector.

        Args:
            vehicle_counts: Current YOLO counts per lane.
                            Keys: "N", "S", "E", "W". Missing keys → 0.
            predicted_densities: XGBoost 60-second ahead predictions per lane.
                            Keys: "N", "S", "E", "W". Missing keys → 0.
            current_phase:  0 = North+South green, 1 = East+West green.
            time_in_phase:  Seconds elapsed since last phase switch (float).
            ambulance_detected: True if an active ambulance is present.
            ambulance_direction: Lane index (0=N, 1=S, 2=E, 3=W) or -1 if none.

        Returns:
            np.ndarray of shape (12,), dtype float32, all values in [0.0, 1.0].
        """
        lanes = ["N", "S", "E", "W"]

        # Features 0-3: YOLO vehicle counts, normalized
        counts = np.array(
            [float(vehicle_counts.get(lane, 0)) for lane in lanes], dtype=np.float32
        )
        norm_counts = np.clip(counts / MAX_QUEUE, 0.0, 1.0)

        # Features 4-7: XGBoost predicted densities, normalized
        preds = np.array(
            [float(predicted_densities.get(lane, 0)) for lane in lanes], dtype=np.float32
        )
        norm_preds = np.clip(preds / MAX_QUEUE, 0.0, 1.0)

        # Feature 8: current phase (already 0 or 1)
        norm_phase = float(int(current_phase == 1))  # enforce 0.0 or 1.0

        # Feature 9: time in phase, capped at 60 s
        norm_time = float(np.clip(time_in_phase / 60.0, 0.0, 1.0))

        # Feature 10: ambulance present flag
        norm_amb = 1.0 if ambulance_detected else 0.0

        # Feature 11: ambulance direction (0 if none)
        if ambulance_detected and ambulance_direction in (0, 1, 2, 3):
            norm_amb_dir = float(ambulance_direction) / 3.0
        else:
            norm_amb_dir = 0.0

        state = np.array(
            [
                norm_counts[0], norm_counts[1], norm_counts[2], norm_counts[3],
                norm_preds[0],  norm_preds[1],  norm_preds[2],  norm_preds[3],
                norm_phase,
                norm_time,
                norm_amb,
                norm_amb_dir,
            ],
            dtype=np.float32,
        )

        # Safety: guarantee all values are in [0, 1]
        state = np.clip(state, 0.0, 1.0)
        return state

    def encode_from_dict(self, detections: dict) -> np.ndarray:
        """Convenience wrapper that accepts a single flat dict.

        Expected dict keys (all optional, default to 0 / False):
            vehicle_counts      : dict[str, int]
            predicted_densities : dict[str, float]
            current_phase       : int   (0 or 1)
            time_in_phase       : float (seconds)
            ambulance_detected  : bool
            ambulance_direction : int   (0-3 or -1)

        Example:
            state = encoder.encode_from_dict({
                "vehicle_counts": {"N": 5, "S": 2, "E": 8, "W": 0},
                "predicted_densities": {"N": 6.0, "S": 2.5, "E": 9.0, "W": 0.5},
                "current_phase": 0,
                "time_in_phase": 22.0,
                "ambulance_detected": False,
                "ambulance_direction": -1,
            })
        """
        return self.encode(
            vehicle_counts      = detections.get("vehicle_counts", {}),
            predicted_densities = detections.get("predicted_densities", {}),
            current_phase       = int(detections.get("current_phase", 0)),
            time_in_phase       = float(detections.get("time_in_phase", 0.0)),
            ambulance_detected  = bool(detections.get("ambulance_detected", False)),
            ambulance_direction = int(detections.get("ambulance_direction", -1)),
        )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    encoder = StateEncoder()

    test_cases = [
        {
            "label": "Normal traffic, NS green, 20 s into phase",
            "detections": {
                "vehicle_counts": {"N": 7, "S": 3, "E": 12, "W": 1},
                "predicted_densities": {"N": 8.0, "S": 3.5, "E": 13.0, "W": 1.2},
                "current_phase": 0,
                "time_in_phase": 20.0,
                "ambulance_detected": False,
                "ambulance_direction": -1,
            },
        },
        {
            "label": "Ambulance present in East lane, EW green",
            "detections": {
                "vehicle_counts": {"N": 2, "S": 5, "E": 0, "W": 4},
                "predicted_densities": {"N": 2.5, "S": 5.0, "E": 1.0, "W": 4.2},
                "current_phase": 1,
                "time_in_phase": 5.0,
                "ambulance_detected": True,
                "ambulance_direction": 2,  # East
            },
        },
        {
            "label": "All-zero input (edge case)",
            "detections": {},
        },
    ]

    print("=" * 65)
    print("StateEncoder Smoke Test")
    print("=" * 65)

    for tc in test_cases:
        state = encoder.encode_from_dict(tc["detections"])
        print(f"\n{tc['label']}")
        labels = [
            "count_N", "count_S", "count_E", "count_W",
            "pred_N",  "pred_S",  "pred_E",  "pred_W",
            "phase",   "time",    "amb",     "amb_dir",
        ]
        for name, val in zip(labels, state):
            print(f"  {name:<12}: {val:.4f}")
        assert state.shape == (12,), "Shape must be (12,)"
        assert state.dtype == np.float32, "dtype must be float32"
        assert np.all(state >= 0.0) and np.all(state <= 1.0), "All values must be in [0,1]"

    print("\n✅  All state encoder tests passed!")
