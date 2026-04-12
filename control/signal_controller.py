"""
optimization/signal_controller.py
───────────────────────────────────
SignalController — deployed bridge between Model 1 (vehicle detector)
and the DQN agent.

The agent now outputs one of 224 actions encoding both:
  • which lane gets green  (direction: N / S / E / W)
  • how long the green lasts (duration: 5, 6, 7 … 60 whole seconds)

Public API
──────────
    sc = SignalController()
    result = sc.decide(lane_counts)
    # → {
    #     "laneN": 0,   "laneS": 37,  "laneE": 0,  "laneW": 0,
    #     "direction": "S",  "duration": 37,  "mode": "dqn"
    #   }

    # After the real cycle completes, call online_update() to keep learning:
    sc.online_update(
        prev_lane_counts = {"laneN": 5, "laneS": 12, "laneE": 3, "laneW": 1},
        action_taken     = 91,   # the flat action index that was used
        reward           = 0.62,
        next_lane_counts = {"laneN": 5, "laneS": 2,  "laneE": 4, "laneW": 2},
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.DQN.environment import (  # noqa: E402
    TrafficEnv,
    encode_action,
    MIN_GREEN,
    MAX_GREEN,
    ACTION_SIZE,
)
from training.DQN.dqn_agent import DQNAgent  # noqa: E402
from training.DQN.replay_buffer import ReplayBuffer  # noqa: E402

try:
    import config as cfg
    CYCLE_TIMEOUT = cfg.CYCLE_TIMEOUT
    DQN_WEIGHTS_PATH = cfg.DQN_WEIGHTS_PATH
    GAMMA = cfg.GAMMA
    BATCH_SIZE = cfg.DQN_ONLINE_BATCH_SIZE
    ONLINE_LR = cfg.DQN_ONLINE_LEARNING_RATE
    ONLINE_BUFFER_SIZE = cfg.DQN_ONLINE_BUFFER_SIZE
    ONLINE_SAVE_EVERY = cfg.DQN_ONLINE_SAVE_EVERY
    LANE_KEYS = cfg.LANE_KEYS
    DIR_LABELS = cfg.DIRECTIONS
except (ImportError, AttributeError):
    CYCLE_TIMEOUT = 120
    DQN_WEIGHTS_PATH = ROOT / "models" / "dqn_signal_optimizer.pt"
    GAMMA = 0.95
    BATCH_SIZE = 32
    ONLINE_LR = 1e-4
    ONLINE_BUFFER_SIZE = 5_000
    ONLINE_SAVE_EVERY = 100
    LANE_KEYS = ("laneN", "laneS", "laneE", "laneW")
    DIR_LABELS = ("N", "S", "E", "W")

logger = logging.getLogger(__name__)


class SignalController:
    """
    Wraps the trained DQN for deployment on the Raspberry Pi.

    Responsibilities
    ────────────────
    1. Convert Model-1 lane_counts dict → normalised state vector.
    2. Ask the agent which action (direction + duration) to take.
    3. Return a green-time dict ready to be sent over serial to Arduino.
    4. Optionally refine weights online after each real cycle.
    5. Fall back to proportional timing when weights are absent.
    """

    def __init__(
        self,
        weights_path: str | Path = DQN_WEIGHTS_PATH,
        device:       str = "cpu",
        online_lr:    float = ONLINE_LR,
    ) -> None:
        self._weights_path = Path(weights_path)
        self._cycle_count = 0
        self._use_dqn = False

        self._env = TrafficEnv()   # used only for state normalisation
        self._agent = DQNAgent(
            state_size=self._env.state_size,
            action_size=ACTION_SIZE,
            lr=online_lr,
            gamma=GAMMA,
            batch_size=BATCH_SIZE,
            device=device,
        )
        self._buffer = ReplayBuffer(capacity=ONLINE_BUFFER_SIZE)

        self._load_weights()

    # ── primary decision API ──────────────────────────────────────────────────

    def decide(self, lane_counts: dict[str, int]) -> dict:
        """
        Given current vehicle counts, return the optimal signal decision.

        Parameters
        ----------
        lane_counts : {"laneN": int, "laneS": int, "laneE": int, "laneW": int}

        Returns
        -------
        dict with keys:
            laneN, laneS, laneE, laneW  – green seconds (0 = stays red)
            direction                   – label of the active green lane ("N"/"S"/"E"/"W")
            duration                    – chosen green time in whole seconds
            action                      – flat action index (useful for online_update)
            mode                        – "dqn" or "proportional"
        """
        counts = self._parse_counts(lane_counts)

        if self._use_dqn:
            action, direction, duration = self._dqn_decide(counts)

            # ── 1. direction safeguard (shielding) ───────────────────────
            # Prevent the DQN picking a practically empty lane when another
            # lane is heavily backed up.
            busiest_direction = int(np.argmax(counts))
            busiest_queue = float(counts[busiest_direction])
            chosen_queue = float(counts[direction])

            if chosen_queue < (busiest_queue * 0.5) and busiest_queue > 0:
                logger.info(
                    "Direction shield applied: DQN picked %s (q=%d), overriding to busiest %s (q=%d)",
                    DIR_LABELS[direction], int(chosen_queue),
                    DIR_LABELS[busiest_direction], int(busiest_queue)
                )
                direction = busiest_direction

            # ── 2. duration floor safeguard ──────────────────────────────
            queue_in_chosen = float(counts[direction])
            if queue_in_chosen > 0:
                # ~2s per vehicle at 1.5 veh/s discharge, clamped to [MIN, MAX]
                min_duration = int(np.clip(
                    round(queue_in_chosen / 1.5), MIN_GREEN, MAX_GREEN
                ))
                if duration < min_duration:
                    logger.debug(
                        "Duration floor applied: %s → %ds (queue=%d)",
                        DIR_LABELS[direction], duration, int(queue_in_chosen),
                    )
                    duration = min_duration

            # Repackage potentially shielded action
            action = encode_action(direction, duration)

        else:
            action, direction, duration = self._proportional_decide(counts)

        # Build per-lane green-time dict: only the chosen lane gets a non-zero value
        green_times = {key: 0 for key in LANE_KEYS}
        green_times[LANE_KEYS[direction]] = duration

        decision = {
            **green_times,
            "direction": DIR_LABELS[direction],
            "duration":  duration,
            "action":    action,
            "mode":      self.mode,
        }
        print("SignalController.decide: lane_counts=",
              lane_counts, "decision=", decision)
        return decision

    # ── online learning hook ──────────────────────────────────────────────────

    def online_update(
        self,
        prev_lane_counts: dict[str, int],
        action_taken:     int,
        reward:           float,
        next_lane_counts: dict[str, int],
        done:             bool = False,
    ) -> None:
        """
        Call after each real intersection cycle to keep the model learning.

        Parameters
        ----------
        prev_lane_counts : counts before the green phase started
        action_taken     : the flat action index returned by decide()
        reward           : observed reward (e.g. throughput − waiting)
        next_lane_counts : counts after the cycle completed
        done             : True at the end of a deployment session
        """
        if not self._use_dqn:
            return

        prev_state = self._make_state(self._parse_counts(prev_lane_counts))
        next_state = self._make_state(self._parse_counts(next_lane_counts))

        self._buffer.push(prev_state, action_taken, reward, next_state, done)
        self._agent.train_step(self._buffer)

        self._cycle_count += 1
        if self._cycle_count % ONLINE_SAVE_EVERY == 0:
            self._agent.save(self._weights_path)
            logger.info(
                "Online weights saved after cycle %d → %s",
                self._cycle_count, self._weights_path,
            )

    # ── internal: decision paths ──────────────────────────────────────────────

    def _dqn_decide(
        self, counts: np.ndarray
    ) -> tuple[int, int, int]:
        """
        Ask the DQN for the best (direction, duration) pair.

        Returns
        -------
        (action, direction, duration)
        """
        state = self._make_state(counts)
        action, direction, duration = self._agent.select_action_decoded(
            state, epsilon=0.0
        )
        return action, direction, duration

    def _proportional_decide(
        self, counts: np.ndarray
    ) -> tuple[int, int, int]:
        """
        Fallback: give green to the busiest lane for a proportionally
        calculated duration.

        Duration = (busiest_count / total_count) × CYCLE_TIMEOUT,
                   clamped to [MIN_GREEN, MAX_GREEN].
        """
        total = float(counts.sum())
        direction = int(np.argmax(counts))

        if total == 0:
            duration = MIN_GREEN
        else:
            raw = (counts[direction] / total) * CYCLE_TIMEOUT
            duration = int(np.clip(round(raw), MIN_GREEN, MAX_GREEN))

        action = encode_action(direction, duration)
        return action, direction, duration

    # ── internal: state helpers ───────────────────────────────────────────────

    def _make_state(self, counts: np.ndarray) -> np.ndarray:
        """Build normalised 6-D state from raw counts."""
        self._env.queues = counts.astype(np.float32)
        return self._env._get_state()

    def _parse_counts(self, lane_counts: dict[str, int]) -> np.ndarray:
        return np.array(
            [lane_counts.get(k, 0) for k in LANE_KEYS], dtype=np.float32
        )

    # ── weight loading ────────────────────────────────────────────────────────

    def _load_weights(self) -> None:
        if self._weights_path.exists():
            try:
                self._agent.load(self._weights_path)
                self._use_dqn = True
                logger.info("DQN weights loaded from %s", self._weights_path)
            except Exception as exc:
                logger.warning(
                    "Could not load DQN weights (%s) — using proportional fallback.",
                    exc,
                )
                self._use_dqn = False
        else:
            logger.warning(
                "No weights file at %s — using proportional fallback until trained.",
                self._weights_path,
            )
            self._use_dqn = False

    # ── diagnostics ──────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return "dqn" if self._use_dqn else "proportional"

    def status(self) -> dict:
        return {
            "mode":        self.mode,
            "cycle_count": self._cycle_count,
            "buffer_size": len(self._buffer),
            "weights":     str(self._weights_path),
            "action_size": ACTION_SIZE,
        }

    def top_actions_for(self, lane_counts: dict[str, int], top_k: int = 5) -> list[dict]:
        """
        Return the top-k DQN actions for the given lane counts — useful for
        debugging and the GUI dashboard.

        Example output:
            [{"action": 91, "direction": "S", "duration": 20, "q": 4.21}, ...]
        """
        if not self._use_dqn:
            return []
        state = self._make_state(self._parse_counts(lane_counts))
        return self._agent.top_actions(state, top_k=top_k)
