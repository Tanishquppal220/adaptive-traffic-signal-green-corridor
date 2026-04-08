"""
optimization/environment.py
────────────────────────────
TrafficEnv – traffic-signal simulation with (direction × duration) actions.

Action Encoding  (224 total actions)
─────────────────────────────────────
  action = direction * N_DURATIONS + (duration - MIN_GREEN)

  direction = action // N_DURATIONS                →  0=N, 1=S, 2=E, 3=W
  duration  = (action  % N_DURATIONS) + MIN_GREEN  →  5, 6, 7 … 60 seconds

  Examples:
    action   0  →  North,  5 s
    action   1  →  North,  6 s
    action  55  →  North, 60 s
    action  56  →  South,  5 s
    action 111  →  South, 60 s
    action 223  →  West,  60 s

State  (6 floats, all normalised to [0, 1])
────────────────────────────────────────────
  [N_count, S_count, E_count, W_count, current_phase_idx, elapsed_fraction]

Reward
──────
  + throughput_score  – fraction of the green lane's QUEUE that was actually cleared
  − waiting_penalty   – 0.05 × total vehicles still waiting across all lanes
  − duration_cost     – 0.001 × chosen green seconds
      (mild penalty prevents the agent always choosing 60 s for tiny queues)
"""

from __future__ import annotations

import numpy as np

# ── timing constants (keep in sync with config.py) ────────────────────────────
MIN_GREEN = 5    # shortest legal green  (seconds)
MAX_GREEN = 60   # longest  legal green  (seconds)
N_DURATIONS = MAX_GREEN - MIN_GREEN + 1   # 56  (5, 6, 7 … 60)
N_DIRECTIONS = 4
ACTION_SIZE = N_DIRECTIONS * N_DURATIONS  # 224

# ── simulation knobs ───────────────────────────────────────────────────────────
ARRIVAL_RATE_MEAN = 0.8    # vehicles / second / lane  (Poisson λ, off-peak)
ARRIVAL_RATE_PEAK = 2.0    # vehicles / second / lane  (peak hour)
DISCHARGE_RATE = 1.5    # vehicles / second leaving the green lane
MAX_QUEUE = 30     # hard cap on queue length per lane
WAITING_PENALTY = 0.01   # reward deduction per waiting vehicle
DURATION_COST = 0.001  # reward deduction per second of chosen green
MAX_VEHICLES_NORM = 30.0   # divisor used to normalise counts → [0, 1]


# ── codec helpers ──────────────────────────────────────────────────────────────

def decode_action(action: int) -> tuple[int, int]:
    """
    Convert a flat action index → (direction_index, green_seconds).

    Parameters
    ----------
    action : int  in [0, 223]

    Returns
    -------
    direction : int  in {0, 1, 2, 3}    → N / S / E / W
    duration  : int  in {5, 6, … 60}    → seconds
    """
    assert 0 <= action < ACTION_SIZE, (
        f"Action {action} is out of range [0, {ACTION_SIZE - 1}]"
    )
    direction = action // N_DURATIONS
    duration = (action % N_DURATIONS) + MIN_GREEN
    return direction, duration


def encode_action(direction: int, duration: int) -> int:
    """
    Inverse of decode_action.
    Useful for constructing actions in tests and the online-update hook.
    """
    assert 0 <= direction < N_DIRECTIONS, f"direction {direction} not in [0,3]"
    assert MIN_GREEN <= duration <= MAX_GREEN, (
        f"duration {duration} not in [{MIN_GREEN}, {MAX_GREEN}]"
    )
    return direction * N_DURATIONS + (duration - MIN_GREEN)


# ── environment ────────────────────────────────────────────────────────────────

class TrafficEnv:
    """
    Gym-like traffic simulation — no gymnasium dependency required.

    Key behaviour
    -------------
    • The agent picks action ∈ [0, 223], which encodes both WHICH lane gets
      green AND for HOW LONG (5–60 whole seconds).
    • The simulation runs for exactly `duration` seconds per step:
        - The chosen lane discharges at DISCHARGE_RATE vehicles/second.
        - All four lanes accumulate Poisson arrivals for `duration` seconds.
    • A longer green clears more of the active lane but lets other lanes
      build up — the agent must learn the right trade-off.

    Usage
    -----
    env   = TrafficEnv()
    state = env.reset()
    for _ in range(200):
        action                         = agent.select_action(state, epsilon)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            state = env.reset()
    """

    DIRECTIONS = ("N", "S", "E", "W")

    def __init__(
        self,
        max_steps: int = 100,
        peak_hour: bool = False,
        seed:      int | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.arrival_rate = ARRIVAL_RATE_PEAK if peak_hour else ARRIVAL_RATE_MEAN
        self.rng = np.random.default_rng(seed)

        # mutable state — initialised properly in reset()
        self.queues:        np.ndarray = np.zeros(4, dtype=np.float32)
        self.current_phase: int = 0
        self.elapsed:       float = 0.0
        self.step_count:    int = 0

    # ── public interface ───────────────────────────────────────────────────────

    def reset(
        self,
        initial_counts: dict[str, int] | None = None,
    ) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        initial_counts : dict, optional
            Seed queue lengths from real detector output, e.g.:
            {"laneN": 5, "laneS": 2, "laneE": 7, "laneW": 0}
            If omitted, random queues in [0, 10] are used.
        """
        if initial_counts:
            self.queues = np.array(
                [
                    initial_counts.get("laneN", 0),
                    initial_counts.get("laneS", 0),
                    initial_counts.get("laneE", 0),
                    initial_counts.get("laneW", 0),
                ],
                dtype=np.float32,
            )
        else:
            self.queues = self.rng.integers(0, 11, size=4).astype(np.float32)

        self.current_phase = int(self.rng.integers(0, 4))
        self.elapsed = 0.0
        self.step_count = 0
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Advance the simulation by one traffic phase.

        The action encodes both direction and duration:
          direction, duration = decode_action(action)

        Simulation logic
        ────────────────
        1. Poisson arrivals land in ALL four lanes for `duration` seconds.
        2. The chosen lane discharges at DISCHARGE_RATE * duration vehicles.
        3. Reward = throughput − waiting_penalty * queue_total − duration_cost * s

        Parameters
        ----------
        action : int  in [0, 223]

        Returns
        -------
        next_state  : np.ndarray  shape (6,)  – normalised
        reward      : float
        done        : bool
        info        : dict  – human-readable breakdown for logging
        """
        direction, duration = decode_action(action)

        # 1. Arrivals in every lane ─────────────────────────────────────────────
        arrivals = self.rng.poisson(
            self.arrival_rate * duration, size=4
        ).astype(np.float32)
        self.queues = np.clip(self.queues + arrivals, 0, MAX_QUEUE)

        # 2. Discharge the green lane ──────────────────────────────────────────
        queue_before = float(self.queues[direction])
        max_discharge = DISCHARGE_RATE * duration
        discharged = float(min(queue_before, max_discharge))
        self.queues[direction] = max(0.0, self.queues[direction] - discharged)

        # 3. Reward ────────────────────────────────────────────────────────────
        #    throughput = fraction of the QUEUE cleared (not fraction of
        #    discharge capacity). This incentivises picking a duration long
        #    enough to actually drain the chosen lane.
        throughput = discharged / max(1.0, queue_before)
        total_waiting = float(self.queues.sum())

        # 🔥 new idea: scale penalty with congestion
        congestion = total_waiting / 20.0   # normalize (~0–6)

        reward = (
            +3.0 * throughput
            - (0.02 + 0.02 * congestion) * total_waiting   # adaptive penalty
            - 0.0001 * duration
        )


        reward = reward / 5.0
        reward = max(-1, min(1, reward))

        # 4. Update phase tracking ─────────────────────────────────────────────
        self.current_phase = direction
        self.elapsed += duration
        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "direction":     self.DIRECTIONS[direction],
            "duration_s":    duration,
            "discharged":    discharged,
            "total_waiting": total_waiting,
            "throughput":    round(throughput, 4),
            "queues":        self.queues.copy(),
            "reward":        round(reward, 4),
        }
        return self._get_state(), reward, done, info

    def seed_from_detector(self, lane_counts: dict[str, int]) -> None:
        """
        Hot-update queue sizes from the live vehicle detector
        without triggering a full episode reset.
        """
        self.queues[0] = min(lane_counts.get("laneN", 0), MAX_QUEUE)
        self.queues[1] = min(lane_counts.get("laneS", 0), MAX_QUEUE)
        self.queues[2] = min(lane_counts.get("laneE", 0), MAX_QUEUE)
        self.queues[3] = min(lane_counts.get("laneW", 0), MAX_QUEUE)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """Return the normalised 6-D state vector."""
        return np.array(
            [
                self.queues[0] / MAX_VEHICLES_NORM,   # N count  [0,1]
                self.queues[1] / MAX_VEHICLES_NORM,   # S count  [0,1]
                self.queues[2] / MAX_VEHICLES_NORM,   # E count  [0,1]
                self.queues[3] / MAX_VEHICLES_NORM,   # W count  [0,1]
                self.current_phase / 3.0,              # phase    [0,1]
                min(self.elapsed / 120.0, 1.0),        # elapsed  [0,1]
            ],
            dtype=np.float32,
        )

    @property
    def state_size(self) -> int:
        return 6

    @property
    def action_size(self) -> int:
        return ACTION_SIZE   # 224
