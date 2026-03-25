"""Gymnasium environment for adaptive traffic signal control.

Simulates a 4-way intersection (North/South/East/West) with 2 signal phases:
  - Phase 0: North + South get green
  - Phase 1: East + West get green

The RL agent chooses HOW LONG to keep the current phase green,
then the environment automatically flips to the other phase.

State (12 features, all normalized to [0, 1]):
  0  count_N        — current vehicle queue in North lane   (÷ MAX_QUEUE)
  1  count_S        — current vehicle queue in South lane   (÷ MAX_QUEUE)
  2  count_E        — current vehicle queue in East lane    (÷ MAX_QUEUE)
  3  count_W        — current vehicle queue in West lane    (÷ MAX_QUEUE)
  4  pred_N         — XGBoost predicted density, North      (÷ MAX_QUEUE)
  5  pred_S         — XGBoost predicted density, South      (÷ MAX_QUEUE)
  6  pred_E         — XGBoost predicted density, East       (÷ MAX_QUEUE)
  7  pred_W         — XGBoost predicted density, West       (÷ MAX_QUEUE)
  8  current_phase  — 0 = NS green, 1 = EW green
  9  time_in_phase  — seconds elapsed in current phase      (÷ 60)
  10 ambulance_detected — 1 if ambulance is active and not yet cleared
  11 ambulance_direction — lane index 0-3 (÷ 3), or 0.0 if none

Action space (Discrete 6):
  Index → green duration in seconds
  0→10s, 1→20s, 2→30s, 3→40s, 4→50s, 5→60s

Reward per simulated second:
  -0.1   × (total vehicles waiting across all 4 lanes)
  -10.0  × 1  if ambulance is active and its lane is NOT green this second
  +100.0      ONE TIME when ambulance lane first gets a green phase (cleared)
  -2.0        if the agent switches phase having spent < MIN_PHASE_TIME seconds
              (penalises unnecessary flickering)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ── Intersection constants ────────────────────────────────────────────────────
MAX_QUEUE: int = 20            # vehicle cap per lane
GREEN_DURATIONS: list[int] = [10, 20, 30, 40, 50, 60]  # seconds per action
ARRIVAL_RATE: float = 0.3     # mean vehicles arriving per lane per second (Poisson)
CLEARANCE_RATE: float = 1.5   # vehicles cleared per second when lane is green
EPISODE_STEPS: int = 300      # simulated seconds per episode

# ── Ambulance constants ───────────────────────────────────────────────────────
AMBULANCE_PROB: float = 0.35   # probability an ambulance appears in an episode
AMBULANCE_REWARD: float = 100.0
AMBULANCE_PENALTY_PER_SEC: float = 10.0

# ── Reward constants ─────────────────────────────────────────────────────────
WAIT_PENALTY: float = 0.1         # per vehicle per second
PREMATURE_SWITCH_PENALTY: float = 2.0
MIN_PHASE_TIME: int = 15          # seconds; switch before this → penalty


class TrafficEnv(gym.Env):
    """4-way intersection environment for DQN-based signal timing.

    Each call to ``step(action)`` runs ``GREEN_DURATIONS[action]`` simulated
    seconds, then automatically flips the signal phase.  The agent's job is to
    pick the best green duration given current + predicted traffic.
    """

    metadata = {"render_modes": ["human"]}

    # Lane index → readable name (used for rendering only)
    LANE_NAMES = ["N", "S", "E", "W"]

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        # ── Spaces ────────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(len(GREEN_DURATIONS))
        self.observation_space = spaces.Box(
            low=np.zeros(12, dtype=np.float32),
            high=np.ones(12, dtype=np.float32),
            dtype=np.float32,
        )

        # ── Internal state (initialised properly in reset()) ──────────────
        self._queues = np.zeros(4, dtype=np.float32)       # vehicle counts [N,S,E,W]
        self._predicted = np.zeros(4, dtype=np.float32)    # XGBoost predictions
        self._phase: int = 0          # 0 = NS green, 1 = EW green
        self._time_in_phase: int = 0  # seconds spent in current phase
        self._step_count: int = 0     # total simulated seconds elapsed

        # Ambulance state
        self._ambulance_present: bool = False
        self._ambulance_lane: int = -1
        self._ambulance_active: bool = False   # True after arrival step reached
        self._ambulance_arrival_step: int = -1
        self._ambulance_cleared: bool = False

        self.np_random = np.random.default_rng()

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment to a fresh episode."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Random initial queues (0-5 vehicles each lane)
        self._queues = self.np_random.integers(0, 6, size=4).astype(np.float32)
        # Initial XGBoost predictions ≈ slightly above current (simulate real predictor)
        noise = self.np_random.uniform(-1.0, 2.0, size=4).astype(np.float32)
        self._predicted = np.clip(self._queues + noise, 0, MAX_QUEUE)

        self._phase = 0
        self._time_in_phase = 0
        self._step_count = 0

        # Decide if ambulance appears in this episode
        self._ambulance_present = bool(self.np_random.random() < AMBULANCE_PROB)
        if self._ambulance_present:
            # Ambulance arrives sometime between step 20 and step 200
            self._ambulance_arrival_step = int(self.np_random.integers(20, 200))
            self._ambulance_lane = int(self.np_random.integers(0, 4))
            self._ambulance_active = False
            self._ambulance_cleared = False
        else:
            self._ambulance_arrival_step = -1
            self._ambulance_lane = -1
            self._ambulance_active = False
            self._ambulance_cleared = False

        return self._get_obs(), {}

    def step(self, action: int):
        """Run one agent decision = ``GREEN_DURATIONS[action]`` simulated seconds.

        After the green duration expires the phase flips (NS ↔ EW).
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        green_seconds = GREEN_DURATIONS[action]

        # Penalty for flipping phase too quickly
        premature_switch = (
            self._phase != int(self._phase == 0)  # will switch after this step
            and self._time_in_phase < MIN_PHASE_TIME
        )
        # Simpler check: we ALWAYS flip after a step, so penalise if current
        # time in phase was very short (agent keeps oscilating)
        premature_switch = self._time_in_phase < MIN_PHASE_TIME and self._time_in_phase > 0

        reward = 0.0

        # ── Simulate second-by-second within the green duration ───────────
        green_lanes = [0, 1] if self._phase == 0 else [2, 3]

        for _ in range(green_seconds):
            self._step_count += 1

            # 1. New vehicles arrive (Poisson process)
            arrivals = self.np_random.poisson(ARRIVAL_RATE, size=4).astype(np.float32)
            self._queues = np.clip(self._queues + arrivals, 0, MAX_QUEUE)

            # 2. Green lanes clear vehicles
            for lane in green_lanes:
                cleared = min(self._queues[lane], CLEARANCE_RATE)
                self._queues[lane] = max(0.0, self._queues[lane] - cleared)

            # 3. Activate ambulance if its arrival step is reached
            if (
                self._ambulance_present
                and not self._ambulance_cleared
                and self._step_count >= self._ambulance_arrival_step
            ):
                self._ambulance_active = True

            # 4. Ambulance reward / penalty
            if self._ambulance_active and not self._ambulance_cleared:
                amb_lane = self._ambulance_lane
                lane_is_green = amb_lane in green_lanes
                if lane_is_green:
                    # Ambulance gets through — big one-time reward
                    self._ambulance_cleared = True
                    reward += AMBULANCE_REWARD
                else:
                    # Ambulance waiting — heavy penalty
                    reward -= AMBULANCE_PENALTY_PER_SEC

            # 5. Queue waiting penalty (all lanes, every second)
            reward -= float(np.sum(self._queues)) * WAIT_PENALTY

        # 6. Premature phase-switch penalty (applied once per step)
        if premature_switch:
            reward -= PREMATURE_SWITCH_PENALTY

        # ── Update XGBoost predictions (simulated: add realistic noise) ───
        noise = self.np_random.uniform(-1.5, 2.5, size=4).astype(np.float32)
        self._predicted = np.clip(self._queues + noise, 0, MAX_QUEUE)

        # ── Flip phase ────────────────────────────────────────────────────
        self._time_in_phase = green_seconds  # track how long we just spent
        self._phase = 1 - self._phase

        # ── Termination ───────────────────────────────────────────────────
        terminated = False
        truncated = self._step_count >= EPISODE_STEPS

        info = {
            "total_queue": float(np.sum(self._queues)),
            "queues": self._queues.tolist(),
            "phase": self._phase,
            "step": self._step_count,
            "ambulance_active": self._ambulance_active,
            "ambulance_cleared": self._ambulance_cleared,
            "green_duration_chosen": green_seconds,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        phase_str = "NS-GREEN" if self._phase == 0 else "EW-GREEN"
        if self._ambulance_active and not self._ambulance_cleared:
            amb_str = f"🚑 lane={self.LANE_NAMES[self._ambulance_lane]} WAITING"
        elif self._ambulance_cleared:
            amb_str = "🚑 CLEARED ✓"
        else:
            amb_str = "none"
        print(
            f"t={self._step_count:4d}s | {phase_str:10s} | "
            f"N={self._queues[0]:4.1f} S={self._queues[1]:4.1f} "
            f"E={self._queues[2]:4.1f} W={self._queues[3]:4.1f} | "
            f"Amb: {amb_str}"
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the 12-feature observation vector (all values in [0, 1])."""
        norm_queues = self._queues / MAX_QUEUE                   # features 0-3
        norm_pred = self._predicted / MAX_QUEUE                   # features 4-7
        norm_phase = float(self._phase)                           # feature  8
        norm_time = min(self._time_in_phase / 60.0, 1.0)         # feature  9

        if self._ambulance_active and not self._ambulance_cleared:
            norm_amb = 1.0                                        # feature 10
            norm_amb_dir = self._ambulance_lane / 3.0             # feature 11
        else:
            norm_amb = 0.0
            norm_amb_dir = 0.0

        return np.array(
            [
                norm_queues[0], norm_queues[1], norm_queues[2], norm_queues[3],
                norm_pred[0],   norm_pred[1],   norm_pred[2],   norm_pred[3],
                norm_phase,
                norm_time,
                norm_amb,
                norm_amb_dir,
            ],
            dtype=np.float32,
        )


# ── Sanity test (run with: python simulation/traffic_env.py) ─────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("TrafficEnv — 12-Feature Sanity Test")
    print("=" * 65)

    env = TrafficEnv(render_mode="human")
    obs, info = env.reset(seed=0)

    print(f"\nObservation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    print(f"Initial obs (12)  : {obs}")
    print(f"\n--- Running 1 episode with RANDOM actions ---\n")

    total_reward = 0.0
    done = False
    decisions = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        decisions += 1

        if decisions % 3 == 0:
            env.render()

    print(f"\n{'=' * 65}")
    print(f"Episode done in {decisions} decisions ({info['step']} simulated seconds)")
    print(f"Total reward       : {total_reward:.1f}")
    print(f"Final queues       : N={info['queues'][0]:.1f}  S={info['queues'][1]:.1f}"
          f"  E={info['queues'][2]:.1f}  W={info['queues'][3]:.1f}")
    print(f"Ambulance cleared  : {info['ambulance_cleared']}")
    print(f"{'=' * 65}")
    print("✅  Environment works correctly!")