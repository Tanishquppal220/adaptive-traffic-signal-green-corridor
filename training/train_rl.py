"""DQN Training Script — Adaptive Traffic Signal Controller.

Trains a Deep Q-Network (DQN) agent using Stable-Baselines3 on
the TrafficEnv Gymnasium environment.

Usage:
    python training/train_rl.py

Output:
    training/checkpoints/signal_policy.zip  ← the trained model
    training/checkpoints/learning_curve.png ← reward plot
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works without a display)
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Make sure the repo root is on the path so we can import simulation/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.traffic_env import TrafficEnv

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "signal_policy")
CURVE_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "learning_curve.png")

# ── Hyperparameters ───────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 200_000      # training steps — ~5-10 min on a laptop
LEARNING_RATE   = 1e-3
BUFFER_SIZE     = 100_000      # replay buffer size
LEARNING_STARTS = 2_000        # steps before learning begins
BATCH_SIZE      = 128
GAMMA           = 0.99         # discount factor
TARGET_UPDATE   = 500          # steps between target network syncs
EXPLORATION_FRAC = 0.4         # fraction of training with epsilon > final
EPS_START       = 1.0
EPS_END         = 0.05
NET_ARCH        = [256, 256]   # two hidden layers of 256 neurons
SEED            = 42


# ── Callback: tracks per-episode rewards and prints progress ──────────────────
class EpisodeRewardTracker(BaseCallback):
    """Logs episode rewards and prints a summary every PRINT_EVERY episodes."""

    PRINT_EVERY = 100  # print a line every N completed episodes

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self._ep_reward: float = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
            n = len(self.episode_rewards)

            if n % self.PRINT_EVERY == 0:
                recent = self.episode_rewards[-self.PRINT_EVERY :]
                avg = np.mean(recent)
                best = np.max(recent)
                print(
                    f"  [ep {n:5d}]  avg reward (last {self.PRINT_EVERY}): "
                    f"{avg:9.1f}  |  best: {best:9.1f}  |  "
                    f"steps: {self.num_timesteps:8d}"
                )
        return True  # continue training


# ── Fixed-timer baseline (always picks 30 s green) ───────────────────────────
def evaluate_fixed_baseline(
    green_seconds: int = 30, n_episodes: int = 100
) -> tuple[float, float]:
    """Evaluate the naive fixed-timer policy.

    Args:
        green_seconds: Always use this green duration.
        n_episodes:    Number of evaluation episodes.

    Returns:
        (mean_reward, std_reward)
    """
    # Action index for green_seconds (must be in GREEN_DURATIONS)
    from simulation.traffic_env import GREEN_DURATIONS
    action = GREEN_DURATIONS.index(green_seconds)

    env = TrafficEnv()
    rewards: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# ── Trained agent evaluation ─────────────────────────────────────────────────
def evaluate_agent(
    model: DQN, n_episodes: int = 100
) -> tuple[float, float, int]:
    """Evaluate the trained DQN agent.

    Returns:
        (mean_reward, std_reward, ambulances_cleared_count)
    """
    env = TrafficEnv()
    rewards: list[float] = []
    cleared = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            total += r
            done = terminated or truncated
        rewards.append(total)
        if info.get("ambulance_cleared", False):
            cleared += 1
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards)), cleared


# ── Learning curve plot ───────────────────────────────────────────────────────
def plot_learning_curve(
    episode_rewards: list[float],
    baseline_reward: float,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    # Raw rewards (faint)
    ax.plot(
        episode_rewards,
        alpha=0.25,
        color="royalblue",
        linewidth=0.7,
        label="Episode reward (raw)",
    )

    # 50-episode smoothed trend
    window = 50
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, window - 1 + len(smoothed)),
            smoothed,
            color="royalblue",
            linewidth=2.5,
            label=f"Smoothed ({window}-ep avg)",
        )

    # Baseline reference line
    ax.axhline(
        y=baseline_reward,
        color="tomato",
        linestyle="--",
        linewidth=2,
        label=f"Fixed-30s baseline ({baseline_reward:.0f})",
    )

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Total Episode Reward", fontsize=13)
    ax.set_title("DQN Training — Adaptive Traffic Signal Control", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n📊 Learning curve saved → {save_path}")


# ── Main training routine ─────────────────────────────────────────────────────
def train() -> DQN:
    print("=" * 65)
    print("  DQN Training — Adaptive Traffic Signal Controller")
    print("=" * 65)

    # Step 1: baseline
    print("\n[1/4] Measuring fixed-30s baseline (100 episodes)…")
    baseline_mean, baseline_std = evaluate_fixed_baseline(30, n_episodes=100)
    print(f"      Fixed-30s baseline : {baseline_mean:9.1f} ± {baseline_std:.1f}")

    # Step 2: build environment
    print("\n[2/4] Creating training environment…")
    env = Monitor(TrafficEnv())
    print(f"      Observation space  : {env.observation_space}")
    print(f"      Action space       : {env.action_space}")

    # Step 3: build DQN
    print("\n[3/4] Building DQN model…")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        target_update_interval=TARGET_UPDATE,
        exploration_fraction=EXPLORATION_FRAC,
        exploration_initial_eps=EPS_START,
        exploration_final_eps=EPS_END,
        policy_kwargs={"net_arch": NET_ARCH},
        verbose=0,
        seed=SEED,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"      Parameters         : {n_params:,}")

    # Step 4: train
    print(f"\n[4/4] Training for {TOTAL_TIMESTEPS:,} steps…")
    print(f"      (progress printed every {EpisodeRewardTracker.PRINT_EVERY} episodes)\n")

    tracker = EpisodeRewardTracker()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=tracker, progress_bar=False)

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅  Model saved → {MODEL_SAVE_PATH}.zip")

    # Step 5: evaluate trained agent
    print("\n[5/5] Evaluating trained agent vs baseline (100 episodes each)…")
    agent_mean, agent_std, amb_cleared = evaluate_agent(model, n_episodes=100)

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Fixed-30s baseline : {baseline_mean:9.1f}  ±{baseline_std:.1f}")
    print(f"  Trained DQN agent  : {agent_mean:9.1f}  ±{agent_std:.1f}")
    improvement = agent_mean - baseline_mean
    sign = "+" if improvement >= 0 else ""
    verdict = "✅ DQN beats fixed baseline!" if improvement > 0 else "⚠️  Needs more training"
    print(f"  Improvement        : {sign}{improvement:.1f}  → {verdict}")
    print(f"  Ambulances cleared : {amb_cleared}/100 episodes")
    print("=" * 65)

    # Plot
    plot_learning_curve(tracker.episode_rewards, baseline_mean, CURVE_SAVE_PATH)

    return model


if __name__ == "__main__":
    train()