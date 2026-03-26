"""Standalone evaluation script — RL agent vs fixed-timer baselines.

Loads the trained DQN model (signal_policy.zip) and benchmarks it against
two fixed-timer baselines (always 20 s green and always 30 s green).

Usage:
    python training/evaluate.py

Output (printed to stdout):
    Comparison table with mean reward, std, and ambulance clearance rate.

Output (saved file):
    training/checkpoints/comparison.png  ← bar chart

Requirements:
    • Run training/train_rl.py first to create signal_policy.zip.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.traffic_env import TrafficEnv, GREEN_DURATIONS

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")
MODEL_PATH      = os.path.join(CHECKPOINT_DIR, "signal_policy.zip")
CHART_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "comparison.png")

N_EPISODES = 200   # episodes per policy (more → more reliable estimate)


# ── Policy runners ────────────────────────────────────────────────────────────

def run_fixed_policy(green_seconds: int, n_episodes: int = N_EPISODES):
    """Always pick the same green duration."""
    action = GREEN_DURATIONS.index(green_seconds)
    env = TrafficEnv()
    rewards: list[float] = []
    amb_cleared = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        info: dict = {}
        while not done:
            obs, r, terminated, truncated, info = env.step(action)
            total += r
            done = terminated or truncated
        rewards.append(total)
        if info.get("ambulance_cleared", False):
            amb_cleared += 1
    env.close()
    return np.mean(rewards), np.std(rewards), amb_cleared


def run_rl_policy(model: DQN, n_episodes: int = N_EPISODES):
    """Let the DQN agent decide every step."""
    env = TrafficEnv()
    rewards: list[float] = []
    amb_cleared = 0
    action_histogram = {d: 0 for d in GREEN_DURATIONS}

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, r, terminated, truncated, info = env.step(action)
            total += r
            done = terminated or truncated
            action_histogram[GREEN_DURATIONS[action]] += 1
        rewards.append(total)
        if info.get("ambulance_cleared", False):
            amb_cleared += 1

    env.close()
    return np.mean(rewards), np.std(rewards), amb_cleared, action_histogram


# ── Bar-chart comparison ──────────────────────────────────────────────────────

def plot_comparison(
    labels: list[str],
    means: list[float],
    stds: list[float],
    cleared: list[int],
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # — Left: reward comparison —
    colors = ["#e57373", "#ef9a9a", "#42a5f5"]  # reds for baselines, blue for RL
    bars = axes[0].bar(labels, means, yerr=stds, color=colors,
                       capsize=8, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Mean Episode Reward", fontsize=14)
    axes[0].set_ylabel("Total Reward per Episode", fontsize=12)
    axes[0].grid(axis="y", alpha=0.35)

    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + abs(std) + 50,
            f"{mean:.0f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # — Right: ambulance clearance rate —
    rates = [c / N_EPISODES * 100 for c in cleared]
    axes[1].bar(labels, rates, color=colors, edgecolor="white", linewidth=1.2)
    axes[1].set_title("Ambulance Clearance Rate", fontsize=14)
    axes[1].set_ylabel("% Episodes Ambulance Cleared", fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.35)

    for i, (label, rate) in enumerate(zip(labels, rates)):
        axes[1].text(i, rate + 1.5, f"{rate:.1f}%",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"RL Agent vs Fixed-Timer Baselines  ({N_EPISODES} episodes each)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📊 Comparison chart saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate() -> None:
    print("=" * 65)
    print("  Evaluation — RL Agent vs Fixed-Timer Baselines")
    print("=" * 65)

    # 1. Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌  Model not found at: {MODEL_PATH}")
        print("    Run  python training/train_rl.py  first.")
        sys.exit(1)

    # 2. Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)
    print("✅  Model loaded.\n")

    results: dict[str, tuple] = {}

    # 3. Fixed-20s baseline
    print(f"[1/3] Fixed-20s policy ({N_EPISODES} episodes)…")
    m20, s20, c20 = run_fixed_policy(20)
    results["Fixed 20 s"] = (m20, s20, c20)
    print(f"      mean={m20:.1f}  std={s20:.1f}  amb_cleared={c20}/{N_EPISODES}")

    # 4. Fixed-30s baseline
    print(f"\n[2/3] Fixed-30s policy ({N_EPISODES} episodes)…")
    m30, s30, c30 = run_fixed_policy(30)
    results["Fixed 30 s"] = (m30, s30, c30)
    print(f"      mean={m30:.1f}  std={s30:.1f}  amb_cleared={c30}/{N_EPISODES}")

    # 5. RL agent
    print(f"\n[3/3] DQN RL agent ({N_EPISODES} episodes)…")
    mrl, srl, crl, action_hist = run_rl_policy(model)
    results["DQN Agent"] = (mrl, srl, crl)
    print(f"      mean={mrl:.1f}  std={srl:.1f}  amb_cleared={crl}/{N_EPISODES}")

    # 6. Print results table
    print("\n" + "=" * 65)
    print(f"  {'Policy':<14}  {'Mean Reward':>14}  {'Std':>8}  {'Amb. Cleared':>14}")
    print(f"  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*14}")
    for label, (mean, std, cleared) in results.items():
        pct = cleared / N_EPISODES * 100
        print(f"  {label:<14}  {mean:>14.1f}  {std:>8.1f}  {cleared:>5}/{N_EPISODES}  ({pct:.0f}%)")
    print("=" * 65)

    # Improvement over fixed-30s
    improvement = mrl - m30
    if improvement > 0:
        print(f"\n✅  DQN is {improvement:.1f} points BETTER than fixed-30s baseline.")
    else:
        print(f"\n⚠️  DQN is {abs(improvement):.1f} points WORSE than fixed-30s baseline. "
              f"Try more training steps.")

    # 7. Action distribution
    print("\n  DQN action distribution:")
    total_actions = sum(action_hist.values())
    for duration, count in action_hist.items():
        bar = "█" * int(count / total_actions * 40)
        print(f"    {duration:2d}s : {bar} ({count/total_actions*100:.1f}%)")

    # 8. Plot
    labels = list(results.keys())
    means  = [results[l][0] for l in labels]
    stds   = [results[l][1] for l in labels]
    clears = [results[l][2] for l in labels]
    plot_comparison(labels, means, stds, clears, CHART_SAVE_PATH)


if __name__ == "__main__":
    evaluate()
