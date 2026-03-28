"""
optimization/train.py
──────────────────────
Offline training script for the 224-action DQN.

Designed for:
  • Google Colab T4 GPU  →  python -m  train --device cuda
  • GitHub Codespaces     →  python -m  train

Typical runtime
───────────────
  Colab T4     : ~15 min  for 100 k steps
  Codespaces   : ~50 min  for 100 k steps  (2-core CPU)

Quick smoke-test (verify the code runs)
───────────────────────────────────────
  python -m  train --steps 500 --log-interval 100

Full training
─────────────
  python -m  train --steps 100000 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from  environment    import TrafficEnv, ACTION_SIZE, decode_action
from  replay_buffer  import ReplayBuffer
from  dqn_agent      import DQNAgent

# ── config (with safe fallback if config.py isn't available) ──────────────────
try:
    import config as cfg
    GAMMA              = cfg.GAMMA
    LEARNING_RATE      = cfg.LEARNING_RATE
    EPSILON_START      = cfg.EPSILON_START
    EPSILON_END        = cfg.EPSILON_END
    EPSILON_DECAY      = cfg.EPSILON_DECAY
    REPLAY_BUFFER_SIZE = cfg.REPLAY_BUFFER_SIZE
    BATCH_SIZE         = cfg.BATCH_SIZE
    TARGET_UPDATE_FREQ = cfg.TARGET_UPDATE_FREQ
    DQN_WEIGHTS_PATH   = cfg.DQN_WEIGHTS_PATH
except (ImportError, AttributeError):
    GAMMA              = 0.95
    LEARNING_RATE      = 1e-3
    EPSILON_START      = 1.0
    EPSILON_END        = 0.05
    EPSILON_DECAY      = 0.9995
    REPLAY_BUFFER_SIZE = 50_000
    BATCH_SIZE         = 64
    TARGET_UPDATE_FREQ = 500
    DQN_WEIGHTS_PATH   = ROOT / "models" / "dqn_signal_optimizer.pt"


# ── training loop ──────────────────────────────────────────────────────────────

def train(
    total_steps:   int  = 100_000,
    device:        str  = "cpu",
    log_interval:  int  = 1_000,
    save_path:     Path = DQN_WEIGHTS_PATH,
    peak_hour:     bool = False,
) -> None:

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*65}")
    print(f"  DQN Traffic Signal Optimizer — Training")
    print(f"  action_size  = {ACTION_SIZE}  (4 directions × 56 durations)")
    print(f"  device       = {device}")
    print(f"  total_steps  = {total_steps:,}")
    print(f"  batch_size   = {BATCH_SIZE}")
    print(f"  save_path    → {save_path}")
    print(f"{'─'*65}\n")

    env    = TrafficEnv(peak_hour=peak_hour, seed=42)
    buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
    agent  = DQNAgent(
        state_size  = env.state_size,
        action_size = env.action_size,   # 224
        lr          = LEARNING_RATE,
        gamma       = GAMMA,
        batch_size  = BATCH_SIZE,
        device      = device,
    )

    epsilon        = EPSILON_START
    state          = env.reset()
    episode_reward = 0.0
    episode_count  = 0

    reward_window: list[float] = []
    loss_window:   list[float] = []
    best_avg_reward             = float("-inf")

    t0 = time.time()

    for step in range(1, total_steps + 1):

        # ── ε-greedy action ────────────────────────────────────────────────────
        action = agent.select_action(state, epsilon)

        # ── step environment ───────────────────────────────────────────────────
        next_state, reward, done, info = env.step(action)
        buffer.push(state, action, reward, next_state, done)

        state          = next_state
        episode_reward += reward

        # ── learn ─────────────────────────────────────────────────────────────
        loss = agent.train_step(buffer)
        if loss:
            loss_window.append(loss)

        # ── sync target ────────────────────────────────────────────────────────
        if step % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        # ── decay ε ────────────────────────────────────────────────────────────
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # ── episode boundary ───────────────────────────────────────────────────
        if done:
            reward_window.append(episode_reward)
            episode_count  += 1
            episode_reward  = 0.0
            state           = env.reset()

        # ── logging ────────────────────────────────────────────────────────────
        if step % log_interval == 0:
            avg_r   = np.mean(reward_window[-50:]) if reward_window else 0.0
            avg_l   = np.mean(loss_window[-200:])  if loss_window  else 0.0
            elapsed = time.time() - t0
            sps     = step / elapsed

            # decode the last action for human-readable display
            d, dur = decode_action(action)
            dirs   = ("N", "S", "E", "W")

            print(
                f"  step {step:>7,} | ε={epsilon:.4f} | "
                f"avg_r={avg_r:+.3f} | loss={avg_l:.5f} | "
                f"last=({dirs[d]},{dur}s) | "
                f"eps={episode_count:,} | {sps:.0f} sps"
            )

            # ── save if this is the best checkpoint so far ─────────────────────
            if avg_r > best_avg_reward and len(reward_window) >= 10:
                best_avg_reward = avg_r
                agent.save(save_path)
                print(f"    ✓ new best  avg_reward={avg_r:+.3f}  → {save_path.name}")

    # ── final save ─────────────────────────────────────────────────────────────
    final_path = save_path.parent / "dqn_signal_optimizer_final.pt"
    agent.save(final_path)

    elapsed = time.time() - t0
    print(f"\n{'─'*65}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Best avg reward   : {best_avg_reward:+.3f}")
    print(f"  Best weights      → {save_path}")
    print(f"  Final weights     → {final_path}")
    print(f"{'─'*65}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DQN traffic-signal optimizer (224 actions)"
    )
    p.add_argument("--steps",        type=int,  default=100_000,
                   help="Total environment steps (default: 100 000)")
    p.add_argument("--device",       type=str,  default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="PyTorch device (use 'cuda' on Colab T4)")
    p.add_argument("--log-interval", type=int,  default=1_000,
                   help="Print stats every N steps")
    p.add_argument("--peak-hour",    action="store_true",
                   help="Use peak-hour arrival rates (harder training scenario)")
    p.add_argument("--save-path",    type=Path,
                   default=DQN_WEIGHTS_PATH,
                   help="Where to save the best checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        total_steps  = args.steps,
        device       = args.device,
        log_interval = args.log_interval,
        save_path    = args.save_path,
        peak_hour    = args.peak_hour,
    )
