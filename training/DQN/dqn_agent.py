"""
optimization/dqn_agent.py
──────────────────────────
Lightweight DQN — now with 224-action output (4 directions × 56 durations).

QNetwork architecture
─────────────────────
  Linear(6 → 128) → ReLU
  Linear(128 → 64) → ReLU
  Linear(64 → 224)

  ~42 k parameters — still loads in <5 ms on Raspberry Pi 4 CPU.
  (Previous 4-action network was ~4 k params; 224 outputs need a slightly
   wider hidden layer to represent all (direction, duration) Q-values well.)

Action codec (imported from environment to keep a single source of truth)
──────────────────────────────────────────────────────────────────────────
  decode_action(action) → (direction: int, duration: int)
  encode_action(direction, duration) → action: int
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from  replay_buffer import ReplayBuffer
from  environment   import ACTION_SIZE, decode_action, encode_action


# ── Q-network ──────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Maps a 6-D normalised state vector to Q-values for all 224 actions.

    The hidden layers are slightly wider than the original 4-action network
    because the agent must learn a 2-D Q-surface (direction × duration)
    rather than a 1-D one.
    """

    def __init__(self, state_size: int = 6, action_size: int = ACTION_SIZE) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # shape: (batch, 224)


# ── DQN agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double-DQN style agent.

    The online network chooses actions; the frozen target network is used
    only for computing stable training targets (synced every N steps).

    Parameters
    ----------
    state_size    : int    – state vector dimension (6)
    action_size   : int    – total discrete actions (224)
    lr            : float  – Adam learning rate
    gamma         : float  – discount factor
    batch_size    : int    – mini-batch size per train_step
    device        : str    – "cuda" | "cpu" | "mps"
    """

    def __init__(
        self,
        state_size:  int   = 6,
        action_size: int   = ACTION_SIZE,
        lr:          float = 1e-3,
        gamma:       float = 0.95,
        batch_size:  int   = 64,
        device:      str   = "cpu",
    ) -> None:
        self.state_size  = state_size
        self.action_size = action_size
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.device      = torch.device(device)

        # online network (trained at every step)
        self.online = QNetwork(state_size, action_size).to(self.device)
        # target network (frozen snapshot, periodically hard-synced)
        self.target = QNetwork(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss — stable for RL

        self._train_steps = 0

    # ── action selection ───────────────────────────────────────────────────────

    def select_action(
        self,
        state:   np.ndarray,
        epsilon: float = 0.0,
    ) -> int:
        """
        ε-greedy selection over all 224 (direction, duration) actions.

        Parameters
        ----------
        state   : np.ndarray shape (6,)
        epsilon : float  – exploration probability (0.0 = fully greedy)

        Returns
        -------
        action : int  in [0, 223]
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.action_size))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online(state_t)          # (1, 224)
        return int(q_values.argmax(dim=1).item())

    def select_action_decoded(
        self,
        state:   np.ndarray,
        epsilon: float = 0.0,
    ) -> tuple[int, int, int]:
        """
        Convenience wrapper — returns (action, direction, duration).

        action    : int  in [0, 223]   – flat index
        direction : int  in {0,1,2,3}  – which lane gets green
        duration  : int  in {5…60}     – green time in whole seconds
        """
        action    = self.select_action(state, epsilon)
        direction, duration = decode_action(action)
        return action, direction, duration

    # ── learning ───────────────────────────────────────────────────────────────

    def train_step(self, buffer: ReplayBuffer) -> float:
        """
        Sample one mini-batch from the replay buffer and do one gradient step.

        Returns
        -------
        loss : float  (0.0 if buffer not yet ready)
        """
        if not buffer.is_ready(self.batch_size):
            return 0.0

        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)           # (B, 6)
        a  = torch.LongTensor(actions).unsqueeze(1).to(self.device)   # (B, 1)
        r  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # (B, 1)
        ns = torch.FloatTensor(next_states).to(self.device)      # (B, 6)
        d  = torch.FloatTensor(dones).unsqueeze(1).to(self.device)    # (B, 1)

        # Q(s, a)  — current estimate for the taken action
        current_q = self.online(s).gather(1, a)                  # (B, 1)

        # r + γ · max_a' Q_target(s', a')   [zeroed if terminal]
        with torch.no_grad():
            max_next_q = self.target(ns).max(dim=1, keepdim=True).values
            target_q   = r + self.gamma * max_next_q * (1.0 - d)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._train_steps += 1
        return float(loss.item())

    def update_target(self) -> None:
        """Hard-copy online → target network weights."""
        self.target.load_state_dict(self.online.state_dict())

    # ── diagnostic helpers ─────────────────────────────────────────────────────

    def top_actions(
        self,
        state:    np.ndarray,
        top_k:    int = 5,
    ) -> list[dict]:
        """
        Return the top-k highest Q-value actions decoded for inspection.

        Useful during training to verify the agent is learning sensible timing.

        Example output:
            [{"action": 91, "direction": "S", "duration": 20, "q": 4.21}, ...]
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.online(state_t).squeeze(0).cpu().numpy()   # (224,)

        top_indices = np.argsort(q_vals)[::-1][:top_k]
        directions  = ("N", "S", "E", "W")
        results     = []
        for idx in top_indices:
            d, dur = decode_action(int(idx))
            results.append({
                "action":    int(idx),
                "direction": directions[d],
                "duration":  dur,
                "q":         round(float(q_vals[idx]), 4),
            })
        return results

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist network weights + optimiser state to disk."""
        torch.save(
            {
                "online_state_dict":    self.online.state_dict(),
                "target_state_dict":    self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_steps":          self._train_steps,
                "action_size":          self.action_size,   # saved for safety check
            },
            str(path),
        )

    def load(self, path: str | Path) -> None:
        """
        Load weights from disk.

        map_location='cpu' ensures weights trained on a Colab T4 GPU
        load correctly on the Raspberry Pi CPU.
        """
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)

        # guard: warn if the saved model has a different action space
        saved_action_size = ckpt.get("action_size", None)
        if saved_action_size and saved_action_size != self.action_size:
            raise ValueError(
                f"Saved model has action_size={saved_action_size} "
                f"but this agent expects action_size={self.action_size}. "
                f"Did you try to load an old 4-action model into the new 224-action agent?"
            )

        self.online.load_state_dict(ckpt["online_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._train_steps = ckpt.get("train_steps", 0)

        self.online.to(self.device)
        self.target.to(self.device)
