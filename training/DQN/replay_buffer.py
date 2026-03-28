from __future__ import annotations
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, state_size: int = 6) -> None:
        self.capacity = capacity
        self.state_size = state_size
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self._buffer.append((
            np.asarray(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buffer)} transitions, need {batch_size}")
        indices = np.random.choice(
            len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.stack(next_states),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self._buffer) >= batch_size
