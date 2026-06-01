"""Trace storage — records and manages trace history.

Stores the full trace stream and provides sliding windows
for training observer agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


class TraceStore:
    """Ring-buffer for trace history with sliding-window access.

    Stores: (t, true_state, observed_state, cluster_id).
    """

    def __init__(self, max_size: int | None = None):
        """Initialize trace store.

        Args:
            max_size: Maximum number of steps to store.
                None = unlimited (grows unbounded).
        """
        self.max_size = max_size
        self._data: List[dict] = []
        self._observed_states: List[int] = []
        self._true_states: List[int] = []
        self._cluster_ids: List[int] = []
        self._t = 0

    def append(
        self,
        true_state: int,
        observed_state: int,
        cluster_id: int | None,
        action: int | None = None,
        reward: float | None = None,
    ) -> None:
        """Append a step to the trace.

        Args:
            true_state: The actual hidden state.
            observed_state: What agents observe (may be noisy).
            cluster_id: Cluster membership of the true state, or None.
            action: Action taken (for policy learning), or None.
            reward: Reward received (for policy learning), or None.
        """
        record = {
            "t": self._t,
            "true_state": true_state,
            "observed_state": observed_state,
            "cluster_id": cluster_id if cluster_id is not None else -1,
            "action": action if action is not None else -1,
            "reward": reward if reward is not None else 0.0,
        }
        self._data.append(record)
        self._observed_states.append(observed_state)
        self._true_states.append(true_state)
        self._cluster_ids.append(cluster_id if cluster_id is not None else -1)
        self._t += 1

        # Trim if exceeding max_size
        if self.max_size is not None and len(self._data) > self.max_size:
            self._data = self._data[-self.max_size:]
            self._observed_states = self._observed_states[-self.max_size:]
            self._true_states = self._true_states[-self.max_size:]
            self._cluster_ids = self._cluster_ids[-self.max_size:]

    def __len__(self) -> int:
        return len(self._data)

    @property
    def current_step(self) -> int:
        return self._t

    def get_window(
        self,
        length: int,
        include_current: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the most recent window of trace data.

        Args:
            length: Number of recent steps to include.
            include_current: If True, include the most recent step.
                If False, exclude it (for prediction targets).

        Returns:
            Tuple of (true_states, observed_states, cluster_ids),
            each an array of shape (length,) or less if not enough data.
        """
        n = len(self._data)
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        if include_current:
            end = n
        else:
            end = n - 1

        start = max(0, end - length)
        return (
            np.array(self._true_states[start:end]),
            np.array(self._observed_states[start:end]),
            np.array(self._cluster_ids[start:end]),
        )

    def get_history_window(
        self,
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get a history window for prediction training.

        Returns (input_window, target_state) where input_window
        is the observed states for the last `window_size` steps
        (excluding current), and target_state is the most recent
        observed state.

        Args:
            window_size: Number of past steps to include as input.

        Returns:
            (input_window, target) or (None, None) if not enough data.
        """
        n = len(self._data)
        if n < window_size + 1:
            return None, None

        input_window = np.array(self._observed_states[-(window_size + 1):-1])
        target_state = self._observed_states[-1]

        return input_window, target_state

    def get_observed_sequence(self) -> np.ndarray:
        """Get all observed states as an array (O(1) — uses cached list)."""
        return np.array(self._observed_states)

    def get_true_sequence(self) -> np.ndarray:
        """Get all true states as an array (O(1) — uses cached list)."""
        return np.array(self._true_states)

    def get_cluster_sequence(self) -> np.ndarray:
        """Get all cluster IDs as an array (O(1) — uses cached list)."""
        return np.array(self._cluster_ids)

    def get_previous_observed(self, n_back: int = 1) -> int | None:
        """Get the n-th previous observed state without building a full array.

        Args:
            n_back: How many steps back (1 = most recent, 2 = second most recent, etc.)

        Returns:
            The observed state at that position, or None if not enough data.
        """
        if len(self._observed_states) < n_back:
            return None
        return self._observed_states[-n_back]

    def get_observed_at(self, idx: int) -> int:
        """Get observed state at a specific index (O(1)).

        Args:
            idx: Index into the observed states list (supports negative indexing).

        Returns:
            Observed state at that index.
        """
        return self._observed_states[idx]

    def get_observed_slice(self, start: int, end: int) -> List[int]:
        """Get a slice of observed states as a list (O(end-start), not O(t)).

        Args:
            start: Start index (inclusive).
            end: End index (exclusive).

        Returns:
            List of observed states in that range.
        """
        return self._observed_states[start:end]

    def to_dataframe(self) -> pd.DataFrame:
        """Export all traces as a pandas DataFrame."""
        return pd.DataFrame(self._data)

    def to_csv(self, path: Path) -> None:
        """Write traces to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def to_parquet(self, path: Path) -> None:
        """Write traces to Parquet (if pyarrow is available)."""
        df = self.to_dataframe()
        df.to_parquet(path, index=False)