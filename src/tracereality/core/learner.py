"""Training utilities for neural observers.

Implements online mini-batch training, count-based learning
(for ablation), and learning rate scheduling.

Key optimization: observers are trained round-robin, one per step,
instead of all observers every step. This is valid because each
observer learns from the same trace stream, and spreading updates
over steps maintains the same total number of gradient steps.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from .observer import NeuralObserver
from ..trace_store import TraceStore


class OnlineLearner:
    """Manages online mini-batch training for a population of observers.

    Observers are trained round-robin (one per step) to keep per-step
    cost O(1) instead of O(n_observers).
    """

    def __init__(
        self,
        batch_size: int = 64,
        history_window_size: int = 3,
        use_history_window: bool = True,
    ):
        """Initialize the learner.

        Args:
            batch_size: Number of samples per training step.
            history_window_size: Number of past states to use as input
                (if use_history_window is True).
            use_history_window: If True, use sliding window of past states
                as input. If False, use only the current state.
        """
        self.batch_size = batch_size
        self.history_window_size = history_window_size
        self.use_history_window = use_history_window
        self._observer_idx = 0

    def train_single_observer(
        self,
        observers: List[NeuralObserver],
        trace_store: TraceStore,
        rng: np.random.Generator,
    ) -> float:
        """Train a single observer on a mini-batch from recent trace history.

        Trains only one observer per call (round-robin). This keeps the
        per-step cost O(1) instead of O(n_observers).

        Args:
            observers: List of observer agents. One is selected round-robin.
            trace_store: TraceStore with accumulated history.
            rng: NumPy random generator for reproducibility.

        Returns:
            Loss value for the trained observer, or 0.0 if not enough data.
        """
        n = len(trace_store)
        if n < self.history_window_size + 2:
            return 0.0

        # Select observer round-robin
        obs = observers[self._observer_idx]
        self._observer_idx = (self._observer_idx + 1) % len(observers)

        # Sample batch from recent history (last half of trace)
        start_idx = max(0, n // 2)
        available = n - self.history_window_size - 1
        if available <= start_idx:
            available = n - self.history_window_size - 1
            start_idx = max(0, available - self.batch_size)

        n_indices = min(self.batch_size, available - start_idx)
        if n_indices <= 0:
            return 0.0

        indices = rng.integers(start_idx, available, size=n_indices)

        # Build input-output pairs using O(1) list access (no full array copy)
        if self.use_history_window:
            inputs = np.zeros((len(indices), self.history_window_size), dtype=np.int64)
        else:
            inputs = np.zeros(len(indices), dtype=np.int64)

        targets = np.zeros(len(indices), dtype=np.int64)

        for i, idx in enumerate(indices):
            if self.use_history_window:
                window = trace_store.get_observed_slice(idx, idx + self.history_window_size)
                inputs[i] = np.array(window, dtype=np.int64)
            else:
                inputs[i] = trace_store.get_observed_at(idx + self.history_window_size - 1)
            targets[i] = trace_store.get_observed_at(idx + self.history_window_size)

        # Train the selected observer
        loss = obs.train_step(inputs, targets)
        return loss

    def train_batch(
        self,
        observers: List[NeuralObserver],
        trace_store: TraceStore,
        rng: np.random.Generator,
    ) -> List[float]:
        """Train all observers on a mini-batch from recent trace history.

        DEPRECATED: Use train_single_observer for O(1) per-step cost.
        This method trains ALL observers and is O(n_observers).

        Args:
            observers: List of observer agents to train.
            trace_store: TraceStore with accumulated history.
            rng: NumPy random generator for reproducibility.

        Returns:
            List of loss values, one per observer.
        """
        n = len(trace_store)
        if n < self.history_window_size + 2:
            return [0.0] * len(observers)

        # Sample batch from recent history (last half of trace)
        start_idx = max(0, n // 2)
        available = n - self.history_window_size - 1
        if available <= start_idx:
            available = n - self.history_window_size - 1
            start_idx = max(0, available - self.batch_size)

        n_indices = min(self.batch_size, available - start_idx)
        if n_indices <= 0:
            return [0.0] * len(observers)

        indices = rng.integers(start_idx, available, size=n_indices)

        # Build input-output pairs using O(1) list access (no full array copy)
        if self.use_history_window:
            inputs = np.zeros((len(indices), self.history_window_size), dtype=np.int64)
        else:
            inputs = np.zeros(len(indices), dtype=np.int64)

        targets = np.zeros(len(indices), dtype=np.int64)

        for i, idx in enumerate(indices):
            if self.use_history_window:
                window = trace_store.get_observed_slice(idx, idx + self.history_window_size)
                inputs[i] = np.array(window, dtype=np.int64)
            else:
                inputs[i] = trace_store.get_observed_at(idx + self.history_window_size - 1)
            targets[i] = trace_store.get_observed_at(idx + self.history_window_size)

        # Train each observer
        losses = []
        for obs in observers:
            loss = obs.train_step(inputs, targets)
            losses.append(loss)

        return losses


class CountBasedLearner:
    """Count-based Markov estimator (for ablation studies).

    Maintains a simple transition count matrix and produces
    maximum-likelihood estimates of transition probabilities.
    """

    def __init__(self, n_states: int, history_window_size: int = 1):
        """Initialize count-based learner.

        Args:
            n_states: Number of states.
            history_window_size: Number of past states to condition on
                (for higher-order Markov models). Currently only 1 is supported.
        """
        self.n_states = n_states
        self.history_window_size = history_window_size
        self.counts = np.zeros((n_states, n_states), dtype=np.float64)
        self.total_updates = 0

    def update(self, from_state: int, to_state: int) -> None:
        """Update transition counts.

        Args:
            from_state: Current state.
            to_state: Next state.
        """
        self.counts[from_state, to_state] += 1.0
        self.total_updates += 1

    def predict(self, from_state: int) -> np.ndarray:
        """Predict next-state distribution from counts.

        Args:
            from_state: Current state.

        Returns:
            Probability distribution of shape (n_states,).
        """
        row = self.counts[from_state]
        if row.sum() == 0:
            return np.ones(self.n_states) / self.n_states
        return row / row.sum()

    def get_transition_matrix_estimate(self) -> np.ndarray:
        """Get the maximum-likelihood transition matrix.

        Returns:
            Matrix of shape (n_states, n_states).
        """
        P_est = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            P_est[i] = self.predict(i)
        return P_est