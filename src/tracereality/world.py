"""Markov world process.

Generates trace streams from a hidden transition matrix with
optional observation noise.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class MarkovWorld:
    """A Markov chain world with hidden structure.

    The world has N states with a clustered transition matrix.
    Optionally adds observation noise so agents see a noisy
    version of the true state.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        cluster_ids: np.ndarray | None = None,
        noise_level: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the world.

        Args:
            transition_matrix: P_world of shape (n_states, n_states),
                row-stochastic.
            cluster_ids: Optional array of shape (n_states,) with
                cluster_id per state.
            noise_level: Probability of observing a random state
                instead of the true state (0 = no noise).
            rng: NumPy random generator for reproducibility.
        """
        self.P = transition_matrix
        self.n_states = transition_matrix.shape[0]
        self.cluster_ids = cluster_ids
        self.noise_level = noise_level
        self.rng = rng if rng is not None else np.random.default_rng()

        # Validate transition matrix
        if self.P.ndim == 2:
            assert np.allclose(self.P.sum(axis=1), 1.0), \
                "Transition matrix rows must sum to 1"
        elif self.P.ndim == 3:
            assert np.allclose(self.P.sum(axis=2), 1.0), \
                "Transition matrix (state, action, next_state) must have rows summing to 1"

        self._current_state: int | None = None

    def reset(self, initial_state: int | None = None) -> int:
        """Reset the world to an initial state.

        Args:
            initial_state: If given, start at this state.
                Otherwise, sample from the stationary distribution.

        Returns:
            The initial true state.
        """
        if initial_state is not None:
            self._current_state = initial_state
        else:
            # Start at a random state
            self._current_state = self.rng.integers(0, self.n_states)
        return self._current_state

    def step(self) -> Tuple[int, int, int | None]:
        """Advance the world by one step (no action, Markov chain).

        Returns:
            Tuple of (true_state, observed_state, cluster_id).
            cluster_id is None if cluster_ids was not provided.
        """
        return self.step_with_action(action=0)

    def step_with_action(self, action: int = 0) -> Tuple[int, int, int | None]:
        """Advance the world by one step with an action (MDP mode).

        If the world has a 2D transition matrix (no actions), the action
        is ignored. If it has a 3D MDP transition matrix, the action
        selects the transition row.

        Args:
            action: Index of the action to take (default: 0/stay).

        Returns:
            Tuple of (true_state, observed_state, cluster_id).
        """
        assert self._current_state is not None, \
            "World must be reset before stepping"

        # Determine if we have action-dependent transitions
        if self.P.ndim == 3:
            probs = self.P[self._current_state, action]
        else:
            probs = self.P[self._current_state]

        # Transition to next state
        next_state = self.rng.choice(self.n_states, p=probs)
        self._current_state = next_state

        # Apply observation noise
        if self.noise_level > 0 and self.rng.uniform() < self.noise_level:
            observed_state = self.rng.integers(0, self.n_states)
        else:
            observed_state = next_state

        cluster_id = self.cluster_ids[next_state] if self.cluster_ids is not None else None

        return next_state, observed_state, cluster_id

    @property
    def current_state(self) -> int | None:
        return self._current_state

    def get_transition_matrix(self) -> np.ndarray:
        """Return the true transition matrix (copy)."""
        return self.P.copy()

    def get_stationary_distribution(self) -> np.ndarray:
        """Compute and return the stationary distribution."""
        from .state_space import compute_stationary_distribution
        return compute_stationary_distribution(self.P)