"""Action definitions, reward functions, and MDP transition matrix builder.

Adds a small discrete action set, a reward/utility function,
and builds action-dependent transition matrices for the MDP.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# --- Action Set ---

ACTIONS: List[str] = ["stay", "left", "right"]
"""Three discrete actions: stay in cluster, move to previous cluster, move to next cluster."""

N_ACTIONS: int = len(ACTIONS)
ACTION_TO_IDX: dict = {a: i for i, a in enumerate(ACTIONS)}


def build_mdp_transition_matrix(
    base_P: np.ndarray,
    cluster_ids: np.ndarray,
    n_actions: int = 3,
    self_loop_bias: float = 0.3,
    within_cluster_bias: float = 0.5,
    action_strength: float = 0.3,
    noise_level: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convert a base Markov transition matrix into an MDP with action-dependent transitions.

    Strategy: For each state i and each action a, define P[i, a, :] as:
      - a=0 (stay):   Same as base_P[i] (tend to stay in cluster)
      - a=1 (left):   Shift probability mass to states in the *previous* cluster
      - a=2 (right):  Shift probability mass to states in the *next* cluster

    Args:
        base_P: Base transition matrix of shape (n_states, n_states).
        cluster_ids: Array of shape (n_states,) with cluster assignments.
        n_actions: Number of discrete actions (default: 3).
        self_loop_bias: Self-loop probability bias.
        within_cluster_bias: Within-cluster transition bias.
        action_strength: How strongly actions shift probability mass (0-1).
        noise_level: Uniform noise in each row for exploration.
        rng: NumPy random generator.

    Returns:
        MDP transition tensor of shape (n_states, n_actions, n_states).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_states = base_P.shape[0]
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    # Build cluster-to-cluster transition template from base_P
    # For each state, compute what fraction goes to each cluster
    P_mdp = np.zeros((n_states, n_actions, n_states))

    for i in range(n_states):
        current_cluster = cluster_ids[i]

        # Base transition (stay action)
        # Start from base_P[i] but enforce self-loop and within-cluster bias
        base_row = base_P[i].copy()

        # Action 0 (stay): same as base, which already has cluster structure
        P_mdp[i, 0] = _normalize_row(base_row, n_states, noise_level, rng)

        # Action 1 (left): shift mass toward previous cluster
        prev_cluster = (current_cluster - 1) % n_clusters
        P_mdp[i, 1] = _action_shifted_row(
            base_row, cluster_ids, i, prev_cluster,
            action_strength, noise_level, n_states, rng,
        )

        # Action 2 (right): shift mass toward next cluster
        next_cluster = (current_cluster + 1) % n_clusters
        P_mdp[i, 2] = _action_shifted_row(
            base_row, cluster_ids, i, next_cluster,
            action_strength, noise_level, n_states, rng,
        )

    return P_mdp


def _normalize_row(
    row: np.ndarray,
    n_states: int,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add uniform noise and renormalize a transition row."""
    noise = rng.uniform(0, noise_level, size=n_states)
    row = row + noise
    row = row / row.sum()
    return row


def _action_shifted_row(
    base_row: np.ndarray,
    cluster_ids: np.ndarray,
    state_idx: int,
    target_cluster: int,
    action_strength: float,
    noise_level: float,
    n_states: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a transition row biased toward a target cluster."""
    mask = (cluster_ids == target_cluster)
    n_target = mask.sum()

    # Start from base row
    row = base_row.copy()

    if n_target == 0:
        return _normalize_row(row, n_states, noise_level, rng)

    # Pull probability mass from non-target states toward target cluster
    mass_to_shift = action_strength * (1.0 - row[mask].sum())
    non_target_indices = np.where(~mask)[0]

    if len(non_target_indices) > 0:
        # Take proportional mass from each non-target state
        for j in non_target_indices:
            shift = row[j] * action_strength
            row[j] -= shift
            mass_to_shift += shift

        # Distribute shifted mass uniformly across target cluster
        row[mask] += mass_to_shift / n_target

    return _normalize_row(row, n_states, noise_level, rng)


# --- Reward Function ---


def compute_reward(
    state: int,
    action: int,
    next_state: int,
    cluster_id_state: int | None,
    cluster_id_next: int | None,
    *,
    stability_reward: float = 0.5,
    goal_cluster_reward: float = 1.0,
    action_cost: float = 0.05,
    goal_cluster_id: int = 0,
) -> float:
    """Compute reward for a (state, action, next_state) transition.

    Args:
        state: Current state.
        action: Action taken (integer index).
        next_state: Resulting next state.
        cluster_id_state: Cluster of current state (can be None).
        cluster_id_next: Cluster of next state (can be None).
        stability_reward: Reward for staying in same cluster.
        goal_cluster_reward: Reward for reaching goal cluster.
        action_cost: Penalty for non-stay actions.
        goal_cluster_id: Which cluster is the "goal".

    Returns:
        Scalar reward value.
    """
    r = 0.0

    # Stability: staying in same cluster
    if cluster_id_state is not None and cluster_id_next is not None:
        if cluster_id_state == cluster_id_next:
            r += stability_reward

        # Goal cluster bonus
        if cluster_id_next == goal_cluster_id:
            r += goal_cluster_reward

    # Action cost: penalize non-stay actions
    if action != 0:  # "stay" is 0
        r -= action_cost

    return r


# --- Utility Functions ---


def get_action_name(action_idx: int) -> str:
    """Get human-readable action name."""
    if 0 <= action_idx < N_ACTIONS:
        return ACTIONS[action_idx]
    return f"action_{action_idx}"


def sample_action(action_probs: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an action from a probability distribution.

    Args:
        action_probs: Probability distribution over actions.
        rng: NumPy random generator.

    Returns:
        Sampled action index.
    """
    return int(rng.choice(N_ACTIONS, p=action_probs))