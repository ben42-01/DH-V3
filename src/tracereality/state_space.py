"""State space and clustered transition matrix construction.

Defines the hidden Markov world structure with clustered states,
self-loop biases, and controlled stochastic transitions.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def assign_clusters(n_states: int, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each state to a cluster.

    States are assigned randomly but each cluster gets roughly equal size.

    Args:
        n_states: Number of states (0..N-1).
        n_clusters: Number of clusters.
        rng: NumPy random generator.

    Returns:
        Array of shape (n_states,) with cluster_id for each state.
    """
    # Assign states round-robin to ensure balanced clusters
    cluster_ids = np.arange(n_clusters).repeat(
        np.array([n_states // n_clusters] * n_clusters) +
        (np.arange(n_clusters) < n_states % n_clusters).astype(int)
    )
    rng.shuffle(cluster_ids)
    return cluster_ids


def build_clustered_transition_matrix(
    n_states: int,
    cluster_ids: np.ndarray,
    self_loop_bias: float = 0.3,
    within_cluster_bias: float = 0.5,
    between_cluster_noise: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a clustered transition matrix P_world.

    The matrix has the following structure:
    - self_loop_bias: probability of staying in the same state.
    - within_cluster_bias: probability of transitioning to another state
      in the same cluster (distributed among them).
    - between_cluster_noise: residual probability mass for transitions
      to states in other clusters.

    Each row sums to 1. The matrix is ergodic if all clusters are
    reachable (ensured by between_cluster_noise > 0).

    Args:
        n_states: Number of states.
        cluster_ids: Array of shape (n_states,) with cluster_id per state.
        self_loop_bias: Probability of staying in the same state.
        within_cluster_bias: Probability mass for within-cluster transitions
            (excluding self-loop).
        between_cluster_noise: Probability mass for between-cluster transitions.
        rng: NumPy random generator for reproducibility.

    Returns:
        Transition matrix of shape (n_states, n_states), row-stochastic.
    """
    if rng is None:
        rng = np.random.default_rng()

    P = np.zeros((n_states, n_states))

    for i in range(n_states):
        current_cluster = cluster_ids[i]

        # Identify states in the same cluster (excluding self)
        same_cluster_mask = (cluster_ids == current_cluster)
        same_cluster_indices = np.where(same_cluster_mask)[0]
        same_cluster_others = same_cluster_indices[same_cluster_indices != i]

        # Identify states in different clusters
        diff_cluster_indices = np.where(~same_cluster_mask)[0]

        # Assign probabilities
        # Self-loop
        P[i, i] = self_loop_bias

        # Within-cluster (to other states in same cluster)
        if len(same_cluster_others) > 0:
            per_within = within_cluster_bias / len(same_cluster_others)
            P[i, same_cluster_others] = per_within
        else:
            # If this state is the only one in its cluster, redistribute
            # self-loop gets both self_loop_bias and within_cluster_bias
            P[i, i] += within_cluster_bias

        # Between-cluster
        if len(diff_cluster_indices) > 0:
            per_between = between_cluster_noise / len(diff_cluster_indices)
            P[i, diff_cluster_indices] = per_between
        else:
            # If all states are in one cluster, add between-cluster mass
            # to other states
            other_indices = np.setdiff1d(np.arange(n_states), [i])
            if len(other_indices) > 0:
                per_other = between_cluster_noise / len(other_indices)
                P[i, other_indices] += per_other
            else:
                P[i, i] += between_cluster_noise

        # Add small random noise to break symmetry and ensure ergodicity
        noise = rng.uniform(0, 0.01, size=n_states)
        P[i, :] += noise
        # Renormalize
        P[i, :] /= P[i, :].sum()

    return P


def compute_stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a Markov chain.

    Solves πP = π subject to sum(π) = 1.

    Args:
        P: Transition matrix of shape (n_states, n_states).

    Returns:
        Stationary distribution π of shape (n_states,).
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()
    return pi


def check_ergodicity(P: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a transition matrix is ergodic (irreducible + aperiodic).

    A sufficient condition: all entries of P^n are > 0 for some n.

    Args:
        P: Transition matrix.
        tol: Tolerance for zero detection.

    Returns:
        True if ergodic.
    """
    n = P.shape[0]
    # Check irreducibility: every state reachable from every other state
    # by checking (I + P)^(n-1) has all positive entries
    Q = np.eye(n) + P
    Q_power = np.linalg.matrix_power(Q, n - 1)
    return np.all(Q_power > tol)


def get_cluster_centroids(P: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    """Compute cluster-level aggregated transition matrix.

    Groups states by cluster and computes the average transition
    distribution from each cluster.

    Args:
        P: Transition matrix of shape (n_states, n_states).
        cluster_ids: Array of shape (n_states,) with cluster_id per state.

    Returns:
        Cluster transition matrix of shape (n_clusters, n_states).
    """
    n_clusters = len(np.unique(cluster_ids))
    centroids = np.zeros((n_clusters, P.shape[1]))
    for c in range(n_clusters):
        mask = cluster_ids == c
        centroids[c] = P[mask].mean(axis=0)
    return centroids