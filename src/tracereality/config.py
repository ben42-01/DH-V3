"""Configuration management for TraceReality experiments.

All experimental parameters are defined here, with JSON serialization
for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """All configurable parameters for a TraceReality experiment.

    Serializes to/from JSON for reproducibility.
    """

    # --- World parameters ---
    n_states: int = 80
    n_clusters: int = 8
    n_steps: int = 100_000

    # --- Observer parameters ---
    n_observers: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 128
    embedding_dim: int = 32
    n_layers: int = 2
    use_history_window: bool = True
    history_window_size: int = 3

    # --- Evolution parameters ---
    selection_interval: int = 5000
    selection_top_k: int = 10
    selection_bottom_k: int = 10
    mutation_rate_mean: float = 0.01
    mutation_rate_std: float = 0.005
    mutate_weights: bool = True
    mutate_learning_rate: bool = False

    # --- Observation noise ---
    noise_level: float = 0.05
    use_noisy_observations: bool = True

    # --- Experiment control ---
    seeds: List[int] = field(default_factory=lambda: [42])
    mode: str = "nn"  # "nn" or "count"
    with_evolution: bool = True
    with_selection: bool = True
    with_mutation: bool = True
    prefix: str = "exp"

    # --- World matrix params ---
    self_loop_bias: float = 0.3
    within_cluster_bias: float = 0.5
    between_cluster_noise: float = 0.2

    # --- Policy learning params ---
    use_policy_learning: bool = False
    n_actions: int = 3
    action_strength: float = 0.3
    discount_factor: float = 0.99
    policy_learning_rate: float = 1e-3
    stability_reward: float = 0.5
    goal_cluster_reward: float = 1.0
    action_cost: float = 0.05
    goal_cluster_id: int = 0
    fitness_reward_weight: float = 1.0
    fitness_acc_weight: float = 0.2

    # --- Analysis intervals ---
    analyze_every: int = 1000  # steps between full population metric computations (KL, matrices, etc.)
    log_every: int = 100       # steps between lightweight console log updates (running accuracy only)
    matrix_interval: int = 10000  # steps between full transition matrix parameter snapshots
    n_bootstrap_samples: int = 1000  # for confidence intervals

    def to_json(self, path: Path) -> None:
        """Serialize config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def clone_with_updates(self, **kwargs) -> "ExperimentConfig":
        """Create a new config with overridden parameters."""
        d = asdict(self)
        d.update(kwargs)
        return ExperimentConfig(**d)