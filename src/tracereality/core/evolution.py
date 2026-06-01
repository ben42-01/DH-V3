"""Evolutionary dynamics for observer populations.

Implements fitness-based selection, mutation, replication,
and population management for the observer agents.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .observer import NeuralObserver
from ..actions import N_ACTIONS, sample_action
from ..config import ExperimentConfig
from ..trace_store import TraceStore


class EvolutionEngine:
    """Manages evolutionary dynamics for a population of observers.

    Performs periodic selection (keep top-k, replace bottom-k),
    mutation (weight perturbation), and replication with logging.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        rng: np.random.Generator,
    ):
        """Initialize the evolution engine.

        Args:
            config: Experiment configuration.
            rng: NumPy random generator for reproducibility.
        """
        self.config = config
        self.rng = rng
        self.generation = 0
        self.log: List[dict] = []

    def compute_fitness(
        self,
        observer: NeuralObserver,
        trace_store: TraceStore,
    ) -> float:
        """Compute fitness for an observer.

        When policy learning is enabled:
            fitness = reward_weight * cumulative_reward + acc_weight * recent_accuracy
        Otherwise:
            fitness = recent prediction accuracy

        Args:
            observer: The observer to evaluate.
            trace_store: TraceStore with accumulated history.

        Returns:
            Fitness value (higher = better).
        """
        if self.config.use_policy_learning:
            # Reward-based fitness
            reward_fitness = observer.cumulative_reward
            acc_weight = self.config.fitness_acc_weight
            reward_weight = self.config.fitness_reward_weight

            if len(observer.accuracy_history) == 0:
                return reward_weight * reward_fitness

            recent_acc = np.mean(observer.accuracy_history[-100:])
            return reward_weight * reward_fitness + acc_weight * recent_acc

        # Standard accuracy-based fitness
        if len(observer.accuracy_history) == 0:
            return 0.0

        recent_acc = np.mean(observer.accuracy_history[-100:])
        return float(recent_acc)

    def evaluate_population(
        self,
        observers: List[NeuralObserver],
        trace_store: TraceStore,
    ) -> List[float]:
        """Compute fitness for all observers.

        Args:
            observers: List of observers.
            trace_store: TraceStore with accumulated history.

        Returns:
            List of fitness values.
        """
        fitnesses = []
        for obs in observers:
            f = self.compute_fitness(obs, trace_store)
            obs.record_fitness(f)
            fitnesses.append(f)
        return fitnesses

    def select(
        self,
        observers: List[NeuralObserver],
        fitnesses: List[float],
    ) -> Tuple[List[NeuralObserver], List[int], List[int]]:
        """Perform selection: keep top-k, discard bottom-k.

        Args:
            observers: Current population.
            fitnesses: Fitness values for each observer.

        Returns:
            Tuple of (surviving_observers, kept_indices, discarded_indices).
        """
        n = len(observers)
        top_k = min(self.config.selection_top_k, n)
        bottom_k = min(self.config.selection_bottom_k, n)

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        kept_indices = sorted_indices[:top_k].tolist()
        discarded_indices = sorted_indices[-bottom_k:].tolist()

        survivors = [observers[i] for i in kept_indices]

        return survivors, kept_indices, discarded_indices

    def replicate(
        self,
        survivors: List[NeuralObserver],
        target_size: int,
        discarded_count: int,
    ) -> List[NeuralObserver]:
        """Replicate survivors to fill population back to target size.

        Args:
            survivors: Surviving observers after selection.
            target_size: Desired population size.
            discarded_count: Number of observers that were removed.

        Returns:
            New population of size target_size.
        """
        new_population = list(survivors)
        next_id = max(obs.id for obs in survivors) + 1

        # Replicate survivors with mutations to fill the gap
        while len(new_population) < target_size:
            # Pick a random survivor to clone
            parent = survivors[self.rng.integers(0, len(survivors))]
            child = parent.clone(next_id)
            next_id += 1

            # Apply mutation to the child
            if self.config.with_mutation:
                mutation_rate = self.rng.normal(
                    self.config.mutation_rate_mean,
                    self.config.mutation_rate_std,
                )
                mutation_rate = max(0.001, mutation_rate)  # Clamp positive
                child.set_mutation_rate(mutation_rate)
                child.mutate(magnitude=mutation_rate)

            new_population.append(child)

        return new_population

    def evolve(
        self,
        observers: List[NeuralObserver],
        trace_store: TraceStore,
    ) -> Tuple[List[NeuralObserver], dict]:
        """Run one evolutionary step: evaluate, select, replicate, mutate.

        Args:
            observers: Current population.
            trace_store: TraceStore with accumulated history.

        Returns:
            Tuple of (new_population, evolution_log_entry).
        """
        self.generation += 1

        # 1. Evaluate fitness
        fitnesses = self.evaluate_population(observers, trace_store)

        # 2. Selection
        if self.config.with_selection:
            survivors, kept_idx, discarded_idx = self.select(observers, fitnesses)

            # 3. Replicate to fill population
            new_population = self.replicate(
                survivors,
                len(observers),
                len(discarded_idx),
            )
        else:
            # No selection: keep all, but still mutate
            new_population = list(observers)
            kept_idx = list(range(len(observers)))
            discarded_idx = []

        # 4. Log evolution event
        log_entry = {
            "generation": self.generation,
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
            "n_survivors": len(kept_idx),
            "n_discarded": len(discarded_idx),
            "mutation_rates": [obs.mutation_rate for obs in new_population],
        }
        self.log.append(log_entry)

        return new_population, log_entry

    def get_evolution_log(self) -> pd.DataFrame:
        """Get evolution log as a DataFrame.

        Returns:
            DataFrame with evolution history.
        """
        import pandas as pd
        return pd.DataFrame(self.log)

    def should_evolve(self, step: int) -> bool:
        """Check if evolution should occur at this step.

        Args:
            step: Current simulation step.

        Returns:
            True if evolution should occur.
        """
        return (
            step > 0
            and step % self.config.selection_interval == 0
            and self.config.with_evolution
        )