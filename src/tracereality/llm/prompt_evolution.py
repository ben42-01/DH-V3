"""Prompt evolution engine for LLM observers.

Manages the evolutionary loop: evaluate fitness, select top variants,
mutate and replicate survivors.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .llm_observer import LLMObserver
from .fitness import FitnessComputer
from ..trace_store import TraceStore


class PromptEvolutionEngine:
    """Manages evolution of LLM observer prompts/system messages.

    The evolutionary loop:
    1. Collect outputs from all variants on a set of trace fragments
    2. Compute fitness (alignment with trace structure)
    3. Select top-k variants, discard bottom-k
    4. Replicate survivors with mutations
    5. Log everything
    """

    def __init__(
        self,
        fitness_computer: FitnessComputer,
        rng: np.random.Generator,
        selection_top_k: int = 6,
        selection_bottom_k: int = 6,
        mutation_rate: float = 0.3,
        fragment_length: int = 10,
        verbose: bool = True,
    ):
        """Initialize the prompt evolution engine.

        Args:
            fitness_computer: Fitness computer for evaluating outputs.
            rng: NumPy random generator for reproducibility.
            selection_top_k: Keep top K variants each generation.
            selection_bottom_k: Discard bottom K variants.
            mutation_rate: Rate for prompt/system message mutations.
            fragment_length: Length of trace fragments to sample.
            verbose: Print per-call progress.
        """
        self.fitness_computer = fitness_computer
        self.rng = rng
        self.selection_top_k = selection_top_k
        self.selection_bottom_k = selection_bottom_k
        self.mutation_rate = mutation_rate
        self.fragment_length = fragment_length
        self.verbose = verbose
        self.generation = 0
        self.log: List[dict] = []

    def sample_fragments(
        self,
        trace_store: TraceStore,
        n_fragments: int,
    ) -> List[List[int]]:
        """Sample random trace fragments from the trace store.

        Args:
            trace_store: TraceStore with accumulated history.
            n_fragments: Number of fragments to sample.

        Returns:
            List of fragment lists (each is a sequence of states).
        """
        n = len(trace_store)
        if n < self.fragment_length:
            return []

        fragments = []
        max_start = n - self.fragment_length
        for _ in range(n_fragments):
            start = self.rng.integers(0, max_start)
            fragment = trace_store.get_observed_slice(start, start + self.fragment_length)
            fragments.append(fragment)
        return fragments

    def evaluate_variants(
        self,
        variants: List[LLMObserver],
        fragments: List[List[int]],
        cluster_sequence: np.ndarray,
    ) -> Dict[int, float]:
        """Evaluate fitness of all variants on the given fragments.

        Args:
            variants: List of LLM observer variants.
            fragments: List of trace fragments to evaluate on.
            cluster_sequence: True cluster IDs for each step (for comparison).

        Returns:
            Dict mapping variant ID to fitness score.
        """
        fitnesses = {}
        n_total = len(variants) * len(fragments)
        n_done = 0
        for variant in variants:
            if self.verbose:
                print(f"    [variant {variant.id}] Querying {len(fragments)} fragments...", end="", flush=True)
            # Get responses for each fragment
            responses = []
            for fragment in fragments:
                response = variant.respond_to_trace_list(fragment)
                responses.append(response)
                n_done += 1
                if self.verbose:
                    print(f" {n_done}/{n_total}", end="", flush=True)

            # Compute fitness
            fitness = self.fitness_computer.compute_fitness(
                responses=responses,
                fragments=fragments,
                cluster_sequence=cluster_sequence,
            )
            variant.record_fitness(fitness)
            fitnesses[variant.id] = fitness
            if self.verbose:
                print(f" fitness={fitness:.4f}")

        return fitnesses

    def select(
        self,
        variants: List[LLMObserver],
        fitnesses: Dict[int, float],
    ) -> Tuple[List[LLMObserver], List[int], List[int]]:
        """Select top-k variants, discard bottom-k.

        Args:
            variants: Current population of variants.
            fitnesses: Dict mapping variant ID to fitness.

        Returns:
            Tuple of (survivors, kept_ids, discarded_ids).
        """
        sorted_variants = sorted(variants, key=lambda v: fitnesses.get(v.id, 0.0), reverse=True)
        top_k = min(self.selection_top_k, len(sorted_variants))
        bottom_k = min(self.selection_bottom_k, len(sorted_variants))

        survivors = sorted_variants[:top_k]
        discarded = sorted_variants[-bottom_k:] if bottom_k > 0 else []

        kept_ids = [v.id for v in survivors]
        discarded_ids = [v.id for v in discarded]

        return survivors, kept_ids, discarded_ids

    def replicate(
        self,
        survivors: List[LLMObserver],
        target_size: int,
    ) -> List[LLMObserver]:
        """Replicate survivors with mutations to reach target population size.

        Args:
            survivors: Surviving observers after selection.
            target_size: Desired population size.

        Returns:
            New population of size target_size.
        """
        new_population = list(survivors)
        next_id = max(v.id for v in survivors) + 1

        while len(new_population) < target_size:
            parent = survivors[self.rng.integers(0, len(survivors))]
            child = parent.clone(next_id)
            next_id += 1
            child.mutate(mutation_rate=self.mutation_rate)
            new_population.append(child)

        return new_population

    def get_evolution_log(self) -> pd.DataFrame:
        """Get the evolution log as a DataFrame.

        Returns:
            DataFrame with generation-by-generation metrics.
        """
        return pd.DataFrame(self.log)

    def evolve(
        self,
        variants: List[LLMObserver],
        trace_store: TraceStore,
        cluster_sequence: np.ndarray,
        n_eval_fragments: int = 20,
    ) -> Tuple[List[LLMObserver], dict]:
        """Run one generation of evolution.

        Args:
            variants: Current population of LLM observer variants.
            trace_store: TraceStore with accumulated history.
            cluster_sequence: True cluster IDs for each step.
            n_eval_fragments: Number of fragments to evaluate on.

        Returns:
            Tuple of (new_population, evolution_log_entry).
        """
        self.generation += 1

        if self.verbose:
            print(f"  [Generation {self.generation}] Sampling fragments...")

        # 1. Sample fragments from trace store
        fragments = self.sample_fragments(trace_store, n_eval_fragments)
        if not fragments:
            if self.verbose:
                print(f"  [Generation {self.generation}] No fragments available (trace too short).")
            return variants, {"generation": self.generation, "error": "no_fragments"}

        if self.verbose:
            print(f"  [Generation {self.generation}] Evaluating {len(variants)} variants...")

        # 2. Evaluate all variants
        fitnesses = self.evaluate_variants(variants, fragments, cluster_sequence)

        # 3. Selection
        survivors, kept_ids, discarded_ids = self.select(variants, fitnesses)

        if self.verbose:
            print(f"  [Generation {self.generation}] Kept ids={kept_ids}, discarded ids={discarded_ids}")

        # 4. Replicate
        new_population = self.replicate(survivors, len(variants))

        # 5. Log
        fitness_values = list(fitnesses.values())
        log_entry = {
            "generation": self.generation,
            "mean_fitness": float(np.mean(fitness_values)),
            "std_fitness": float(np.std(fitness_values)),
            "max_fitness": float(np.max(fitness_values)),
            "min_fitness": float(np.min(fitness_values)),
            "n_survivors": len(survivors),
            "n_discarded": len(discarded_ids),
            "kept_ids": kept_ids,
            "discarded_ids": discarded_ids,
        }
        self.log.append(log_entry)

        if self.verbose:
            print(f"  [Generation {self.generation}] Done. Mean fitness: {log_entry['mean_fitness']:.4f}, "
                  f"Max: {log_entry['max_fitness']:.4f}")

        return new_population, log_entry