"""Tests for the PromptEvolutionEngine."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tracereality.llm.llm_observer import LLMObserver
from tracereality.llm.fitness import FitnessComputer
from tracereality.llm.prompt_evolution import PromptEvolutionEngine
from tracereality.trace_store import TraceStore


@pytest.fixture
def trace_store_with_data() -> TraceStore:
    """TraceStore pre-filled with 100 steps."""
    ts = TraceStore(max_size=200)
    for i in range(100):
        ts.append(
            true_state=i % 16,
            observed_state=i % 16,
            cluster_id=(i % 16) // 4,  # 4 clusters
        )
    return ts


@pytest.fixture
def mock_variants() -> List[LLMObserver]:
    """Create 8 mocked LLM observers."""
    variants = []
    for i in range(8):
        obs = MagicMock(spec=LLMObserver)
        obs.id = i
        obs.respond_to_trace_list.return_value = f"Response from observer {i}"
        obs.record_fitness = MagicMock()
        obs.clone.return_value = MagicMock(spec=LLMObserver)
        obs.clone.return_value.id = 100 + i
        variants.append(obs)
    return variants


@pytest.fixture
def fitness_computer() -> FitnessComputer:
    return FitnessComputer(use_embeddings=False)


@pytest.fixture
def evolution_engine(fitness_computer, rng) -> PromptEvolutionEngine:
    return PromptEvolutionEngine(
        fitness_computer=fitness_computer,
        rng=rng,
        selection_top_k=4,
        selection_bottom_k=2,
        mutation_rate=0.3,
        fragment_length=5,
    )


class TestPromptEvolutionEngineInit:
    """Test initialization."""

    def test_default_parameters(self, fitness_computer, rng):
        engine = PromptEvolutionEngine(
            fitness_computer=fitness_computer,
            rng=rng,
        )
        assert engine.selection_top_k == 6
        assert engine.selection_bottom_k == 6
        assert engine.mutation_rate == 0.3
        assert engine.fragment_length == 10
        assert engine.generation == 0
        assert engine.log == []

    def test_custom_parameters(self, fitness_computer, rng):
        engine = PromptEvolutionEngine(
            fitness_computer=fitness_computer,
            rng=rng,
            selection_top_k=3,
            selection_bottom_k=1,
            mutation_rate=0.5,
            fragment_length=8,
        )
        assert engine.selection_top_k == 3
        assert engine.selection_bottom_k == 1


class TestSampleFragments:
    """Test fragment sampling."""

    def test_sample_fragments_returns_correct_count(self, evolution_engine, trace_store_with_data):
        fragments = evolution_engine.sample_fragments(trace_store_with_data, n_fragments=5)
        assert len(fragments) == 5

    def test_fragment_length(self, evolution_engine, trace_store_with_data):
        fragments = evolution_engine.sample_fragments(trace_store_with_data, n_fragments=3)
        for f in fragments:
            assert len(f) == evolution_engine.fragment_length

    def test_empty_when_not_enough_data(self, evolution_engine):
        ts = TraceStore(max_size=100)
        for _ in range(3):
            ts.append(0, 0, 0)
        fragments = evolution_engine.sample_fragments(ts, n_fragments=5)
        assert fragments == []  # Not enough data for fragment_length=5

    def test_deterministic_with_same_seed(self, rng, fitness_computer, trace_store_with_data):
        engine1 = PromptEvolutionEngine(fitness_computer=fitness_computer, rng=rng)
        engine2 = PromptEvolutionEngine(fitness_computer=fitness_computer, rng=np.random.default_rng(42))
        f1 = engine1.sample_fragments(trace_store_with_data, 5)
        f2 = engine2.sample_fragments(trace_store_with_data, 5)
        assert f1 == f2


class TestSelect:
    """Test the selection algorithm."""

    def test_select_top_k_survive(self, evolution_engine, mock_variants):
        fitnesses = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5, 5: 0.4, 6: 0.3, 7: 0.2}
        survivors, kept_ids, discarded_ids = evolution_engine.select(mock_variants, fitnesses)
        assert len(survivors) == 4  # top_k=4
        assert kept_ids == [0, 1, 2, 3]

    def test_select_bottom_k_discarded(self, evolution_engine, mock_variants):
        fitnesses = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5, 5: 0.4, 6: 0.3, 7: 0.2}
        _, _, discarded_ids = evolution_engine.select(mock_variants, fitnesses)
        assert len(discarded_ids) == 2  # bottom_k=2
        assert 6 in discarded_ids
        assert 7 in discarded_ids

    def test_select_handles_equal_fitness(self, evolution_engine, mock_variants):
        fitnesses = {i: 0.5 for i in range(8)}
        survivors, _, _ = evolution_engine.select(mock_variants, fitnesses)
        assert len(survivors) == 4

    def test_select_respects_population_size(self, fitness_computer, rng):
        engine = PromptEvolutionEngine(
            fitness_computer=fitness_computer,
            rng=rng,
            selection_top_k=10,
            selection_bottom_k=3,
        )
        variants = [MagicMock(spec=LLMObserver, id=i) for i in range(5)]
        fitnesses = {i: float(i) for i in range(5)}
        survivors, _, discarded = engine.select(variants, fitnesses)
        assert len(survivors) == 5  # Can't keep more than exist
        assert len(discarded) == 0  # Can't discard more than exist


class TestReplicate:
    """Test population replication."""

    def test_replicate_to_target_size(self, evolution_engine, mock_variants):
        survivors = mock_variants[:4]
        new_pop = evolution_engine.replicate(survivors, target_size=8)
        assert len(new_pop) == 8

    def test_replicate_preserves_survivors(self, evolution_engine, mock_variants):
        survivors = mock_variants[:4]
        new_pop = evolution_engine.replicate(survivors, target_size=8)
        # First 4 should be the original survivors
        for i, s in enumerate(survivors):
            assert new_pop[i] is s

    def test_replicate_children_have_new_ids(self, evolution_engine, mock_variants):
        survivors = mock_variants[:4]
        new_pop = evolution_engine.replicate(survivors, target_size=8)
        for i in range(4, 8):
            assert new_pop[i].clone.called


class TestEvolve:
    """Test the full evolve cycle."""

    def test_evolve_increments_generation(self, evolution_engine, trace_store_with_data, mock_variants):
        cluster_seq = np.array([0]*100)
        evolution_engine.evolve(
            variants=mock_variants,
            trace_store=trace_store_with_data,
            cluster_sequence=cluster_seq,
            n_eval_fragments=3,
        )
        assert evolution_engine.generation == 1

    def test_evolve_logs_entry(self, evolution_engine, trace_store_with_data, mock_variants):
        cluster_seq = np.array([0]*100)
        _, log_entry = evolution_engine.evolve(
            variants=mock_variants,
            trace_store=trace_store_with_data,
            cluster_sequence=cluster_seq,
            n_eval_fragments=3,
        )
        assert "generation" in log_entry
        assert "mean_fitness" in log_entry
        assert "max_fitness" in log_entry
        assert len(evolution_engine.log) == 1

    def test_evolve_returns_population(self, evolution_engine, trace_store_with_data, mock_variants):
        cluster_seq = np.array([0]*100)
        new_pop, _ = evolution_engine.evolve(
            variants=mock_variants,
            trace_store=trace_store_with_data,
            cluster_sequence=cluster_seq,
            n_eval_fragments=3,
        )
        assert len(new_pop) == len(mock_variants)

    def test_evolve_handles_no_fragments(self, evolution_engine, mock_variants):
        ts = TraceStore(max_size=100)
        ts.append(0, 0, 0)
        cluster_seq = np.array([0])
        new_pop, log_entry = evolution_engine.evolve(
            variants=mock_variants,
            trace_store=ts,
            cluster_sequence=cluster_seq,
            n_eval_fragments=3,
        )
        assert log_entry.get("error") == "no_fragments"
        assert new_pop is mock_variants  # Returned as-is


class TestGetEvolutionLog:
    """Test the get_evolution_log method."""

    def test_empty_log_returns_empty_dataframe(self, evolution_engine):
        df = evolution_engine.get_evolution_log()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_log_after_evolution(self, evolution_engine, trace_store_with_data, mock_variants):
        cluster_seq = np.array([0]*100)
        evolution_engine.evolve(mock_variants, trace_store_with_data, cluster_seq, n_eval_fragments=3)
        df = evolution_engine.get_evolution_log()
        assert len(df) == 1
        assert "mean_fitness" in df.columns