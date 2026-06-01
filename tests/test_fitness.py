"""Tests for the FitnessComputer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tracereality.llm.fitness import FitnessComputer


class TestFitnessComputerInit:
    """Test initialization."""

    def test_default_parameters(self):
        fc = FitnessComputer()
        assert fc.embedding_model_name == "all-MiniLM-L6-v2"
        assert fc.use_embeddings is True
        assert fc._embedder is None

    def test_no_embeddings_mode(self):
        fc = FitnessComputer(use_embeddings=False)
        assert fc.use_embeddings is False

    def test_custom_model(self):
        fc = FitnessComputer(embedding_model_name="custom-model")
        assert fc.embedding_model_name == "custom-model"


class TestFitnessComputerHeuristicFeatures:
    """Test the heuristic fallback features."""

    def test_heuristic_features_shape(self):
        fc = FitnessComputer(use_embeddings=False)
        texts = ["short text", "a longer text with more words here"]
        features = fc._heuristic_features(texts)
        assert features.shape == (2, 5)

    def test_heuristic_features_values(self):
        fc = FitnessComputer(use_embeddings=False)
        features = fc._heuristic_features(["hello world"])
        # [word_count, char_count, vocab_diversity, sentence_count, punct_rate]
        assert features[0, 0] == 2  # word count
        assert features[0, 1] == 11  # char count
        assert features[0, 2] == 1.0  # vocab diversity (2 unique / 2 total)
        assert features[0, 3] == 1  # sentence count
        assert features[0, 4] == 0.0  # punctuation rate

    def test_heuristic_features_empty_text(self):
        fc = FitnessComputer(use_embeddings=False)
        features = fc._heuristic_features([""])
        assert features[0, 0] == 0
        assert features[0, 2] == 0.0  # Avoids division by zero


class TestFitnessComputerGetFragmentClusters:
    """Test extracting cluster labels from fragments."""

    def test_single_cluster_fragment(self, sample_cluster_sequence):
        fc = FitnessComputer(use_embeddings=False)
        fragments = [[0, 1, 2, 3]]  # All cluster 0
        labels = fc._get_fragment_clusters(fragments, sample_cluster_sequence)
        assert labels.tolist() == [0]

    def test_multi_cluster_fragments(self, sample_cluster_sequence):
        fc = FitnessComputer(use_embeddings=False)
        fragments = [
            [0, 1, 2, 3],     # Should be cluster 0
            [4, 5, 6, 7],     # Should be cluster 1
            [8, 9, 10, 11],   # Should be cluster 2
        ]
        labels = fc._get_fragment_clusters(fragments, sample_cluster_sequence)
        assert labels.tolist() == [0, 1, 2]

    def test_empty_fragment(self, sample_cluster_sequence):
        fc = FitnessComputer(use_embeddings=False)
        labels = fc._get_fragment_clusters([[]], sample_cluster_sequence)
        assert labels.tolist() == [0]

    def test_fragment_out_of_bounds(self, sample_cluster_sequence):
        fc = FitnessComputer(use_embeddings=False)
        # fragment indices beyond the cluster_sequence length
        fragments = [[100, 200]]
        labels = fc._get_fragment_clusters(fragments, sample_cluster_sequence)
        assert labels.tolist() == [0]


class TestFitnessComputerComputeFitness:
    """Test the main fitness computation."""

    def test_zero_fitness_with_few_responses(self):
        fc = FitnessComputer(use_embeddings=False)
        fitness = fc.compute_fitness(
            responses=["only one"],
            fragments=[[0, 1]],
            cluster_sequence=np.array([0, 1]),
        )
        assert fitness == 0.0

    def test_zero_fitness_with_single_cluster(self, sample_cluster_sequence):
        fc = FitnessComputer(use_embeddings=False)
        # All fragments map to same cluster
        fragments = [[0, 1], [2, 3], [4, 5]]
        # Use a cluster sequence where all are the same cluster
        single_cluster = np.zeros_like(sample_cluster_sequence)
        fitness = fc.compute_fitness(
            responses=["response A", "response B", "response C"],
            fragments=fragments,
            cluster_sequence=single_cluster,
        )
        assert fitness == 0.0

    def test_heuristic_mode_returns_float(self, sample_trace_fragments, sample_cluster_sequence):
        """In heuristic mode, fitness computation should return a float."""
        fc = FitnessComputer(use_embeddings=False)
        responses = [
            "This sequence shows a clear pattern of alternating values.",
            "I notice these numbers are increasing steadily over time.",
            "The symbols here seem to repeat in a cyclic manner.",
            "This looks like random noise with no discernible structure.",
            "There is a definite upward trend in this data.",
        ]
        fitness = fc.compute_fitness(
            responses=responses,
            fragments=sample_trace_fragments[:5],
            cluster_sequence=sample_cluster_sequence,
        )
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0


class TestFitnessComputerStability:
    """Test stability computation."""

    def test_identical_responses_max_stability(self):
        fc = FitnessComputer(use_embeddings=False)
        # In heuristic mode, identical texts should give identical features
        stability = fc.compute_stability(["hello world", "hello world"])
        assert stability == pytest.approx(1.0)

    def test_single_response_zero_stability(self):
        fc = FitnessComputer(use_embeddings=False)
        stability = fc.compute_stability(["only one"])
        assert stability == 0.0

    def test_empty_list_zero_stability(self):
        fc = FitnessComputer(use_embeddings=False)
        stability = fc.compute_stability([])
        assert stability == 0.0

    def test_different_responses_lower_stability(self):
        fc = FitnessComputer(use_embeddings=False)
        stability = fc.compute_stability([
            "hello world",
            "quantum physics mechanics formalism",
            "completely unrelated topic discussion",
        ])
        assert 0.0 <= stability <= 1.0


class TestFitnessComputerEmbedder:
    """Test the lazy-loaded embedding model."""

    @patch("tracereality.llm.fitness.SentenceTransformer")
    def test_embedder_loaded_on_demand(self, mock_transformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_transformer.return_value = mock_model

        fc = FitnessComputer(use_embeddings=True)
        # First access triggers loading
        embedder = fc._get_embedder()
        assert embedder is not None
        mock_transformer.assert_called_once_with("all-MiniLM-L6-v2")

    @patch("tracereality.llm.fitness.SentenceTransformer")
    def test_embedder_fallback_on_error(self, mock_transformer):
        mock_transformer.side_effect = Exception("Model not found")

        fc = FitnessComputer(use_embeddings=True)
        embedder = fc._get_embedder()
        assert embedder is None
        assert fc.use_embeddings is False  # Should have fallen back