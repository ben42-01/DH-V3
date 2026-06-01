"""Fitness computation for LLM observer outputs.

Computes how well LLM responses align with hidden trace structure
using text embeddings, clustering, and comparison with true clusters.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans


class FitnessComputer:
    """Computes fitness of LLM observer outputs.

    Fitness is based on how well the LLM's output patterns align with
    the hidden cluster structure in the trace data.

    The pipeline:
    1. Embed LLM outputs using sentence-transformers
    2. Cluster the embeddings
    3. Compare cluster assignments to true trace clusters
    4. Return alignment score (ARI or NMI)
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True,
    ):
        """Initialize the fitness computer.

        Args:
            embedding_model_name: Name of the sentence-transformers model.
            use_embeddings: If False, use simple heuristic (output length, etc.)
                instead of embeddings. Useful for testing without a GPU.
        """
        self.embedding_model_name = embedding_model_name
        self.use_embeddings = use_embeddings
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                print(f"  Warning: Could not load embedding model: {e}")
                print(f"  Falling back to heuristic fitness computation")
                self.use_embeddings = False
        return self._embedder

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            Array of embeddings of shape (n_texts, embedding_dim).
        """
        embedder = self._get_embedder()
        if embedder is None:
            return self._heuristic_features(texts)
        embeddings = embedder.encode(texts, show_progress_bar=False)
        return np.array(embeddings)

    def _heuristic_features(self, texts: List[str]) -> np.ndarray:
        """Compute simple heuristic features from texts.

        Used when sentence-transformers is not available.

        Args:
            texts: List of text strings.

        Returns:
            Feature array of shape (n_texts, 5).
        """
        features = []
        for text in texts:
            words = text.split()
            sentences = text.split(".")
            features.append([
                len(words),  # Length in words
                len(text),   # Length in chars
                len(set(words)) / max(len(words), 1),  # Vocabulary diversity
                len(sentences),  # Number of sentences
                sum(1 for c in text if c in "!?.") / max(len(sentences), 1),  # Punctuation rate
            ])
        return np.array(features)

    def _get_fragment_clusters(
        self,
        fragments: List[List[int]],
        cluster_sequence: np.ndarray,
    ) -> np.ndarray:
        """Get cluster labels for each fragment.

        For each fragment, returns the majority cluster ID among its states.

        Args:
            fragments: List of state index sequences.
            cluster_sequence: Cluster ID for each step in the trace.

        Returns:
            Array of cluster labels, one per fragment.
        """
        labels = []
        for fragment in fragments:
            if len(fragment) == 0:
                labels.append(0)
            else:
                # Get cluster for each state in the fragment
                clusters = [cluster_sequence[s] for s in fragment if s < len(cluster_sequence)]
                if len(clusters) == 0:
                    labels.append(0)
                else:
                    # Majority vote
                    labels.append(int(np.bincount(clusters).argmax()))
        return np.array(labels)

    def compute_fitness(
        self,
        responses: List[str],
        fragments: List[List[int]],
        cluster_sequence: np.ndarray,
    ) -> float:
        """Compute fitness based on alignment between output clusters and true clusters.

        Args:
            responses: LLM responses for each fragment.
            fragments: The trace fragments that generated each response.
            cluster_sequence: True cluster IDs for each step.

        Returns:
            Fitness score (0 to 1, higher = better alignment).
        """
        if len(responses) < 2:
            return 0.0

        # Get true cluster labels for each fragment
        true_labels = self._get_fragment_clusters(fragments, cluster_sequence)
        n_clusters = len(np.unique(true_labels))

        if n_clusters < 2:
            return 0.0

        # Embed responses as feature vectors
        features = self._embed_texts(responses)

        if features.shape[0] < 2:
            return 0.0

        # Cluster the features into the same number of clusters as true clusters
        n_clusters = min(n_clusters, features.shape[0] - 1)
        if n_clusters < 2:
            return 0.0

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(features)

        # Compare to true cluster labels
        ari = adjusted_rand_score(true_labels, pred_labels)

        # Also compute NMI
        nmi = normalized_mutual_info_score(true_labels, pred_labels)

        # Combined fitness: mean of ARI and NMI, clamped to [0, 1]
        fitness = max(0.0, (ari + nmi) / 2.0)
        return float(fitness)

    def compute_stability(self, responses: List[str]) -> float:
        """Compute output stability (lower = more stable).

        Measures how similar responses are to each other.
        High similarity = stable attractor state.

        Args:
            responses: List of responses.

        Returns:
            Stability score (0 to 1, higher = more stable).
        """
        if len(responses) < 2:
            return 0.0

        features = self._embed_texts(responses)
        if features.shape[0] < 2:
            return 0.0

        # Compute pairwise cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = features / norms
        similarities = normalized @ normalized.T

        # Upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(len(similarities), k=1)
        if len(triu_indices[0]) == 0:
            return 0.0

        mean_sim = similarities[triu_indices].mean()
        return float(mean_sim)