"""Analysis and metrics computation for LLM observer experiments.

Computes narrative cluster emergence, alignment with world clusters,
fitness over generations, and divergence between LLM lineages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

from .llm_observer import LLMObserver
from .fitness import FitnessComputer


class LLMAnalyzer:
    """Computes all metrics for LLM observer experiments."""

    def __init__(self, fitness_computer: FitnessComputer):
        """Initialize the analyzer.

        Args:
            fitness_computer: Fitness computer for embedding/comparison.
        """
        self.fitness = fitness_computer
        self.generation_metrics: List[dict] = []

    def analyze_generation(
        self,
        variants: List[LLMObserver],
        generation: int,
    ) -> dict:
        """Compute metrics for a generation.

        Args:
            variants: Current population of variants.
            generation: Generation number.

        Returns:
            Dictionary of metrics.
        """
        # Fitness statistics
        fitnesses = [v.current_fitness for v in variants]
        avg_fitnesses = [v.avg_fitness for v in variants]

        # Collect recent outputs for analysis
        all_outputs = []
        for v in variants:
            all_outputs.extend(v.output_history)

        # Output statistics
        output_lengths = [len(o.split()) for o in all_outputs if o]
        unique_words_per_output = [
            len(set(o.split())) / max(len(o.split()), 1)
            for o in all_outputs if o
        ]

        metrics = {
            "generation": generation,
            "n_variants": len(variants),
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
            "mean_avg_fitness": float(np.mean(avg_fitnesses)),
            "n_total_outputs": len(all_outputs),
            "mean_output_length": float(np.mean(output_lengths)) if output_lengths else 0.0,
            "mean_vocab_diversity": float(np.mean(unique_words_per_output)) if unique_words_per_output else 0.0,
        }

        # Prompt diversity (edit distance between system messages)
        if len(variants) > 1:
            prompt_similarities = self._compute_prompt_similarity(variants)
            metrics["mean_prompt_similarity"] = prompt_similarities["mean"]
            metrics["std_prompt_similarity"] = prompt_similarities["std"]

        self.generation_metrics.append(metrics)
        return metrics

    def _compute_prompt_similarity(
        self,
        variants: List[LLMObserver],
    ) -> Dict[str, float]:
        """Compute pairwise similarity between variant prompts.

        Uses Jaccard similarity on word sets.

        Args:
            variants: List of variants.

        Returns:
            Dict with mean and std of pairwise similarities.
        """
        similarities = []
        n = len(variants)
        for i in range(n):
            words_i = set(variants[i].system_message.lower().split())
            for j in range(i + 1, n):
                words_j = set(variants[j].system_message.lower().split())
                if len(words_i | words_j) == 0:
                    sim = 0.0
                else:
                    sim = len(words_i & words_j) / len(words_i | words_j)
                similarities.append(sim)

        if not similarities:
            return {"mean": 0.0, "std": 0.0}

        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
        }

    def analyze_narrative_clusters(
        self,
        variants: List[LLMObserver],
        true_cluster_sequence: np.ndarray,
    ) -> dict:
        """Analyze narrative clusters formed by LLM outputs.

        Args:
            variants: List of variants.
            true_cluster_sequence: True cluster IDs for each step.

        Returns:
            Dict with narrative cluster metrics.
        """
        # Collect all outputs with their variant IDs
        all_outputs = []
        for v in variants:
            for output in v.output_history:
                all_outputs.append((v.id, output))

        if len(all_outputs) < 3:
            return {"error": "not_enough_outputs", "n_outputs": len(all_outputs)}

        outputs_only = [o for _, o in all_outputs]

        # Embed and cluster outputs
        features = self.fitness._embed_texts(outputs_only)
        if features.shape[0] < 3:
            return {"error": "embedding_failed", "n_outputs": len(outputs_only)}

        # Use silhouette score to find optimal number of clusters
        from sklearn.metrics import silhouette_score

        best_n_clusters = 2
        best_score = -1
        max_clusters = min(10, features.shape[0] - 1)

        for n in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            if len(set(labels)) > 1:
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n

        # Final clustering with optimal n
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # Count outputs per cluster
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        # Per-cluster sample outputs
        cluster_samples = {}
        for c in unique:
            indices = np.where(cluster_labels == c)[0]
            sample_idx = indices[0]
            cluster_samples[int(c)] = {
                "size": int(counts[np.where(unique == c)[0][0]]),
                "sample_output": outputs_only[sample_idx][:200],
                "variant_id": all_outputs[sample_idx][0],
            }

        return {
            "n_clusters_found": best_n_clusters,
            "silhouette_score": float(best_score),
            "cluster_sizes": cluster_sizes,
            "cluster_samples": cluster_samples,
            "n_total_outputs": len(outputs_only),
        }

    def compute_alignment(
        self,
        variants: List[LLMObserver],
        trace_fragments: List[List[int]],
        cluster_sequence: np.ndarray,
    ) -> dict:
        """Compute alignment between LLM outputs and true clusters.

        Args:
            variants: List of variants.
            trace_fragments: Trace fragments used for evaluation.
            cluster_sequence: True cluster IDs.

        Returns:
            Dict with alignment metrics.
        """
        all_responses = []
        all_fragments = []

        for variant in variants:
            for fragment in trace_fragments[:5]:  # Subset for speed
                response = variant.respond_to_trace_list(fragment)
                all_responses.append(response)
                all_fragments.append(fragment)

        if len(all_responses) < 2:
            return {"error": "not_enough_responses"}

        # Get true labels
        true_labels = self.fitness._get_fragment_clusters(all_fragments, cluster_sequence)

        # Embed and cluster responses
        features = self.fitness._embed_texts(all_responses)
        n_clusters = len(np.unique(true_labels))

        if n_clusters < 2 or features.shape[0] < 2:
            return {"error": "insufficient_data"}

        n_clusters = min(n_clusters, features.shape[0] - 1)
        if n_clusters < 2:
            return {"error": "need_more_clusters"}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(features)

        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)

        return {
            "adjusted_rand_index": float(ari),
            "normalized_mutual_info": float(nmi),
            "n_true_clusters": n_clusters,
            "n_responses": len(all_responses),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export generation metrics as DataFrame."""
        return pd.DataFrame(self.generation_metrics)

    def save_metrics(
        self,
        output_dir: Path,
        prefix: str,
        variants: List[LLMObserver],
        evolution_log: pd.DataFrame,
        trace_fragments: List[List[int]],
        cluster_sequence: np.ndarray,
    ) -> None:
        """Save all metrics to CSV/JSON files.

        Args:
            output_dir: Output directory.
            prefix: Experiment prefix.
            variants: List of variants.
            evolution_log: Evolution log DataFrame.
            trace_fragments: Trace fragments for alignment analysis.
            cluster_sequence: True cluster IDs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generation metrics
        df_gen = self.to_dataframe()
        if len(df_gen) > 0:
            df_gen.to_csv(output_dir / f"{prefix}_generation_metrics.csv", index=False)

        # Evolution log
        if len(evolution_log) > 0:
            evolution_log.to_csv(output_dir / f"{prefix}_evolution_log.csv", index=False)

        # Narrative cluster analysis
        narrative = self.analyze_narrative_clusters(variants, cluster_sequence)
        with open(output_dir / f"{prefix}_narrative_clusters.json", "w") as f:
            # Convert numpy types for JSON serialization
            json.dump(narrative, f, indent=2, default=str)

        # Alignment analysis
        alignment = self.compute_alignment(variants, trace_fragments, cluster_sequence)
        with open(output_dir / f"{prefix}_alignment_metrics.json", "w") as f:
            json.dump(alignment, f, indent=2, default=str)

        # Variant states
        variant_states = [v.to_dict() for v in variants]
        with open(output_dir / f"{prefix}_variant_states.json", "w") as f:
            json.dump(variant_states, f, indent=2, default=str)

        # Sample outputs grouped by variant
        sample_outputs = {}
        for v in variants:
            sample_outputs[v.id] = {
                "system_message": v.system_message,
                "prompt_template": v.prompt_template,
                "fitness_history": v.fitness_history,
                "sample_outputs": v.output_history[-10:] if v.output_history else [],
            }
        with open(output_dir / f"{prefix}_sample_outputs.json", "w") as f:
            json.dump(sample_outputs, f, indent=2, default=str)