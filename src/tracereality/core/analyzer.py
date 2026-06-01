"""Research-grade metrics computation and statistical analysis.

Computes per-observer metrics, population-level metrics,
cluster recovery, information-theoretic measures, and
statistical tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

from .observer import NeuralObserver
from ..config import ExperimentConfig
from ..state_space import get_cluster_centroids


class Analyzer:
    """Computes all research-grade metrics for an experiment run."""

    def __init__(self, config: ExperimentConfig):
        """Initialize the analyzer.

        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.observer_metrics: List[dict] = []
        self.population_metrics: List[dict] = []
        self.cluster_recovery_metrics: List[dict] = []
        self.info_theoretic_metrics: List[dict] = []

    def _get_true_transition_matrix(self, true_P: np.ndarray) -> np.ndarray:
        """Extract the 2D transition matrix from true_P (handles 2D and 3D).

        For MDP (3D) matrices, uses the 'stay' action (index 0) for KL comparison.

        Args:
            true_P: Either 2D (n_states, n_states) or 3D (n_states, n_actions, n_states).

        Returns:
            2D transition matrix of shape (n_states, n_states).
        """
        if true_P.ndim == 3:
            return true_P[:, 0, :]  # Use "stay" action
        return true_P

    def compute_observer_metrics(
        self,
        observer: NeuralObserver,
        true_P: np.ndarray,
        cluster_ids: np.ndarray | None = None,
    ) -> dict:
        """Compute all metrics for a single observer.

        Args:
            observer: The observer to evaluate.
            true_P: True world transition matrix (2D or 3D).
            cluster_ids: Cluster assignments for states (optional).

        Returns:
            Dictionary of metrics.
        """
        true_P_2d = self._get_true_transition_matrix(true_P)
        obs_P = observer.get_transition_matrix_estimate()
        n_states = obs_P.shape[0]

        # KL divergence: observer vs world (row-wise, mean)
        kl_divs = []
        for i in range(n_states):
            p = true_P_2d[i]
            q = obs_P[i]
            # Avoid log(0) by adding small epsilon
            eps = 1e-10
            p_safe = np.clip(p, eps, 1.0)
            q_safe = np.clip(q, eps, 1.0)
            kl = np.sum(p_safe * np.log(p_safe / q_safe))
            kl_divs.append(kl)
        kl_world = float(np.mean(kl_divs))

        # Accuracy metrics
        avg_acc = observer.average_accuracy
        recent_acc = float(np.mean(observer.accuracy_history[-1000:])) if len(observer.accuracy_history) >= 1000 else avg_acc

        # Stability
        param_variance = observer.get_param_variance(n_recent=10)

        # Fitness
        current_fitness = observer.current_fitness
        avg_fitness = float(np.mean(observer.fitness_history)) if observer.fitness_history else 0.0

        # Model complexity (number of parameters)
        n_params = sum(p.numel() for p in observer.model.parameters())

        # Effective sample size (total training steps)
        n_train_steps = observer.step_count

        metrics = {
            "observer_id": observer.id,
            "avg_accuracy": avg_acc,
            "recent_accuracy": recent_acc,
            "kl_divergence_world": kl_world,
            "param_variance": param_variance,
            "current_fitness": current_fitness,
            "avg_fitness": avg_fitness,
            "n_params": n_params,
            "n_train_steps": n_train_steps,
            "mutation_rate": observer.mutation_rate,
            "n_fitness_records": len(observer.fitness_history),
            "n_param_snapshots": len(observer.param_snapshots),
        }

        # Add cluster recovery metrics if cluster_ids available
        if cluster_ids is not None:
            recovery = self._compute_cluster_recovery(obs_P, cluster_ids, true_P)
            metrics.update(recovery)

        return metrics

    def _compute_cluster_recovery(
        self,
        obs_P: np.ndarray,
        cluster_ids: np.ndarray,
        true_P: np.ndarray,
    ) -> dict:
        """Compute how well observer transitions recover cluster structure.

        Args:
            obs_P: Observer's estimated transition matrix.
            cluster_ids: True cluster assignments.
            true_P: True transition matrix (2D or 3D).

        Returns:
            Dictionary of cluster recovery metrics.
        """
        true_P_2d = self._get_true_transition_matrix(true_P)
        n_clusters = len(np.unique(cluster_ids))
        n_states = obs_P.shape[0]

        # Cluster the observer's transition matrix rows using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        obs_clusters = kmeans.fit_predict(obs_P)

        # Compare to true cluster assignments
        ari = adjusted_rand_score(cluster_ids, obs_clusters)
        nmi = normalized_mutual_info_score(cluster_ids, obs_clusters)

        # Compute cluster centroid alignment
        true_centroids = get_cluster_centroids(true_P_2d, cluster_ids)
        obs_centroids = get_cluster_centroids(obs_P, obs_clusters)

        # Find best match between true and observed clusters (Hungarian-like)
        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                cost_matrix[i, j] = np.linalg.norm(true_centroids[i] - obs_centroids[j])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        centroid_alignment_error = float(cost_matrix[row_ind, col_ind].mean())

        return {
            "cluster_ari": ari,
            "cluster_nmi": nmi,
            "centroid_alignment_error": centroid_alignment_error,
        }

    def compute_population_metrics(
        self,
        observers: List[NeuralObserver],
        true_P: np.ndarray,
        step: int,
    ) -> dict:
        """Compute population-level metrics across all observers.

        Args:
            observers: List of all observers.
            true_P: True world transition matrix.
            step: Current simulation step.

        Returns:
            Dictionary of population metrics.
        """
        # Gather per-observer metrics
        all_metrics = [
            self.compute_observer_metrics(obs, true_P) for obs in observers
        ]
        df = pd.DataFrame(all_metrics)

        # Population-level summaries
        metrics = {
            "step": step,
            "n_observers": len(observers),
            "mean_accuracy": df["avg_accuracy"].mean(),
            "std_accuracy": df["avg_accuracy"].std(),
            "max_accuracy": df["avg_accuracy"].max(),
            "min_accuracy": df["avg_accuracy"].min(),
            "mean_kl_world": df["kl_divergence_world"].mean(),
            "std_kl_world": df["kl_divergence_world"].std(),
            "mean_fitness": df["current_fitness"].mean(),
            "std_fitness": df["current_fitness"].std(),
            "mean_param_variance": df["param_variance"].mean(),
            "std_param_variance": df["param_variance"].std(),
            "mean_mutation_rate": df["mutation_rate"].mean(),
        }

        # Pairwise KL divergence between observers
        pairwise_kls = self._compute_pairwise_kl(observers)
        metrics["mean_pairwise_kl"] = pairwise_kls["mean"]
        metrics["std_pairwise_kl"] = pairwise_kls["std"]

        # If cluster metrics available
        if "cluster_ari" in df.columns:
            metrics["mean_cluster_ari"] = df["cluster_ari"].mean()
            metrics["std_cluster_ari"] = df["cluster_ari"].std()

        self.population_metrics.append(metrics)
        return metrics

    def _compute_pairwise_kl(
        self,
        observers: List[NeuralObserver],
    ) -> dict:
        """Compute pairwise KL divergence between all observer pairs.

        Args:
            observers: List of observers.

        Returns:
            Dictionary with mean and std of pairwise KL.
        """
        kls = []
        n = len(observers)
        for i in range(n):
            Pi = observers[i].get_transition_matrix_estimate()
            for j in range(i + 1, n):
                Pj = observers[j].get_transition_matrix_estimate()
                # Symmetric KL: (KL(Pi||Pj) + KL(Pj||Pi)) / 2
                eps = 1e-10
                Pi_safe = np.clip(Pi, eps, 1.0)
                Pj_safe = np.clip(Pj, eps, 1.0)
                kl_ij = np.sum(Pi_safe * np.log(Pi_safe / Pj_safe), axis=1).mean()
                kl_ji = np.sum(Pj_safe * np.log(Pj_safe / Pi_safe), axis=1).mean()
                kls.append((kl_ij + kl_ji) / 2)

        if not kls:
            return {"mean": 0.0, "std": 0.0}

        return {
            "mean": float(np.mean(kls)),
            "std": float(np.std(kls)),
        }

    def compute_information_theoretic(
        self,
        observers: List[NeuralObserver],
        true_P: np.ndarray,
    ) -> dict:
        """Compute expected KL divergence between world and observer models.

        NOTE: This was previously (incorrectly) labeled as "mutual information".
        The computed quantity is actually the negative expected KL divergence:
            E_i[ KL( p_true(·|i) || p_pred(·|i) ) ]
        which is >= 0 by construction. A higher value means the observer's
        model diverges more from the true world (worse fit).

        Args:
            observers: List of observers.
            true_P: True world transition matrix (2D or 3D).

        Returns:
            Dictionary with expected KL divergence metrics.
        """
        true_P_2d = self._get_true_transition_matrix(true_P)
        n_states = true_P_2d.shape[0]

        # True stationary distribution
        from ..state_space import compute_stationary_distribution
        pi_true = compute_stationary_distribution(true_P_2d)

        # Compute expected KL divergence: E_i[ KL(p_true(·|i) || p_pred(·|i)) ]
        ekls = []
        compressions = []

        for obs in observers:
            obs_P = obs.get_transition_matrix_estimate()

            # Expected KL divergence (>= 0 by construction)
            ekl = 0.0
            for i in range(n_states):
                p_true_given_i = true_P_2d[i]
                p_pred_given_i = obs_P[i]
                eps = 1e-10
                # KL(p_true || p_pred) for row i
                p_safe = np.clip(p_true_given_i, eps, 1.0)
                q_safe = np.clip(p_pred_given_i, eps, 1.0)
                kl_i = np.sum(p_safe * np.log(p_safe / q_safe))
                ekl += pi_true[i] * kl_i
            ekls.append(ekl)

            # Compression: model complexity vs accuracy
            n_params = sum(p.numel() for p in obs.model.parameters())
            acc = obs.average_accuracy
            compressions.append({
                "n_params": n_params,
                "accuracy": acc,
                "expected_kl_divergence": ekl,
            })

        metrics = {
            "mean_expected_kl_divergence": float(np.mean(ekls)),
            "std_expected_kl_divergence": float(np.std(ekls)),
            "compression_tradeoff": compressions,
        }

        self.info_theoretic_metrics.append(metrics)
        return metrics

    def run_statistical_tests(
        self,
        condition_metrics: Dict[str, List[dict]],
    ) -> dict:
        """Run statistical tests comparing different conditions.

        Args:
            condition_metrics: Dict mapping condition names to
                lists of metric dicts (one per run).

        Returns:
            Dictionary of test results.
        """
        results = {}

        for metric_key in ["mean_accuracy", "mean_kl_world", "mean_pairwise_kl"]:
            groups = []
            group_names = []
            for name, runs in condition_metrics.items():
                values = [r[metric_key] for r in runs if metric_key in r]
                if len(values) > 0:
                    groups.append(values)
                    group_names.append(name)

            if len(groups) >= 2:
                # ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                results[f"{metric_key}_anova"] = {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                }

                # Pairwise t-tests
                pairwise = []
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                        pairwise.append({
                            "group1": group_names[i],
                            "group2": group_names[j],
                            "t_statistic": float(t_stat),
                            "p_value": float(p_val),
                            "significant": bool(p_val < 0.05),
                        })
                results[f"{metric_key}_pairwise_ttest"] = pairwise

        return results

    def compute_confidence_intervals(
        self,
        values: np.ndarray,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence intervals.

        Args:
            values: Array of values.
            confidence: Confidence level (default 0.95).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        n_bootstrap = self.config.n_bootstrap_samples
        boot_means = np.zeros(n_bootstrap)
        n = len(values)
        rng = np.random.default_rng(42)

        for i in range(n_bootstrap):
            sample = rng.choice(values, size=n, replace=True)
            boot_means[i] = np.mean(sample)

        alpha = 1 - confidence
        lower = np.percentile(boot_means, 100 * alpha / 2)
        upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

        return float(lower), float(upper)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all computed metrics as a DataFrame.

        Returns:
            DataFrame with all metrics.
        """
        rows = []
        for pm in self.population_metrics:
            row = dict(pm)
            rows.append(row)
        return pd.DataFrame(rows)

    def save_metrics(
        self,
        output_dir: Path,
        prefix: str,
        observers: List[NeuralObserver],
        true_P: np.ndarray,
        cluster_ids: np.ndarray | None = None,
    ) -> None:
        """Save all metrics to CSV files.

        Args:
            output_dir: Output directory.
            prefix: Experiment prefix.
            observers: List of observers.
            true_P: True transition matrix.
            cluster_ids: Cluster assignments.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Per-observer metrics (final)
        obs_metrics = [
            self.compute_observer_metrics(obs, true_P, cluster_ids)
            for obs in observers
        ]
        df_obs = pd.DataFrame(obs_metrics)
        df_obs.to_csv(output_dir / f"{prefix}_observer_metrics.csv", index=False)

        # Population metrics (over time)
        df_pop = self.to_dataframe()
        if len(df_pop) > 0:
            df_pop.to_csv(output_dir / f"{prefix}_population_metrics.csv", index=False)

        # Summary
        if len(df_pop) > 0:
            summary = {
                "final_mean_accuracy": df_pop["mean_accuracy"].iloc[-1],
                "final_std_accuracy": df_pop["std_accuracy"].iloc[-1],
                "final_mean_kl_world": df_pop["mean_kl_world"].iloc[-1],
                "final_std_kl_world": df_pop["std_kl_world"].iloc[-1],
                "final_mean_fitness": df_pop["mean_fitness"].iloc[-1],
                "best_accuracy": df_pop["max_accuracy"].max(),
                "best_kl": df_pop["mean_kl_world"].min(),
            }
            if "mean_cluster_ari" in df_pop.columns:
                summary["final_mean_cluster_ari"] = df_pop["mean_cluster_ari"].iloc[-1]

            df_summary = pd.DataFrame([summary])
            df_summary.to_csv(output_dir / f"{prefix}_metrics_summary.csv", index=False)