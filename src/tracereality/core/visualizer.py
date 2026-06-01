"""Publication-quality visualization for TraceReality experiments.

Generates all figures with confidence bands, proper labels,
and saves metadata for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .observer import NeuralObserver
from ..config import ExperimentConfig


# Global matplotlib style for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "figure.figsize": (8, 6),
})


class Visualizer:
    """Generates all publication-quality plots for an experiment."""

    def __init__(self, config: ExperimentConfig, output_dir: Path, prefix: str):
        """Initialize the visualizer.

        Args:
            config: Experiment configuration.
            output_dir: Directory to save plots.
            prefix: Experiment prefix for filenames.
        """
        self.config = config
        self.output_dir = output_dir
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Consistent color palette
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def save_plot(
        self,
        fig: plt.Figure,
        name: str,
        caption: str = "",
    ) -> None:
        """Save a figure with metadata.

        Args:
            fig: Matplotlib figure.
            name: Plot filename (without extension).
            caption: Description/caption for the plot.
        """
        try:
            fig.tight_layout()
        except Exception:
            pass  # Some complex layouts don't support tight_layout
        fig.savefig(self.output_dir / f"{self.prefix}_{name}.png")
        fig.savefig(self.output_dir / f"{self.prefix}_{name}.pdf", format="pdf")

        # Save metadata
        meta = {
            "name": name,
            "caption": caption,
            "config": {
                "n_states": self.config.n_states,
                "n_clusters": self.config.n_clusters,
                "n_observers": self.config.n_observers,
            },
        }
        with open(self.output_dir / f"{self.prefix}_{name}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        plt.close(fig)

    def plot_accuracy_vs_time(
        self,
        population_metrics: pd.DataFrame,
    ) -> plt.Figure:
        """Plot prediction accuracy over time with confidence bands.

        Args:
            population_metrics: DataFrame with population metrics over time.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots()

        steps = population_metrics["step"].values
        mean_acc = population_metrics["mean_accuracy"].values
        std_acc = population_metrics["std_accuracy"].values

        ax.plot(steps, mean_acc, color=self.colors[0], label="Mean Accuracy")
        ax.fill_between(
            steps,
            mean_acc - std_acc,
            mean_acc + std_acc,
            alpha=0.2,
            color=self.colors[0],
            label="±1 SD",
        )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Prediction Accuracy")
        ax.set_title("Observer Prediction Accuracy Over Time")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1)

        self.save_plot(
            fig,
            "accuracy_vs_time",
            "Mean prediction accuracy over time, with ±1 standard deviation "
            "across the observer population."
        )
        return fig

    def plot_divergence_vs_time(
        self,
        population_metrics: pd.DataFrame,
    ) -> plt.Figure:
        """Plot KL divergence over time with confidence bands.

        Args:
            population_metrics: DataFrame with population metrics.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots()

        steps = population_metrics["step"].values
        mean_kl = population_metrics["mean_kl_world"].values
        std_kl = population_metrics["std_kl_world"].values

        ax.plot(steps, mean_kl, color=self.colors[1], label="Mean KL(World || Observer)")
        ax.fill_between(
            steps,
            mean_kl - std_kl,
            mean_kl + std_kl,
            alpha=0.2,
            color=self.colors[1],
            label="±1 SD",
        )

        # Also plot pairwise KL if available
        if "mean_pairwise_kl" in population_metrics.columns:
            pairwise_kl = population_metrics["mean_pairwise_kl"].values
            ax.plot(
                steps, pairwise_kl, color=self.colors[2],
                linestyle="--", label="Mean Pairwise KL",
            )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("KL Divergence")
        ax.set_title("Observer Divergence from World and Each Other")
        ax.legend(loc="upper right")

        self.save_plot(
            fig,
            "divergence_vs_time",
            "KL divergence between observer models and the true world transition "
            "matrix, plus pairwise divergence between observers."
        )
        return fig

    def plot_world_heatmap(
        self,
        true_P: np.ndarray,
        cluster_ids: np.ndarray | None = None,
    ) -> plt.Figure:
        """Plot heatmap of the true world transition matrix.

        Args:
            true_P: True transition matrix.
            cluster_ids: Cluster assignments for coloring.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots()

        # Rearrange by cluster if available
        if cluster_ids is not None:
            sort_idx = np.argsort(cluster_ids)
            P_sorted = true_P[sort_idx][:, sort_idx]
            # Add cluster boundaries
            unique_clusters = np.unique(cluster_ids)
            boundaries = []
            for c in unique_clusters:
                indices = np.where(cluster_ids[sort_idx] == c)[0]
                if len(indices) > 0:
                    boundaries.append(indices[0])
            boundaries.append(len(cluster_ids))
        else:
            P_sorted = true_P
            boundaries = None

        im = ax.imshow(P_sorted, cmap="viridis", aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Transition Probability")

        if boundaries:
            for b in boundaries:
                ax.axhline(b - 0.5, color="white", linewidth=1, linestyle="--")
                ax.axvline(b - 0.5, color="white", linewidth=1, linestyle="--")

        ax.set_xlabel("Next State")
        ax.set_ylabel("Current State")
        ax.set_title("World Transition Matrix")

        self.save_plot(
            fig,
            "world_heatmap",
            "Heatmap of the true world transition matrix. States are sorted "
            "by cluster membership. Dashed lines indicate cluster boundaries."
        )
        return fig

    def plot_observer_heatmaps(
        self,
        observers: List[NeuralObserver],
        true_P: np.ndarray,
        max_plots: int = 6,
    ) -> plt.Figure:
        """Plot sample observer transition matrices.

        Args:
            observers: List of observers.
            true_P: True transition matrix for comparison.
            max_plots: Maximum number of observer heatmaps to show.

        Returns:
            Matplotlib figure.
        """
        n_obs = min(len(observers), max_plots)
        fig, axes = plt.subplots(2, n_obs, figsize=(4 * n_obs, 8))

        if n_obs == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_obs):
            obs_P = observers[i].get_transition_matrix_estimate()

            # True matrix (row 0)
            im0 = axes[0, i].imshow(true_P, cmap="viridis", aspect="auto", interpolation="nearest")
            axes[0, i].set_title(f"World (Reference)")
            axes[0, i].set_xlabel("Next State")
            axes[0, i].set_ylabel("Current State")

            # Observer matrix (row 1)
            im1 = axes[1, i].imshow(obs_P, cmap="viridis", aspect="auto", interpolation="nearest")
            axes[1, i].set_title(f"Observer {observers[i].id}")
            axes[1, i].set_xlabel("Next State")
            axes[1, i].set_ylabel("Current State")

            if i == 0:
                axes[0, i].set_ylabel("True")
                axes[1, i].set_ylabel("Observer")

        # Colorbar for each row using its own axes reference
        fig.colorbar(im0, ax=axes[0, :], label="Probability", shrink=0.8, pad=0.02)
        fig.colorbar(im1, ax=axes[1, :], label="Probability", shrink=0.8, pad=0.02)

        fig.suptitle("Observer vs World Transition Matrices", fontsize=14, y=1.02)
        fig.tight_layout()

        self.save_plot(
            fig,
            "observer_heatmaps",
            "Comparison of true world transition matrix with learned "
            "observer transition matrices."
        )
        return fig

    def plot_cluster_recovery(
        self,
        observers: List[NeuralObserver],
        true_cluster_ids: np.ndarray,
    ) -> plt.Figure:
        """Plot t-SNE of observer transition matrices colored by cluster.

        Args:
            observers: List of observers.
            true_cluster_ids: True cluster assignments for states.

        Returns:
            Matplotlib figure.
        """
        n_clusters = len(np.unique(true_cluster_ids))
        n_states = true_cluster_ids.shape[0]

        # Collect all observer transition matrices as feature vectors
        all_features = []
        for obs in observers:
            P = obs.get_transition_matrix_estimate()
            all_features.append(P.ravel())
        all_features = np.array(all_features)

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(observers) - 1))
        embedded = tsne.fit_transform(all_features)

        # Color by observer performance
        accuracies = [obs.average_accuracy for obs in observers]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: t-SNE colored by accuracy
        sc1 = axes[0].scatter(
            embedded[:, 0], embedded[:, 1],
            c=accuracies, cmap="viridis", s=80, alpha=0.8,
            edgecolors="black", linewidth=0.5,
        )
        axes[0].set_xlabel("t-SNE Dimension 1")
        axes[0].set_ylabel("t-SNE Dimension 2")
        axes[0].set_title("Observer Models (colored by accuracy)")
        plt.colorbar(sc1, ax=axes[0], label="Prediction Accuracy")

        # Plot 2: PCA of transition matrices
        pca = PCA(n_components=2)
        pca_embedded = pca.fit_transform(all_features)

        sc2 = axes[1].scatter(
            pca_embedded[:, 0], pca_embedded[:, 1],
            c=accuracies, cmap="viridis", s=80, alpha=0.8,
            edgecolors="black", linewidth=0.5,
        )
        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        axes[1].set_title("Observer Models (PCA projection)")
        plt.colorbar(sc2, ax=axes[1], label="Prediction Accuracy")

        self.save_plot(
            fig,
            "cluster_recovery",
            "Dimensionality reduction of observer transition matrices. "
            "t-SNE and PCA projections colored by prediction accuracy."
        )
        return fig

    def plot_fitness_landscape(
        self,
        observers: List[NeuralObserver],
    ) -> plt.Figure:
        """Plot fitness landscape: accuracy vs mutation rate.

        Args:
            observers: List of observers.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots()

        mutation_rates = [obs.mutation_rate for obs in observers]
        accuracies = [obs.average_accuracy for obs in observers]
        n_params = [sum(p.numel() for p in obs.model.parameters()) for obs in observers]

        sc = ax.scatter(
            mutation_rates, accuracies,
            c=n_params, cmap="plasma", s=80, alpha=0.8,
            edgecolors="black", linewidth=0.5,
        )
        ax.set_xlabel("Mutation Rate")
        ax.set_ylabel("Prediction Accuracy")
        ax.set_title("Fitness Landscape: Accuracy vs Mutation Rate")
        plt.colorbar(sc, ax=ax, label="Number of Parameters")

        # Add trend line
        if len(mutation_rates) > 1:
            z = np.polyfit(mutation_rates, accuracies, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(mutation_rates), max(mutation_rates), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, label="Trend")

        ax.legend()

        self.save_plot(
            fig,
            "fitness_landscape",
            "Fitness landscape showing relationship between mutation rate, "
            "prediction accuracy, and model complexity."
        )
        return fig

    def plot_evolution_fitness(
        self,
        evolution_log: pd.DataFrame,
    ) -> plt.Figure:
        """Plot fitness over evolutionary generations.

        Args:
            evolution_log: DataFrame with evolution history.

        Returns:
            Matplotlib figure.
        """
        if len(evolution_log) == 0:
            return None

        fig, ax = plt.subplots()

        generations = evolution_log["generation"].values
        mean_fitness = evolution_log["mean_fitness"].values
        std_fitness = evolution_log["std_fitness"].values
        max_fitness = evolution_log["max_fitness"].values

        ax.plot(generations, mean_fitness, color=self.colors[0], label="Mean Fitness")
        ax.fill_between(
            generations,
            mean_fitness - std_fitness,
            mean_fitness + std_fitness,
            alpha=0.2,
            color=self.colors[0],
            label="±1 SD",
        )
        ax.plot(generations, max_fitness, color=self.colors[3], linestyle="--", label="Max Fitness")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Evolutionary Fitness Over Generations")
        ax.legend(loc="lower right")

        self.save_plot(
            fig,
            "evolution_fitness",
            "Mean and maximum fitness over evolutionary generations, "
            "with ±1 standard deviation."
        )
        return fig

    def plot_all(
        self,
        population_metrics: pd.DataFrame,
        evolution_log: pd.DataFrame,
        observers: List[NeuralObserver],
        true_P: np.ndarray,
        cluster_ids: np.ndarray | None = None,
    ) -> None:
        """Generate all standard plots for an experiment.

        Args:
            population_metrics: DataFrame with population metrics.
            evolution_log: DataFrame with evolution history.
            observers: List of observers.
            true_P: True transition matrix.
            cluster_ids: Cluster assignments.
        """
        self.plot_accuracy_vs_time(population_metrics)
        self.plot_divergence_vs_time(population_metrics)
        self.plot_world_heatmap(true_P, cluster_ids)
        self.plot_observer_heatmaps(observers, true_P)
        self.plot_cluster_recovery(observers, cluster_ids)
        self.plot_fitness_landscape(observers)
        self.plot_evolution_fitness(evolution_log)