"""Publication-quality visualization for LLM observer experiments.

Generates plots for narrative evolution, alignment with true clusters,
fitness over generations, and sample outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from ..config import ExperimentConfig


# Style for publication-quality figures
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
})


class LLMVisualizer:
    """Generates all plots for LLM observer experiments."""

    def __init__(self, config: ExperimentConfig, output_dir: Path, prefix: str):
        """Initialize the visualizer.

        Args:
            config: Experiment configuration.
            output_dir: Directory to save plots.
            prefix: Experiment prefix.
        """
        self.config = config
        self.output_dir = output_dir
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def save_plot(self, fig: plt.Figure, name: str, caption: str = "") -> None:
        """Save a figure with metadata."""
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(self.output_dir / f"{self.prefix}_{name}.png")
        fig.savefig(self.output_dir / f"{self.prefix}_{name}.pdf", format="pdf")

        meta = {
            "name": name,
            "caption": caption,
            "config": {
                "n_states": self.config.n_states,
                "n_clusters": self.config.n_clusters,
                "n_variants": self.config.n_observers,  # Reuse field
            },
        }
        with open(self.output_dir / f"{self.prefix}_{name}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        plt.close(fig)

    def plot_fitness_over_generations(
        self,
        generation_metrics: pd.DataFrame,
    ) -> plt.Figure:
        """Plot fitness metrics over generations.

        Args:
            generation_metrics: DataFrame with per-generation metrics.

        Returns:
            Matplotlib figure.
        """
        if len(generation_metrics) == 0:
            return None

        fig, ax = plt.subplots()

        gens = generation_metrics["generation"].values
        mean_fit = generation_metrics["mean_fitness"].values
        std_fit = generation_metrics["std_fitness"].values
        max_fit = generation_metrics["max_fitness"].values

        ax.plot(gens, mean_fit, color=self.colors[0], label="Mean Fitness")
        ax.fill_between(
            gens,
            mean_fit - std_fit,
            mean_fit + std_fit,
            alpha=0.2,
            color=self.colors[0],
            label="±1 SD",
        )
        ax.plot(gens, max_fit, color=self.colors[3], linestyle="--", label="Max Fitness")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("LLM Observer Fitness Over Generations")
        ax.legend(loc="lower right")

        self.save_plot(
            fig,
            "fitness_over_generations",
            "Mean and maximum fitness of LLM observer variants over evolutionary generations."
        )
        return fig

    def plot_output_diversity(
        self,
        generation_metrics: pd.DataFrame,
    ) -> plt.Figure:
        """Plot output diversity metrics over generations.

        Args:
            generation_metrics: DataFrame with per-generation metrics.

        Returns:
            Matplotlib figure.
        """
        if len(generation_metrics) == 0:
            return None

        fig, ax1 = plt.subplots()
        gens = generation_metrics["generation"].values

        # Output length
        if "mean_output_length" in generation_metrics.columns:
            ax1.plot(
                gens, generation_metrics["mean_output_length"].values,
                color=self.colors[0], marker="o", label="Mean Output Length",
            )
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Mean Output Length (words)", color=self.colors[0])
            ax1.tick_params(axis="y", labelcolor=self.colors[0])

        # Vocabulary diversity on second axis
        if "mean_vocab_diversity" in generation_metrics.columns:
            ax2 = ax1.twinx()
            ax2.plot(
                gens, generation_metrics["mean_vocab_diversity"].values,
                color=self.colors[1], marker="s", label="Vocab Diversity",
            )
            ax2.set_ylabel("Vocabulary Diversity", color=self.colors[1])
            ax2.tick_params(axis="y", labelcolor=self.colors[1])

        ax1.set_title("LLM Output Diversity Over Generations")
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

        self.save_plot(
            fig,
            "output_diversity",
            "Mean output length and vocabulary diversity of LLM responses over generations."
        )
        return fig

    def plot_narrative_evolution(
        self,
        variant_states: List[dict],
        generation_metrics: pd.DataFrame,
    ) -> plt.Figure:
        """Plot t-SNE of variant prompts to show narrative evolution.

        Args:
            variant_states: List of variant state dicts.
            generation_metrics: DataFrame with per-generation metrics.

        Returns:
            Matplotlib figure.
        """
        if len(variant_states) < 3:
            return None

        # Extract all system messages and their fitness values
        messages = [v["system_message"] for v in variant_states]
        fitnesses = [v["fitness_history"][-1] if v["fitness_history"] else 0.0 for v in variant_states]

        # Simple word-count-based features (no embedding model dependency)
        features = np.zeros((len(messages), 10))
        for i, msg in enumerate(messages):
            words = msg.lower().split()
            for j, w in enumerate(set(words)):
                if j < 10:
                    features[i, j] = words.count(w)

        if features.shape[0] < 3:
            return None

        # t-SNE
        perplexity = min(5, len(messages) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=max(2, perplexity))
        embedded = tsne.fit_transform(features)

        fig, ax = plt.subplots()
        sc = ax.scatter(
            embedded[:, 0], embedded[:, 1],
            c=fitnesses, cmap="viridis", s=100, alpha=0.8,
            edgecolors="black", linewidth=0.5,
        )
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title("Variant Prompt Space (colored by fitness)")
        plt.colorbar(sc, ax=ax, label="Fitness")

        self.save_plot(
            fig,
            "narrative_evolution",
            "t-SNE projection of variant system messages, colored by fitness. "
            "Convergence indicates stable narrative attractors."
        )
        return fig

    def plot_prompt_similarity(
        self,
        generation_metrics: pd.DataFrame,
    ) -> plt.Figure:
        """Plot prompt similarity over generations.

        Args:
            generation_metrics: DataFrame with per-generation metrics.

        Returns:
            Matplotlib figure.
        """
        if "mean_prompt_similarity" not in generation_metrics.columns:
            return None

        fig, ax = plt.subplots()
        gens = generation_metrics["generation"].values
        mean_sim = generation_metrics["mean_prompt_similarity"].values
        std_sim = generation_metrics["std_prompt_similarity"].values

        ax.plot(gens, mean_sim, color=self.colors[4], marker="o", label="Mean Similarity")
        ax.fill_between(
            gens,
            mean_sim - std_sim,
            mean_sim + std_sim,
            alpha=0.2,
            color=self.colors[4],
        )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Prompt Similarity (Jaccard)")
        ax.set_title("Prompt Convergence Over Generations")
        ax.set_ylim(0, 1)

        self.save_plot(
            fig,
            "prompt_similarity",
            "Jaccard similarity between variant system messages. "
            "Increasing similarity indicates convergence to shared narratives."
        )
        return fig

    def plot_alignment(
        self,
        generation_metrics: pd.DataFrame,
    ) -> plt.Figure:
        """Plot alignment metrics if available.

        Args:
            generation_metrics: DataFrame with per-generation metrics.

        Returns:
            Matplotlib figure.
        """
        # If alignment data was stored in the metrics, plot it
        # Alignment is computed at the end, so this is a placeholder
        # for when alignment-over-time data becomes available
        return None

    def plot_all(
        self,
        generation_metrics: pd.DataFrame,
        variant_states: List[dict],
    ) -> None:
        """Generate all standard plots.

        Args:
            generation_metrics: DataFrame with per-generation metrics.
            variant_states: List of variant state dicts.
        """
        self.plot_fitness_over_generations(generation_metrics)
        self.plot_output_diversity(generation_metrics)
        self.plot_narrative_evolution(variant_states, generation_metrics)
        self.plot_prompt_similarity(generation_metrics)