"""Simulation summarizer that produces Markdown reports.

Generates two complementary summary files after a core simulation completes:
1. Technical report — detailed metrics, stats, and observations
2. Plain-language explanation — what happened and what it means, accessible to non-experts
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .observer import NeuralObserver
from ..config import ExperimentConfig


def _fmt(val: float, decimals: int = 4) -> str:
    """Format a float for display."""
    return f"{val:.{decimals}f}"


def _fmt_pct(val: float, decimals: int = 1) -> str:
    """Format as percentage."""
    return f"{val * 100:.{decimals}f}%"


def _trend_description(values: List[float], improvement_is_good: bool = True) -> str:
    """Describe the trend of a metric over time."""
    if len(values) < 2:
        return "insufficient data"
    first_half = np.mean(values[: len(values) // 2])
    second_half = np.mean(values[len(values) // 2 :])
    delta = second_half - first_half
    if abs(delta) < 1e-6:
        return "stable (no significant change)"
    if improvement_is_good:
        if delta > 0:
            return f"improved by {abs(delta):.4f} over the run"
        else:
            return f"declined by {abs(delta):.4f} over the run"
    else:
        if delta < 0:
            return f"improved (decreased) by {abs(delta):.4f} over the run"
        else:
            return f"worsened (increased) by {abs(delta):.4f} over the run"


def _evolution_narrative(evo_log: pd.DataFrame) -> str:
    """Generate a narrative about evolution dynamics."""
    if len(evo_log) == 0:
        return "Evolution was disabled for this simulation."

    lines = []
    n_gens = len(evo_log)
    lines.append(f"- **{n_gens} evolutionary generations** occurred during the simulation.")

    # Fitness trajectory
    mean_fitness = evo_log["mean_fitness"].values
    max_fitness = evo_log["max_fitness"].values
    lines.append(
        f"- Mean population fitness started at **{_fmt(mean_fitness[0])}** "
        f"and ended at **{_fmt(mean_fitness[-1])}** "
        f"({_trend_description(list(mean_fitness))})."
    )
    lines.append(
        f"- Best observer fitness started at **{_fmt(max_fitness[0])}** "
        f"and peaked at **{_fmt(max(max_fitness))}**."
    )

    # Fitness variance — indication of selection pressure working
    std_fitness = evo_log["std_fitness"].values
    avg_std = float(np.mean(std_fitness))
    lines.append(
        f"- Average fitness variance across generations: **{_fmt(avg_std)}** "
        + ("(moderate diversity in the population)." if avg_std > 0.01 else "(population is relatively homogeneous).")
    )

    # Survivor / discard ratio
    avg_survivors = int(evo_log["n_survivors"].mean())
    avg_discarded = int(evo_log["n_discarded"].mean())
    lines.append(
        f"- Each generation kept ~**{avg_survivors}** observers and "
        f"replaced ~**{avg_discarded}** with mutated copies of survivors."
    )

    # Mutation rate behavior
    if "mutation_rates" in evo_log.columns:
        # mutation_rates is a list per row — flatten to get overall statistics
        all_rates = []
        for row in evo_log["mutation_rates"]:
            if isinstance(row, list):
                all_rates.extend(row)
        if all_rates:
            lines.append(
                f"- Mutation rates ranged from **{_fmt(min(all_rates))}** to **{_fmt(max(all_rates))}** "
                f"across all observers and generations."
            )

    return "\n".join(lines)


def _observer_diversity_narrative(observers: List[NeuralObserver]) -> str:
    """Describe diversity across the observer population."""
    if not observers:
        return ""

    accs = [obs.average_accuracy for obs in observers]
    fits = [obs.current_fitness for obs in observers]
    rates = [obs.mutation_rate for obs in observers]

    lines = []
    lines.append(f"- **Population size:** {len(observers)} observers")
    lines.append(
        f"- **Accuracy range:** {_fmt_pct(min(accs))} – {_fmt_pct(max(accs))} "
        f"(mean: {_fmt_pct(float(np.mean(accs)))}, std: {_fmt_pct(float(np.std(accs)))})"
    )
    lines.append(
        f"- **Fitness range:** {_fmt(min(fits))} – {_fmt(max(fits))} "
        f"(mean: {_fmt(float(np.mean(fits)))})"
    )
    if rates:
        lines.append(
            f"- **Mutation rates:** {_fmt(min(rates))} – {_fmt(max(rates))} "
            f"(mean: {_fmt(float(np.mean(rates)))})"
        )
    return "\n".join(lines)


def generate_technical_summary(
    config: ExperimentConfig,
    observers: List[NeuralObserver],
    final_pop_metrics: Dict,
    info_metrics: Dict,
    evolution_log: pd.DataFrame,
    pop_metrics_history: pd.DataFrame,
    cluster_ids: Optional[np.ndarray] = None,
    n_states: Optional[int] = None,
    ergodic: bool = True,
    elapsed_time_seconds: float = 0.0,
) -> str:
    """Generate a detailed technical Markdown summary of the simulation.

    Args:
        config: The experiment configuration used.
        observers: The final observer population.
        final_pop_metrics: Dict from analyzer.compute_population_metrics at final step.
        info_metrics: Dict from analyzer.compute_information_theoretic.
        evolution_log: DataFrame from evolution.get_evolution_log().
        pop_metrics_history: DataFrame from analyzer.to_dataframe().
        cluster_ids: Cluster assignments (optional).
        n_states: Number of states in the world (optional, falls back to config).
        ergodic: Whether the world was ergodic.
        elapsed_time_seconds: Wall-clock time for the simulation.

    Returns:
        Markdown string with the technical summary.
    """
    n_states = n_states or config.n_states
    n_obs = len(observers)

    md = []
    md.append("# TraceReality Core Simulation — Technical Report")
    md.append("")
    md.append(f"*Generated automatically after simulation completion*  ")
    md.append(f"*Run time: {elapsed_time_seconds:.1f}s*  ")
    md.append("")

    # ── 1. Experiment Configuration ──
    md.append("## 1. Experiment Configuration")
    md.append("")
    md.append("| Parameter | Value |")
    md.append("|-----------|-------|")

    config_rows = [
        ("World states", str(config.n_states)),
        ("Clusters", str(config.n_clusters)),
        ("Observers", str(config.n_observers)),
        ("Simulation steps", f"{config.n_steps:,}"),
        ("Mode", config.mode),
        ("Evolution enabled", str(config.with_evolution)),
        ("Selection enabled", str(config.with_selection)),
        ("Mutation enabled", str(config.with_mutation)),
        ("Selection interval (steps)", str(config.selection_interval)),
        ("Top-k keep", str(config.selection_top_k)),
        ("Bottom-k discard", str(config.selection_bottom_k)),
        ("Mutation rate mean", _fmt(config.mutation_rate_mean)),
        ("Mutation rate std", _fmt(config.mutation_rate_std)),
        ("Learning rate", _fmt(config.learning_rate)),
        ("Hidden dimension", str(config.hidden_dim)),
        ("Embedding dimension", str(config.embedding_dim)),
        ("History window", str(config.history_window_size) if config.use_history_window else "disabled"),
        ("Observation noise", _fmt(config.noise_level) if config.use_noisy_observations else "disabled"),
        ("Self-loop bias", _fmt(config.self_loop_bias)),
        ("Within-cluster bias", _fmt(config.within_cluster_bias)),
        ("Between-cluster noise", _fmt(config.between_cluster_noise)),
        ("World ergodic", str(ergodic)),
    ]
    for param, val in config_rows:
        md.append(f"| {param} | {val} |")
    md.append("")

    # ── 2. Final Population Metrics ──
    md.append("## 2. Final Population Metrics")
    md.append("")
    md.append("| Metric | Mean | Std | Min | Max |")
    md.append("|--------|------|-----|-----|-----|")

    acc = final_pop_metrics.get("mean_accuracy", 0.0)
    acc_std = final_pop_metrics.get("std_accuracy", 0.0)
    acc_min = final_pop_metrics.get("min_accuracy", 0.0)
    acc_max = final_pop_metrics.get("max_accuracy", 0.0)
    md.append(f"| Accuracy | {_fmt_pct(acc)} | {_fmt_pct(acc_std)} | {_fmt_pct(acc_min)} | {_fmt_pct(acc_max)} |")

    kl = final_pop_metrics.get("mean_kl_world", 0.0)
    kl_std = final_pop_metrics.get("std_kl_world", 0.0)
    md.append(f"| KL(World ‖ Observer) | {_fmt(kl)} | {_fmt(kl_std)} | — | — |")

    fit = final_pop_metrics.get("mean_fitness", 0.0)
    fit_std = final_pop_metrics.get("std_fitness", 0.0)
    md.append(f"| Fitness | {_fmt(fit)} | {_fmt(fit_std)} | — | — |")

    pairwise_kl = final_pop_metrics.get("mean_pairwise_kl", None)
    if pairwise_kl is not None:
        pairwise_kl_std = final_pop_metrics.get("std_pairwise_kl", 0.0)
        md.append(f"| Pairwise KL (observer diversity) | {_fmt(pairwise_kl)} | {_fmt(pairwise_kl_std)} | — | — |")

    param_var = final_pop_metrics.get("mean_param_variance", None)
    if param_var is not None:
        param_var_std = final_pop_metrics.get("std_param_variance", 0.0)
        md.append(f"| Param variance (stability) | {_fmt(param_var)} | {_fmt(param_var_std)} | — | — |")

    if "mean_cluster_ari" in final_pop_metrics:
        ari = final_pop_metrics["mean_cluster_ari"]
        ari_std = final_pop_metrics.get("std_cluster_ari", 0.0)
        md.append(f"| Cluster ARI (structure recovery) | {_fmt(ari)} | {_fmt(ari_std)} | — | — |")

    md.append("")

    # ── 3. Expected KL Divergence (Model Mismatch) ──
    md.append("## 3. Expected KL Divergence")
    md.append("")
    ekl = info_metrics.get("mean_expected_kl_divergence", 0.0)
    ekl_std = info_metrics.get("std_expected_kl_divergence", 0.0)
    md.append(f"- **Expected KL Divergence** E_i[ KL(p_true‖p_pred) ]: **{_fmt(ekl)}** ± {_fmt(ekl_std)}")
    md.append("  - Expected (stationary-weighted) KL divergence between the true world")
    md.append("    and each observer's model, per state.")
    md.append("  - Lower is better: a small value means the observer's transition model")
    md.append("    closely matches the true world dynamics across all states.")
    md.append("  - This is closely related to the mean KL reported in Section 2; both")
    md.append("    converge to the same quantity for a well-sampled stationary distribution.")
    md.append("")

    # ── 4. Evolution Dynamics ──
    md.append("## 4. Evolution Dynamics")
    md.append("")
    md.append(_evolution_narrative(evolution_log))
    md.append("")

    # ── 5. Observer Population Diversity ──
    md.append("## 5. Observer Population (Final State)")
    md.append("")
    md.append(_observer_diversity_narrative(observers))
    md.append("")

    # ── 6. Metric Trajectory Summary ──
    if len(pop_metrics_history) > 1:
        md.append("## 6. Metric Trajectories")
        md.append("")
        md.append("Key metrics were sampled at regular intervals throughout the simulation. "
                  "Below is a summary of how they evolved:")
        md.append("")

        for col, label, good_is_up in [
            ("mean_accuracy", "Accuracy", True),
            ("mean_kl_world", "KL(World‖Observer)", False),
            ("mean_fitness", "Fitness", True),
            ("mean_pairwise_kl", "Pairwise KL (diversity)", None),
        ]:
            if col in pop_metrics_history.columns:
                vals = pop_metrics_history[col].dropna().values
                if len(vals) > 0:
                    trend = _trend_description(list(vals), improvement_is_good=good_is_up) if good_is_up is not None else ""
                    start_v = vals[0]
                    end_v = vals[-1]
                    best_v = max(vals) if good_is_up is not None and good_is_up else min(vals)
                    if isinstance(best_v, float):
                        best_v = _fmt(best_v)
                    line = f"- **{label}**: {_fmt(start_v)} → {_fmt(end_v)} (best: {best_v})"
                    if trend:
                        line += f" — {trend}"
                    md.append(line)

        md.append("")

    # ── 7. Interpretation Notes ──
    md.append("## 7. Technical Interpretation")
    md.append("")

    # Accuracy interpretation
    n_states_baseline = 1.0 / n_states
    if acc > n_states_baseline * 2:
        md.append(f"- **Above-chance prediction:** Observers achieved {_fmt_pct(acc)} accuracy vs. "
                  f"{_fmt_pct(n_states_baseline)} random baseline ({n_states} states). "
                  "The population has learned meaningful predictive structure.")
    else:
        md.append(f"- **Near-chance prediction:** Accuracy is {_fmt_pct(acc)} vs. "
                  f"{_fmt_pct(n_states_baseline)} random baseline. "
                  "Observers may not have learned meaningful structure yet, "
                  "or the world may be too complex for the current model capacity.")

    # KL interpretation
    if kl < 0.1:
        md.append(f"- **Excellent world model:** KL divergence of {_fmt(kl)} indicates "
                  "observers' internal transition matrices closely match the true world dynamics.")
    elif kl < 0.5:
        md.append(f"- **Moderate world model:** KL divergence of {_fmt(kl)} indicates "
                  "observers capture broad transition patterns but miss finer structure.")
    else:
        md.append(f"- **Poor world model:** KL divergence of {_fmt(kl)} indicates "
                  "observers' learned dynamics differ substantially from the true world.")

    # Evolution effectiveness
    if config.with_evolution and len(evolution_log) > 0:
        first_gen_fit = evolution_log["mean_fitness"].iloc[0]
        last_gen_fit = evolution_log["mean_fitness"].iloc[-1]
        if last_gen_fit > first_gen_fit:
            md.append(f"- **Evolution was effective:** Population fitness improved from {_fmt(first_gen_fit)} "
                      f"to {_fmt(last_gen_fit)} over {len(evolution_log)} generations.")
        else:
            md.append(f"- **Evolution had limited impact:** Population fitness did not significantly improve "
                      f"({_fmt(first_gen_fit)} → {_fmt(last_gen_fit)}). "
                      "This may indicate the population converged early or selection pressure was too weak.")

    # Expected KL (model mismatch) — note: this is redundant with KL above but kept for compatibility
    if ekl < 0.1:
        md.append(f"- **Excellent world model:** Expected KL divergence of {_fmt(ekl)} confirms "
                  "observers' models closely match the true world dynamics.")
    elif ekl < 0.5:
        md.append(f"- **Moderate world model:** Expected KL divergence of {_fmt(ekl)} confirms "
                  "observers capture broad transition patterns but miss finer structure "
                  "(consistent with the mean KL in Section 2).")
    else:
        md.append(f"- **Poor world model:** Expected KL divergence of {_fmt(ekl)} indicates "
                  "a substantial mismatch between observers' models and the true world.")

    # Cluster recovery (if available)
    if "mean_cluster_ari" in final_pop_metrics:
        ari = final_pop_metrics["mean_cluster_ari"]
        if ari > 0.7:
            md.append(f"- **Excellent cluster recovery (ARI={_fmt(ari)}):** Observers' internal representations "
                      "accurately recover the true cluster structure of the world.")
        elif ari > 0.3:
            md.append(f"- **Partial cluster recovery (ARI={_fmt(ari)}):** Some cluster structure detected "
                      "but incomplete.")
        else:
            md.append(f"- **Poor cluster recovery (ARI={_fmt(ari)}):** Observers did not recover "
                      "the underlying cluster structure.")

    md.append("")

    return "\n".join(md)


def generate_plain_language_summary(
    config: ExperimentConfig,
    observers: List[NeuralObserver],
    final_pop_metrics: Dict,
    info_metrics: Dict,
    evolution_log: pd.DataFrame,
    pop_metrics_history: pd.DataFrame,
    cluster_ids: Optional[np.ndarray] = None,
    ergodic: bool = True,
    elapsed_time_seconds: float = 0.0,
) -> str:
    """Generate a plain-language Markdown explanation of the simulation results.

    Args:
        Same as generate_technical_summary.

    Returns:
        Markdown string with an accessible summary.
    """
    n_states = config.n_states
    n_obs = len(observers)
    n_steps = config.n_steps

    acc = final_pop_metrics.get("mean_accuracy", 0.0)
    kl = final_pop_metrics.get("mean_kl_world", 0.0)
    fit = final_pop_metrics.get("mean_fitness", 0.0)
    ekl = info_metrics.get("mean_expected_kl_divergence", 0.0)
    random_baseline = 1.0 / n_states

    md = []
    md.append("# TraceReality Core Simulation — What Happened? (Plain-Language Summary)")
    md.append("")
    md.append("*This summary explains the simulation results in simple terms, with minimal jargon.*")
    md.append("")

    # ── 1. What did we do? ──
    md.append("## 🧪 What Did We Do?")
    md.append("")
    md.append(f"We created a **miniature artificial world** with {n_states} different states "
              f"(think of them as rooms in a building, or positions on a board). "
              f"These states were organized into **{config.n_clusters} groups (clusters)** — "
              f"like neighborhoods in a city where you're more likely to move between nearby places.")

    if config.with_evolution:
        md.append("")
        md.append(f"We then placed **{n_obs} AI agents (\"observers\")** into this world. "
                  f"Each agent's job was to **learn the rules** of how the world transitions from one state to another — "
                  f"basically, to predict \"if I'm in state X, what's most likely to happen next?\"")
        md.append("")
        md.append(f"But here's the twist: we added **evolution**. Every {config.selection_interval} steps, "
                  f"the worst-performing agents were replaced with slightly mutated copies of the best ones. "
                  f"This is similar to natural selection — agents that make better predictions "
                  f"get to \"reproduce,\" while poor predictors get淘汰.")
    else:
        md.append("")
        md.append(f"We placed **{n_obs} AI agents (\"observers\")** into this world. "
                  f"Each agent tried to learn the transition rules. "
                  f"**Evolution was disabled** — agents learned independently without selection or mutation.")

    if config.mode == "count":
        md.append("")
        md.append("The agents used a **simple counting strategy**: they kept track of how often each "
                  "state followed another, building up a frequency table over time.")
    else:
        md.append("")
        md.append("The agents used **small neural networks** (simple AI models) to learn the transition patterns. "
                  "Each network has a few thousand adjustable parameters.")

    md.append("")

    # ── 2. What happened? ──
    md.append("## 📊 What Happened?")
    md.append("")

    # Accuracy story
    md.append(f"### Prediction Accuracy")
    md.append("")
    md.append(f"By the end of the simulation, the average agent predicted the next state correctly "
              f"**{_fmt_pct(acc)} of the time**.")

    if acc > random_baseline * 3:
        md.append(f"To put this in perspective: if the agents were guessing randomly, "
                  f"they'd only be right **{_fmt_pct(random_baseline)} of the time** "
                  f"(since there are {n_states} possible states). So the agents "
                  f"learned **{acc / random_baseline:.0f}x better than random guessing** — "
                  f"they genuinely figured out the world's patterns!")
    elif acc > random_baseline * 1.5:
        md.append(f"Random guessing would give **{_fmt_pct(random_baseline)}%** "
                  f"(since there are {n_states} options). The agents performed "
                  f"better than random, suggesting they learned **some** of the underlying structure.")
    else:
        md.append(f"Random guessing would give about **{_fmt_pct(random_baseline)}%**. "
                  f"The agents performed close to this baseline, suggesting the world's patterns "
                  f"were difficult to capture with the current setup.")

    md.append("")

    # KL / world model story
    md.append(f"### Internal World Model Quality")
    md.append("")
    if kl < 0.05:
        md.append(f"Beyond just predicting the next step, we compared each agent's internal model of the world "
                  f"to the actual world. The **KL divergence** (a measure of model mismatch) was **{_fmt(kl)}**, "
                  f"which is **very low**. This means the agents' internal mental models closely match "
                  f"the true rules of the world.")
        md.append("")
        md.append("**Analogy:** It's like someone learning the rules of chess so well that they can "
                  f"predict their opponent's move with near-perfect accuracy.")
    elif kl < 0.3:
        md.append(f"The **KL divergence** (model mismatch) was **{_fmt(kl)}** — moderately low. "
                  f"Agents captured the broad strokes of the world's dynamics but missed some "
                  f"finer details.")
        md.append("")
        md.append("**Analogy:** It's like someone who knows the main roads of a city but "
                  f"doesn't know the shortcuts and alleys.")
    else:
        md.append(f"The **KL divergence** was **{_fmt(kl)}**, indicating a noticeable gap between "
                  f"the agents' understanding and the actual world dynamics.")
        md.append("")
        md.append("**Analogy:** The agents understood some patterns but had significant blind spots "
                  f"in their model of the world.")

    md.append("")

    # Evolution story (if enabled)
    if config.with_evolution and len(evolution_log) > 0:
        first_fit = evolution_log["mean_fitness"].iloc[0]
        last_fit = evolution_log["mean_fitness"].iloc[-1]
        n_gens = len(evolution_log)

        md.append(f"### Evolution at Work")
        md.append("")
        if last_fit > first_fit * 1.1:
            md.append(f"The evolutionary process **improved the population** over {n_gens} generations. "
                      f"Average fitness rose from **{_fmt(first_fit)}** to **{_fmt(last_fit)}** — "
                      f"a gain of **{((last_fit / first_fit) - 1) * 100:.0f}%**.")
            md.append("")
            md.append("**What this means:** Evolution is working as intended. Better predictors are surviving "
                      f"and passing their \"genes\" (neural network weights) to the next generation, "
                      f"while weaker predictors are淘汰.")
        else:
            md.append(f"Evolution ran for {n_gens} generations, but the population's fitness "
                      f"remained relatively stable ({_fmt(first_fit)} → {_fmt(last_fit)}).")
            md.append("")
            md.append("**What this means:** Either the population reached a ceiling (all agents are "
                      f"similarly good), selection pressure was too weak, or mutations were too "
                      f"small/destructive to drive improvement.")

        md.append("")

    # Diversity story
    accs = [obs.average_accuracy for obs in observers]
    if max(accs) - min(accs) > 0.1:
        md.append(f"### Population Diversity")
        md.append("")
        md.append(f"The observer population remained **diverse**: accuracies ranged from "
                  f"**{_fmt_pct(min(accs))}** to **{_fmt_pct(max(accs))}**. "
                  f"This is healthy — it means different agents have discovered different strategies, "
                  f"which provides raw material for evolution to work with.")
        md.append("")

    # Cluster recovery
    if "mean_cluster_ari" in final_pop_metrics:
        ari = final_pop_metrics["mean_cluster_ari"]
        md.append(f"### Structure Discovery")
        md.append("")
        if ari > 0.7:
            md.append(f"The agents not only learned transition probabilities, they also **discovered "
                      f"the underlying group structure** of the world. The Adjusted Rand Index (ARI) "
                      f"of **{_fmt(ari)}** indicates excellent recovery of the true clusters.")
            md.append("")
            md.append("**Analogy:** Imagine someone listening to music in a foreign genre and "
                      f"correctly identifying which songs belong together — without being told the categories.")
        elif ari > 0.3:
            md.append(f"The agents partially detected the cluster structure (ARI={_fmt(ari)}), "
                      f"but the recovery was incomplete.")
        else:
            md.append(f"The agents did not successfully recover the underlying cluster structure "
                      f"(ARI={_fmt(ari)}).")
        md.append("")

    # Expected KL divergence (model mismatch)
    md.append(f"### Model Mismatch (Expected KL Divergence)")
    md.append("")
    md.append(f"The **expected KL divergence** — a stationary-weighted average of how much each "
              f"agent's model diverges from the true world — was **{_fmt(ekl)}**. ")
    if ekl < 0.1:
        md.append(f"This is a very small number, confirming that the agents' internal models "
                  f"closely match the actual world dynamics.")
    elif ekl < 0.3:
        md.append(f"This is a moderately low value, indicating the agents have captured the "
                  f"broad structure of the world but miss some finer details.")
    else:
        md.append(f"This value indicates a noticeable gap between the agents' internal models "
                  f"and the true world dynamics.")
    md.append("")
    md.append(f"> **Note:** This metric was previously (incorrectly) labeled as \"mutual information.\" "
              f"It is actually the expected KL divergence and is redundant with the KL(World ‖ Observer) "
              f"reported above. We keep it here for backward compatibility.")
    md.append("")

    # ── 3. What does this mean? ──
    md.append("## 💡 What Does This Mean?")
    md.append("")

    if config.with_evolution and acc > random_baseline * 2:
        md.append(f"✅ **Evolution + learning works.** The combination of individual learning "
                  f"(each agent improving over time) and population-level evolution "
                  f"(survival of the fittest) produced agents that understand the world "
                  f"significantly better than random chance.")
    elif config.with_evolution:
        md.append(f"🔄 **Evolution was limited in its effect.** While agents learned above chance, "
                  f"the evolutionary process didn't dramatically boost performance. "
                  f"This may be due to the problem's difficulty, the model's capacity, "
                  f"or the evolution parameters needing tuning.")
    else:
        md.append(f"🧠 **Individual learning.** With evolution disabled, each agent learned independently. "
                  f"The results show how well a static population can learn without the boost "
                  f"of evolutionary selection.")

    md.append("")
    md.append(f"This simulation demonstrates **how agents can build internal models of a complex, "
              f"probabilistic environment** by observing sequences of events. "
              f"This is relevant to understanding:")
    md.append("")
    md.append("- **How brains might learn** statistical regularities in the environment")
    md.append("- **How populations of AI agents** can collectively improve through evolution")
    md.append("- **How well simple models** can capture structured, clustered dynamics")
    if config.with_evolution:
        md.append("- **The power of natural selection** as an optimization mechanism in artificial systems")
    md.append("")

    # ── 4. Key Takeaway ──
    md.append("## 🎯 Key Takeaway")
    md.append("")
    if acc > random_baseline * 2 and kl < 0.1:
        md.append(f"The observer agents **successfully learned the world's dynamics** with high fidelity. "
                  f"Evolutionary selection produced a population of agents that can predict the "
                  f"world's next state with **{_fmt_pct(acc)} accuracy** "
                  f"and maintain an accurate internal model (KL divergence: {_fmt(kl)}). "
                  f"This shows that even simple neural networks, guided by evolution, "
                  f"can extract meaningful structure from complex environments.")
    elif acc > random_baseline:
        md.append(f"The agents learned **above chance** ({_fmt_pct(acc)} vs. {_fmt_pct(random_baseline)} random), "
                  f"showing they extracted some structure from the environment. "
                  f"The KL divergence of {_fmt(kl)} suggests their internal model captures "
                  f"some — but not all — of the world's dynamics.")
    else:
        md.append(f"The agents struggled to learn the world's dynamics in this configuration. "
                  f"This is valuable information — it tells us the task is challenging and may require "
                  f"more capacity, different learning parameters, or more evolutionary pressure.")

    md.append("")
    md.append("---")
    md.append(f"*Simulation completed in {elapsed_time_seconds:.1f}s with seed(s) {config.seeds}*")
    md.append("")

    return "\n".join(md)


def generate_summary_files(
    output_dir: Path,
    prefix: str,
    config: ExperimentConfig,
    observers: List[NeuralObserver],
    final_pop_metrics: Dict,
    info_metrics: Dict,
    evolution_log: pd.DataFrame,
    pop_metrics_history: pd.DataFrame,
    cluster_ids: Optional[np.ndarray] = None,
    n_states: Optional[int] = None,
    ergodic: bool = True,
    elapsed_time_seconds: float = 0.0,
) -> Tuple[Path, Path]:
    """Generate and save both technical and plain-language summary Markdown files.

    Args:
        output_dir: Directory to save summaries.
        prefix: Experiment filename prefix.
        *args: Passed through to both generator functions.

    Returns:
        Tuple of (technical_report_path, plain_language_report_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Technical summary
    tech_md = generate_technical_summary(
        config=config,
        observers=observers,
        final_pop_metrics=final_pop_metrics,
        info_metrics=info_metrics,
        evolution_log=evolution_log,
        pop_metrics_history=pop_metrics_history,
        cluster_ids=cluster_ids,
        n_states=n_states,
        ergodic=ergodic,
        elapsed_time_seconds=elapsed_time_seconds,
    )
    tech_path = output_dir / f"{prefix}_summary_technical.md"
    tech_path.write_text(tech_md)
    print(f"  Technical summary saved to {tech_path}")

    # Plain-language summary
    plain_md = generate_plain_language_summary(
        config=config,
        observers=observers,
        final_pop_metrics=final_pop_metrics,
        info_metrics=info_metrics,
        evolution_log=evolution_log,
        pop_metrics_history=pop_metrics_history,
        cluster_ids=cluster_ids,
        ergodic=ergodic,
        elapsed_time_seconds=elapsed_time_seconds,
    )
    plain_path = output_dir / f"{prefix}_summary_explained.md"
    plain_path.write_text(plain_md)
    print(f"  Plain-language summary saved to {plain_path}")

    return tech_path, plain_path