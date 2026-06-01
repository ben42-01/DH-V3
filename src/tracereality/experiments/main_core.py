"""Main entry point for TraceReality core experiments.

CLI interface for running experiments with full parameter control,
parameter sweeps, ablation studies, and multiple seeds.

Usage:
    python -m tracereality.experiments.main_core --n-states 80 --seeds 1 2 3 --prefix my_exp
    python main.py --n-states 80 --seeds 1 2 3 --prefix my_exp

Examples:
    # Single run with 3 seeds
    python -m tracereality.experiments.main_core \\
        --n-states 80 --n-clusters 8 --n-observers 40 \\
        --n-steps 100000 --seeds 1 2 3 --prefix my_exp

    # Ablation: no evolution
    python -m tracereality.experiments.main_core \\
        --n-states 80 --no-evolution --prefix ablation_no_evo

    # Parameter sweep
    python -m tracereality.experiments.main_core \\
        --mutation-rate-mean 0.001 0.01 0.1 --prefix sweep_mutation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add src to path for direct script execution
_script_dir = Path(__file__).resolve().parent.parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from tracereality.config import ExperimentConfig
from tracereality.state_space import (
    assign_clusters,
    build_clustered_transition_matrix,
    check_ergodicity,
)
from tracereality.world import MarkovWorld
from tracereality.trace_store import TraceStore
from tracereality.core.observer import NeuralObserver
from tracereality.core.learner import OnlineLearner, CountBasedLearner
from tracereality.core.evolution import EvolutionEngine
from tracereality.core.analyzer import Analyzer
from tracereality.core.summarizer import generate_summary_files
from tracereality.core.visualizer import Visualizer
from tracereality.core.policy_learner import PolicyGradientLearner
from tracereality.actions import (
    build_mdp_transition_matrix,
    compute_reward,
    sample_action,
    N_ACTIONS,
    get_action_name,
)


def setup_environment(seed: int) -> np.random.Generator:
    """Set up all random seeds for reproducibility.

    Args:
        seed: Master seed for all random generators.

    Returns:
        NumPy random generator (seeded).
    """
    # Python random
    import random
    random.seed(seed)

    # NumPy
    rng = np.random.default_rng(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return rng


def run_single_experiment(
    config: ExperimentConfig,
    seed: int,
    output_dir: Path,
) -> Dict:
    """Run a single experiment with the given config and seed.

    Args:
        config: Experiment configuration.
        seed: Random seed for this run.
        output_dir: Directory to save outputs.

    Returns:
        Dictionary of final population metrics.
    """
    print(f"\n{'='*60}")
    print(f"Starting experiment: seed={seed}, prefix={config.prefix}")
    print(f"{'='*60}")

    rng = setup_environment(seed)
    use_policy = config.use_policy_learning

    # --- Setup world ---
    print(f"  Building world: {config.n_states} states, {config.n_clusters} clusters")
    cluster_ids = assign_clusters(config.n_states, config.n_clusters, rng)
    P_world = build_clustered_transition_matrix(
        config.n_states,
        cluster_ids,
        self_loop_bias=config.self_loop_bias,
        within_cluster_bias=config.within_cluster_bias,
        between_cluster_noise=config.between_cluster_noise,
        rng=rng,
    )

    # If policy learning, convert to MDP transition matrix
    if use_policy:
        print(f"  Converting to MDP with {config.n_actions} actions")
        P_world = build_mdp_transition_matrix(
            base_P=P_world,
            cluster_ids=cluster_ids,
            n_actions=config.n_actions,
            action_strength=config.action_strength,
            noise_level=config.between_cluster_noise,
            rng=rng,
        )

    # Verify ergodicity
    if P_world.ndim == 2:
        is_ergodic = check_ergodicity(P_world)
    else:
        # For MDP, check ergodicity of the base (action 0 = stay) transition
        is_ergodic = check_ergodicity(P_world[:, 0, :])
    if not is_ergodic:
        print("  WARNING: World transition matrix is not ergodic")
    else:
        print("  World is ergodic (verified)")

    # Save world matrix
    if P_world.ndim == 3:
        np.savetxt(output_dir / f"{config.prefix}_seed{seed}_world_matrix.csv",
                   P_world[:, 0, :], delimiter=",")
    else:
        np.savetxt(output_dir / f"{config.prefix}_seed{seed}_world_matrix.csv", P_world, delimiter=",")

    world = MarkovWorld(
        transition_matrix=P_world,
        cluster_ids=cluster_ids,
        noise_level=config.noise_level if config.use_noisy_observations else 0.0,
        rng=rng,
    )
    world.reset()

    # --- Setup trace store ---
    trace_store = TraceStore(max_size=config.n_steps)

    # --- Setup observers ---
    observers: List[NeuralObserver] = []
    n_actions = config.n_actions if use_policy else 0
    print(f"  Creating {config.n_observers} observer agents (n_actions={n_actions})")
    for i in range(config.n_observers):
        obs = NeuralObserver(
            observer_id=i,
            n_states=config.n_states,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            history_window_size=config.history_window_size if config.use_history_window else 1,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            mutation_rate=rng.normal(config.mutation_rate_mean, config.mutation_rate_std),
            device="cpu",
            n_actions=n_actions,
        )
        observers.append(obs)

    # --- Setup learner ---
    if config.mode == "nn":
        learner = OnlineLearner(
            batch_size=config.batch_size,
            history_window_size=config.history_window_size if config.use_history_window else 1,
            use_history_window=config.use_history_window,
        )
    else:
        learner = CountBasedLearner(config.n_states, history_window_size=1)

    # --- Setup evolution ---
    evolution = EvolutionEngine(config, rng)

    # --- Setup analyzer ---
    analyzer = Analyzer(config)

    # --- Setup visualizer ---
    viz = Visualizer(config, output_dir, f"{config.prefix}_seed{seed}")

    # --- Setup policy learner (if enabled) ---
    policy_learner = None
    if use_policy:
        policy_learner = PolicyGradientLearner(
            discount_factor=config.discount_factor,
            policy_learning_rate=config.policy_learning_rate,
            update_interval=100,
        )

    # --- Main simulation loop ---
    print(f"  Running simulation for {config.n_steps} steps...")
    start_time = time.time()

    # Running KL counter for incremental logging
    kl_sum = 0.0
    kl_count = 0

    for step in range(config.n_steps):
        if use_policy:
            # --- Policy learning mode ---
            # Each observer selects an action based on its policy
            # We train ONE observer per step (round-robin) on both prediction and policy

            # Get current state from the world
            current_state = world.current_state

            # Pick one observer round-robin to act and learn
            obs_idx = step % len(observers)
            obs = observers[obs_idx]

            # Observer chooses action from its policy
            # Build proper input: if history window is used, repeat the state
            obs.model.eval()
            with torch.no_grad():
                if config.use_history_window and config.history_window_size > 1:
                    state_tensor = torch.full(
                        (1, config.history_window_size), current_state,
                        dtype=torch.long, device=obs.device,
                    )
                else:
                    state_tensor = torch.tensor([current_state], dtype=torch.long, device=obs.device)
                _, action_logits = obs.model(state_tensor)
                action_probs = torch.softmax(action_logits, dim=-1).cpu().numpy()[0]
            action = sample_action(action_probs, rng)

            # Step the world with the chosen action
            next_state, observed_state, next_cluster_id = world.step_with_action(action)

            # Compute reward
            current_cluster = cluster_ids[current_state] if cluster_ids is not None else None
            reward = compute_reward(
                state=current_state,
                action=action,
                next_state=next_state,
                cluster_id_state=current_cluster,
                cluster_id_next=next_cluster_id,
                stability_reward=config.stability_reward,
                goal_cluster_reward=config.goal_cluster_reward,
                action_cost=config.action_cost,
                goal_cluster_id=config.goal_cluster_id,
            )

            # Record transition for policy learning
            policy_learner.record_transition(obs, current_state, action, reward)

            # Store trace with action and reward
            trace_store.append(next_state, observed_state, next_cluster_id, action, reward)

            # Train the observer on next-state prediction (round-robin)
            if step >= config.history_window_size + 1:
                learner.train_single_observer(observers, trace_store, rng)

            # Policy gradient update (every update_interval steps)
            policy_learner.maybe_update(obs)
        else:
            # --- Standard passive prediction mode ---
            true_state, observed_state, cluster_id = world.step()
            trace_store.append(true_state, observed_state, cluster_id)

            # Train ONE observer per step (round-robin)
            if config.mode == "nn" and step >= config.history_window_size + 1:
                learner.train_single_observer(observers, trace_store, rng)

            # Update count-based learner
            if config.mode == "count":
                if step > 0:
                    prev_obs = trace_store.get_previous_observed(2)
                    if prev_obs is not None:
                        learner.update(prev_obs, observed_state)

        # Lightweight logging every log_every steps
        if step % config.log_every == 0 and step > 0:
            accs = [obs.running_accuracy for obs in observers]
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            elapsed = time.time() - start_time
            extra = ""
            if use_policy:
                mean_reward = float(np.mean([obs.cumulative_reward for obs in observers]))
                extra = f" | Reward: {mean_reward:.2f}"
            print(f"    Step {step:7d}/{config.n_steps} | "
                  f"Acc: {mean_acc:.4f} ± {std_acc:.4f}{extra} | "
                  f"Time: {elapsed:.1f}s")

        # Full analysis every analyze_every steps
        if step % config.analyze_every == 0 and step > 0:
            pop_metrics = analyzer.compute_population_metrics(
                observers, P_world, step
            )

            kl_sum += pop_metrics['mean_kl_world']
            kl_count += 1

            for obs in observers:
                obs.snapshot_params()

            elapsed = time.time() - start_time
            extra = ""
            if use_policy:
                mean_reward = float(np.mean([obs.cumulative_reward for obs in observers]))
                extra = f" | Reward: {mean_reward:.2f}"
            print(f"    Step {step:7d}/{config.n_steps} | "
                  f"Acc: {pop_metrics['mean_accuracy']:.4f} ± {pop_metrics['std_accuracy']:.4f} | "
                  f"KL: {pop_metrics['mean_kl_world']:.4f}{extra} | "
                  f"Time: {elapsed:.1f}s")

        # Save transition matrix snapshots every matrix_interval
        if step % config.matrix_interval == 0 and step > 0 and not use_policy:
            for obs in observers:
                obs.snapshot_params()

        # Evolution step
        if evolution.should_evolve(step):
            observers, evo_log = evolution.evolve(observers, trace_store)
            print(f"    EVOLUTION generation {evo_log['generation']}: "
                  f"fitness={evo_log['mean_fitness']:.4f} ± {evo_log['std_fitness']:.4f}")

    # --- Final metrics ---
    print(f"\n  Computing final metrics...")
    final_pop_metrics = analyzer.compute_population_metrics(
        observers, P_world, config.n_steps
    )

    # Information-theoretic metrics
    info_metrics = analyzer.compute_information_theoretic(observers, P_world)
    print(f"  Expected KL: {info_metrics['mean_expected_kl_divergence']:.4f}")

    # --- Save all outputs ---
    print(f"  Saving outputs to {output_dir}...")

    # Save config
    config.to_json(output_dir / f"{config.prefix}_seed{seed}_config.json")

    # Save traces
    trace_store.to_csv(output_dir / f"{config.prefix}_seed{seed}_traces.csv")

    # Save cluster assignments
    np.savetxt(
        output_dir / f"{config.prefix}_seed{seed}_cluster_ids.csv",
        cluster_ids,
        delimiter=",",
        fmt="%d",
    )

    # Save metrics
    analyzer.save_metrics(
        output_dir,
        f"{config.prefix}_seed{seed}",
        observers,
        P_world,
        cluster_ids,
    )

    # Save evolution log
    evo_log_df = evolution.get_evolution_log()
    if len(evo_log_df) > 0:
        evo_log_df.to_csv(
            output_dir / f"{config.prefix}_seed{seed}_evolution_log.csv",
            index=False,
        )

    # Save statistical tests (if multiple conditions)
    # For single run, just save info-theoretic metrics
    with open(output_dir / f"{config.prefix}_seed{seed}_info_theoretic.json", "w") as f:
        json.dump(info_metrics, f, indent=2)

    # --- Generate plots ---
    # For visualization, use the 2D "stay" action matrix (avoids 3D indexing errors)
    viz_true_P = analyzer._get_true_transition_matrix(P_world)
    print(f"  Generating plots...")
    pop_df = analyzer.to_dataframe()
    if len(pop_df) > 0:
        viz.plot_all(
            population_metrics=pop_df,
            evolution_log=evo_log_df,
            observers=observers,
            true_P=viz_true_P,
            cluster_ids=cluster_ids,
        )
        print(f"  Plots saved.")

    elapsed = time.time() - start_time

    # --- Generate summary markdown reports ---
    print(f"  Generating summary reports...")
    generate_summary_files(
        output_dir=output_dir,
        prefix=f"{config.prefix}_seed{seed}",
        config=config,
        observers=observers,
        final_pop_metrics=final_pop_metrics,
        info_metrics=info_metrics,
        evolution_log=evo_log_df,
        pop_metrics_history=pop_df,
        cluster_ids=cluster_ids,
        ergodic=is_ergodic,
        elapsed_time_seconds=elapsed,
    )

    print(f"\n  Run complete in {elapsed:.1f}s")
    print(f"  Final accuracy: {final_pop_metrics['mean_accuracy']:.4f} ± {final_pop_metrics['std_accuracy']:.4f}")
    print(f"  Final KL(World||Obs): {final_pop_metrics['mean_kl_world']:.4f}")

    return final_pop_metrics


def run_parameter_sweep(
    base_config: ExperimentConfig,
    sweep_params: Dict[str, List],
    seeds: List[int],
    output_dir: Path,
) -> Dict:
    """Run a parameter sweep over multiple values.

    Args:
        base_config: Base configuration.
        sweep_params: Dict mapping parameter names to lists of values.
        seeds: List of seeds to run for each parameter combination.
        output_dir: Base output directory.

    Returns:
        Dict of results keyed by parameter combination.
    """
    import itertools

    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    results = {}

    for combo in itertools.product(*param_values):
        combo_config = base_config.clone_with_updates(
            **dict(zip(param_names, combo))
        )
        combo_name = "_".join(f"{n}_{v}" for n, v in zip(param_names, combo))
        combo_dir = output_dir / combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Sweep: {combo_name}")
        print(f"{'='*60}")

        combo_results = []
        for seed in seeds:
            seed_dir = combo_dir / f"seed_{seed}"
            seed_dir.mkdir(exist_ok=True)
            combo_config.prefix = f"sweep_{combo_name}"
            metrics = run_single_experiment(combo_config, seed, seed_dir)
            combo_results.append(metrics)

        results[combo_name] = combo_results

    return results


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="TraceReality: Research-Grade Evolutionary Markov Observers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tracereality.experiments.main_core --n_states 80 --n_clusters 8 --seeds 1 2 3 --prefix my_exp
  python -m tracereality.experiments.main_core --no-evolution --prefix ablation_no_evo
  python -m tracereality.experiments.main_core --mode count --prefix count_baseline
        """,
    )

    # World parameters
    world_group = parser.add_argument_group("World Parameters")
    world_group.add_argument("--n-states", type=int, default=80, help="Number of states (default: 80)")
    world_group.add_argument("--n-clusters", type=int, default=8, help="Number of clusters (default: 8)")
    world_group.add_argument("--n-steps", type=int, default=100000, help="Simulation steps (default: 100000)")
    world_group.add_argument("--self-loop-bias", type=float, default=0.3, help="Self-loop probability (default: 0.3)")
    world_group.add_argument("--within-cluster-bias", type=float, default=0.5, help="Within-cluster transition bias (default: 0.5)")
    world_group.add_argument("--between-cluster-noise", type=float, default=0.2, help="Between-cluster noise (default: 0.2)")

    # Observer parameters
    obs_group = parser.add_argument_group("Observer Parameters")
    obs_group.add_argument("--n-observers", type=int, default=40, help="Number of observer agents (default: 40)")
    obs_group.add_argument("--batch-size", type=int, default=64, help="Training batch size (default: 64)")
    obs_group.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    obs_group.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension (default: 128)")
    obs_group.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension (default: 32)")
    obs_group.add_argument("--no-history-window", action="store_true", help="Disable history window (use only current state)")

    # Evolution parameters
    evo_group = parser.add_argument_group("Evolution Parameters")
    evo_group.add_argument("--selection-interval", type=int, default=5000, help="Steps between evolution events (default: 5000)")
    evo_group.add_argument("--selection-top-k", type=int, default=10, help="Keep top K observers (default: 10)")
    evo_group.add_argument("--selection-bottom-k", type=int, default=10, help="Discard bottom K observers (default: 10)")
    evo_group.add_argument("--mutation-rate-mean", type=float, default=0.01, help="Mean mutation rate (default: 0.01)")
    evo_group.add_argument("--mutation-rate-std", type=float, default=0.005, help="Mutation rate std dev (default: 0.005)")

    # Ablation flags
    ablation_group = parser.add_argument_group("Ablation Controls")
    ablation_group.add_argument("--no-evolution", action="store_true", help="Disable evolution entirely")
    ablation_group.add_argument("--no-selection", action="store_true", help="Disable selection (mutation only)")
    ablation_group.add_argument("--no-mutation", action="store_true", help="Disable mutation (selection only)")
    ablation_group.add_argument("--mode", choices=["nn", "count"], default="nn", help="Learning mode (default: nn)")

    # Noise parameters
    noise_group = parser.add_argument_group("Noise Parameters")
    noise_group.add_argument("--noise-level", type=float, default=0.05, help="Observation noise level (default: 0.05)")
    noise_group.add_argument("--no-noisy-observations", action="store_true", help="Disable observation noise")

    # Policy learning parameters
    policy_group = parser.add_argument_group("Policy Learning Parameters")
    policy_group.add_argument("--use-policy-learning", action="store_true", help="Enable policy learning (actions, rewards, utility-based fitness)")
    policy_group.add_argument("--n-actions", type=int, default=3, help="Number of discrete actions (default: 3)")
    policy_group.add_argument("--discount-factor", type=float, default=0.99, help="Reward discount factor (default: 0.99)")
    policy_group.add_argument("--stability-reward", type=float, default=0.5, help="Reward for staying in same cluster (default: 0.5)")
    policy_group.add_argument("--goal-cluster-reward", type=float, default=1.0, help="Reward for reaching goal cluster (default: 1.0)")
    policy_group.add_argument("--action-cost", type=float, default=0.05, help="Penalty for non-stay actions (default: 0.05)")
    policy_group.add_argument("--goal-cluster-id", type=int, default=0, help="Which cluster is the goal (default: 0)")

    # Experiment control
    exp_group = parser.add_argument_group("Experiment Control")
    exp_group.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds (default: 42)")
    exp_group.add_argument("--prefix", type=str, default="exp", help="Output prefix (default: exp)")
    exp_group.add_argument("--output-dir", type=str, default="outputs", help="Output directory (default: outputs)")
    exp_group.add_argument("--analyze-every", type=int, default=1000, help="Steps between analysis (default: 1000)")

    return parser.parse_args(argv)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Build config
    config = ExperimentConfig(
        n_states=args.n_states,
        n_clusters=args.n_clusters,
        n_steps=args.n_steps,
        n_observers=args.n_observers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        use_history_window=not args.no_history_window,
        self_loop_bias=args.self_loop_bias,
        within_cluster_bias=args.within_cluster_bias,
        between_cluster_noise=args.between_cluster_noise,
        selection_interval=args.selection_interval,
        selection_top_k=args.selection_top_k,
        selection_bottom_k=args.selection_bottom_k,
        mutation_rate_mean=args.mutation_rate_mean,
        mutation_rate_std=args.mutation_rate_std,
        with_evolution=not args.no_evolution,
        with_selection=not args.no_selection,
        with_mutation=not args.no_mutation,
        mode=args.mode,
        noise_level=args.noise_level,
        use_noisy_observations=not args.no_noisy_observations,
        seeds=args.seeds,
        prefix=args.prefix,
        analyze_every=args.analyze_every,
        use_policy_learning=args.use_policy_learning,
        n_actions=args.n_actions,
        discount_factor=args.discount_factor,
        stability_reward=args.stability_reward,
        goal_cluster_reward=args.goal_cluster_reward,
        action_cost=args.action_cost,
        goal_cluster_id=args.goal_cluster_id,
    )

    # Setup output directory
    output_dir = Path(args.output_dir) / args.prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save base config
    config.to_json(output_dir / f"{args.prefix}_config.json")

    # Print experiment summary
    print(f"\n{'='*60}")
    print(f"TraceReality Core Experiment")
    print(f"{'='*60}")
    print(f"  States:     {config.n_states} in {config.n_clusters} clusters")
    print(f"  Observers:  {config.n_observers}")
    print(f"  Steps:      {config.n_steps:,}")
    print(f"  Mode:       {config.mode}")
    print(f"  Evolution:  {config.with_evolution}")
    print(f"    Selection: {config.with_selection} (top {config.selection_top_k}, bottom {config.selection_bottom_k})")
    print(f"    Mutation:  {config.with_mutation} (rate={config.mutation_rate_mean} ± {config.mutation_rate_std})")
    print(f"  Seeds:      {config.seeds}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # Run experiments for each seed
    all_results = []
    for seed in config.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        metrics = run_single_experiment(config, seed, seed_dir)
        all_results.append(metrics)

    # Summary across seeds
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"Summary Across Seeds")
        print(f"{'='*60}")
        for key in ["mean_accuracy", "mean_kl_world", "mean_fitness"]:
            values = [r[key] for r in all_results]
            print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        # Run statistical tests if we have multiple conditions
        # For single condition, save summary
        summary = {
            "n_runs": len(all_results),
            "seeds": config.seeds,
            "config": {
                "n_states": config.n_states,
                "n_clusters": config.n_clusters,
                "n_observers": config.n_observers,
                "with_evolution": config.with_evolution,
                "with_selection": config.with_selection,
                "with_mutation": config.with_mutation,
                "mode": config.mode,
            },
            "results": {
                "mean_accuracy": {
                    "mean": float(np.mean([r["mean_accuracy"] for r in all_results])),
                    "std": float(np.std([r["mean_accuracy"] for r in all_results])),
                },
                "mean_kl_world": {
                    "mean": float(np.mean([r["mean_kl_world"] for r in all_results])),
                    "std": float(np.std([r["mean_kl_world"] for r in all_results])),
                },
            },
        }
        with open(output_dir / f"{args.prefix}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()