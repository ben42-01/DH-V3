"""Main entry point for TraceReality LLM observer experiments.

CLI interface for running LLM-based observer experiments with
evolutionary prompt engineering via Ollama.

Usage:
    python -m tracereality.experiments.main_llm --model llama3 --n-states 40 --n-clusters 4 --n-variants 12 --generations 20
    python main_llm.py --model mistral --n-states 80 --n-clusters 8 --n-variants 20 --generations 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
from tracereality.llm.ollama_client import OllamaClient
from tracereality.llm.llm_observer import LLMObserver
from tracereality.llm.prompt_evolution import PromptEvolutionEngine
from tracereality.llm.fitness import FitnessComputer
from tracereality.llm.analyzer import LLMAnalyzer
from tracereality.llm.visualizer import LLMVisualizer


DEFAULT_SYSTEM_MESSAGE = (
    "You are seeing a sequence of symbols. Say something about it."
)

DEFAULT_PROMPT_TEMPLATE = (
    "Sequence: {trace_fragment}\n"
    "What do you notice about this sequence?"
)


def setup_environment(seed: int) -> np.random.Generator:
    """Set up random seeds for reproducibility.

    Args:
        seed: Master seed.

    Returns:
        NumPy random generator.
    """
    import random as py_random
    py_random.seed(seed)
    rng = np.random.default_rng(seed)
    return rng


def check_ollama(model: str) -> None:
    """Verify Ollama is running and the model is available.

    Args:
        model: Model name to check.

    Raises:
        RuntimeError: If Ollama is not available or model not found.
    """
    client = OllamaClient(model=model, timeout=10)
    try:
        available = client.check_available()
        if not available:
            print(f"  Warning: Model '{model}' not found locally.")
            print(f"  It will be pulled automatically on first use, or run: ollama pull {model}")
    except RuntimeError as e:
        print(f"  Warning: {e}")
        print("  Continuing anyway; Ollama will be contacted when generation is needed.")


def create_initial_variants(
    n_variants: int,
    model: str,
    temperature: float,
    max_tokens: int,
    ollama_base_url: str,
    ollama_timeout: int,
    num_ctx: int,
    rng: np.random.Generator,
) -> List[LLMObserver]:
    """Create the initial population of LLM observer variants.

    All start from the same base but with small random mutations.

    Args:
        n_variants: Number of variants to create.
        model: Ollama model name.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        ollama_base_url: Ollama server URL.
        ollama_timeout: Request timeout.
        num_ctx: Context window size in tokens (lower = less GPU compute).
        rng: NumPy random generator.

    Returns:
        List of LLMObserver instances.
    """
    variants = []

    # Slightly vary the initial system message for each variant
    base_adjectives = ["", "interesting ", "curious ", "simple ", "basic ", "raw ", ""]
    base_verbs = ["comment on", "describe", "talk about", "examine", "look at", "observe"]

    for i in range(n_variants):
        adj = rng.choice(base_adjectives)
        verb = rng.choice(base_verbs)
        system = f"You are seeing a sequence of symbols. {adj.capitalize()}{verb} it briefly."

        # Slight variation in prompt template
        templates = [
            "Sequence: {trace_fragment}\nWhat do you notice?",
            "Here is a sequence: {trace_fragment}\nDescribe what you see.",
            "Look at this sequence: {trace_fragment}\nWhat comes to mind?",
            "Sequence: {trace_fragment}\nSay something about this pattern.",
            "Observe: {trace_fragment}\nWhat do you think?",
        ]
        template = rng.choice(templates)

        observer = LLMObserver(
            observer_id=i,
            system_message=system,
            prompt_template=template,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            ollama_base_url=ollama_base_url,
            ollama_timeout=ollama_timeout,
            num_ctx=num_ctx,
        )
        variants.append(observer)

    return variants


def run_llm_experiment(
    config: ExperimentConfig,
    llm_config: dict,
    seed: int,
    output_dir: Path,
) -> Dict:
    """Run a single LLM observer experiment.

    Args:
        config: Base experiment config (for world parameters).
        llm_config: LLM-specific parameters.
        seed: Random seed.
        output_dir: Output directory.

    Returns:
        Dictionary of final metrics.
    """
    print(f"\n{'='*60}")
    print(f"Starting LLM experiment: seed={seed}, model={llm_config['model']}")
    print(f"{'='*60}")

    rng = setup_environment(seed)

    # --- Setup world (reuse core components) ---
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

    is_ergodic = check_ergodicity(P_world)
    print(f"  World ergodic: {is_ergodic}")

    world = MarkovWorld(
        transition_matrix=P_world,
        cluster_ids=cluster_ids,
        noise_level=config.noise_level if config.use_noisy_observations else 0.0,
        rng=rng,
    )
    world.reset()

    # --- Setup trace store ---
    trace_store = TraceStore(max_size=config.n_steps)

    # --- Run world to generate traces ---
    print(f"  Generating {config.n_steps} trace steps...")
    for _ in range(config.n_steps):
        true_state, observed_state, cluster_id = world.step()
        trace_store.append(true_state, observed_state, cluster_id)
    print(f"  Generated {len(trace_store)} traces.")

    # --- Setup LLM components ---
    print(f"  Creating {llm_config['n_variants']} LLM observer variants...")
    variants = create_initial_variants(
        n_variants=llm_config["n_variants"],
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"],
        ollama_base_url=llm_config["ollama_base_url"],
        ollama_timeout=llm_config["ollama_timeout"],
        num_ctx=llm_config.get("num_ctx", 512),
        rng=rng,
    )

    # Setup fitness computer
    fitness_computer = FitnessComputer(
        embedding_model_name=llm_config["embedding_model"],
        use_embeddings=llm_config.get("use_embeddings", True),
    )

    # Setup evolution engine
    evolution = PromptEvolutionEngine(
        fitness_computer=fitness_computer,
        rng=rng,
        selection_top_k=llm_config["selection_top_k"],
        selection_bottom_k=llm_config["selection_bottom_k"],
        mutation_rate=llm_config["mutation_rate"],
        fragment_length=llm_config["fragment_length"],
    )

    # Setup analyzer
    analyzer = LLMAnalyzer(fitness_computer=fitness_computer)

    # Setup visualizer
    viz = LLMVisualizer(config, output_dir, f"{config.prefix}_seed{seed}")

    # Get cluster sequence for fitness computation
    cluster_sequence = trace_store.get_cluster_sequence()

    # --- Evolutionary loop ---
    n_generations = llm_config["generations"]
    n_eval_fragments = llm_config.get("n_eval_fragments", 10)

    print(f"  Running {n_generations} generations...")
    start_time = time.time()

    for gen in range(n_generations):
        # Sample new trace fragments as the world keeps running
        # (advance the world to get fresh data)
        for _ in range(config.n_steps // n_generations):
            true_state, observed_state, cluster_id = world.step()
            trace_store.append(true_state, observed_state, cluster_id)

        # Run one evolution step
        variants, evo_log = evolution.evolve(
            variants=variants,
            trace_store=trace_store,
            cluster_sequence=cluster_sequence,
            n_eval_fragments=n_eval_fragments,
        )

        # Analyze generation
        gen_metrics = analyzer.analyze_generation(variants, gen)

        # Log progress
        elapsed = time.time() - start_time
        print(f"    Gen {gen:3d}/{n_generations} | "
              f"Fitness: {evo_log['mean_fitness']:.4f} ± {evo_log['std_fitness']:.4f} | "
              f"Max: {evo_log['max_fitness']:.4f} | "
              f"Time: {elapsed:.1f}s")

    # --- Final analysis ---
    print(f"\n  Computing final metrics...")

    # Trace fragments for analysis
    analysis_fragments = []
    for _ in range(20):
        if len(trace_store) > llm_config["fragment_length"]:
            start = rng.integers(0, len(trace_store) - llm_config["fragment_length"])
            frag = trace_store.get_observed_slice(start, start + llm_config["fragment_length"])
            analysis_fragments.append(frag)
        else:
            analysis_fragments.append(trace_store.get_observed_sequence()[:llm_config["fragment_length"]].tolist())

    # --- Save outputs ---
    print(f"  Saving outputs to {output_dir}...")

    # Save configs
    config.to_json(output_dir / f"{config.prefix}_seed{seed}_config.json")
    llm_config_path = output_dir / f"{config.prefix}_seed{seed}_llm_config.json"
    with open(llm_config_path, "w") as f:
        json.dump(llm_config, f, indent=2, default=str)

    # Save traces
    trace_store.to_csv(output_dir / f"{config.prefix}_seed{seed}_traces.csv")
    np.savetxt(
        output_dir / f"{config.prefix}_seed{seed}_cluster_ids.csv",
        cluster_ids,
        delimiter=",",
        fmt="%d",
    )

    # Save metrics and analysis
    analyzer.save_metrics(
        output_dir=output_dir,
        prefix=f"{config.prefix}_seed{seed}",
        variants=variants,
        evolution_log=evolution.get_evolution_log(),
        trace_fragments=analysis_fragments,
        cluster_sequence=cluster_sequence,
    )

    # --- Generate plots ---
    print(f"  Generating plots...")
    gen_df = analyzer.to_dataframe()
    variant_states = [v.to_dict() for v in variants]
    viz.plot_all(
        generation_metrics=gen_df,
        variant_states=variant_states,
    )
    print(f"  Plots saved.")

    elapsed = time.time() - start_time
    print(f"\n  Experiment complete in {elapsed:.1f}s")
    print(f"  Final mean fitness: {evolution.log[-1]['mean_fitness']:.4f}" if evolution.log else "")

    return {
        "mean_fitness": evolution.log[-1]["mean_fitness"] if evolution.log else 0.0,
        "max_fitness": evolution.log[-1]["max_fitness"] if evolution.log else 0.0,
        "n_generations": n_generations,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TraceReality LLM Observer Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # World parameters
    parser.add_argument("--n-states", type=int, default=40, help="Number of states (default: 40)")
    parser.add_argument("--n-clusters", type=int, default=4, help="Number of clusters (default: 4)")
    parser.add_argument("--n-steps", type=int, default=10000, help="World trace steps (default: 10000)")
    parser.add_argument("--noise-level", type=float, default=0.05, help="Observation noise (default: 0.05)")
    parser.add_argument("--no-noisy-observations", action="store_true", help="Disable observation noise")

    # LLM parameters
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b",
                        help="Ollama model (default: qwen2.5:1.5b). "
                             "Use --list-models to see available. "
                             "2GB VRAM -> qwen2.5:1.5b (1GB) or qwen2.5:3b (1.9GB). "
                             "llama3.1:8b (4.9GB) WILL NOT FIT — spills to CPU.")
    parser.add_argument("--list-models", action="store_true",
                        help="List available Ollama models and exit")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max response tokens (default: 256)")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--num-ctx", type=int, default=512,
                        help="LLM context window size in tokens (default: 512). "
                             "Lower = less GPU compute per call. 512 is fine for short traces.")
    parser.add_argument("--ollama-timeout", type=int, default=120, help="Ollama request timeout (default: 120)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="Embedding model for fitness (default: all-MiniLM-L6-v2)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Use heuristic features instead of embeddings")

    # Evolution parameters
    parser.add_argument("--n-variants", type=int, default=4, help="Number of LLM variants (default: 4) — each variant × eval fragments = total LLM calls per generation. Keep low on laptops.")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations (default: 5)")
    parser.add_argument("--mutation-rate", type=float, default=0.3, help="Prompt mutation rate (default: 0.3)")
    parser.add_argument("--selection-top-k", type=int, default=2, help="Keep top K variants (default: 2)")
    parser.add_argument("--selection-bottom-k", type=int, default=1, help="Discard bottom K variants (default: 1)")
    parser.add_argument("--fragment-length", type=int, default=6, help="Trace fragment length (default: 6)")
    parser.add_argument("--n-eval-fragments", type=int, default=3,
                        help="Fragments per evaluation (default: 3). Total LLM calls per gen = variants × eval_fragments.")

    # Experiment control
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds (default: 42)")
    parser.add_argument("--prefix", type=str, default="llm_exp", help="Output prefix (default: llm_exp)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory (default: outputs)")

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        try:
            client = OllamaClient(base_url=args.ollama_base_url, timeout=10)
            models = client.list_models()
            if not models:
                print("No models found. Pull one with: ollama pull <model>")
                return
            print(f"\nAvailable Ollama models ({len(models)} found):")
            print(f"  {'NAME':<25} {'SIZE':<12} {'PARAMETERS':<15} {'QUANT'}")
            print(f"  {'-'*25} {'-'*12} {'-'*15} {'-'*10}")
            for m in sorted(models, key=lambda x: -x.get("size", 0)):
                name = m.get("name", "?")
                size = m.get("size", 0)
                details = m.get("details", {})
                param_size = details.get("parameter_size", "?")
                quant = details.get("quantization_level", "?")
                size_str = f"{size / 1e9:.1f}GB" if size else "?"
                print(f"  {name:<25} {size_str:<12} {param_size:<15} {quant}")
            print("\nUse --model <name> to select a model (e.g. --model llama3.1:8b)")
        except RuntimeError as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running (ollama serve)")
        return

    # Build base config (world parameters)
    config = ExperimentConfig(
        n_states=args.n_states,
        n_clusters=args.n_clusters,
        n_steps=args.n_steps,
        noise_level=args.noise_level,
        use_noisy_observations=not args.no_noisy_observations,
        seeds=args.seeds,
        prefix=args.prefix,
    )

    # LLM-specific config
    llm_config = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "ollama_base_url": args.ollama_base_url,
        "ollama_timeout": args.ollama_timeout,
        "num_ctx": args.num_ctx,
        "embedding_model": args.embedding_model,
        "use_embeddings": not args.no_embeddings,
        "n_variants": args.n_variants,
        "generations": args.generations,
        "mutation_rate": args.mutation_rate,
        "selection_top_k": args.selection_top_k,
        "selection_bottom_k": args.selection_bottom_k,
        "fragment_length": args.fragment_length,
        "n_eval_fragments": args.n_eval_fragments,
    }

    # Setup output directory
    output_dir = Path(args.output_dir) / args.prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print experiment summary
    print(f"\n{'='*60}")
    print(f"TraceReality LLM Observer Experiment")
    print(f"{'='*60}")
    print(f"  World: {config.n_states} states in {config.n_clusters} clusters, {config.n_steps} steps")
    print(f"  Model: {llm_config['model']} (temp={llm_config['temperature']})")
    print(f"  Variants: {llm_config['n_variants']} for {llm_config['generations']} generations")
    print(f"  Evolution: top {llm_config['selection_top_k']}, bottom {llm_config['selection_bottom_k']}")
    print(f"  Mutation rate: {llm_config['mutation_rate']}")
    print(f"  Embeddings: {llm_config['use_embeddings']} ({llm_config['embedding_model']})")
    print(f"  Seeds: {config.seeds}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Check Ollama availability
    check_ollama(llm_config["model"])

    # Save configs
    config.to_json(output_dir / f"{args.prefix}_config.json")
    with open(output_dir / f"{args.prefix}_llm_config.json", "w") as f:
        json.dump(llm_config, f, indent=2, default=str)

    # Run experiments for each seed
    all_results = []
    for seed in config.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        metrics = run_llm_experiment(config, llm_config, seed, seed_dir)
        all_results.append(metrics)

    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"Summary Across Seeds")
        print(f"{'='*60}")
        for key in ["mean_fitness", "max_fitness"]:
            values = [r[key] for r in all_results]
            print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()