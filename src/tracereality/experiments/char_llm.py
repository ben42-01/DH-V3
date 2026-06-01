"""Character-Level LLM Observer — perceptual interface emergence in token-space.

Instead of asking the LLM to "describe" or "comment" on trace fragments
in natural language (which activates billions of pre-trained linguistic
attractors), we:

1. Encode each world state as a single ASCII character (e.g. !"#$%...)
2. Feed trace fragments as raw character strings
3. Constrain the LLM to output **one character only**
4. Start from the "I" attractor — the LLM outputs "I" for everything,
   treating the world as a single uniform interface
5. Prompt-evolve toward output characters that cluster with true
   world clusters

This strips away ALL pre-trained linguistic structure. The LLM cannot
rely on grammar, semantics, or narrative patterns. It must forge a
genuinely new mapping from input tokens to output tokens — the same
problem the core NN observers solve from scratch.

Usage:
    uv run python -m tracereality.experiments.char_llm \\
        --model qwen2.5:1.5b \\
        --n-states 80 \\
        --n-clusters 8 \\
        --n-variants 8 \\
        --generations 10 \\
        --no-embeddings

Hypothesis (H1):
    Under evolutionary pressure, the LLM will transition from outputting
    a uniform character ("I") to outputting character-groups that correlate
    with the hidden cluster structure — demonstrating emergence of a
    perceptual interface from raw character processing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tracereality.config import ExperimentConfig
from tracereality.state_space import (
    assign_clusters,
    build_clustered_transition_matrix,
    check_ergodicity,
)
from tracereality.world import MarkovWorld
from tracereality.trace_store import TraceStore
from tracereality.llm.ollama_client import OllamaClient

# ---------------------------------------------------------------------------
# Character encoding: map world states [0..N-1] ↔ printable ASCII chars
# We skip space, use ! (33) through ~ (126) = 94 possible chars.
# ---------------------------------------------------------------------------

def state_to_char(state: int) -> str:
    """Map a state index to a single printable ASCII character."""
    return chr(33 + state)  # 33 = '!', 34 = '"', ...

def char_to_state(char: str) -> int:
    """Reverse: character → state index."""
    return ord(char) - 33

def trace_to_chars(trace: List[int]) -> str:
    """Convert a list of state indices to a compact character string."""
    return "".join(state_to_char(s) for s in trace)

ALLOWED_OUTPUT_CHARS = set(chr(i) for i in range(33, 127))  # ! through ~


class CharLLMObserver:
    """An LLM observer that outputs a single character.

    The observer sees trace fragments as raw character strings
    (e.g. "!@#$%") and must emit one character in response.
    The output character *is* the perceptual interface — it
    represents how the observer "sees" the trace.

    Fitness is measured by how well output characters cluster
    according to the true world cluster structure.
    """

    def __init__(
        self,
        observer_id: int,
        system_message: str,
        output_character: str = "I",
        model: str = "qwen2.5:1.5b",
        temperature: float = 0.7,
        max_tokens: int = 8,
        ollama_base_url: str = "http://localhost:11434",
        ollama_timeout: int = 120,
        num_ctx: int = 512,
    ):
        self.id = observer_id
        self.system_message = system_message
        self.output_character = output_character  # the "interface" char
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OllamaClient(
            base_url=ollama_base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=ollama_timeout,
            num_ctx=num_ctx,
        )

        # History
        self.output_history: List[str] = []   # raw LLM outputs
        self.extracted_chars: List[str] = []  # single characters extracted
        self.fitness_history: List[float] = []

    def respond(self, trace_chars: str) -> str:
        """Ask the LLM to output a single character in response to a trace.

        Args:
            trace_chars: Compact character string, e.g. "!@#$%"

        Returns:
            The single output character (first printable char from LLM).
        """
        prompt = (
            f"Sequence: {trace_chars}\n"
            f"Emit one character:"
        )
        raw = self.client.generate(
            system_message=self.system_message,
            user_prompt=prompt,
        )
        self.output_history.append(raw)

        # Extract first allowed character from the response
        char = self._extract_first_char(raw)
        self.extracted_chars.append(char)
        return char

    def _extract_first_char(self, text: str) -> str:
        """Extract the first printable ASCII character from LLM output."""
        for ch in text.strip():
            if ch in ALLOWED_OUTPUT_CHARS:
                return ch
        # Fallback: return the configured output character
        return self.output_character

    def record_fitness(self, fitness: float) -> None:
        self.fitness_history.append(fitness)

    @property
    def current_fitness(self) -> float:
        return self.fitness_history[-1] if self.fitness_history else 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "system_message": self.system_message,
            "output_character": self.output_character,
            "model": self.model,
            "fitness_history": self.fitness_history,
            "sample_outputs": self.output_history[-10:] if self.output_history else [],
            "sample_chars": self.extracted_chars[-20:] if self.extracted_chars else [],
        }


def compute_char_fitness(
    output_chars: List[str],
    fragments: List[List[int]],
    cluster_sequence: np.ndarray,
    use_heuristic: bool = True,
) -> float:
    """Compute fitness: how well do output chars cluster with true clusters?

    Maps each unique output char to a "predicted cluster" and compares
    to the true cluster labels via ARI.

    Args:
        output_chars: Single characters from LLM, one per fragment.
        fragments: The trace fragments (state lists) that generated each.
        cluster_sequence: True cluster IDs for each step.

    Returns:
        ARI score in [0, 1], higher = better alignment.
    """
    from sklearn.metrics import adjusted_rand_score

    if len(set(output_chars)) < 2 or len(output_chars) < 2:
        return 0.0

    # Get true cluster for each fragment (majority vote)
    true_labels = []
    for frag in fragments:
        if not frag:
            true_labels.append(0)
        else:
            clusters = [cluster_sequence[s] for s in frag if s < len(cluster_sequence)]
            if not clusters:
                true_labels.append(0)
            else:
                true_labels.append(int(np.bincount(clusters).argmax()))

    true_labels = np.array(true_labels)

    if len(np.unique(true_labels)) < 2:
        return 0.0

    # Map output chars to numeric labels
    unique_chars = list(set(output_chars))
    char_to_label = {c: i for i, c in enumerate(unique_chars)}
    pred_labels = np.array([char_to_label[c] for c in output_chars])

    if len(np.unique(pred_labels)) < 2:
        return 0.0

    ari = adjusted_rand_score(true_labels, pred_labels)
    return max(0.0, float(ari))


def run_char_experiment(
    config: ExperimentConfig,
    llm_config: dict,
    seed: int,
    output_dir: Path,
) -> Dict:
    """Run a single character-level LLM observer experiment."""
    print(f"\n{'='*60}")
    print(f"Char-LLM experiment: seed={seed}, model={llm_config['model']}")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    # --- Build world ---
    print(f"  Building world: {config.n_states} states, {config.n_clusters} clusters")
    cluster_ids = assign_clusters(config.n_states, config.n_clusters, rng)
    P_world = build_clustered_transition_matrix(
        config.n_states, cluster_ids,
        self_loop_bias=config.self_loop_bias,
        within_cluster_bias=config.within_cluster_bias,
        between_cluster_noise=config.between_cluster_noise,
        rng=rng,
    )
    world = MarkovWorld(P_world, cluster_ids, noise_level=0.0, rng=rng)
    world.reset()

    # --- Generate trace ---
    trace_store = TraceStore(max_size=config.n_steps)
    print(f"  Generating {config.n_steps} trace steps...")
    for _ in range(config.n_steps):
        true_state, observed_state, cluster_id = world.step()
        trace_store.append(true_state, observed_state, cluster_id)
    print(f"  Generated {len(trace_store)} traces.")
    cluster_sequence = trace_store.get_cluster_sequence()

    # --- Create character-LLM observers ---
    n_variants = llm_config["n_variants"]
    print(f"  Creating {n_variants} char-LLM observers (starting from 'I')...")

    # Base system message: minimal, no linguistic framing
    base_system = "You see characters. Output one character."

    # Slight variations in the system message
    variants = []
    system_variants = [
        base_system,
        "You see symbols. Emit a single character.",
        "Observe the characters. Reply with one character.",
        "Sequence of symbols. Output one character.",
        "Characters appear. Respond with one character.",
        "Look at the string. Say one character.",
    ]

    for i in range(n_variants):
        sys_msg = system_variants[i % len(system_variants)]
        # Each variant starts with a different "interface" character
        # Some start with "I", some with random chars to seed diversity
        if i < 3:
            start_char = "I"  # The uniform interface hypothesis
        else:
            # Seed some diversity from the available ASCII chars
            start_char = chr(33 + (i * 7) % 94)

        obs = CharLLMObserver(
            observer_id=i,
            system_message=sys_msg,
            output_character=start_char,
            model=llm_config["model"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
            ollama_base_url=llm_config["ollama_base_url"],
            ollama_timeout=llm_config["ollama_timeout"],
            num_ctx=llm_config.get("num_ctx", 512),
        )
        variants.append(obs)

    # --- Evolutionary loop ---
    n_generations = llm_config["generations"]
    n_eval = llm_config["n_eval_fragments"]
    fragment_len = llm_config["fragment_length"]
    mutation_rate = llm_config["mutation_rate"]

    print(f"  Running {n_generations} generations...")
    evolution_log = []
    start_time = time.time()

    for gen in range(n_generations):
        # Sample trace fragments
        fragments = []
        for _ in range(n_eval * n_variants):
            if len(trace_store) > fragment_len:
                s = rng.integers(0, len(trace_store) - fragment_len)
                frag = trace_store.get_observed_slice(s, s + fragment_len)
                fragments.append(frag)
            else:
                fragments.append(trace_store.get_observed_sequence()[:fragment_len].tolist())

        # Split fragments evenly among variants
        frags_per_variant = len(fragments) // n_variants

        # Evaluate each variant
        all_chars = []
        all_frags = []
        fitnesses = {}

        for idx, variant in enumerate(variants):
            v_frags = fragments[idx * frags_per_variant:(idx + 1) * frags_per_variant]
            if not v_frags:
                continue

            print(f"    [variant {variant.id}] Querying {len(v_frags)} fragments...",
                  end="", flush=True)
            chars = []
            for f in v_frags:
                char_str = trace_to_chars(f)
                c = variant.respond(char_str)
                chars.append(c)
                all_chars.append(c)
                all_frags.append(f)

            fitness = compute_char_fitness(chars, v_frags, cluster_sequence)
            variant.record_fitness(fitness)
            fitnesses[variant.id] = fitness

            char_set = set(chars)
            print(f" chars={''.join(sorted(char_set))} fitness={fitness:.4f}")

        # --- Evolution: select and mutate ---
        sorted_variants = sorted(variants, key=lambda v: fitnesses.get(v.id, 0.0), reverse=True)

        # Keep top half
        keep = max(2, n_variants // 2)
        survivors = sorted_variants[:keep]

        # Generate new variants by mutating system messages of survivors
        new_variants = list(survivors)
        while len(new_variants) < n_variants:
            parent = survivors[rng.integers(0, len(survivors))]
            child_id = max(v.id for v in new_variants) + 1

            # Mutate: slightly tweak the system message
            sys_words = parent.system_message.split()
            if random.random() < mutation_rate and len(sys_words) > 3:
                idx = random.randint(0, len(sys_words) - 1)
                word_swaps = {
                    "one": "single", "single": "one",
                    "character": "symbol", "symbol": "character",
                    "output": "emit", "emit": "send",
                    "respond": "reply", "reply": "respond",
                    "see": "observe", "observe": "see",
                    "Reply": "Respond", "Respond": "Reply",
                    "Output": "Emit", "Emit": "Output",
                }
                if sys_words[idx] in word_swaps:
                    sys_words[idx] = word_swaps[sys_words[idx]]
                new_sys = " ".join(sys_words)
            else:
                new_sys = parent.system_message

            obs = CharLLMObserver(
                observer_id=child_id,
                system_message=new_sys,
                output_character=parent.output_character,
                model=llm_config["model"],
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"],
                ollama_base_url=llm_config["ollama_base_url"],
                ollama_timeout=llm_config["ollama_timeout"],
                num_ctx=llm_config.get("num_ctx", 512),
            )
            new_variants.append(obs)

        variants = new_variants

        # --- Log ---
        fit_vals = list(fitnesses.values())
        log_entry = {
            "generation": gen + 1,
            "mean_fitness": float(np.mean(fit_vals)) if fit_vals else 0.0,
            "max_fitness": float(np.max(fit_vals)) if fit_vals else 0.0,
            "std_fitness": float(np.std(fit_vals)) if fit_vals else 0.0,
            "n_survivors": keep,
        }
        evolution_log.append(log_entry)

        elapsed = time.time() - start_time
        print(f"  [Gen {gen+1}/{n_generations}] Mean fitness: {log_entry['mean_fitness']:.4f}, "
              f"Max: {log_entry['max_fitness']:.4f} | {elapsed:.1f}s")

    # --- Final analysis ---
    print(f"\n  Saving results...")

    # Collect all unique characters used by final variants
    final_chars = {}
    for v in variants:
        final_chars[v.id] = {
            "system": v.system_message,
            "fitness": v.current_fitness,
            "output_chars": v.extracted_chars[-50:] if v.extracted_chars else [],
            "unique_chars": list(set(v.extracted_chars)) if v.extracted_chars else [],
        }

    # Save
    results = {
        "config": llm_config,
        "evolution_log": evolution_log,
        "final_variants": final_chars,
    }

    with open(output_dir / f"char_llm_seed{seed}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save evolution log as CSV
    pd.DataFrame(evolution_log).to_csv(
        output_dir / f"char_llm_seed{seed}_evolution.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Final mean fitness: {log_entry['mean_fitness']:.4f}")

    return {
        "mean_fitness": log_entry["mean_fitness"],
        "max_fitness": log_entry["max_fitness"],
        "n_generations": n_generations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Char-LLM: Perceptual Interface Emergence in Token Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # World
    parser.add_argument("--n-states", type=int, default=40, help="States (default: 40)")
    parser.add_argument("--n-clusters", type=int, default=4, help="Clusters (default: 4)")
    parser.add_argument("--n-steps", type=int, default=10000, help="Trace steps (default: 10000)")

    # LLM
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b", help="Ollama model")
    parser.add_argument("--list-models", action="store_true", help="List Ollama models and exit")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=8, help="Max output tokens (default: 8 — only need 1 char)")
    parser.add_argument("--num-ctx", type=int, default=512)
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--ollama-timeout", type=int, default=120)

    # Evolution
    parser.add_argument("--n-variants", type=int, default=8, help="Number of LLM variants (default: 8)")
    parser.add_argument("--generations", type=int, default=10, help="Generations (default: 10)")
    parser.add_argument("--mutation-rate", type=float, default=0.3, help="System msg mutation rate")
    parser.add_argument("--fragment-length", type=int, default=6, help="Trace fragment length (default: 6)")
    parser.add_argument("--n-eval-fragments", type=int, default=4, help="Fragments per variant per gen")

    # Output
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--prefix", type=str, default="char_llm", help="Output prefix")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    # --list-models
    if args.list_models:
        import requests
        try:
            r = requests.get(f"{args.ollama_base_url}/api/tags", timeout=10)
            models = r.json().get("models", [])
            print(f"\nAvailable Ollama models ({len(models)}):")
            for m in sorted(models, key=lambda x: -x.get("size", 0)):
                name = m["name"]
                size = m["size"] / 1e9
                print(f"  {name:<25} {size:.1f}GB")
        except Exception as e:
            print(f"Error: {e}")
        return

    output_dir = Path(args.output_dir) / args.prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        n_states=args.n_states,
        n_clusters=args.n_clusters,
        n_steps=args.n_steps,
        seeds=[args.seed],
        prefix=args.prefix,
    )

    llm_config = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_ctx": args.num_ctx,
        "ollama_base_url": args.ollama_base_url,
        "ollama_timeout": args.ollama_timeout,
        "n_variants": args.n_variants,
        "generations": args.generations,
        "mutation_rate": args.mutation_rate,
        "fragment_length": args.fragment_length,
        "n_eval_fragments": args.n_eval_fragments,
    }

    print(f"\n{'='*60}")
    print(f"Char-LLM: Perceptual Interface from Token Space")
    print(f"{'='*60}")
    print(f"  World: {config.n_states} states in {config.n_clusters} clusters")
    print(f"  Model: {llm_config['model']}")
    print(f"  Variants: {llm_config['n_variants']} × {llm_config['generations']} gens")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    run_char_experiment(config, llm_config, args.seed, output_dir)

    print(f"\nDone! Results in {output_dir}")


if __name__ == "__main__":
    main()