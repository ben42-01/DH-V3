"""Predict-Char: LLM as next-character predictor.

The NN observers work because they predict P(next_state | current_state).
No labels, no names, no "semantics" — just prediction. The clusters emerge
naturally from the world's transition structure: states that predict the
same next states ARE the same "thing."

This script tests if the LLM can do the same thing in character-space:
- Input: raw characters representing a trace window
- Predict: the next character
- No ARI, no labels, no prompt evolution
- Just ask "what comes next?" and measure accuracy

If the LLM can predict next-char better than random, its hidden states
contain genuine information about the world's structure — an "interface"
in Hoffman's sense, but forged entirely through prediction pressure.

Usage:
    uv run python -m tracereality.experiments.predict_char --model qwen2.5:1.5b
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

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


# --- Character encoding: state index → single char ---
def s2c(state: int) -> str:
    return chr(33 + state)

def trace_to_str(states: List[int]) -> str:
    return "".join(s2c(s) for s in states)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict-Char: LLM as next-character predictor"
    )
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b")
    parser.add_argument("--n-states", type=int, default=40)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--n-tests", type=int, default=50, help="Number of test predictions")
    parser.add_argument("--window", type=int, default=6, help="Context window length")
    parser.add_argument("--num-ctx", type=int, default=512)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- Build world ---
    print(f"Building world: {args.n_states} states, {args.n_clusters} clusters...")
    cluster_ids = assign_clusters(args.n_states, args.n_clusters, rng)
    P = build_clustered_transition_matrix(args.n_states, cluster_ids, rng=rng)
    world = MarkovWorld(P, cluster_ids, noise_level=0.0, rng=rng)
    world.reset()

    # --- Generate trace ---
    trace = TraceStore(max_size=args.n_steps)
    for _ in range(args.n_steps):
        true_state, obs_state, cid = world.step()
        trace.append(true_state, obs_state, cid)
    print(f"Generated {len(trace)} steps")

    # --- Setup LLM client ---
    client = OllamaClient(
        model=args.model,
        temperature=0.0,  # deterministic predictions
        max_tokens=16,
        timeout=30,
        num_ctx=args.num_ctx,
        cooldown=args.cooldown,
    )

    # --- Test predictions ---
    print(f"\nTesting {args.n_tests} next-char predictions (window={args.window})...")
    max_start = len(trace) - args.window - 1

    correct = 0
    results = []

    for i in range(args.n_tests):
        start = rng.integers(0, max_start)
        context_states = trace.get_observed_slice(start, start + args.window)
        actual_next = trace.get_observed_at(start + args.window)

        context_str = trace_to_str(context_states)
        actual_char = s2c(actual_next)

        # Ask LLM to predict next character
        prompt = f"Next char after \"{context_str}\":"
        response = client.generate(
            system_message="You are a next-character predictor. Only output the predicted character, nothing else.",
            user_prompt=prompt,
        )
        predicted = response.strip()[0] if response.strip() else "?"

        is_correct = predicted == actual_char
        if is_correct:
            correct += 1

        results.append({
            "trial": i,
            "context": context_str,
            "actual": actual_char,
            "predicted": predicted,
            "correct": is_correct,
        })

        if (i + 1) % 10 == 0 or i == 0:
            acc = correct / (i + 1)
            print(f"  {i+1:3d}/{args.n_tests} accuracy={acc:.3f} "
                  f"last: ctx='{context_str}' actual='{actual_char}' predicted='{predicted}'")

    # --- Results ---
    accuracy = correct / args.n_tests
    random_baseline = 1.0 / args.n_states

    print(f"\n{'='*50}")
    print(f"Results")
    print(f"{'='*50}")
    print(f"  Next-char accuracy: {accuracy:.4f} ({correct}/{args.n_tests})")
    print(f"  Random baseline:    {random_baseline:.4f}")
    print(f"  Improvement:        {accuracy / random_baseline:.1f}x better than random")
    print(f"  World:              {args.n_states} states, {args.n_clusters} clusters")
    print(f"  Model:              {args.model}")

    # Save
    out_dir = Path("outputs") / "predict_char"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / f"predict_char_seed{args.seed}.csv", index=False)

    # Summary
    summary = {
        "accuracy": accuracy,
        "random_baseline": random_baseline,
        "n_tests": args.n_tests,
        "n_states": args.n_states,
        "n_clusters": args.n_clusters,
        "model": args.model,
        "window": args.window,
    }
    import json
    with open(out_dir / f"predict_char_seed{args.seed}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()