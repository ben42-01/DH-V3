"""Predict-Next: LLM as next-state predictor with prompt evolution.

Key insight: feed raw number traces, give the LLM a clear prediction
goal, and evolve prompts to maximize accuracy. This aligns with the
LLM's pre-trained pattern-recognition capabilities instead of fighting
them.

CRITICAL: All variants share ONE OllamaClient so the cooldown
is truly global — one request at a time, no parallel firing.

Usage:
    uv run python -m tracereality.experiments.predict_next \\
        --model qwen2.5:1.5b --cooldown 1.0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tracereality.state_space import assign_clusters, build_clustered_transition_matrix
from tracereality.world import MarkovWorld
from tracereality.trace_store import TraceStore
from tracereality.llm.ollama_client import OllamaClient
from tracereality.llm.openrouter_client import OpenRouterClient

BASE_PROMPTS = [
    "You are a next-state predictor for a Markov chain. Each number [0-39] is a hidden state. You see the last 6 visited states. Your job: predict the next state. Output ONLY a single integer 0-39. No explanation.",
    "You observe a stream of states from a Markov process. Given a window of 6 consecutive states, predict the 7th. Rules: (1) States that repeat tend to stay in the same cluster. (2) Adjacent states are correlated. Output ONLY the next state number.",
    "You track an evolving Markov chain. Input: 6 observed states. Output: the most likely 7th state. The world has 40 states organized into clusters. States within a cluster tend to follow each other. Output ONLY a number 0-39.",
    "You are a Markov observer. You see 6 consecutive states from a 40-state Markov chain with clustered transitions. Predict the next state. Self-loops are common — if a state repeats, it may continue. Output ONLY the predicted next state.",
    "Analyze this sequence of 6 Markov chain states [0-39] and predict the 7th. Consider: transitions are clustered, self-loops are biased high. Return ONLY the predicted next state as a single number.",
    "Markov chain predictor: given 6 states, output the next. Clusters matter: states in the same group tend to transition to each other. Output ONLY a number 0-39. No explanations or reasoning.",
]


class Predictor:
    """One LLM next-state predictor.

    All predictors share ONE OllamaClient — this is essential.
    The cooldown lives in the client, so one shared client means
    all requests are globally serialized: wait → response → wait → ...
    """

    def __init__(self, oid: int, system: str, n_states: int, client: OllamaClient):
        self.id = oid
        self.system = system
        self.n_states = n_states
        self.client = client  # Shared — cooldown is global across ALL predictors
        self.fitness: List[float] = []
        self.log: List[dict] = []

    def predict(self, trace: List[int]) -> int:
        """Predict next state. Returns -1 on parse failure."""
        s = ", ".join(str(x) for x in trace)
        raw = self.client.generate(
            system_message=self.system,
            user_prompt=f"Sequence: {s}\nPredict the next state:",
        )
        input_set = set(trace)
        candidates = []
        for tok in raw.strip().split():
            c = tok.strip(".,!?;:\"'()[]{}")
            # Skip parenthesized numbers like "(1)" from rule references
            if c.startswith("(") or c.endswith(")") or c.startswith("[") or c.endswith("]"):
                continue
            try:
                v = int(c)
                if 0 <= v < self.n_states and v not in input_set:
                    candidates.append(v)
            except ValueError:
                continue
        # Prefer the LAST candidate (models put answer at end)
        pred = candidates[-1] if candidates else -1
        self.log.append({"trace": trace, "raw": raw, "pred": pred})
        return pred

    def last_raw(self) -> str:
        """Get the most recent raw LLM response (for logging)."""
        return self.log[-1]["raw"] if self.log else ""


def mutate(msg: str, rate: float = 0.3) -> str:
    words = msg.split()
    swaps = {
        "observe": "watch", "watch": "analyze", "predict": "forecast",
        "sequence": "trace", "trace": "stream", "number": "value",
        "value": "symbol", "symbol": "state", "state": "number",
        "accurate": "precise", "next": "following", "following": "next",
    }
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in swaps and random.random() < rate:
            r = swaps[lw]
            words[i] = r if w.islower() else r.capitalize()
    return " ".join(words)


def main():
    p = argparse.ArgumentParser(description="Next-state predictor with prompt evolution")
    p.add_argument("--provider", default="ollama", choices=["ollama", "openrouter"],
                   help="LLM provider: ollama (local GPU) or openrouter (cloud API)")
    p.add_argument("--model", default="qwen2.5:1.5b",
                   help="Model name. For Ollama use e.g. qwen2.5:1.5b. "
                        "For OpenRouter use e.g. gpt-4o-mini (default).")
    p.add_argument("--list-free-models", action="store_true",
                   help="List OpenRouter free/cheap models and exit")
    p.add_argument("--n-states", type=int, default=40)
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=10000)
    p.add_argument("--n-variants", type=int, default=6)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--win", type=int, default=6, help="Context window")
    p.add_argument("--n-eval", type=int, default=10, help="Trials per variant/gen")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--cooldown", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", default="predict_next")
    p.add_argument("--verbose", action="store_true", help="Print raw LLM responses")
    args = p.parse_args()

    # Handle --list-free-models
    if args.list_free_models:
        print("\nRecommended OpenRouter models for next-token prediction:")
        for name, desc in OpenRouterClient.list_recommended_models():
            print(f"  {name:<45} {desc}")
        print("\nSet OPENROUTER_API_KEY env var and use --provider openrouter --model <name>")
        return

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    baseline = 1.0 / args.n_states

    # --- World ---
    print(f"World: {args.n_states} states, {args.n_clusters} clusters")
    cids = assign_clusters(args.n_states, args.n_clusters, rng)
    P = build_clustered_transition_matrix(args.n_states, cids, rng=rng)
    world = MarkovWorld(P, cids, noise_level=0.0, rng=rng)
    world.reset()
    ts = TraceStore(max_size=args.n_steps * 2)
    for _ in range(args.n_steps):
        s, o, c = world.step()
        ts.append(s, o, c)
    print(f"Trace: {len(ts)} steps  (random baseline: {baseline:.4f})")

    # --- ONE shared client — cooldown is GLOBAL (serializes ALL requests) ---
    if args.provider == "openrouter":
        print(f"Creating {args.n_variants} predictors with ONE shared OpenRouter client...")
        shared_client = OpenRouterClient(
            model=args.model or "gpt-4o-mini",
            temperature=args.temperature, max_tokens=16,
            timeout=30, cooldown=args.cooldown,
        )
    else:
        print(f"Creating {args.n_variants} predictors with ONE shared Ollama client...")
        shared_client = OllamaClient(
            model=args.model, temperature=args.temperature, max_tokens=16,
            timeout=30, num_ctx=512, cooldown=args.cooldown,
        )
    variants = [
        Predictor(i, BASE_PROMPTS[i % len(BASE_PROMPTS)], args.n_states, shared_client)
        for i in range(args.n_variants)
    ]

    # --- Evolution ---
    out = Path("outputs") / args.prefix
    out.mkdir(parents=True, exist_ok=True)
    evo = []
    t0 = time.time()
    all_raw_logs = []  # Collect all raw responses

    for gen in range(args.generations):
        print(f"\n--- Gen {gen+1}/{args.generations} ---")
        mx = len(ts) - args.win - 1
        fits = {}

        for v in variants:
            ok = 0
            for _ in range(args.n_eval):
                st = rng.integers(0, mx)
                frag = ts.get_observed_slice(st, st + args.win)
                actual = ts.get_observed_at(st + args.win)
                v.predict(frag)

                # Log raw response if verbose
                entry = v.log[-1]
                log_entry = {
                    "gen": gen + 1,
                    "variant_id": v.id,
                    "system": v.system[:60],
                    "trace": str(entry["trace"]),
                    "actual": actual,
                    "predicted": entry["pred"],
                    "correct": entry["pred"] == actual,
                    "raw": entry["raw"][:200],
                }
                all_raw_logs.append(log_entry)

                if args.verbose:
                    print(f"    gen={gen+1} v{v.id} trace={str(frag)[:20]}... "
                          f"actual={actual} pred={entry['pred']} "
                          f"raw='{entry['raw'][:80]}'")

                if entry["pred"] == actual:
                    ok += 1

            acc = ok / args.n_eval
            v.fitness.append(acc)
            fits[v.id] = acc
            uniq = set(x["pred"] for x in v.log[-args.n_eval:])
            print(f"  v{v.id:02d} acc={acc:.3f}  preds={sorted(uniq)[:8]}  "
                  f"'{v.system[:55]}'")

        # Select + replicate (children share the same client)
        ranked = sorted(variants, key=lambda v: fits[v.id], reverse=True)
        keep = max(2, args.n_variants // 2)
        surv = ranked[:keep]
        kids = list(surv)
        while len(kids) < args.n_variants:
            parent = surv[rng.integers(0, len(surv))]
            nid = max(v.id for v in kids) + 1
            kids.append(Predictor(nid, mutate(parent.system), args.n_states, shared_client))
        variants = kids

        vals = list(fits.values())
        e = {"gen": gen+1, "mean": float(np.mean(vals)), "max": float(np.max(vals)),
             "min": float(np.min(vals)), "std": float(np.std(vals)),
             "baseline": baseline, "x_random": float(np.max(vals) / baseline)}
        evo.append(e)
        print(f"  -> mean={e['mean']:.4f} max={e['max']:.4f} "
              f"({e['x_random']:.1f}x random)  [{time.time()-t0:.0f}s]")

    # --- Save ---
    final_fits = [v.fitness[-1] for v in variants if v.fitness]
    # Save raw prediction log
    if all_raw_logs:
        pd.DataFrame(all_raw_logs).to_csv(
            out / f"{args.prefix}_seed{args.seed}_raw.csv", index=False)

    pd.DataFrame(evo).to_csv(out / f"{args.prefix}_seed{args.seed}_evo.csv", index=False)

    results = {
        "config": vars(args),
        "baseline": baseline,
        "final_mean": float(np.mean(final_fits)) if final_fits else 0.0,
        "final_max": float(np.max(final_fits)) if final_fits else 0.0,
        "best_prompt": variants[int(np.argmax([v.fitness[-1] if v.fitness else 0 for v in variants]))].system,
    }
    with open(out / f"{args.prefix}_seed{args.seed}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Done  [{time.time()-t0:.0f}s]")
    print(f"Final max fitness: {results['final_max']:.4f} "
          f"({results['final_max']/baseline:.1f}x random)")
    print(f"Results: {out}/")


if __name__ == "__main__":
    main()