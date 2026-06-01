"""
Motif Analyzer for Utility-Driven Observers.

Detects action motifs in observer policies:
  - Flip-flops:  2-state action cycles (s1 -a-> s2 -b-> s1)
  - Counters:    k-state action cycles (s1 -a-> s2 -b-> ... -k-> s1)
  - Gates:       conditional branching (same state, different actions
                 depending on cluster membership or other context)

Usage:
  python motif_analyzer.py \\
    --trace outputs/utility_exp_gen_100/seed_1/traces.csv \\
    --world-matrix outputs/utility_exp_gen_100/seed_1/world_matrix.csv \\
    --cluster-ids outputs/utility_exp_gen_100/seed_1/cluster_ids.csv \\
    --output outputs/utility_exp_gen_100/seed_1/motifs/ \\
    --n-actions 3 --n-states 80
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_trace(trace_path: str) -> pd.DataFrame:
    """Load trace CSV. Expects columns: t,true_state,...,action,..."""
    df = pd.read_csv(trace_path)
    required = {"true_state", "action"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trace CSV missing columns: {missing}. Found: {list(df.columns)}")
    return df


def load_world_matrix(path: str, n_states: int, n_actions: int = 3) -> np.ndarray:
    """Load world transition matrix from CSV.

    Handles:
      - 2D (n_states, n_states) — passive mode
      - 2D (n_states, n_states * n_actions) — flattened MDP, reshape to 3D
      - 3D (n_states, n_actions, n_states) — unflattened MDP
    Returns the 3D MDP matrix of shape (n_states, n_actions, n_states).
    """
    mat = np.loadtxt(path, delimiter=",")
    if mat.ndim == 3:
        return mat  # already (n_states, n_actions, n_states)
    if mat.shape == (n_states, n_states * n_actions):
        # Flattened MDP: reshape to (n_states, n_actions, n_states)
        return mat.reshape(n_states, n_actions, n_states)
    if mat.shape == (n_states, n_states):
        # Passive mode: repeat across actions for compatibility
        return np.repeat(mat[np.newaxis, :, :], n_actions, axis=0).transpose(1, 0, 2)
    # Fallback: assume it's already 2D and just trim
    return mat[:n_states, :n_states]


def load_cluster_ids(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.int64)


# ---------------------------------------------------------------------------
# Deterministic policy from trace (majority action per state)
# ---------------------------------------------------------------------------

def infer_policy_from_trace(df: pd.DataFrame) -> np.ndarray:
    """Infer the majority action per state from the trace.

    Returns array of shape (max_state+1,) where policy[s] = most-frequent action.
    """
    max_s = int(df["true_state"].max())
    counts = [Counter() for _ in range(max_s + 1)]
    for _, row in df.iterrows():
        s = int(row["true_state"])
        a = int(row["action"])
        counts[s][a] += 1

    policy = np.full(len(counts), -1, dtype=np.int64)
    for s, c in enumerate(counts):
        if c:
            policy[s] = c.most_common(1)[0][0]
        else:
            policy[s] = 0
    return policy


# ---------------------------------------------------------------------------
# Stochastic transition lookup (probabilistic, not argmax)
# ---------------------------------------------------------------------------

def build_stochastic_transitions(
    world_matrix_3d: np.ndarray,
    n_actions: int,
    cluster_ids: np.ndarray,
) -> Dict[int, Dict[int, List[Tuple[int, float, int]]]]:
    """Build a probabilistic next-state lookup from the 3D MDP matrix.

    Returns: transitions[state][action] = [(next_state, prob, next_cluster), ...]
    Sorted by probability descending, with zeros removed.
    """
    n_states = world_matrix_3d.shape[0]
    transitions: Dict[int, Dict[int, List[Tuple[int, float, int]]]] = {}

    for s in range(n_states):
        transitions[s] = {}
        for a in range(min(n_actions, world_matrix_3d.shape[1])):
            probs = world_matrix_3d[s, a]
            entries = []
            for ns in range(n_states):
                p = float(probs[ns])
                if p > 1e-8:
                    c = int(cluster_ids[ns]) if ns < len(cluster_ids) else -1
                    entries.append((ns, p, c))
            entries.sort(key=lambda x: -x[1])
            transitions[s][a] = entries
    return transitions


def build_deterministic_lookup(
    world_matrix_3d: np.ndarray,
    n_actions: int,
    cluster_ids: np.ndarray,
) -> Dict[int, Dict[int, int]]:
    """Deterministic: most-likely next state per (state, action)."""
    lookup: Dict[int, Dict[int, int]] = {}
    for s in range(world_matrix_3d.shape[0]):
        lookup[s] = {}
        for a in range(min(n_actions, world_matrix_3d.shape[1])):
            lookup[s][a] = int(np.argmax(world_matrix_3d[s, a]))
    return lookup


# ---------------------------------------------------------------------------
# Motif detectors (trace-based, data-driven)
# ---------------------------------------------------------------------------

def find_flip_flops_from_trace(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    min_occurrences: int = 20,
) -> List[Dict]:
    """Detect flip-flops directly from trace: frequent 2-state cycles.

    Counts (s1, a, s2) triples and (s2, b, s1) triples — if both are
    frequent, it's a flip-flop.

    Returns list of dicts with: s1, s2, a1, a2, count_forward, count_backward.
    """
    # Count (prev_state, action, state) transitions
    triple_counts: Counter = Counter()
    prev_s = None
    prev_a = None
    for _, row in df.iterrows():
        s = int(row["true_state"])
        a = int(row["action"])
        if prev_s is not None:
            triple_counts[(prev_s, prev_a, s)] += 1
        prev_s = s
        prev_a = a

    # Find reciprocal pairs
    flip_flops = []
    checked = set()
    for (s1, a1, s2), cnt_fwd in triple_counts.items():
        if s1 == s2:
            continue
        if (s1, s2) in checked or (s2, s1) in checked:
            continue
        # Look for the reverse transition (s2, *, s1)
        # Find the most common action that goes s2 -> s1
        best_a2 = None
        best_cnt_bwd = 0
        for a2 in range(int(df["action"].max()) + 1):
            cnt_bwd = triple_counts.get((s2, a2, s1), 0)
            if cnt_bwd > best_cnt_bwd:
                best_cnt_bwd = cnt_bwd
                best_a2 = a2

        if cnt_fwd >= min_occurrences and best_cnt_bwd >= min_occurrences:
            flip_flops.append({
                "s1": int(s1),
                "s2": int(s2),
                "a1": int(a1),
                "a2": int(best_a2),
                "count_forward": int(cnt_fwd),
                "count_backward": int(best_cnt_bwd),
                "clusters": (
                    int(cluster_ids[s1]) if s1 < len(cluster_ids) else -1,
                    int(cluster_ids[s2]) if s2 < len(cluster_ids) else -1,
                ),
            })
            checked.add((s1, s2))

    flip_flops.sort(key=lambda x: -(x["count_forward"] + x["count_backward"]))
    return flip_flops


def find_counters_from_trace(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    min_k: int = 3,
    max_k: int = 15,
    min_edge_count: int = 10,
) -> List[Dict]:
    """Detect k-state cycles from trace by constructing a frequency graph.

    Builds a directed graph where edge weights = (state, action, next_state) counts.
    Finds cycles of length >= min_k using DFS.

    Returns list of dicts with: cycle_states, cycle_actions, min_edge_count.
    """
    # Build frequency graph: edges[s1][s2] = [(action, count), ...]
    edges: Dict[int, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    prev_s = None
    prev_a = None
    for _, row in df.iterrows():
        s = int(row["true_state"])
        a = int(row["action"])
        if prev_s is not None:
            edges[prev_s][s][a] += 1
        prev_s = s
        prev_a = a

    # Convert to adjacency: for each s1, get the best (action, next_state) pair
    adj: Dict[int, List[Tuple[int, int, int]]] = {}
    for s1 in edges:
        adj[s1] = []
        for s2 in edges[s1]:
            best_a, best_cnt = edges[s1][s2].most_common(1)[0]
            if best_cnt >= min_edge_count:
                adj[s1].append((s2, int(best_a), best_cnt))
        # Sort by frequency descending
        adj[s1].sort(key=lambda x: -x[2])

    # DFS for cycles
    counters = []
    visited_global = set()

    for start in range(len(adj)):
        if start in visited_global or start not in adj:
            continue
        dfs_stack = [(start, [start], [])]  # (current, path_states, path_actions)
        seen_in_path = {start: 0}

        while dfs_stack:
            curr, path_s, path_a = dfs_stack[-1]
            if curr not in adj:
                dfs_stack.pop()
                if path_s:
                    seen_in_path.pop(curr, None)
                continue

            # Find next unvisited outgoing neighbor
            found_next = False
            for nxt, act, _ in adj[curr]:
                if nxt == start and len(path_s) >= min_k and len(path_s) <= max_k:
                    # Found a cycle back to start
                    cycle_states = list(path_s)
                    cycle_actions = list(path_a)
                    counters.append({
                        "start_state": int(start),
                        "cycle_states": [int(s) for s in cycle_states],
                        "cycle_actions": [int(a) for a in cycle_actions],
                        "length": len(cycle_states),
                        "action_sequence": "-".join(str(a) for a in cycle_actions),
                        "clusters": [int(cluster_ids[s]) if s < len(cluster_ids) else -1
                                     for s in cycle_states],
                    })
                    visited_global.update(cycle_states)
                    # Pop all to end this branch
                    while dfs_stack:
                        dfs_stack.pop()
                    found_next = True
                    break
                elif nxt not in seen_in_path and nxt not in visited_global:
                    # Extend path
                    seen_in_path[nxt] = len(path_s)
                    dfs_stack.append((nxt, path_s + [nxt], path_a + [int(act)]))
                    # Remove the edge so we don't reuse it
                    adj[curr] = [(s, a, c) for s, a, c in adj[curr] if s != nxt]
                    found_next = True
                    break

            if not found_next:
                dfs_stack.pop()
                if path_s:
                    seen_in_path.pop(curr, None)

    return counters


def find_gates_from_trace(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    min_alternations: int = 10,
) -> List[Dict]:
    """Detect gate-like states from trace.

    A gate is a state where the observer takes different actions
    that lead to *different clusters*. Detected by counting:
      (state, action) -> distribution over next clusters

    Returns list of dicts with keys:
      state, actions_by_cluster, total_visits, entropy
    """
    # Count (state, action, next_cluster) from the trace itself
    # (using the *observed* next state's cluster)
    counts: Dict[int, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for _, row in df.iterrows():
        s = int(row["true_state"])
        a = int(row["action"])
        rc = int(row["cluster_id"])  # next cluster (as recorded in trace)
        counts[s][a][rc] += 1

    gates = []
    for state, action_dict in counts.items():
        if len(action_dict) < 2:
            continue

        # For each action, find the dominant next cluster
        action_cluster = {}
        for a, cluster_counter in action_dict.items():
            dominant_c, count = cluster_counter.most_common(1)[0]
            action_cluster[a] = (int(dominant_c), int(count))

        # Check if any two actions lead to different clusters
        action_items = list(action_cluster.items())
        found_gate = False
        for i in range(len(action_items)):
            for j in range(i + 1, len(action_items)):
                a1, (c1, cnt1) = action_items[i]
                a2, (c2, cnt2) = action_items[j]
                if c1 != c2 and cnt1 >= min_alternations and cnt2 >= min_alternations:
                    # This is a gate
                    gate_entry = {
                        "state": int(state),
                        "n_actions": len(action_dict),
                        "actions": {
                            str(a): {
                                "dominant_cluster": int(c),
                                "count": int(cnt),
                                "cluster_distribution": {
                                    str(rc): int(c2)
                                    for rc, c2 in action_dict[a].items()
                                },
                            }
                            for a, (c, cnt) in action_cluster.items()
                        },
                    }
                    gates.append(gate_entry)
                    found_gate = True
                    break
            if found_gate:
                break

    gates.sort(key=lambda x: -len(x["actions"]))
    return gates


# ---------------------------------------------------------------------------
# Action-cluster effect (computed from trace directly)
# ---------------------------------------------------------------------------

def compute_action_cluster_effect(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    n_actions: int,
) -> Dict[int, Dict]:
    """For each action, compute what fraction stays vs switches clusters.

    Uses the trace's *recorded* cluster_id (the next cluster after the action).
    """
    counts: Dict[int, Dict[str, int]] = {
        a: {"same": 0, "diff": 0, "total": 0} for a in range(n_actions)
    }

    for _, row in df.iterrows():
        s = int(row["true_state"])
        a = int(row["action"])
        next_c = int(row["cluster_id"])  # already recorded in trace
        current_c = int(cluster_ids[s]) if s < len(cluster_ids) else -1

        if a in counts:
            counts[a]["total"] += 1
            if next_c == current_c:
                counts[a]["same"] += 1
            else:
                counts[a]["diff"] += 1

    result = {}
    for a in range(n_actions):
        c = counts[a]
        total = c["total"]
        result[a] = {
            "same_cluster": c["same"],
            "diff_cluster": c["diff"],
            "total": total,
            "frac_same": c["same"] / total if total > 0 else 0.0,
            "frac_diff": c["diff"] / total if total > 0 else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# Per-cluster action summary
# ---------------------------------------------------------------------------

def compute_cluster_action_stats(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    n_actions: int,
) -> Dict:
    """Compute which actions are preferred in each cluster."""
    # For each cluster, count actions taken FROM states in that cluster
    cluster_action_counts: Dict[int, Counter] = defaultdict(Counter)
    for _, row in df.iterrows():
        s = int(row["true_state"])
        a = int(row["action"])
        c = int(cluster_ids[s]) if s < len(cluster_ids) else -1
        cluster_action_counts[c][a] += 1

    result = {}
    for c, counter in sorted(cluster_action_counts.items()):
        total = sum(counter.values())
        result[str(c)] = {
            "total_actions": total,
            "action_distribution": {
                str(a): round(cnt / total, 4) if total > 0 else 0.0
                for a, cnt in sorted(counter.items())
            },
            "dominant_action": int(counter.most_common(1)[0][0]),
        }
    return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_motifs(
    trace_df: pd.DataFrame,
    world_matrix: np.ndarray,
    cluster_ids: np.ndarray,
    n_actions: int,
    min_flipflop_occurrences: int = 20,
    min_counter_k: int = 3,
    max_counter_k: int = 15,
    min_counter_edge: int = 10,
    min_gate_alternations: int = 10,
) -> Dict:
    """Run all motif detectors directly from trace data.

    Unlike the earlier deterministic approach, this uses the *actual*
    observed (state, action, next_state, next_cluster) sequences from
    the trace, which already captures the stochastic dynamics.

    Args:
        trace_df: Trace dataframe with columns: true_state, action, cluster_id, ...
        world_matrix: World transition matrix (for reference, shape).
        cluster_ids: Cluster assignment per state.
        n_actions: Number of discrete actions.
        min_flipflop_occurrences: Min count for both directions of a flip-flop.
        min_counter_k: Minimum cycle length for counters.
        max_counter_k: Max cycle walk length.
        min_counter_edge: Minimum edge weight in counter graph.
        min_gate_alternations: Min occurrences for gate detection.

    Returns:
        Dictionary with all motif results.
    """
    n_states = len(cluster_ids)

    # Majority-action policy (for reference only)
    policy = infer_policy_from_trace(trace_df)

    # Build deterministic lookup (for top triples)
    # Use the 3D matrix if available, otherwise fallback
    if world_matrix.ndim == 3:
        det_lookup = build_deterministic_lookup(world_matrix, n_actions, cluster_ids)
    else:
        # 2D matrix — all actions lead to same next state
        det_lookup = {}
        for s in range(n_states):
            det_lookup[s] = {}
            for a in range(n_actions):
                det_lookup[s][a] = int(np.argmax(world_matrix[s]))

    # 1. Top (state, action, next_state) triples from trace
    triple_counts: Counter = Counter()
    for _, row in trace_df.iterrows():
        s = int(row["true_state"])
        a = int(row["action"])
        ns = det_lookup.get(s, {}).get(a, -1)
        triple_counts[(s, a, ns)] += 1
    top_triples = triple_counts.most_common(20)

    # 2. Flip-flops (data-driven from trace)
    flip_flops = find_flip_flops_from_trace(
        trace_df, cluster_ids, min_occurrences=min_flipflop_occurrences
    )

    # 3. Counters (data-driven from trace)
    counters = find_counters_from_trace(
        trace_df, cluster_ids,
        min_k=min_counter_k, max_k=max_counter_k,
        min_edge_count=min_counter_edge,
    )

    # 4. Gates (data-driven from trace)
    gates = find_gates_from_trace(
        trace_df, cluster_ids, min_alternations=min_gate_alternations,
    )

    # 5. Action → cluster effect (data-driven from trace)
    action_cluster_effect = compute_action_cluster_effect(
        trace_df, cluster_ids, n_actions,
    )

    # 6. Per-cluster action preferences
    cluster_action_stats = compute_cluster_action_stats(
        trace_df, cluster_ids, n_actions,
    )

    # Action entropy per state: how mixed is the action selection?
    state_action_entropy = {}
    state_counts = Counter()
    state_action_counts: Dict[int, Counter] = defaultdict(Counter)
    for _, row in trace_df.iterrows():
        s = int(row["true_state"])
        a = int(row["action"])
        state_action_counts[s][a] += 1
        state_counts[s] += 1

    entropies = []
    for s, total in state_counts.items():
        if total < 10:
            continue
        dist = [state_action_counts[s][a] / total for a in range(n_actions)]
        dist = [p for p in dist if p > 0]
        if len(dist) > 1:
            entropy = -sum(p * np.log2(p) for p in dist)
            entropies.append(entropy)

    mean_action_entropy = float(np.mean(entropies)) if entropies else 0.0

    return {
        "top_triples": [
            {"state": int(t[0]), "action": int(t[1]),
             "next_state": int(t[2]), "count": int(c)}
            for t, c in top_triples
        ],
        "flip_flops": flip_flops,
        "counters": counters,
        "gates": gates,
        "action_cluster_effect": action_cluster_effect,
        "cluster_action_stats": cluster_action_stats,
        "stats": {
            "n_flip_flops": len(flip_flops),
            "n_counters": len(counters),
            "n_gates": len(gates),
            "n_unique_triples": len(triple_counts),
            "action_distribution": {
                str(a): int(np.sum(policy == a)) for a in range(n_actions)
            },
            "mean_action_entropy": mean_action_entropy,
            "n_states_visited": len(state_counts),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motif Analyzer for Utility-Driven Observers",
    )
    parser.add_argument("--trace", required=True, help="Path to trace CSV")
    parser.add_argument("--world-matrix", required=True,
                        help="Path to world matrix CSV (2D or 3D)")
    parser.add_argument("--cluster-ids", required=True,
                        help="Path to cluster IDs CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-actions", type=int, default=3,
                        help="Number of discrete actions (default: 3)")
    parser.add_argument("--n-states", type=int, default=80,
                        help="Number of states (default: 80)")
    parser.add_argument("--min-flipflop-occurrences", type=int, default=20)
    parser.add_argument("--min-counter-k", type=int, default=3)
    parser.add_argument("--max-counter-k", type=int, default=15)
    parser.add_argument("--min-counter-edge", type=int, default=10)
    parser.add_argument("--min-gate-alternations", type=int, default=10)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    print("Loading data...")
    trace_df = load_trace(args.trace)
    print(f"  Trace: {len(trace_df)} rows")

    world_matrix = load_world_matrix(args.world_matrix, args.n_states)
    print(f"  World matrix shape: {world_matrix.shape}")

    cluster_ids = load_cluster_ids(args.cluster_ids)
    print(f"  Cluster IDs: {len(cluster_ids)} states, "
          f"{len(np.unique(cluster_ids))} clusters")

    policy = infer_policy_from_trace(trace_df)
    action_dist = Counter(policy)
    print(f"  Majority-action policy: {dict(sorted(action_dist.items()))}")
    print(f"  (States with no visits defaulted to action 0)")

    print("\nAnalyzing motifs...")
    motifs = analyze_motifs(
        trace_df=trace_df,
        world_matrix=world_matrix,
        cluster_ids=cluster_ids,
        n_actions=args.n_actions,
        min_flipflop_occurrences=args.min_flipflop_occurrences,
        min_counter_k=args.min_counter_k,
        max_counter_k=args.max_counter_k,
        min_counter_edge=args.min_counter_edge,
        min_gate_alternations=args.min_gate_alternations,
    )

    # --- Save ---
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    motifs_path = out_dir / "motifs.json"
    with open(motifs_path, "w") as f:
        json.dump(motifs, f, indent=2, default=str)

    triples_path = out_dir / "top_triples.csv"
    with open(triples_path, "w") as f:
        f.write("state,action,next_state,count\n")
        for t in motifs["top_triples"]:
            f.write(f"{t['state']},{t['action']},{t['next_state']},{t['count']}\n")

    summary = {
        "n_trace_rows": len(trace_df),
        "n_states": args.n_states,
        "n_clusters": len(np.unique(cluster_ids)),
        "n_actions": args.n_actions,
        "action_distribution": motifs["stats"]["action_distribution"],
        "n_flip_flops": motifs["stats"]["n_flip_flops"],
        "n_counters": motifs["stats"]["n_counters"],
        "n_gates": motifs["stats"]["n_gates"],
        "n_unique_triples": motifs["stats"]["n_unique_triples"],
        "mean_action_entropy": motifs["stats"]["mean_action_entropy"],
        "n_states_visited": motifs["stats"]["n_states_visited"],
        "top_triple": motifs["top_triples"][0] if motifs["top_triples"] else None,
        "action_cluster_effect": motifs["action_cluster_effect"],
        "cluster_action_stats": motifs["cluster_action_stats"],
    }
    summary_path = out_dir / "motif_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print ---
    print(f"\n{'='*60}")
    print("Motif Analysis Summary")
    print(f"{'='*60}")
    print(f"  Flip-flops detected:  {summary['n_flip_flops']}")
    print(f"  Counters detected:    {summary['n_counters']}")
    print(f"  Gates detected:       {summary['n_gates']}")
    print(f"  Unique (s,a,ns) triples: {summary['n_unique_triples']}")
    print(f"  States visited:       {summary['n_states_visited']}/{args.n_states}")
    print(f"  Mean action entropy:  {summary['mean_action_entropy']:.4f} bits")
    print(f"  Action dist (states): {summary['action_distribution']}")

    print(f"\n  Action → Cluster effect (from trace):")
    for a, eff in sorted(summary["action_cluster_effect"].items()):
        print(f"    Action {a}: {eff['frac_same']:.1%} same-cluster, "
              f"{eff['frac_diff']:.1%} cross-cluster "
              f"(n={eff['total']})")

    print(f"\n  Cluster → Action preferences:")
    for c, stats in sorted(summary["cluster_action_stats"].items()):
        dom_a = int(stats["dominant_action"])
        frac = max(stats["action_distribution"].values())
        print(f"    Cluster {c}: dominant action {dom_a} ({frac:.1%})")

    if motifs["flip_flops"]:
        print(f"\n  Top flip-flops:")
        for ff in motifs["flip_flops"][:5]:
            print(f"    s{ff['s1']} -a{ff['a1']}-> s{ff['s2']} -a{ff['a2']}-> s{ff['s1']} "
                  f"({ff['count_forward']}+{ff['count_backward']} times)")

    if motifs["counters"]:
        print(f"\n  Top counters:")
        for cnt in motifs["counters"][:5]:
            print(f"    {cnt['action_sequence']} ({cnt['length']} states)")

    if motifs["gates"]:
        print(f"\n  Top gates:")
        for g in motifs["gates"][:5]:
            actions_str = ", ".join(
                f"a{a}→cluster{c['dominant_cluster']}"
                for a, c in sorted(g["actions"].items())
            )
            print(f"    State {g['state']}: {actions_str}")

    if summary["top_triple"]:
        t = summary["top_triple"]
        print(f"\n  Top triple: s{t['state']} -a{t['action']}-> s{t['next_state']} "
              f"({t['count']} times)")

    print(f"\n  Results saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()