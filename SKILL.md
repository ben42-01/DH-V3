# TraceReality Analysis Skill

A concise reference for analyzing TraceReality core simulation results. Use this instead of reading the full codebase.

---

## 1. Quick Start — Running Experiments

```bash
# Basic run (80 states, 8 clusters, 40 observers, 100k steps, 1 seed)
python -m tracereality.experiments.main_core --n-states 80 --n-clusters 8 --n-observers 40 --n-steps 100000 --seeds 1 --prefix my_exp

# With larger population and evolution tuning
python -m tracereality.experiments.main_core --n-states 80 --n-clusters 8 --n-observers 100 --selection-top-k 50 --selection-bottom-k 50 --n-steps 100000 --seeds 1 --prefix my_exp

# Multiple seeds for statistical significance
python -m tracereality.experiments.main_core --n-states 80 --n-clusters 8 --n-observers 40 --n-steps 100000 --seeds 1 2 3 4 5 --prefix my_exp

# Ablation: no evolution
python -m tracereality.experiments.main_core --n-states 80 --no-evolution --prefix ablation_no_evo

# Ablation: count-based (no neural networks)
python -m tracereality.experiments.main_core --n-states 80 --mode count --prefix count_baseline
```

### Key CLI Parameters

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--n-states` | 80 | Number of world states |
| `--n-clusters` | 8 | Hidden cluster structure |
| `--n-observers` | 40 | Population size |
| `--n-steps` | 100000 | Simulation length |
| `--selection-interval` | 5000 | Steps between evolution events. Generations = n_steps / selection_interval |
| `--selection-top-k` | 10 | Best observers kept per generation |
| `--selection-bottom-k` | 10 | Worst observers replaced per generation |
| `--mutation-rate-mean` | 0.01 | Mean mutation strength |
| `--mutation-rate-std` | 0.005 | Mutation strength diversity |
| `--no-evolution` | false | Disable all evolution |
| `--mode` | "nn" | "nn" (neural) or "count" (frequency table) |
| `--seeds` | [42] | Random seeds for reproducibility |

### Generations Formula
```
generations = n_steps / selection_interval
```
Example: `--n-steps 100000 --selection-interval 2000` → 50 generations.

---

## 2. Output File Layout

Every run with prefix `my_exp` and seed `1` produces:

```
outputs/my_exp/
  my_exp_config.json              # Full experiment config
  my_exp_summary.json             # Aggregate across seeds (only if >1 seed)
  seed_1/
    my_exp_seed1_config.json      # Per-seed config copy
    my_exp_seed1_world_matrix.csv # True transition matrix (80×80)
    my_exp_seed1_cluster_ids.csv  # True cluster assignments
    my_exp_seed1_traces.csv       # Full step-by-step trace log
    my_exp_seed1_evolution_log.csv # Evolution generation log
    my_exp_seed1_observer_metrics.csv   # Per-observer metrics
    my_exp_seed1_population_metrics.csv # Population metrics over time
    my_exp_seed1_metrics_summary.csv    # One-row final summary
    my_exp_seed1_info_theoretic.json    # Expected KL divergence metrics
    my_exp_seed1_summary_technical.md   # ← READ THIS FIRST (detailed tech report)
    my_exp_seed1_summary_explained.md   # ← READ THIS SECOND (plain English)
    *.png / *.pdf / *.meta.json         # Plots
```

---

## 3. Reading Results — The Two Summary Files

### `*_summary_technical.md` (auto-generated)
Contains everything needed for analysis:

- **Section 1:** Full experiment configuration table
- **Section 2:** Final population metrics table (accuracy, KL, fitness, diversity, stability)
- **Section 3:** Expected KL divergence
- **Section 4:** Evolution dynamics narrative (generations, fitness trajectory, variance)
- **Section 5:** Observer population diversity stats
- **Section 6:** Metric trajectories over time
- **Section 7:** Technical interpretation (above-chance prediction, world model quality, etc.)

### `*_summary_explained.md` (auto-generated)
Plain-language version for non-experts:
- What did we do? (accessible explanation)
- What happened? (accuracy, world model, evolution, diversity)
- What does this mean?
- Key takeaway

---

## 4. Core Metrics & Interpretation

### Accuracy
- **Random baseline** = 1 / n_states (e.g. 1/80 = 1.25%)
- **Above-chance** = > 2× random baseline → observers learned real structure
- **15%+** on 80 states = 12× better than random → strong learning

### KL Divergence (World ‖ Observer)
- Measures how different the observers' internal model is from the true world
- **< 0.05:** Excellent — models closely match reality
- **0.05 – 0.5:** Moderate — broad patterns captured, finer details missed
- **> 0.5:** Poor — observers don't understand the world well
- **Trend:** should decrease over time if learning is working

### Fitness
- Higher = better predictors survive
- **Trend increasing** = evolution is working
- **Flat** = population converged or evolution is ineffective
- Compare first generation vs last generation

### Pairwise KL (Observer Diversity)
- Measures how different observers are from each other
- **High** = diverse population (good for evolution to have material to select on)
- **Low** = homogeneous population (may indicate convergence or successful selection)
- Can be tracked over time: does diversity increase or decrease with evolution?

### Expected KL Divergence (Model Mismatch)
- Stationary-weighted average KL(p_true(·|i) ‖ p_pred(·|i)) across all states
- **< 0.05:** Excellent — models closely match reality
- **0.05 – 0.5:** Moderate — broad patterns captured, finer details missed
- **> 0.5:** Poor — observers don't understand the world well
- Closely related to the mean KL(World ‖ Observer) metric; both converge for well-sampled stationary distributions
- **Note:** This was previously (incorrectly) labeled as "mutual information" in older runs

### Cluster ARI (Adjusted Rand Index)
- How well observers recovered the hidden cluster structure
- **> 0.7:** Excellent cluster recovery
- **0.3 – 0.7:** Partial recovery
- **< 0.3:** Poor recovery (or no recovery)
- Only computed when cluster metrics are available in population_metrics.csv

### Param Variance (Stability)
- Variance of observer parameters over time
- **~0:** Very stable (parameters barely changing) — may indicate convergence
- **> 0:** Parameters still adapting

---

## 5. CSV Data Files — Column Breakdown

### `population_metrics.csv` (sampled every `analyze_every` steps)
| Column | Meaning |
|--------|---------|
| step | Simulation step |
| n_observers | Population size |
| mean_accuracy / std_accuracy | Population prediction accuracy |
| max_accuracy / min_accuracy | Best/worst observer accuracy |
| mean_kl_world / std_kl_world | Average observer vs world model mismatch |
| mean_fitness / std_fitness | Population fitness |
| mean_param_variance / std_param_variance | Parameter stability |
| mean_mutation_rate | Average mutation rate |
| mean_pairwise_kl / std_pairwise_kl | Observer-to-observer diversity |
| mean_cluster_ari / std_cluster_ari | Cluster recovery (if available) |

### `observer_metrics.csv` (final state per observer)
| Column | Meaning |
|--------|---------|
| observer_id | Unique ID |
| avg_accuracy / recent_accuracy | Prediction accuracy |
| kl_divergence_world | Model vs world mismatch |
| param_variance | Parameter stability |
| current_fitness / avg_fitness | Fitness scores |
| mutation_rate | How much weights are perturbed |
| n_train_steps | How many training updates this observer received |
| cluster_ari / cluster_nmi | Cluster recovery metrics (if available) |

### `evolution_log.csv` (one row per generation)
| Column | Meaning |
|--------|---------|
| generation | Generation number |
| mean_fitness / std_fitness | Population fitness stats |
| max_fitness / min_fitness | Best/worst in population |
| n_survivors | Number kept |
| n_discarded | Number replaced |
| mutation_rates | List of all mutation rates (JSON-like) |

---

## 6. Common Analysis Patterns

### A. "Did evolution help?"
1. Compare `*_summary_technical.md` Section 4 (evolution dynamics)
2. Check if fitness increased from first to last generation
3. Check if accuracy improved more than in no-evolution baseline
4. Run with `--no-evolution` flag and compare final accuracy/KL

### B. "Are clusters denser?"
1. Look at `*_summary_technical.md` Section 2 for Cluster ARI (if available)
2. If not available, load `observer_metrics.csv` and check `cluster_ari` column
3. Compare cluster_recovery plot (PNG)
4. Run with higher `--n-observers` (e.g. 100) and `--selection-top-k 50` to see if more observers → better recovery

### C. "Did the population converge or diversify?"
1. Check **Pairwise KL** in Section 2: increasing = diversifying, decreasing = converging
2. Check **Param variance**: ~0 = very stable, higher = adapting
3. Look at Section 6 (Metric Trajectories) for pairwise KL trend
4. Compare mutation_rate range in Section 5

### D. "Compare two experiments"
1. Run both with same seeds
2. Compare their `*_summary_technical.md` Section 2 side by side
3. Compare the metrics_summary.csv final row
4. For formal stats, use the `*_summary_explained.md` narrative differences

### E. "Is the result statistically significant?"
1. Run with `--seeds 1 2 3 4 5` (at least 3-5 seeds)
2. The aggregate `*_summary.json` shows mean ± std across seeds
3. Compare mean values accounting for std overlap
4. More seeds = more confidence

---

## 7. Quick Analysis Snippets

### Load and plot accuracy over time
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/my_exp/seed_1/my_exp_seed1_population_metrics.csv")
plt.plot(df["step"], df["mean_accuracy"], label="Mean Accuracy")
plt.fill_between(df["step"],
                 df["mean_accuracy"] - df["std_accuracy"],
                 df["mean_accuracy"] + df["std_accuracy"],
                 alpha=0.3)
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

### Compare two experiments' final metrics
```python
import pandas as pd

exp1 = pd.read_csv("outputs/exp_a/seed_1/exp_a_seed1_metrics_summary.csv")
exp2 = pd.read_csv("outputs/exp_b/seed_1/exp_b_seed1_metrics_summary.csv")
comparison = pd.concat([exp1, exp2], keys=["exp_a", "exp_b"])
print(comparison)
```

### Load and inspect evolution log
```python
import pandas as pd

evo = pd.read_csv("outputs/my_exp/seed_1/my_exp_seed1_evolution_log.csv")
print(f"Generations: {len(evo)}")
print(f"Fitness: {evo['mean_fitness'].iloc[0]:.4f} → {evo['mean_fitness'].iloc[-1]:.4f}")
print(f"Best ever: {evo['max_fitness'].max():.4f}")
```

### Load observer metrics and find best/worst
```python
import pandas as pd

obs = pd.read_csv("outputs/my_exp/seed_1/my_exp_seed1_observer_metrics.csv")
best = obs.loc[obs["avg_accuracy"].idxmax()]
worst = obs.loc[obs["avg_accuracy"].idxmin()]
print(f"Best: observer {best['observer_id']} — acc={best['avg_accuracy']:.4f}, kl={best['kl_divergence_world']:.4f}")
print(f"Worst: observer {worst['observer_id']} — acc={worst['avg_accuracy']:.4f}, kl={worst['kl_divergence_world']:.4f}")
```

---

## 8. Key Files to Analyze (by an LLM Agent)

When asked to "analyze the results" of a TraceReality experiment, an LLM agent should:

1. **Read** `*_summary_technical.md` — the single most informative file
2. **Read** `*_summary_explained.md` — for the plain-language interpretation
3. **If more detail needed:**
   - `population_metrics.csv` — full time-series data
   - `observer_metrics.csv` — per-observer breakdown
   - `evolution_log.csv` — generation-by-generation evolution
   - `info_theoretic.json` — expected KL divergence details
4. **If visual inspection needed:**
   - `*_accuracy_vs_time.png`
   - `*_divergence_vs_time.png`
   - `*_cluster_recovery.png` (if clusters were analyzed)
   - `*_evolution_fitness.png`
   - `*_observer_heatmaps.png` — observer vs world transition matrices

**Do NOT read:**
- `traces.csv` (massive — millions of rows)
- `world_matrix.csv` (dense 80×80 matrix — use the heatmap instead)
- Source code files (use this SKILL.md instead)

---

## 9. Project Architecture (Minimal)

```
src/tracereality/
  config.py            # All experiment parameters
  state_space.py       # Build world transition matrix with clusters
  world.py             # Markov world simulator (step function)
  trace_store.py       # Logs all traces to CSV
  core/
    observer.py        # Neural network predictor agent
    learner.py         # Training logic (online learning)
    evolution.py       # Selection + mutation + replication
    analyzer.py        # All metrics computation
    summarizer.py      # Generates the two Markdown summary files
    visualizer.py      # Publication-quality plots
  experiments/
    main_core.py       # CLI entry point for core experiments
```

---

## 10. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Accuracy ~= 1/n_states | Not enough steps | Increase `--n-steps` |
| Fitness never improves | Evolution disabled or too weak | Check `--no-evolution`, lower selection interval |
| KL divergence stays high | Observers not learning world structure | Increase `--n-observers` or `--hidden-dim` |
| Cluster ARI not in output | Cluster metrics not computed | Normal — only appears when `analyze_every` triggers cluster recovery computation |
| Evolution log file is huge | Selection interval too small | Increase `--selection-interval` (generations = n_steps / interval) |
| "World is not ergodic" warning | Random matrix state | Usually harmless; rare with proper clustering params |
| Expected KL ~= mean KL(World‖Observer) | Normal — these two metrics are closely related and converge for well-sampled stationary distributions | Use either metric; both measure the same model mismatch |

---

*Created for TraceReality — for LLM agents and humans to analyze evolutionary simulation results without reading the full codebase.*