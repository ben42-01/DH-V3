---
sidebar_position: 3
title: Running Experiments
---

# Running Experiments

## Quick Start

```bash
# Basic run: 80 states, 40 observers, 100k steps, seed 1
python -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 100000 \
  --seeds 1 \
  --prefix my_first_exp
```

Results appear in: `outputs/my_first_exp/seed_1/`

---

## Common Experiment Patterns

### Pattern 1: Baseline (Single Seed)
Best for quick testing and understanding a single run:

```bash
python -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 100000 \
  --seeds 1 \
  --prefix baseline_exp
```

**Output**: One seed_1 directory with results and visualizations

---

### Pattern 2: Statistical Validation (Multiple Seeds)
Run the same configuration with different random seeds to assess stability:

```bash
python -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 100000 \
  --seeds 1 2 3 4 5 \
  --prefix stable_exp
```

**Output**: 
- `outputs/stable_exp/seed_1/`
- `outputs/stable_exp/seed_2/`
- ... (5 total)
- `outputs/stable_exp/stable_exp_summary.json` (aggregated stats)

**Analysis**: Compare mean ± std accuracy, KL, etc. across seeds to assess robustness.

---

### Pattern 3: Ablation — No Evolution
Disable evolution to see pure learning (no selection/mutation):

```bash
python -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 100000 \
  --seeds 1 2 3 \
  --no-evolution \
  --prefix no_evo_baseline
```

**Compare to**: `stable_exp` above

**Question answered**: How much does evolution help? (Difference between runs)

---

### Pattern 4: Ablation — Count-Based Baseline
Use simple frequency tables instead of neural networks:

```bash
python -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 100000 \
  --seeds 1 2 3 \
  --mode count \
  --prefix count_baseline
```

**Compare to**: Neural network runs

**Question answered**: Is the neural architecture necessary? How much does it help?

---

### Pattern 5: Scaling — More Observers
Test if larger populations learn better:

```bash
python -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 100 \
  --selection-top-k 50 \
  --selection-bottom-k 50 \
  --n-steps 100000 \
  --seeds 1 2 3 \
  --prefix more_observers_exp
```

**Compare to**: 40-observer runs

**Question answered**: Does population size matter? (Often yes: more diversity)

---

### Pattern 6: Scaling — More Steps
Test if longer simulations improve performance:

```bash
python -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 200000 \
  --seeds 1 2 3 \
  --prefix longer_exp
```

**Compare to**: 100k-step runs

**Question answered**: Are we reaching a plateau? Is there room to improve?

---

### Pattern 7: Scaling — Harder World
Test if observers can learn more complex worlds:

```bash
python -m tracereality.experiments.main_core \
  --n-states 160 \
  --n-clusters 16 \
  --n-observers 80 \
  --n-steps 200000 \
  --seeds 1 2 3 \
  --prefix harder_world_exp
```

**Compare to**: 80-state runs

**Question answered**: How does complexity scale? Does accuracy degrade? By how much?

---

### Pattern 8: Aggressive Evolution
Test evolution with stronger selection pressure:

```bash
python -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --selection-top-k 20 \
  --selection-bottom-k 20 \
  --mutation-rate-mean 0.02 \
  --mutation-rate-std 0.01 \
  --n-steps 100000 \
  --seeds 1 2 3 \
  --prefix aggressive_evo_exp
```

**Compare to**: Standard evolution parameters

**Question answered**: Does stronger selection → faster learning? Risk of premature convergence?

---

## CLI Parameters Reference

### World Configuration
| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `--n-states` | 80 | int | Number of world states |
| `--n-clusters` | 8 | int | Number of hidden clusters |
| `--self-loop-bias` | 0.3 | float | P(stay in same state) |
| `--within-cluster-bias` | 0.5 | float | P(transition within cluster) |
| `--between-cluster-noise` | 0.2 | float | P(random cross-cluster) |
| `--observation-noise` | 0.05 | float | Noise in observations |

### Population & Evolution
| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `--n-observers` | 40 | int | Population size |
| `--n-steps` | 100000 | int | Total simulation steps |
| `--selection-interval` | 5000 | int | Steps between generations |
| `--selection-top-k` | 10 | int | Best observers to keep |
| `--selection-bottom-k` | 10 | int | Worst observers to replace |
| `--mutation-rate-mean` | 0.01 | float | Mean mutation strength |
| `--mutation-rate-std` | 0.005 | float | Std of mutation strength |
| `--no-evolution` | false | flag | Disable evolution entirely |

### Observer Learning
| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `--mode` | "nn" | str | "nn" or "count" |
| `--learning-rate` | 0.001 | float | Gradient descent step size |
| `--hidden-dim` | 128 | int | NN hidden layer size |
| `--embedding-dim` | 32 | int | State embedding dimension |
| `--history-window` | 3 | int | Past steps to include |

### Execution & Output
| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `--seeds` | [42] | list | Random seeds (space-separated) |
| `--prefix` | "exp" | str | Output directory name |
| `--verbose` | false | flag | Print detailed logs |
| `--no-plots` | false | flag | Skip visualization generation |

---

## Understanding Output Structure

For each run, you get:

```
outputs/my_exp/
├── my_exp_config.json                # Full config
├── my_exp_summary.json               # Aggregate stats (if >1 seed)
└── seed_1/
    ├── my_exp_seed1_config.json      # Per-seed config
    ├── my_exp_seed1_world_matrix.csv # True world (80×80)
    ├── my_exp_seed1_traces.csv       # Full trace log
    ├── my_exp_seed1_evolution_log.csv # Generation log
    ├── my_exp_seed1_observer_metrics.csv # Per-observer final stats
    ├── my_exp_seed1_population_metrics.csv # Population over time
    ├── my_exp_seed1_summary_explained.md  # ← READ THIS FIRST
    ├── my_exp_seed1_summary_technical.md  # ← READ THIS SECOND
    └── *.png / *.pdf                # Visualizations
```

### Key Files to Read
1. **`*_summary_explained.md`**: Plain-language overview
2. **`*_summary_technical.md`**: Detailed metrics and interpretation
3. **`*_config.json`**: Verify parameters used
4. **`*_population_metrics.csv`**: Plot your own figures

---

## Tips for Systematic Exploration

### 1. Start Small
```bash
# Quick sanity check
python -m tracereality.experiments.main_core \
  --n-states 40 --n-observers 20 --n-steps 10000 --seeds 1
```

### 2. Establish Baseline
```bash
# Standard configuration with multiple seeds
python -m tracereality.experiments.main_core \
  --n-states 80 --n-clusters 8 --n-observers 40 \
  --n-steps 100000 --seeds 1 2 3 --prefix baseline
```

### 3. Run Ablations
```bash
# Disable evolution
python ... --no-evolution --prefix baseline_no_evo

# Use count baseline
python ... --mode count --prefix baseline_count
```

### 4. Systematic Ablations
```bash
# Vary observer count
for n in 20 40 80 160; do
  python ... --n-observers $n --prefix scaling_obs_${n}
done

# Vary simulation length
for steps in 50000 100000 200000; do
  python ... --n-steps $steps --prefix scaling_steps_${steps}
done
```

### 5. Compare Results
Use the summary files to create comparison tables:

| Experiment | Accuracy | KL | Fitness | Notes |
|-----------|----------|----|---------+-------|
| baseline | 18.3% | 0.086 | 0.052 | Standard |
| no_evo | 12.1% | 0.245 | N/A | Pure learning |
| count | 8.5% | 0.512 | N/A | Simple baseline |
| more_obs | 19.2% | 0.078 | 0.061 | Better with size |

---

## Troubleshooting

### Run takes too long
- Reduce `--n-states` or `--n-observers`
- Reduce `--n-steps`
- Run with fewer seeds (just 1 for testing)

### Low accuracy (< 5%)
- Increase `--n-steps` (simulation too short)
- Check `--n-clusters` (maybe too many clusters?)
- Try disabling `--no-evolution` to see if evolution helps

### High KL divergence (> 0.5)
- Increase `--learning-rate`
- Increase `--hidden-dim`
- Use more observers (`--n-observers`)

### Population converged (all observers identical)
- Increase `--mutation-rate-mean`
- Reduce `--selection-interval` (more frequent evolution)
- Use more observers
