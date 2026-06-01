---
sidebar_position: 5
title: Results Interpretation
---

# How to Interpret Results

## Quick Start: Reading Your First Result

After running an experiment, you'll find in `outputs/my_exp/seed_1/`:

1. **Read first**: `my_exp_seed1_summary_explained.md` — Plain English explanation
2. **Read second**: `my_exp_seed1_summary_technical.md` — Detailed metrics
3. **Explore**: PNG visualizations (accuracy curves, heatmaps, fitness landscapes)
4. **Analyze**: CSV files if you want to do custom analysis

---

## The Summary Files

### Plain-Language Summary (`*_summary_explained.md`)

This is written for non-specialists. It answers four questions:

**What did we do?**
> "We simulated a world with 80 states organized into 8 hidden clusters. A population of 40 neural networks tried to learn state transitions through evolution over 100,000 steps."

**What happened?**
> "The population evolved successfully. Accuracy improved from 1.25% (random baseline) to 18.3%, and KL divergence dropped from 0.8 to 0.09, indicating the observers learned accurate world models."

**What does this mean?**
> "The population discovered meaningful patterns in the world. Evolution was effective — observers that predict better survived, spread their genes, and improved over 19 generations."

**Key takeaway?**
> "Neural networks can learn hidden Markov structure through evolutionary selection, achieving ~15× random baseline accuracy and nearly perfect internal models."

---

### Technical Summary (`*_summary_technical.md`)

Contains everything needed for deep analysis:

#### Section 1: Configuration Table
Verify the exact parameters used:
```
World states: 80
Clusters: 8
Observers: 40
Simulation steps: 100,000
Mode: nn (neural networks)
Evolution: enabled
```

#### Section 2: Final Population Metrics
One-row summary of key statistics:
```
Accuracy: 18.3% ± 0.0%         ← Population converged on similar accuracy
KL(World ‖ Observer): 0.086    ← Excellent world model match
Fitness: 0.052 ± 0.091         ← Mean fitness (high variance in details)
Pairwise KL: 0.051 ± 0.018     ← Moderate observer diversity
```

#### Section 3: Information-Theoretic Metrics
```
Mutual Information: -0.0864 nats
Expected KL Divergence: 0.086 (state-weighted average)
```

#### Section 4: Evolution Dynamics
```
Generations: 19
Initial fitness: 0.0418
Final fitness: 0.2013
Improvement: 0.1595 (impressive!)
Best fitness achieved: 0.2144
```

#### Section 5: Observer Population
Final population snapshot:
- Size: 40
- Accuracy range: 18.2% – 18.3% (tight convergence)
- Fitness range: 0.0 – 0.213 (some variance in details)
- Mutation rates: 0.001 – 0.018

#### Section 6: Metric Trajectories
How metrics evolved over time:
```
Accuracy:    0.0179 → 0.1826  (improvement: +0.0580)
KL:          0.8223 → 0.0864  (improvement: -0.2009)
Fitness:     0.0000 → 0.0520  (improvement: +0.0145)
Diversity:   0.0173 → 0.0509  (mixed)
```

#### Section 7: Interpretation
Human-readable conclusions:
- "Above-chance prediction" ✓ (18.3% vs 1.2% baseline)
- "Excellent world model" ✓ (KL 0.086)
- "Evolution was effective" ✓ (fitness improved)
- Statistical significance if applicable

---

## Visual Inspection: The PNG Plots

### Accuracy vs Time
**What it shows**: Accuracy over simulation steps

**Good plot**:
- Steep rise early (rapid learning)
- Levels off mid-run (plateau)
- Flat late (convergence)

**Bad plots**:
- Flat from start (no learning)
- Noisy with no trend (learning unstable)
- Decreasing (something went wrong)

**What to do**: If your accuracy plateaus early and low, try:
- Increase `--n-steps`
- Decrease `--n-clusters` (easier world)
- Increase `--learning-rate`

---

### KL Divergence vs Time
**What it shows**: Model mismatch over time

**Good plot**:
- Steep decrease early (learning the world)
- Levels off around 0.05–0.2
- Stable late

**Bad plots**:
- Flat and high (> 0.5) from start
- Increasing (learning going backward)

**What to do**: If KL stays high:
- Increase `--hidden-dim` (more network capacity)
- Increase `--n-steps` (more learning time)
- Use more observers (`--n-observers`)

---

### Fitness Landscape
**What it shows**: Distribution of observer fitnesses

**Good plot**:
- Peak on the right (most observers fit)
- Narrow distribution (population converged)

**Bad plots**:
- Flat distribution (no evolution pressure)
- All zeros (population not learning)

---

### Cluster Recovery
**What it shows**: How well observers found hidden clusters

**Good plot**:
- ARI > 0.7 (excellent recovery)
- Cluster assignments match true clusters

**Bad plots**:
- ARI < 0.3 (failed to find structure)
- Random-looking clusters

**Interpretation**: If accuracy is high but cluster recovery is low, observers learned the dynamics but not the structure. If both are high, observers understood the full hidden world.

---

### Evolution Fitness Over Generations
**What it shows**: Fitness trajectory across generations

**Good plot**:
- Steep rise early (selection pressure effective)
- Levels off mid-run (convergence)
- Stable late

**Bad plots**:
- Flat (no selection pressure)
- Noisy with no trend (selection parameters bad)
- Decreasing (evolution broken)

**Analysis**:
```
Generation 0: fitness = 0.04
Generation 19: fitness = 0.20
Improvement: 5× in 19 generations
Per-generation gain: 5^(1/19) ≈ 1.08 (8% per generation)
```

This is healthy evolutionary progress.

---

## CSV Data Files: Deep Dive

### `population_metrics.csv`
Sampled every ~5000 steps. Columns:

| Column | What it means |
|--------|---------------|
| step | Simulation timestep |
| n_observers | Population size |
| mean_accuracy | Population average accuracy |
| std_accuracy | Accuracy standard deviation |
| mean_kl_world | Average KL divergence |
| mean_fitness | Average fitness |
| std_fitness | Fitness standard deviation |
| mean_pairwise_kl | Observer diversity |
| mean_param_variance | Weight stability |

**Analysis**: Plot these yourself to see trends more clearly than PNG plots.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('population_metrics.csv')
plt.plot(df['step'], df['mean_accuracy'])
plt.xlabel('Step')
plt.ylabel('Mean Accuracy')
plt.title('Population Learning Trajectory')
plt.savefig('my_analysis.png')
```

---

### `observer_metrics.csv`
Final state of each observer (40 rows for 40 observers):

| Column | What it means |
|--------|---------------|
| observer_id | ID 0-39 |
| avg_accuracy | Final accuracy |
| kl_divergence_world | Model mismatch |
| current_fitness | Final fitness score |
| mutation_rate | How much it mutates |
| n_train_steps | Training updates received |

**Analysis**: Find the best and worst observers:

```python
df = pd.read_csv('observer_metrics.csv')
best = df.loc[df['avg_accuracy'].idxmax()]
worst = df.loc[df['avg_accuracy'].idxmin()]
print(f"Best accuracy: {best['avg_accuracy']:.1%}")
print(f"Worst accuracy: {worst['avg_accuracy']:.1%}")
```

---

### `evolution_log.csv`
One row per generation (20 rows for 20 generations):

| Column | What it means |
|--------|---------------|
| generation | Gen number 0-19 |
| mean_fitness | Population average fitness |
| max_fitness | Best fitness in generation |
| n_survivors | Kept observers |
| n_discarded | Replaced observers |

**Analysis**: Calculate generation-to-generation improvement:

```python
df = pd.read_csv('evolution_log.csv')
df['fitness_gain'] = df['mean_fitness'].diff()
print("Fitness per generation:")
print(df[['generation', 'mean_fitness', 'fitness_gain']])
```

---

### `world_matrix.csv`
The true world transition matrix (80×80):

Each row i, column j contains P(next = j | current = i).

**Use for**:
- Understanding the ground truth world
- Comparing to observer models
- Visualizing cluster structure

---

### `cluster_ids.csv`
Ground truth cluster assignments (80 rows):

```
state_id, cluster_id
0, 0
1, 0
2, 1
...
79, 7
```

**Use for**: Evaluating cluster recovery metrics.

---

## Comparison Across Runs

### Comparing Two Experiments
After running experiment A and experiment B, create a comparison table:

```python
import json

# Load summaries
with open('outputs/exp_a/seed_1/exp_a_seed1_metrics_summary.csv') as f:
    a = json.load(f)
with open('outputs/exp_b/seed_1/exp_b_seed1_metrics_summary.csv') as f:
    b = json.load(f)

print("Experiment A vs B")
print(f"Accuracy:      {a['accuracy']:.1%} vs {b['accuracy']:.1%}")
print(f"KL Divergence: {a['kl_divergence']:.4f} vs {b['kl_divergence']:.4f}")
print(f"Fitness:       {a['fitness']:.4f} vs {b['fitness']:.4f}")

# Determine winner
if a['accuracy'] > b['accuracy']:
    print("✓ A wins on accuracy")
else:
    print("✓ B wins on accuracy")
```

### Multi-Seed Comparison
If you ran with seeds 1, 2, 3:

```python
import pandas as pd
import numpy as np

results = []
for seed in [1, 2, 3]:
    df = pd.read_csv(f'seed_{seed}/metrics_summary.csv')
    results.append(df)

df_combined = pd.concat(results)
print(f"Mean accuracy: {df_combined['accuracy'].mean():.1%} ± {df_combined['accuracy'].std():.1%}")
print(f"Mean KL:       {df_combined['kl'].mean():.4f} ± {df_combined['kl'].std():.4f}")
```

---

## Common Patterns & What They Mean

### Pattern 1: High Accuracy, Low KL
**Result**: 18% accuracy, KL 0.086

**Interpretation**: 
- ✓ Population learned well
- ✓ Internal models match reality
- ✓ Evolution worked

**Conclusion**: Success! The experiment achieved its goals.

---

### Pattern 2: Moderate Accuracy, High KL
**Result**: 12% accuracy, KL 0.45

**Interpretation**:
- ~ Population learned some patterns
- ✗ Internal models are misspecified
- ~ Observers predict somewhat but don't understand why

**Conclusion**: Partial success. Maybe try:
- Longer training
- Simpler world
- Bigger networks

---

### Pattern 3: Low Accuracy, High KL
**Result**: 2% accuracy, KL 1.2

**Interpretation**:
- ✗ No learning above random
- ✗ Observers failed to model the world
- ✗ Evolution didn't help

**Conclusion**: Failure. Debug by:
- Disable evolution: is pure learning working?
- Use count baseline: does simplicity help?
- Reduce world complexity

---

### Pattern 4: Accuracy Plateaus Early & Low
**Result**: Accuracy rises to 5%, then flat from step 20k onward

**Interpretation**:
- ~ Initial learning worked
- ✗ Population converged prematurely
- ✗ Evolution lost diversity

**Conclusion**: Convergence trap. Try:
- Increase mutation rate
- Reduce selection pressure
- Add more observers
- Use smaller selection interval (more frequent evolution)

---

### Pattern 5: Accuracy Flat from Start
**Result**: Accuracy stays at 1.25% (random baseline) for entire run

**Interpretation**:
- ✗ Population failed to learn anything
- ✗ Either learning is broken or world is too hard

**Conclusion**: Major problem. Try:
1. Disable evolution first:
   ```bash
   python -m tracereality.experiments.main_core \
     --n-states 80 --no-evolution --prefix debug_no_evo
   ```
   If this learns: evolution is the problem
   If this doesn't learn: learning is the problem

2. If learning is broken:
   - Reduce `--n-states` to 20 (easier world)
   - Increase `--learning-rate` to 0.01
   - Increase `--hidden-dim` to 256

---

### Pattern 6: Noisy/Unstable Curves
**Result**: Accuracy bounces around with no clear trend

**Interpretation**:
- ~ Some learning happening
- ✗ Population unstable or high variance
- ~ Evolution parameters may be aggressive

**Conclusion**: Try:
- Use multiple seeds for averaging
- Reduce mutation rate
- Reduce selection pressure
- Increase population size

---

## Advanced Analysis

### Calculating Generations Needed
From evolution_log.csv:
```
Total generations = steps / selection_interval
Example: 100,000 steps / 5,000 interval = 20 generations
```

### Calculating Learning Efficiency
```
Accuracy improvement = final - initial
Generations taken = total_generations
Per-generation improvement = (final / initial) ^ (1 / generations)

Example:
- Initial: 1.2%
- Final: 18.3%
- Generations: 20
- Ratio: 18.3 / 1.2 = 15.25
- Per-gen growth: 15.25^(1/20) = 1.149 (14.9% per generation)
```

### Assessing Evolution Effectiveness
Compare with no-evolution baseline:
```
Advantage = (fitness_with_evo - fitness_no_evo) / fitness_no_evo

If advantage > 1.5x: Evolution helps significantly
If advantage 1.0-1.5x: Evolution helps moderately  
If advantage < 1.0x: Evolution hurts or doesn't help
```

---

## When to Worry (Troubleshooting)

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Accuracy stays at baseline | Learning broken | Disable evolution; try count baseline |
| Accuracy good, KL high | Model misspecified | Increase hidden_dim; more steps |
| Population converged too early | Aggressive selection | Reduce top-k/bottom-k; increase mutation |
| Fitness doesn't improve | Selection pressure too weak | Increase top-k/bottom-k |
| High accuracy but low diversity | Over-convergence | Reduce selection pressure |
| Runs take forever | Too many steps/states | Reduce n_steps or n_states for testing |
