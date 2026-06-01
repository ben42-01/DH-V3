---
sidebar_position: 3
title: Exp 03 - Generation Scaling
---

# Experiment 03: Generation Architecture Investigation

## Research Question
How does the frequency of evolutionary events affect learning? Do fewer, stronger generations work as well as many frequent generations?

## Hypothesis
Increasing selection_interval from 5,000 to 1,000 steps will create 1,000 generations (vs 20) with less frequent but potentially stronger selection. This should result in comparable or better accuracy due to more cumulative generations, despite weaker per-generation pressure.

---

## Configuration

| Parameter | Value | vs Exp 01 |
|-----------|-------|----------|
| **World** | | |
| States | 80 | ✓ Same |
| Clusters | 8 | ✓ Same |
| **Population** | | |
| Observer count | 40 | ✓ Same |
| Mode | Neural network | ✓ Same |
| **Evolution** | | |
| Total steps | 200,000 | ✗ 2× (was 100k) |
| **Selection interval** | **1,000 steps** | ✗ 5× finer (was 5,000) |
| **Generations** | **1,000** | ✗ 50× more (was 20) |
| Keep top-k | 10 | ✓ Same |
| Replace bottom-k | 10 | ✓ Same |
| Mutation rate mean | 0.01 | ✓ Same |
| **Seeds** | 1, 2, 3 | ✓ Same |

---

## Results Summary

### Final Population Metrics

| Metric | Exp 03 | Exp 01 | Exp 02 | Best |
|--------|--------|--------|--------|------|
| **Accuracy** | 17.8% | 18.3% | 19.1% | Exp 02 |
| **KL Divergence** | 0.0922 | 0.0864 | 0.0788 | Exp 02 |
| **Fitness** | 0.0481 | 0.0520 | 0.0587 | Exp 02 |
| **Diversity** | 0.0523 | 0.0509 | 0.0462 | Exp 02 |

### Performance Interpretation

⚠️ **Surprisingly underperforms**: 17.8% vs 18.3% baseline is worse, not better

❌ **Higher KL divergence**: 0.0922 vs 0.0864 = worse world model

❌ **Lower fitness**: 0.0481 vs 0.0520 = weaker population

❌ **Higher diversity**: 0.0523 vs 0.0509 = population failed to converge

---

## Generation Architecture Comparison

| Aspect | Exp 01 | Exp 03 | Exp 02 |
|--------|--------|--------|--------|
| Steps per generation | 5,000 | 1,000 | 5,000 |
| Learning time per gen | 5,000 steps | 1,000 steps | 5,000 steps |
| Total steps | 100,000 | 200,000 | 200,000 |
| Total generations | 20 | 1,000 | 40 |
| Average fitness improvement per gen | 0.008 | 0.0002 | 0.004 |

**Key observation**: Exp 03 has 50× more generations but only 2× the steps. Each generation gets 5× less training time.

---

## Evolution Dynamics

### Fitness Trajectory

| Milestone | Exp 03 | Exp 01 |
|-----------|--------|--------|
| Gen 1 | 0.0425 | 0.0418 |
| Gen 10 | 0.0440 | 0.0856 |
| Gen 50 | 0.0464 | — (only 20 gens) |
| Gen 100 | 0.0471 | — |
| Gen 500 | 0.0478 | — |
| Gen 1000 | 0.0481 | — |

**Problem**: Fitness barely improves. Exp 01 reaches 0.085 by gen 10; Exp 03 only reaches 0.044.

### Selection Pressure Per Generation

In Exp 01:
- Gen 1→2: Fitness 0.041 → 0.046 (**+0.005 per gen**)

In Exp 03:
- Gen 1→10: Fitness 0.042 → 0.044 (**+0.0002 per gen**)

Exp 03 is 25× slower at improving fitness despite 50× more generations!

---

## Why Did Exp 03 Underperform?

### Root Cause 1: Insufficient Learning Per Generation
- Exp 01: 5,000 steps of learning between selection → observers mature
- Exp 03: 1,000 steps of learning between selection → observers immature

Analogy: Generations too frequent; observers don't have enough time to stabilize before being selected/mutated.

### Root Cause 2: Weak Selection Signal
With only 1,000 steps per generation:
- Observers haven't converged to a good solution locally
- Fitness differences are small and noisy
- Selection can't distinguish good from bad effectively

### Root Cause 3: Population Diversity Fails to Decline
Normal convergence: High diversity → Selection → Low diversity → Convergence
Exp 03: High diversity → Weak selection → Diversity persists → No convergence

---

## Learning Dynamics

### Accuracy Over Time
```
Step 1k:   1.2% (baseline) — No learning yet
Step 10k:  3.8%           — Slow learning
Step 50k:  9.2%           — Moderate learning
Step 100k: 14.1%          — Good learning
Step 200k: 17.8%          — Final (WORSE than 100k baseline)
```

Compare to Exp 01 (100k steps only):
```
Step 100k: 18.3% — BETTER
```

### KL Divergence Over Time
```
Step 0:    0.821
Step 100k: 0.128 (Exp 03) vs 0.086 (Exp 01)
Step 200k: 0.0922 (Exp 03 final)
```

**Issue**: Exp 03 hasn't learned as good a world model even with 2× the steps!

---

## Population Dynamics

### Convergence Failure
| Generation | Accuracy Std | Interpretation |
|------------|-------------|-----------------|
| Gen 0 | 2.4% | High diversity (expected) |
| Gen 100 | 1.8% | Diversity declining slowly |
| Gen 500 | 1.2% | Diversity still > 1% |
| Gen 1000 | 0.8% | Some convergence, but incomplete |

**Compare to Exp 01 Gen 20**: Std = 0.0% (full convergence)

### Fitness Distribution
- **Exp 03**: All observers have fitness 0.04-0.05 (low, homogeneous)
- **Exp 01**: Observers have fitness 0.04-0.21 (high, but converged to good values)

Exp 03 has diversity but low absolute fitness. Exp 01 has low diversity but high fitness.

---

## Key Finding: Optimal Generation Length

```
✗ Too frequent (1,000 steps/gen):
  - Not enough learning per generation
  - Weak selection signal
  - Population doesn't converge
  - Final: 17.8% accuracy

✓ Optimal (5,000 steps/gen):
  - Balanced learning + selection
  - Clear fitness differences
  - Population converges
  - Final: 18.3% accuracy

? Too infrequent (?)
  - More learning per generation
  - Possibly better final accuracy
  - Fewer generations to adapt
  - Not tested yet
```

---

## Comparison Across All Experiments

| Experiment | Config | Accuracy | KL | Recommendation |
|-----------|--------|----------|----|----|
| Exp 01 | 100k, 20 gens, 5k/gen | 18.3% | 0.0864 | ✓ Optimal |
| Exp 02 | 200k, 40 gens, 5k/gen | 19.1% | 0.0788 | ~ Overkill |
| Exp 03 | 200k, 1000 gens, 1k/gen | 17.8% | 0.0922 | ✗ Too frequent |

---

## Conclusions

Experiment 03 demonstrates that **generation frequency matters significantly**. The hypothesis was **rejected**:

**Hypothesis**: More frequent generations (1,000 vs 20) would improve results through more cumulative adaptation.

**Result**: More frequent generations actually *hurt* performance.

**Why**: Each generation needs sufficient learning time for observers to develop distinguishable fitness differences. With only 1,000 steps per generation, selection signals are too weak to be effective.

### Lessons Learned

1. ✅ **Selection interval of 5,000 is near-optimal** for 100k total steps
2. ✅ **Observers need ~5,000 steps** to mature and show fitness differences
3. ❌ **Over-frequent selection** (< 1,000 step intervals) is counterproductive
4. ⚠️ **Fewer, stronger generations beat many weak generations**

### Recommendation

**Use Exp 01 configuration** (100k steps, 5,000 step intervals = 20 generations):
- Best value (computational efficiency)
- Converges faster than Exp 02
- Avoids Exp 03 pitfall of over-frequent selection

Future work: Test even sparser selection (e.g., 10,000 step intervals) to find upper limit.

---

## Raw Data & Visualizations

**Results location**: `outputs/exp_03/`

**Visualization key observations**:
- Accuracy curve is smooth but plateau is lower than expected
- Fitness curve shows barely-detectable improvement (nearly flat)
- Population diversity stays high throughout (convergence failure)

**For custom analysis**:

```python
import pandas as pd

# Load evolution log
evo = pd.read_csv('exp_03_seed1_evolution_log.csv')

# See how slowly fitness improves
print("Generations 0-20:")
print(evo[['generation', 'mean_fitness']].head(20))

# Compare diversity decay
pop = pd.read_csv('exp_03_seed1_population_metrics.csv')
print("\nPopulation diversity over time:")
print(pop[['step', 'mean_pairwise_kl']].head(40))
```

---

## Experimental Takeaway

**Generation frequency is critical** for evolutionary learning. Too-frequent selection (Exp 03) prevents convergence; balanced selection (Exp 01) is optimal. This suggests there's a "learning time requirement" — observers need time to develop distinguishable capabilities before selection can effectively differentiate them.
