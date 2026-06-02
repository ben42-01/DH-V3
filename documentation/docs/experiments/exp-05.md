---
sidebar_position: 5
title: Exp 05 - No Evolution Ablation
---

# Experiment 05: No Evolution Ablation

## Research Question
How much does evolution help? What is the contribution of evolutionary selection beyond individual gradient descent learning?

## Hypothesis
Disabling evolution will degrade accuracy compared to the evolved baseline. Pure gradient descent learning alone should still yield above-chance accuracy, but evolution should provide a meaningful boost.

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
| Total steps | 100,000 | ✓ Same |
| **Evolution enabled** | **No** | ✗ Disabled |
| **Seeds** | 1, 2, 3 | ✓ Same |

All other parameters identical to Exp 01 (5k selection interval, 0.01 mutation rate, etc.). The key difference is `--no-evolution` flag, which disables all selection and mutation cycles. Observers learn purely through individual gradient descent.

---

## Results Summary

### Final Population Metrics

| Metric | No Evo (Exp 05) | With Evo (Exp 01) | Difference |
|--------|----------------|-------------------|-----------|
| **Accuracy** | 17.5% | 18.3% | -0.8 pp |
| **KL(World ‖ Observer)** | 0.1004 | 0.0864 | +0.0140 |
| **Fitness** | 0.0000 | 0.0520 | — (disabled) |
| **Pairwise KL** | **0.1352** | 0.0509 | **+0.0843** |
| **Param Variance** | 0.0000 | 0.0000 | Identical |

### Performance Interpretation

✅ **Above-chance prediction**: 17.5% vs 1.25% baseline = **14× better** than random guessing — learning happens even without evolution

✅ **Surprisingly good**: Pure gradient descent achieves 96% of evolved performance

✅ **Highest diversity**: Pairwise KL is **2.7× higher** than evolved baseline — observers explore more strategies

⚠️ **Slightly worse model quality**: KL of 0.1004 vs 0.0864 — internal models are marginally less accurate

---

## Contribution of Evolution

### Accuracy Decomposition

| Component | Accuracy | vs Random |
|-----------|----------|-----------|
| Random baseline | 1.25% | 1× |
| Pure learning (no evolution) | 17.5% | 14× |
| With evolution (Exp 01) | 18.3% | 14.6× |
| **Evolution contribution** | **+0.8 pp** | **+4.6%** |

### KL Decomposition

| Component | KL Divergence | Quality |
|-----------|---------------|---------|
| Pure learning (no evolution) | 0.1004 | Moderate |
| With evolution (Exp 01) | 0.0864 | Excellent |
| **Evolution contribution** | **-0.0140** | — |

### Key Insight
Evolution contributes approximately **4.6% additional accuracy** and **14% better KL divergence**. Most of the learning (96% of accuracy) comes from individual gradient descent, but evolution refines the population toward higher-quality solutions.

---

## Learning Dynamics Without Evolution

### Accuracy Over Time
```
Step 0:   1.8%  — Identical start
Step 50k: 15.9% — Slightly behind evolved (16.2%)
Step 100k: 17.5% — Behind evolved (18.3%)
```

### KL Divergence Over Time
```
Step 0:   0.822 — Identical start
Step 50k: 0.125 — Behind evolved (0.105)
Step 100k: 0.100 — Behind evolved (0.086)
```

### Accuracy Trajectory Comparison
The learning curves are similar in shape, but the evolved population consistently maintains a ~0.5–0.8 pp advantage throughout.

---

## Population Diversity Analysis

### Diversity Comparison (Pairwise KL)

| Point in Time | No Evolution | With Evolution |
|---------------|-------------|----------------|
| **Start** | 0.0173 | 0.0173 |
| **Mid (50k)** | 0.0891 | 0.0682 |
| **End (100k)** | **0.1352** | **0.0509** |

### Why So Diverse?

Without evolution:
1. **No selection pressure** — all observers survive regardless of quality
2. **No convergence force** — no mechanism that drives the population toward a single strategy
3. **Individual drift** — each observer follows its own gradient descent path
4. **Result**: High diversity but no mechanism to amplify the best strategies

In contrast, with evolution:
1. Selection weeds out poor predictors each generation
2. The best strategies are amplified through replication
3. The population converges to a narrow set of successful strategies
4. **Result**: Lower diversity but higher average quality

---

## Fitness Analysis

- **No evolution**: Fitness = 0.0000 for all observers (no evolution cycle runs)
- **With evolution**: Fitness 0.0520 mean, range 0.0000–0.2125
- Fitness is a measure tied to evolutionary selection; without selection, it's not updated

This is expected behavior — fitness only has meaning in the context of evolutionary competition.

---

## Comparison to Other Ablation

| Ablation | Accuracy | KL | vs Random | Key Limitation |
|----------|----------|----|-----------|----------------|
| **No evolution** | 17.5% | 0.100 | 14× | No selection pressure |
| **Standard (Exp 01)** | 18.3% | 0.086 | 14.6× | Baseline |
| **Count baseline** | 0.0% | 0.846 | 0× | No neural learning |

The no-evolution ablation performs far better than the count baseline (which can't learn at all) but falls short of the evolved population.

---

## Key Findings

### 1. Individual Learning is Powerful
Neural networks with gradient descent alone achieve 17.5% accuracy — 14× random baseline. The architecture itself is capable of substantial learning without evolutionary pressure.

### 2. Evolution Adds ~1 pp
Evolution contributes an additional 0.8 percentage points (4.6% relative improvement). This is meaningful but not transformative.

### 3. Evolution Reduces Diversity
Without evolution, diversity is 2.7× higher. Evolution sacrifices exploration for exploitation, converging on the best strategy.

### 4. Evolution Improves Model Quality
KL divergence is 14% better with evolution, indicating that selection refines internal world models even after accuracy plateaus.

### 5. No Evolution is Not Random
The no-evolution population achieves highly consistent results (std 0.04%), showing that gradient descent alone is stable and reproducible.

---

## Answering the Research Question

**How much does evolution help?**

Evolution provides a **~5% relative improvement** in accuracy and **~14% improvement** in model quality (KL divergence). 

The effect is:
- **Real**: Consistent across all 3 seeds
- **Modest**: Most of the learning capacity is in the neural architecture + gradient descent
- **Meaningful**: For applications where every percentage point matters, evolution is valuable

**Is evolution worth the complexity?**

For this problem scale: probably yes — the computational cost of evolution is low relative to total runtime. For larger-scale problems with higher computational costs, pure gradient descent may be sufficient (~96% of evolved performance).

---

## Raw Data & Visualizations

**Results location**: `outputs/no_evo_baseline/`

**Summary files**:
- `no_evo_baseline_summary_explained.md` — Plain English explanation
- `no_evo_baseline_summary_technical.md` — Detailed technical metrics

**Data files**:
- `*_population_metrics.csv` — Population stats over time
- `*_observer_metrics.csv` — Per-observer final state
- `*_metrics_summary.csv` — Final summary

**Note**: No evolution log file exists (evolution was disabled).

---

## For Analysis

```python
import pandas as pd
import json

# Compare no-evo vs evolved
no_evo = pd.read_csv('outputs/no_evo_baseline/seed_1/no_evo_baseline_seed1_population_metrics.csv')
evo = pd.read_csv('outputs/stable_exp/seed_1/stable_exp_seed1_population_metrics.csv')

print(f"No evolution final accuracy: {no_evo['mean_accuracy'].iloc[-1]:.1%}")
print(f"Evolved final accuracy:     {evo['mean_accuracy'].iloc[-1]:.1%}")
print(f"Evolution advantage:        {evo['mean_accuracy'].iloc[-1] - no_evo['mean_accuracy'].iloc[-1]:.1%}")
print(f"KL without evolution:       {no_evo['mean_kl_world'].iloc[-1]:.4f}")
print(f"KL with evolution:          {evo['mean_kl_world'].iloc[-1]:.4f}")
print(f"Diversity without evolution: {no_evo['mean_pairwise_kl'].iloc[-1]:.4f}")
print(f"Diversity with evolution:    {evo['mean_pairwise_kl'].iloc[-1]:.4f}")
```

---

## Conclusion

Experiment 05 confirms that **evolution provides a meaningful but modest improvement** over pure gradient descent learning:

- ✅ Learning without evolution is surprisingly effective (96% of evolved accuracy)
- ✅ Evolution adds ~0.8 pp accuracy and improves model quality by 14%
- ✅ Evolution reduces diversity — the population converges on best strategies
- ⚠️ No evolution leaves population diverse but less optimized

**Recommendation**: Use evolution as standard — the computational overhead is minimal and the gains are consistent. The no-evolution baseline is a useful control but not a replacement.