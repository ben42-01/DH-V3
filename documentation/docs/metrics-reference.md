---
sidebar_position: 4
title: Metrics Reference
---

# Metrics Reference

## Primary Metrics

### Accuracy (% Correct Predictions)
**Definition**: Fraction of test predictions that match the true next state.

```
Accuracy = (# correct predictions) / (# total predictions)
```

**Baseline**: Random guess on N states = 1/N
- 80 states: 1.25% baseline
- 160 states: 0.625% baseline

**Interpretation**:
- **< 2% of baseline**: No learning above random
- **2-5% of baseline**: Some learning, but weak
- **5-10% of baseline**: Meaningful learning
- **> 10% of baseline**: Strong learning (12%+ for 80 states)

**Trend**: Should increase over time if learning works. Plateauing is normal after many generations.

---

### KL Divergence: World ‖ Observer
**Definition**: Kullback-Leibler divergence from the true world transition distribution to the observer's learned distribution.

**Formula**: KL(P_true || P_obs) = Σ_i P_true(i) * log(P_true(i) / P_obs(i))

Measures how different the observer's probability distribution is from the true world.

**Interpretation**:
- **0.0**: Perfect match (impossible in practice due to sampling)
- **< 0.05**: Excellent — observer model closely matches reality
- **0.05 - 0.2**: Very good — most patterns captured
- **0.2 - 0.5**: Moderate — broad structure learned, details missed
- **0.5 - 1.0**: Poor — major mismatch with ground truth
- **> 1.0**: Very poor — observer has wrong model

**Why it matters**:
- Direct measure of model quality
- Lower is better
- Should decrease over evolutionary time

**Related metric**: Mutual Information (technically negative KL for this setup)

---

### Fitness
**Definition**: The survival metric for evolutionary selection. Higher fitness = better chance of reproduction.

In TraceReality: `Fitness = Accuracy` (or weighted combination of accuracy metrics)

**Trend Analysis**:
- **Increasing**: Evolution is effective, selection pressure driving improvement
- **Flat**: Population converged, all observers equally fit (good or bad depending on accuracy level)
- **Decreasing**: Rare; usually indicates a problem

**Population-level view**:
- **Mean fitness**: Average population fitness
- **Max fitness**: Best observer in current generation
- **Variance**: Diversity of fitness values (high = good for evolution, low = convergence)

---

### Pairwise KL (Observer Diversity)
**Definition**: Average KL divergence between all pairs of observers in the population.

**Formula**: Diversity = (1 / C(N,2)) × Σ for all pairs (i,j) of KL(P_i || P_j)

where C(N,2) is the number of unique pairs of N observers.

**Interpretation**:
- **High (> 0.05)**: Diverse population, different prediction strategies
- **Medium (0.01 - 0.05)**: Moderate diversity, some convergence
- **Low (< 0.01)**: Homogeneous population, likely converged

**Trend Analysis**:
- **Increasing**: Population diverging (exploration phase)
- **Decreasing**: Population converging (exploitation phase)
- **Stable**: Population in equilibrium

**What it means**:
- High diversity early: Evolution has material to select on ✓
- High diversity late: Maybe not using selection effectively?
- Low diversity early: Population converged too fast (over-selection?)
- Low diversity late: Normal, successful convergence

---

### Parameter Variance (Network Stability)
**Definition**: Variance of neural network weights across observers.

**Interpretation**:
- **High (> 0.01)**: Different observers have different weight configurations
- **Medium (0.001 - 0.01)**: Some weight variation
- **Low (< 0.001)**: Observers have nearly identical weights (convergence)

**Trend**:
- Should generally decrease over time as selection narrows the population
- Stabilizes near zero when population fully converges

---

## Expected KL Divergence (Model Mismatch)

**Definition**: Weighted average of KL divergences across all states, weighted by the stationary distribution of the world.

**Formula**: Expected KL = Σ_i π_i * KL(P_true(·|i) || P_obs(·|i))

where π_i is the stationary probability of state i (how often state i is visited in the long run).

**Purpose**: Takes into account which states are visited frequently. A mistake on a rare state hurts less than a mistake on a common state.

**Interpretation**: Same as KL Divergence (World ‖ Observer) — essentially measures the same thing but weighted by state frequency.

---

## Cluster Recovery Metrics

### Adjusted Rand Index (ARI)
**Definition**: Measures how well the observer's cluster assignments match the ground-truth clusters.

**Range**: -1 to 1
- **> 0.7**: Excellent cluster recovery
- **0.3 - 0.7**: Moderate recovery
- **0 - 0.3**: Weak recovery
- **< 0**: Performance worse than random assignment

**Interpretation**:
- **High ARI**: Observer discovered the hidden cluster structure
- **Low ARI**: Observer failed to recover clusters (but may still have good accuracy)

**Why it matters**: Separating whether observers learned:
1. The world dynamics (measured by accuracy/KL)
2. The hidden structure (measured by ARI)

Some observers might predict well without understanding clusters, while others recover clusters without predicting perfectly.

---

### Cluster NMI (Normalized Mutual Information)
**Definition**: Information-theoretic measure of cluster similarity.

**Range**: 0 to 1 (higher = better agreement)
- **> 0.7**: Excellent recovery
- **0.3 - 0.7**: Moderate recovery
- **< 0.3**: Poor recovery

**Comparison to ARI**: NMI penalizes splits differently; sometimes more sensitive to fine-grained differences.

---

## Evolution Metrics

### Generations
**Definition**: Number of selection/mutation cycles that occurred.

**Formula**: Generations = floor(n_steps / selection_interval)

**Example**: 100,000 steps with 5,000-step intervals = 20 generations.

**Interpretation**: More generations = more time for evolution to adapt. Typical: 10-50 generations for good results.

---

### Fitness Trajectory
**Definition**: How fitness changes across generations.

**Metrics**:
- **Initial fitness**: Generation 0 (usually near random baseline)
- **Final fitness**: Last generation
- **Best fitness**: Peak fitness reached (may not be final)
- **Fitness improvement**: Final - Initial

**Interpretation**:
- **Steep improvement**: Evolution working effectively
- **Plateau**: Population converged; further improvement unlikely
- **Decline**: Problem with selection or mutation parameters

---

### Selection Pressure
**Definition**: Fraction of population replaced per generation.

**Formula**: Selection Pressure = (k_top + k_bottom) / n_observers

**Example**: 
- 40 observers, keep top 10, replace bottom 10 → 20/40 = 50% pressure

**Higher pressure**:
- Faster initial improvement
- Risk of premature convergence
- May discard useful diversity

**Lower pressure**:
- Slower improvement
- Better diversity preservation
- May waste generations on weak observers

---

## Population Dynamics Metrics

### Population Size
**Definition**: Number of observers at a given time.

**Typically**: Constant throughout run, though could vary (not in current implementation).

---

### Accuracy Spread (Std Dev)
**Definition**: Standard deviation of observer accuracies.

**Formula**: σ_acc = sqrt( (1/N) * Σ_(i=1 to N) (acc_i - μ_acc)² )

Higher values mean observers have different accuracies; lower values mean they're similar.

**Interpretation**:
- **High**: Population has diverse accuracy levels (some good, some bad)
- **Low**: Population homogeneous (all similar accuracy)

**Trend**: Should decrease over time as selection narrows the population toward high-accuracy strategies.

---

### Mutation Rate Distribution
**Definition**: Variation in how much observers mutate each generation.

**Typical values**: 0.001 to 0.05 (weights perturbed by this fraction)

**Tracking**:
- **Mean mutation rate**: Population average
- **Min/max**: Range of mutation rates

**Trend**: Usually stable or slightly increasing as survivors accumulate mutations.

---

## Reading Summary Statistics

### The `*_summary_technical.md` File
Automatically generated after every run. Key sections:

#### Section 1: Experiment Configuration
Full parameter table — verify what you actually ran.

#### Section 2: Final Population Metrics
One-row summary of key metrics at simulation end:

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 18.3% | 0.0% |
| KL(World‖Obs) | 0.0864 | 0.0052 |
| Fitness | 0.0520 | 0.0912 |

Interpretation:
- Low std = population converged
- High accuracy = good learning
- Low KL = good world model

#### Section 3: Information-Theoretic Metrics
Mutual information and related statistics.

#### Section 4: Evolution Dynamics
Summary of evolutionary progress:
- Number of generations
- Initial/final fitness
- Improvement trajectory

#### Section 5: Observer Population
Population statistics (accuracy range, fitness range, mutation rates).

#### Section 6: Metric Trajectories
How key metrics evolved over the entire simulation:

```
Accuracy: 0.0179 → 0.1826 (improved by 0.0580)
KL: 0.8223 → 0.0864 (improved by 0.2009)
```

#### Section 7: Technical Interpretation
English-language explanation of what the results mean.

---

## Metric Relationships

### Accuracy vs. KL Divergence
- **Both should improve together**: Accurate observers ↔ good models
- **May diverge**: Observer predicts well locally but global model is wrong
- **Correlation**: Usually strong (r > 0.8)

### Fitness vs. Diversity
- **High fitness + Low diversity**: Successful convergence (good!)
- **High fitness + High diversity**: Good exploration phase
- **Low fitness + High diversity**: Selection not working
- **Low fitness + Low diversity**: Population stuck in local optimum (bad!)

### Evolution Effectiveness
**Measure**: Compare to no-evolution baseline:
- Run with `--no-evolution` to get pure-learning baseline
- Run standard (with evolution) version
- If standard >> no-evolution: Evolution helps significantly

---

## Common Pitfalls & Interpretation Mistakes

### Mistake 1: "High accuracy = perfect understanding"
- Accuracy of 20% on 80 states = 16× random, still wrong 80% of the time
- Always compare to baseline

### Mistake 2: "KL divergence of 0.1 is good"
- Need context: baseline KL for random model on 80 states ≈ 4.4
- 0.1 is 44× better than random — excellent!

### Mistake 3: "Population converged = evolution failed"
- Low diversity late is **normal** if fitness is high
- Convergence with high fitness = success
- Convergence with low fitness = failure

### Mistake 4: "More generations = better results"
- True up to a point, then diminishing returns
- Typical plateau: 10-30 generations
- Beyond that: diminishing returns unless you change selection pressure

---

## Comparing Experiments

### Checklist for Comparison
```
Experiment A vs B:
☐ Same random seed? (for fair comparison)
☐ Same world config (n_states, n_clusters)?
☐ Same observation noise level?
☐ Same number of steps?
☐ Same population size?
☐ Different only in the factor I'm testing?
```

### Comparison Table Template
| Metric | Exp A | Exp B | Better | Change |
|--------|-------|-------|--------|--------|
| Accuracy | 18.3% | 15.2% | A | +3.1 pp |
| KL | 0.086 | 0.124 | A | -0.038 |
| Fitness | 0.052 | 0.031 | A | +0.021 |
| ARI | 0.45 | 0.38 | A | +0.07 |

Conclude: "Experiment A performs consistently better across all metrics."
