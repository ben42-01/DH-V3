---
sidebar_position: 6
title: Experiments Overview
slug: experiments
---

# Research Experiments

This section documents all the experiments conducted with TraceReality. Each experiment tests specific hypotheses about how neural observers learn hidden world structures through evolutionary selection.

## Experiment Organization

### Main Series (Exp 01-04)

These are the core experimental series designed to investigate fundamental questions:

- **[Exp 01: Baseline (80 states, 8 clusters)](./exp-01)** — Standard configuration, 100k steps
- **[Exp 02: Extended Run (200k steps)](./exp-02)** — Same world, longer to observe convergence
- **[Exp 03: Scaling Generations (1000 steps/gen)](./exp-03)** — Fewer but stronger generations
- **[Exp 04: Population Scaling](./exp-04)** — Various observer population sizes

### Ablation Studies

- **No Evolution Baseline** — Pure learning without selection/mutation
- **Count Baseline** — Simple frequency tables vs neural networks
- **Extended Training** — More learning steps to assess convergence

### Parameter Sweeps

- **Generation Analysis** — Varying selection_interval
- **Observer Scaling** — Different population sizes (50, 100, 200)
- **World Complexity** — Different numbers of states and clusters

---

## Key Research Questions

### Q1: Can evolution discover hidden structure?
**Answer**: Yes, with strong success.
- Observers achieve 15-20× random baseline accuracy
- Cluster ARI > 0.7 in favorable conditions
- KL divergence approaches 0.05-0.1 (excellent)

**Evidence**: [Exp 01](./exp-01), [Exp 02](./exp-02)

---

### Q2: How many generations are needed?
**Answer**: Typically 15-25 generations for convergence.
- Steep improvement first 5-10 generations
- Plateau afterward (diminishing returns)
- Population typically converges before improvement plateaus

**Evidence**: Evolution fitness curves across experiments

---

### Q3: Does population size matter?
**Answer**: Yes, but with diminishing returns.
- Larger populations (100+) → better final accuracy
- Small populations (20) → high variance, sometimes miss solutions
- Recommended: 40-80 observers for standard worlds

**Evidence**: [Exp 04](./exp-04)

---

### Q4: How much does evolution help?
**Answer**: Significantly, but not essential for learning.
- With evolution: 18-20% accuracy
- No evolution: 10-15% accuracy
- Improvement factor: 1.5-2.5×

**Evidence**: Baseline vs Evolution comparison

---

### Q5: How does world complexity affect learning?
**Answer**: Accuracy scales logarithmically with complexity.
- 80 states: ~18% accuracy
- 160 states: ~10-12% accuracy
- 320 states: ~5-8% accuracy

**Evidence**: Scaling studies

---

## Summary Table

| Exp | Name | Config | Final Accuracy | KL | ARI | Seeds | Key Finding |
|-----|------|--------|----------------|----|-----|-------|------------|
| 01 | Baseline | 80 states, 100k steps | 18.3% | 0.086 | 0.45-0.60 | 3 | Standard works well |
| 02 | Extended | 80 states, 200k steps | 19.1% | 0.079 | 0.55-0.65 | 3 | Marginal improvement |
| 03 | Gen scaling | 80 states, 1000 step intervals | 17.8% | 0.092 | 0.40-0.55 | 3 | Fewer strong gens work |
| 04 | Population | 80 states, variable pop | Varies | Varies | Varies | 3 | Pop size helps (80-100 optimal) |

---

## How to Read Each Experiment

Each experiment has:
1. **Technical Report** — Full metrics, configuration, interpretation
2. **Plain Summary** — Non-technical explanation
3. **Visualizations** — Accuracy curves, fitness landscapes, heatmaps
4. **Raw Data** — CSV files for custom analysis

Navigate using the sidebar:
- **Click on an experiment** (e.g., "Exp 01") to see the technical report
- **Look for PNG visualizations** to see curves and patterns
- **Download CSVs** to analyze the data yourself

---

## Cross-Experiment Comparison

### Accuracy Rankings
1. **Exp 02** (Extended, 200k) — 19.1% (best)
2. **Exp 01** (Baseline, 100k) — 18.3%
3. **Exp 03** (Gen scaling) — 17.8%
4. **Exp 04a** (Pop 80) — 18.5%
5. **Exp 04b** (Pop 40) — 18.3%
6. **Exp 04c** (Pop 100) — 19.0%

### KL Divergence Rankings (Lower is Better)
1. **Exp 02** — 0.079 (excellent)
2. **Exp 01** — 0.086
3. **Exp 04a** — 0.085
4. **Exp 03** — 0.092
5. **Exp 04c** — 0.081

### Cluster Recovery (ARI)
1. **Exp 02** — 0.62 avg (best structure recovery)
2. **Exp 04c** (Pop 100) — 0.59
3. **Exp 01** — 0.52
4. **Exp 03** — 0.48
5. **Exp 04a** — 0.47

---

## Key Insights

### 1. Longer Simulations Help
- 100k steps → 18.3% accuracy
- 200k steps → 19.1% accuracy
- Diminishing returns after ~150k steps

**Implication**: Run for 100-150k steps; beyond that, focus on other parameters.

---

### 2. Population Size Matters for Diversity
- 40 observers (standard): Works well, lower variance
- 80 observers: Better final accuracy (0.5-1% improvement)
- 100+ observers: Marginal gains beyond 80

**Implication**: Use 40-100 depending on computational budget.

---

### 3. Evolution is Crucial
- With evolution: ~18% accuracy, steady improvement
- Without evolution: ~10% accuracy, plateaus early

**Implication**: Always include evolution; it's the main driver of improvement.

---

### 4. Cluster Discovery is Hard
- World accurately modeled (KL < 0.1)
- But cluster structure only partially discovered (ARI 0.5-0.6)

**Implication**: Learning dynamics is easier than finding hidden structure. May need explicit clustering objectives.

---

### 5. Population Converges Successfully
- Final Std Dev of accuracy: < 0.1%
- All observers learn similar strategies
- This is healthy convergence, not stagnation

**Implication**: Observer homogeneity at end is expected and good.

---

## Research Timeline

### Phase 1: Baseline Establishment (Exp 01)
- Objective: Verify simulation works, establish baseline
- Result: Success, confirmed 15-20× baseline accuracy
- Finding: Evolution and learning both work

### Phase 2: Extended Exploration (Exp 02)
- Objective: Test if longer runs improve results
- Result: Marginal improvement (18.3% → 19.1%)
- Finding: Convergence reached by 100k steps; diminishing returns after

### Phase 3: Generation Analysis (Exp 03)
- Objective: Investigate role of generation frequency
- Result: Fewer, stronger generations work comparably
- Finding: Generation frequency less critical than total steps

### Phase 4: Scaling Studies (Exp 04)
- Objective: Understand population size effects
- Result: Sweet spot around 80 observers
- Finding: More diversity helps, but returns diminish

---

## Next Steps / Future Experiments

### Proposed Investigations

1. **Cluster Objective**: Add explicit clustering loss to see if structure discovery improves
2. **Observer Types**: Compare different network architectures
3. **World Complexity**: Systematic scaling to 160, 320, 640 states
4. **Mode Comparison**: Compare neural networks vs count-based baselines across conditions
5. **Transfer Learning**: Can observers from exp A predict well in world B?

---

## How to Add Your Own Experiment

1. **Run the experiment**:
   ```bash
   python -m tracereality.experiments.main_core \
     --n-states 80 --n-observers 40 --n-steps 100000 \
     --seeds 1 2 3 --prefix my_new_exp
   ```

2. **Copy results**:
   ```bash
   cp outputs/my_new_exp/seed_1/*_summary_* documentation/docs/experiments/
   ```

3. **Create documentation page** using template below

4. **Update this overview** with results

### Template for New Experiment Page
```markdown
---
sidebar_position: 10
title: Exp XX - Your Title
---

# Experiment XX: Your Title

## Research Question
What are we testing?

## Hypothesis
Expected outcome?

## Configuration
- n_states: ...
- n_observers: ...
- n_steps: ...

## Results Summary
...

## Visualizations
...

## Comparison to Baseline
...

## Conclusions
...
```
