---
sidebar_position: 2
title: Core Concepts
---

# Core Concepts

## The World (Ground Truth)

### States and Transitions
The **world** is a discrete, finite Markov process with:
- **N states** (e.g., 80 distinct configurations)
- **Transition matrix**: A probability distribution P(next_state | current_state)
- **Stationary distribution**: The long-run probability of occupying each state

In TraceReality, the world's transition matrix is fixed before the simulation and never changes. It represents ground truth that observers must learn.

### Cluster Structure
The world has hidden **clusters** — groups of states that share similar transition patterns. For example:
- 80 states organized into 8 clusters
- States within a cluster tend to transition to other states in the same cluster (high within-cluster bias)
- Cross-cluster transitions are less likely (between-cluster noise)

This structure is **not visible** to the observers — they must discover it through observation and evolution.

### Key World Parameters
| Parameter | Typical Value | Meaning |
|-----------|---------------|---------|
| `n_states` | 80 | Number of world states |
| `n_clusters` | 8 | Hidden cluster groups |
| `self_loop_bias` | 0.3 | P(stay in same state) |
| `within_cluster_bias` | 0.5 | P(move to same cluster) |
| `between_cluster_noise` | 0.2 | P(random cross-cluster transition) |

---

## Observers (Learning Agents)

### What Observers Do
Each observer is a **neural network** that learns to predict state transitions:

```
INPUT: [current_state, history_window]
  ↓
NETWORK: embeddings → hidden layers → output layer
  ↓
OUTPUT: predicted_next_state (probability distribution)
```

### Learning Mechanism
Observers learn via **supervised learning**:
1. **Observe** the world's transitions over time
2. **Train** on (current_state, true_next_state) pairs
3. **Update weights** to minimize prediction error
4. **Evaluate** on held-out test data

The learning rate and training dynamics are configurable.

### Network Architecture
| Component | Default | Purpose |
|-----------|---------|---------|
| Embedding dim | 32 | Compressed state representation |
| Hidden dim | 128 | Latent feature extraction |
| History window | 3 | How many past steps to consider |
| Activation | ReLU | Non-linearity in hidden layers |

---

## Evolution (Selection & Mutation)

### The Lifecycle
Each observer has a **fitness score** based on prediction accuracy. Over evolutionary time:

1. **Reproduction**: Best observers (top-k) are copied and kept
2. **Mutation**: Their copies are perturbed (weights randomly changed)
3. **Replacement**: Worst observers (bottom-k) are discarded
4. **Selection interval**: This happens every N steps

### Key Evolution Parameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `selection_interval` | 5000 | Steps between evolution events |
| `selection_top_k` | 10 | How many best observers to keep |
| `selection_bottom_k` | 10 | How many worst observers to replace |
| `mutation_rate_mean` | 0.01 | Average perturbation strength |
| `mutation_rate_std` | 0.005 | Variation in perturbation strength |

### Example Evolution Timeline
```
Generation 0 (step 0):
  - Population: 40 random observers
  - Accuracy: ~1.2% (random baseline for 80 states)

Generation 1 (step 5000):
  - Top 10 observers keep their weights + small mutations
  - Bottom 10 observers are replaced with mutated copies of top 10
  - Accuracy improves to ~5-8%

Generation 2-19 (steps 10000-95000):
  - Repeated selection/mutation cycle
  - Accuracy continues to improve (often to 15-20%)

Final state (step 100000):
  - Population has converged on successful strategies
  - Best observers much better than random baseline
```

### Why Evolution Helps
- **Selection** favors observers that predict well (differential survival)
- **Mutation** provides variation for selection to act on
- **Population diversity** prevents premature convergence
- **Generations** allow cumulative adaptation

---

## Population Dynamics

### Genetic Diversity
The population maintains variation in two ways:

1. **Parameter diversity**: Different observers have different weights
2. **Mutation rates**: Different observers mutate at different rates

Without diversity, evolution would stagnate (no material for selection to act on).

### Convergence vs. Exploration
- **Early generations**: High diversity, rapid fitness improvement
- **Late generations**: May converge (all observers similar) or maintain diversity
- **Healthy populations**: Show both exploration (some diversity) and exploitation (top observers survive)

### Measuring Diversity
- **Pairwise KL divergence**: How different observers' predictions are from each other
- **Parameter variance**: How different their weights are
- **Accuracy spread**: Standard deviation of observer accuracies

---

## Observability and Noise

### What Observers See
Observers don't see states directly — they make **noisy observations**:

```
True state = 42
Noisy observation = 42 + random_noise
Observation noise = 0.05 (5% of max observation value)
```

This adds realism: observers can never see the world perfectly.

### Implications
- Observers must learn robust internal models
- Pure memorization won't work
- Generalization matters

---

## Performance Metrics

### Accuracy
- **Random baseline**: 1 / n_states (e.g., 1/80 ≈ 1.2% for 80 states)
- **Good performance**: > 2× random (> 2.4%)
- **Excellent performance**: > 10× random (> 12%)

### KL Divergence (World ‖ Observer)
Measures how different the observer's learned transition model is from the true world:

- **< 0.05**: Excellent match
- **0.05 - 0.5**: Moderate match (broad patterns learned, details missed)
- **> 0.5**: Poor match

### Fitness
Population fitness = mean prediction accuracy. Trends:
- **Increasing**: Evolution is effective, observers learning
- **Flat**: Population converged or evolution stalled
- **Decreasing**: Problem in simulation or evolution parameters

### Cluster Recovery (ARI - Adjusted Rand Index)
How well observers recovered the hidden cluster structure:
- **> 0.7**: Excellent recovery
- **0.3 - 0.7**: Partial recovery
- **< 0.3**: Little to no recovery

---

## Simulation Parameters: Quick Reference

### World Configuration
```python
--n-states 80              # Number of world states
--n-clusters 8             # Hidden cluster groups
--self-loop-bias 0.3       # P(stay in place)
--within-cluster-bias 0.5  # P(stay in cluster)
--between-cluster-noise 0.2 # Cross-cluster transition prob
```

### Population & Evolution
```python
--n-observers 40           # Population size
--n-steps 100000           # Total simulation steps
--selection-interval 5000  # Steps between generations
--selection-top-k 10       # Keep best observers
--selection-bottom-k 10    # Replace worst observers
--mutation-rate-mean 0.01  # Average mutation strength
```

### Observer Learning
```python
--learning-rate 0.001      # Gradient descent step size
--hidden-dim 128           # Neural network hidden layer size
--embedding-dim 32         # State embedding size
--history-window 3         # Past states to consider
--observation-noise 0.05   # Observation noise level
```

### Execution
```python
--seeds 1 2 3             # Random seeds for reproducibility
--mode nn                 # "nn" (neural) or "count" (baseline)
--prefix my_exp           # Output directory prefix
--no-evolution            # Disable evolution (learning only)
```
