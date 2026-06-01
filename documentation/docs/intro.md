
---
sidebar_position: 2
title: Introduction
---

# `TraceReality` 

Welcome to **TraceReality** — a computational framework for studying how artificial observers learn to model and predict hidden world structures through evolutionary learning.

## What is TraceReality?

TraceReality is a sophisticated simulation platform that explores a fundamental question in cognitive science and machine learning:

> **How well can a population of neural network observers learn the underlying structure of a hidden Markov world through adaptive evolution?**

## The Core Idea

Imagine a hidden world with:
- **States**: A finite set of configurations or patterns
- **Transitions**: Probabilistic rules for how states evolve over time
- **Hidden structure**: Clusters or groups of states that behave similarly

Now imagine a **population of observers** (neural networks) that must:
1. **Observe** the world in action (sequences of state transitions)
2. **Learn** to predict what happens next
3. **Compete** — the best predictors survive to reproduce
4. **Evolve** — successful observers pass mutated copies to the next generation

This creates a feedback loop where evolution selects for observers that best understand the world's true dynamics.

## Key Features

### 🎯 Flexible Experimentation
- Configurable world complexity (number of states, cluster structures)
- Adjustable population size, evolution parameters, and training dynamics
- Multiple observer model types (neural networks, count-based baselines)
- Multi-seed runs for statistical validation

### 📊 Rich Metrics
- **Accuracy**: How often observers predict correctly
- **KL Divergence**: How different observer models are from ground truth
- **Fitness Dynamics**: Evolution progress over generations
- **Population Diversity**: Genetic and phenotypic variation over time
- **Cluster Recovery**: Whether observers discover hidden structure

### 🔍 Comprehensive Output
- Detailed configuration and results tracking
- CSV data for detailed analysis
- Auto-generated technical and plain-language summaries
- Visualizations (accuracy curves, fitness landscapes, heatmaps)

## Research Questions

This framework helps answer questions like:

1. **Can observers learn hidden clusters?** Under what conditions does evolution discover the underlying structure?
2. **How does population size matter?** Does larger population → better learning?
3. **What's the role of evolution?** How much does selection/mutation help vs. pure learning?
4. **Scaling properties**: How does performance change with world complexity?
5. **Diversity vs. Convergence**: Does the population stay diverse or collapse to consensus?

## Quick Navigation

- **[Core Concepts](./core-concepts.md)** — Understand the simulation components
- **[Running Experiments](./running-experiments.md)** — CLI guide and best practices
- **[Results Interpretation](./results-interpretation.md)** — How to read your results
- **[Metrics Reference](./metrics-reference.md)** — Detailed metric definitions
- **[Experiments](./experiments.md)** — Overview of conducted research

## Getting Started

### For the Impatient
```bash
# Run a basic experiment (80 states, 100k steps, 1 seed)
python -m tracereality.experiments.main_core --n-states 80 --n-observers 40 --n-steps 100000 --seeds 1 --prefix my_experiment

# Results appear in outputs/my_experiment/seed_1/
# Read: my_experiment_seed1_summary_explained.md for plain English
#       my_experiment_seed1_summary_technical.md for details
```

### For Systematic Analysis
See [Running Experiments](./running-experiments.md) for:
- Multi-seed statistical runs
- Ablation studies (evolution, baselines)
- Parameter sweeps
- Common analysis patterns

## Project Structure

```
tracereality/
├── src/tracereality/          # Core simulation code
├── tests/                      # Unit tests
├── outputs/                    # Experiment results
├── documentation/              # This documentation site
└── README.md                   # Quick reference
```

## Citation

If you use TraceReality in your research, please cite:

```bibtex
@software{tracereality2024,
  title={TraceReality: Evolutionary Learning in Hidden Markov Worlds},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/dh-v3}
}
```
