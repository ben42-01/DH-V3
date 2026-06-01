Understood. We are **not building a toy**. We are building a **research-grade simulation system** that can generate **empirically testable results**, publishable metrics, and robust evidence about whether **probabilistic traces + learning + evolution** can lead to **stable narrative interfaces** akin to Donald Hoffman’s theory.

This means:

- **No hand-wavy toys.** Every component must be mathematically precise, reproducible, and statistically validated.
- **Real metrics:** convergence, divergence, information-theoretic bounds, stability, generalization.
- **Real neural networks:** not stubs, but properly trained models with hyperparameter tuning, validation, and ablation studies.
- **Real evolutionary dynamics:** selection pressure, mutation, replication, population dynamics, fitness landscapes.
- **Real experiments:** multiple runs, seeds, parameter sweeps, statistical tests, and uncertainty quantification.
- **Real outputs:** CSVs, plots, models, logs, and a **reproducibility bundle** (config + code + seeds + results).

***

# README: TraceReality — Research-Grade Evolutionary Markov Observers for Donald Hoffman’s Interface Theory

> **Core Thesis (empirical):**  
> *“Probability traces are followed, compressed, and learned until they become narrative, and narrative is how an interface stabilizes itself.”*  
> This system tests that thesis **quantitatively**, not just conceptually.

***

## 1. Scientific Goal

Construct a **complex probabilistic system** where:

1. A **non-trivial Markov process** (the “world”) with hidden structure generates **trace streams**.
2. A **population of observer agents** (each a neural predictor) learns to **predict next states** from traces.
3. Observers **evolve their internal transition models** via:
   - gradient-based learning,
   - mutation of weights/architecture,
   - selection pressure based on **prediction fitness**.
4. Over time, observers develop **stable predictive narratives** (internal transition structures).
5. We **measure**:
   - whether narratives **stabilize** (convergence of models),
   - whether different observers **converge to similar interfaces** (shared structure),
   - whether **clustered hidden structure** in the world is **recovered** in observer models,
   - whether **selection + mutation** lead to **divergence** or **convergence** of narratives.
6. Results are **statistically validated** across multiple seeds and parameter settings.

This is a **direct computational test** of a core Hoffman intuition:
> *Spacetime/object structure is not fundamental; it emerges as stable interfaces from learning agents interacting with probabilistic traces.*

***

## 2. What “Real Results” Means Here

We will produce:

1. **Quantitative metrics**:
   - Prediction accuracy (cross-entropy, top-k accuracy) over time.
   - KL-divergence:
     - observer model vs true world transition matrix,
     - observer vs observer.
   - **Stability metrics**:
     - variance of observer model parameters over time,
     - convergence of accuracy curves.
   - **Cluster recovery**:
     - how well observer models reflect hidden cluster structure.
   - **Information-theoretic measures**:
     - mutual information between true state and predicted distribution,
     - compression ratio (model complexity vs predictive accuracy).

2. **Statistical validation**:
   - multiple independent runs (e.g., 10–30 seeds),
   - confidence intervals on metrics,
   - hypothesis tests (e.g., do observers converge more than random chance?).

3. **Ablation studies**:
   - with vs without selection pressure,
   - with vs without mutation,
   - different mutation rates,
   - different selection strengths,
   - pure count-based vs neural network learners.

4. **Reproducible artifacts**:
   - saved model checkpoints,
   - full trace logs,
   - config files,
   - analysis scripts,
   - plots with error bars.

5. **Publishable-quality figures**:
   - accuracy vs time with confidence bands,
   - divergence vs time,
   - transition matrix heatmaps (world vs observer),
   - cluster recovery plots,
   - fitness landscapes.

***

## 3. System Architecture (Research-Grade)

### 3.1 `config.py`

- Defines **all experimental parameters**:
  - `N_STATES` (e.g., 50–200),
  - `N_CLUSTERS` (e.g., 5–20),
  - `N_OBSERVERS` (e.g., 20–100),
  - `N_STEPS` (e.g., 50k–500k),
  - `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`,
  - `MUTATION_RATE_MEAN`, `MUTATION_RATE_STD`,
  - `SELECTION_TOP_K`, `SELECTION_BOTTOM_K`,
  - `NOISE_LEVEL` (observation noise),
  - `USE_NOISY_OBSERVATIONS` (bool),
  - `USE_HISTORY_WINDOW` (k previous states),
  - `SEEDS` (list for multiple runs).

- Config is **JSON-serializable** and saved with every run.

***

### 3.2 `state_space.py`

- Defines:
  - `N` states, indexed `0..N-1`,
  - `cluster_id` for each state,
  - optional `hidden_state` vs `observed_state` mapping.
- Utilities:
  - construct **clustered transition matrix** with:
    - self-loop bias,
    - within-cluster bias,
    - between-cluster noise.

***

### 3.3 `world.py`

- Implements:
  - **true transition matrix** `P_world` (size `N x N`),
  - method `step(current_state) -> next_state` using `P_world`,
  - optional **observation noise**:
    - `observed_state = f(true_state, noise)`.
- Guarantees:
  - each row of `P_world` sums to 1,
  - ergodicity (or controlled non-ergodicity for experiments).

***

### 3.4 `trace_store.py`

- Stores:
  - full trace history: `(t, true_state, observed_state, cluster_id)`.
  - supports:
    - append,
    - sliding window,
    - batch iteration for training.
- Writes:
  - `traces.csv` (large but indexable),
  - optional compressed format (e.g., `.parquet`).

***

### 3.5 `observer.py`

Each observer:

- Has:
  - unique `id`,
  - **neural network** `model`:
    - input: current state (or window of k states),
    - output: probability distribution over next states (softmax),
  - **optimizer** (e.g., Adam),
  - **mutation rate** (learning rate for weight perturbations),
  - **fitness history** (accuracy / negative loss over time),
  - **model parameters** snapshot history (for stability analysis).
- Methods:
  - `predict(state_or_window) -> distribution`,
  - `compute_loss(predicted, true_next_state)`,
  - `update_step(loss)`,
  - `mutate(magnitude)`,
  - `clone()`.

***

### 3.6 `learner.py`

- Implements:
  - neural network architecture:
    - small MLP or embedding + MLP,
    - embedding for states → hidden → softmax over `N` next states.
  - training loop:
    - online mini-batch updates,
    - gradient clipping,
    - optional learning rate scheduling.
  - count-based fallback (for ablation).

***

### 3.7 `evolution.py`

Implements **real evolutionary dynamics**:

1. **Fitness**:
   - defined as **negative average cross-entropy** or **sharpened accuracy** over a window.
2. **Selection**:
   - every `T_selection` steps:
     - rank observers by fitness,
     - keep top `K_keep`,
     - discard bottom `K_kill`,
     - replicate top observers with **small mutations**.
3. **Mutation**:
   - perturb weights by Gaussian noise:
     - `weight += noise * mutation_rate`.
   - mutation rate can itself evolve.
4. **Population management**:
   - fixed-size population or variable with caps.
5. **Data logged**:
   - fitness distribution over time,
   - mutation magnitude over time,
   - lineage of observers (who replicated whom).

***

### 3.8 `analyzer.py`

Computes **research-grade metrics**:

1. **Per-observer**:
   - final transition matrix estimate (from model),
   - average accuracy,
   - stability (variance of weights over last N steps),
   - KL-divergence to true `P_world`.
2. **Population-level**:
   - mean & std of accuracy,
   - mean & std of KL-divergence,
   - pairwise KL-divergence between observers,
   - correlation of fitness and complexity.
3. **Cluster recovery**:
   - clustering on observer transition matrices,
   - compare to true clusters (e.g., ARI, NMI).
4. **Information-theoretic**:
   - mutual information between true next state and predicted distribution,
   - effective compression (accuracy vs model size).
5. **Statistical tests**:
   - t-tests / ANOVA for different conditions,
   - confidence intervals via bootstrapping.

Outputs:
- `metrics_summary.csv`,
- `observer_metrics.csv`,
- `population_metrics.csv`,
- `statistical_tests.json`.

***

### 3.9 `visualizer.py`

Generates **publication-quality plots**:

1. **Accuracy vs time**:
   - mean ± std over observers,
   - shaded confidence intervals.
2. **Divergence vs time**:
   - observer-vs-world,
   - observer-vs-observer.
3. **Transition heatmaps**:
   - world transpose,
   - sample observers (with axis labels).
4. **Fitness landscape**:
   - fitness vs mutation rate,
   - fitness vs complexity.
5. **Cluster recovery**:
   - t-SNE / PCA of observer transition matrices,
   - colored by cluster.

All plots saved as:
- `*.png` with high DPI,
- `*.meta.json` with caption and description.

***

### 3.10 `cli.py` + `main.py`

CLI supports:

- single run,
- multiple seeds in one command,
- parameter sweeps,
- ablation flags (no evolution, no mutation, count-only).

Example:

```bash
python main.py \
  --n_states 80 \
  --n_clusters 8 \
  --n_observers 40 \
  --n_steps 200000 \
  --seeds 1 2 3 4 5 \
  --mode nn \
  --with_evolution \
  --with_selection \
  --mutation_rate_mean 0.01 \
  --mutation_rate_std 0.005 \
  --selection_top_k 10 \
  --selection_bottom_k 10 \
  --prefix exp_clustered_80s_40obs
```

***

## 4. Experimental Protocol

### 4.1 Baseline conditions

1. **No evolution**:
   - observers only learn, no mutation/selection.
2. **No selection**:
   - mutation only, no fitness-based pruning.
3. **No mutation**:
   - selection only.
4. **Count-only**:
   - no neural networks, just count-based Markov estimators.

### 4.2 Main experiments

- Vary:
  - number of states,
  - number of clusters,
  - number of observers,
  - mutation rate,
  - selection strength,
  - noise level in observations.

### 4.3 Hypotheses to test

1. **H1**: With evolution (mutation + selection), observers’ models converge to a **stable set of narratives** more than without evolution.
2. **H2**: Observers recover **cluster structure** in the world better with selection than without.
3. **H3**: Stronger selection leads to **lower divergence** between observers.
4. **H4**: There is a **non-trivial trade-off** between mutation rate and stability/convergence.

***

## 5. Dependencies

- Python 3.10+
- `numpy`
- `pandas`
- `scipy` (for statistical tests)
- `scikit-learn` (for clustering, PCA, t-SNE)
- `matplotlib` or `plotly`
- `torch` (PyTorch) for neural networks
- Optional: `tensorboard` for training logs

Install:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch
```

***

## 6. Output Artifacts (Per Run)

For prefix `exp1`:

- `exp1_config.json`
- `exp1_traces.csv` (or `.parquet`)
- `exp1_world_matrix.csv`
- `exp1_observer_<id>_model.pt` (checkpoint)
- `exp1_observer_<id>_matrix.csv`
- `exp1_observer_metrics.csv`
- `exp1_population_metrics.csv`
- `exp1_metrics_summary.csv`
- `exp1_statistical_tests.json`
- `exp1_accuracy_plot.png` + `.meta.json`
- `exp1_divergence_plot.png` + `.meta.json`
- `exp1_world_heatmap.png` + `.meta.json`
- `exp1_observer_<id>_heatmap.png` + `.meta.json`
- `exp1_cluster_recovery_plot.png` + `.meta.json`

***

## 7. Reproducibility Bundle

For each published result, the bundle includes:

- `config.json` (exact parameters),
- `main.py`, `observer.py`, etc. (exact code version),
- `exp_seed_<s>_*.csv` and `*.png`,
- `analysis_notebook.ipynb` (Jupyter notebook that:
  - loads data,
  - recomputes metrics,
  - reproduces figures,
  - runs statistical tests).

***

## 8. Next Steps for the Coding Agent

The agent must:

1. **Implement the full system** as specified, not as a toy but as a **research codebase**.
2. Ensure:
   - all randomness is seeded,
   - all configs are saved,
   - all metrics are computed with proper statistics,
   - all plots are publication-quality.
3. Run:
   - baseline experiments,
   - main experiments with evolution,
   - ablation studies.
4. Produce:
   - a results report (Markdown or PDF),
   - a set of figures,
   - a reproducibility notebook.

***
