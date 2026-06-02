
# Pattern 1: Multiple Seeds

uv run -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 100000 \
  --seeds 1 2 3 4 5 \
  --prefix stable_exp

# Pattern 2: Ablation — No Evolution
uv run -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 100000 \
  --seeds 1 2 3 \
  --no-evolution \
  --prefix no_evo_baseline

## Pattern 3: Ablation — Count-Based Baseline

uv run -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 100000 \
  --seeds 1 2 3 \
  --mode count \
  --prefix count_baseline_exp

# Pattern 4: Scaling — More Observers

uv run -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 100 \
  --selection-top-k 50 \
  --selection-bottom-k 50 \
  --n-steps 100000 \
  --seeds 1 2 3 \s
  --prefix more_observers_exp

# Pattern 5: Scaling — More Steps
uv run -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 200000 \
  --seeds 1 2 3 \
  --prefix longer_exp

# Pattern 6: Scaling — Harder World
uv run -m tracereality.experiments.main_core \
  --n-states 160 \
  --n-clusters 16 \
  --n-observers 80 \
  --n-steps 200000 \
  --seeds 1 2 3 \
  --prefix harder_world_exp

# Pattern 8: Aggressive Evolution
uv run -m tracereality.experiments.main_core \
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