# TraceReality

> *Can minds evolve their own reality from pure probability?*

## The Big Idea — In Plain English

Imagine you're born into a universe you know nothing about. You can't see it, hear it, or touch it. All you get is a **stream of symbols** — like `5, 3, 7, 7, 2, 9, 1, 1, 1, 8, ...` — appearing one after another, forever.

You don't know:
- What these symbols mean
- Whether there's a pattern
- Whether you should care about them
- What you're supposed to do

All you can do is watch, remember, and try to make sense of them.

Over time, you start to notice that some symbols tend to follow others. `7` is often followed by `7` again. `5` is usually followed by `3` or `8`, but almost never by `1`. You build a **mental model** — a story — about how this symbol-world works.

Now imagine there are **hundreds of you**, each building your own mental model. Some of you are better at predicting what comes next. You get to "survive" and pass your mental model to others. The worse predictors get replaced by slightly mutated copies of the good ones.

**What happens?**

This is the core question TraceReality is built to answer:

> **Will a group of learning, evolving minds converge on a shared "reality" — a stable story about how the world works — even though all they ever see is a stream of meaningless symbols?**

And importantly: **Will that shared story reflect what's actually true about the hidden world?**

## Why Does This Matter?

This is a direct test of a fascinating idea from cognitive scientist **Donald Hoffman**:

> *What we perceive as space, time, and objects isn't reality itself — it's just a useful interface. A desktop icon isn't the file; it's a simplified view that helps you interact with something far more complex.*

TraceReality simulates this theory. The "hidden Markov world" is the true complex reality. The observer agents are minds trying to make sense of it. The "narratives" they develop are the interfaces they build — the mental icons.

If multiple independent minds, starting from scratch, converge on the same interface purely through prediction and evolution... that's powerful evidence that **stable interfaces can emerge naturally**, without being pre-programmed.

## What the System Actually Does

### Part 1: The Core Simulation

1. **We build a hidden world** — A complex system with 50–200 states organized into 5–20 clusters. States within a cluster tend to follow each other. The world has real structure, but it's hidden from the observers.

2. **We create observer agents** — Each agent is a small neural network (a "brain") that tries to predict what symbol comes next. They start dumb — random predictions.

3. **We run the simulation** — Each agent watches the symbol stream, tries to predict, and learns from its mistakes. Over thousands of steps, their predictions improve.

4. **We add evolution** — Every so often, the worst predictors are replaced by mutated copies of the best predictors. This mimics natural selection: better predictors survive and reproduce.

5. **We measure everything** — Do the agents' internal models converge? Do they agree with each other? Do they discover the hidden cluster structure? We track all of this statistically.

### Part 2: The LLM Experiment (Coming After Core)

The same idea, but instead of small neural networks, we use **large language models** (like Llama, Mistral, or Gemma running locally via Ollama).

The twist: **The LLM is never told what to do.** It's shown raw symbol sequences and asked vaguely to "say something about them." No instructions about prediction. No mention of patterns. Just... free-form commentary.

Then we evolve the *prompts* — the instructions given to the LLM — over generations. The prompts that produce commentary most "aligned" with the hidden structure survive and mutate. The rest die out.

**Will the LLM spontaneously develop stable narrative patterns — attractor states — that reflect the hidden world?** That's the experiment.

## Running Experiments

### Quick Start

```bash
# Install dependencies (you need uv installed)
uv sync

# Run a basic core experiment (use -m flag for proper imports)
uv run python -m tracereality.experiments.main_core \
  --n-states 80 \
  --n-clusters 8 \
  --n-observers 40 \
  --n-steps 100000 \
  --seeds 1 2 3 \
  --prefix my_first_run

# Or use the convenience entry point
uv run python main.py --n-states 80 --n-clusters 8 --n-observers 40 --n-steps 100000 --seeds 1 2 3 --prefix my_first_run
```

This runs 3 independent experiments (seeds 1, 2, 3) with 80 world states in 8 clusters, 40 observer agents, for 100,000 time steps each. Results appear in `outputs/my_first_run/`.

### What You Get

For every run, the system produces:
- **Config** — The exact parameters used (saved as JSON)
- **Traces** — The raw symbol stream
- **Metrics** — Prediction accuracy, divergence, stability scores (CSV files)
- **Models** — Saved neural network weights
- **Plots** — Publication-quality figures with confidence bands
- **Statistical tests** — Are the results significant?

### Running Ablations (Understanding What Matters)

To understand *why* things work (or don't), you can disable components:

```bash
# No evolution — agents just learn individually
uv run python -m tracereality.experiments.main_core ... --no-evolution

# No mutation — selection but no random variation
uv run python -m tracereality.experiments.main_core ... --no-mutation

# No selection — mutation but no淘汰 of bad predictors
uv run python -m tracereality.experiments.main_core ... --no-selection

# Count-based learners — no neural networks, just counting frequencies
uv run python -m tracereality.experiments.main_core ... --mode count
```

### Parameter Sweeps

```bash
# Try different mutation rates
uv run python -m tracereality.experiments.main_core \
  --n-states 80 \
  --mutation-rate-mean 0.001 0.01 0.1 \
  --prefix mutation_sweep
```

The system runs all combinations automatically.

## Output Structure

```
outputs/
├── my_first_run/
│   ├── config.json
│   ├── traces.csv
│   ├── world_matrix.csv
│   ├── observer_metrics.csv
│   ├── population_metrics.csv
│   ├── metrics_summary.csv
│   ├── statistical_tests.json
│   ├── accuracy_plot.png
│   ├── divergence_plot.png
│   ├── world_heatmap.png
│   └── ...
└── analysis_notebook.ipynb
```

Double-click the `.ipynb` to open in VS Code or Jupyter — it loads results and regenerates all figures.

## Performance

The initial implementation had **quadratic time growth** (`O(N²)`) — each step rescanned the entire trace history, making 100,000-step runs take 22+ hours.

This was fixed in **Performance Fix #1** (May 2026):

| Step | Before | After |
|------|--------|-------|
| 1,000 | ~83s | ~2.4s |
| 10,000 | ~300s (est) | ~24s |
| 100,000 | ~22-28h | ~4 min |

**What changed:**
1. **Cached trace arrays** — Eliminated O(t) array copies on every access
2. **Running accuracy counters** — Incremental O(1) updates, not full-history recomputation
3. **Round-robin observer training** — Train 1 observer per step instead of all N
4. **Interval-based heavy analysis** — Full KL/transition-matrix computation only every K steps
5. **Lightweight interval logging** — Uses running counters for frequent log output

The science is identical — all outputs are statistically equivalent.

## Project Status

- ✅ **Phase 1: Core System** — In development (neural observers + evolution)
- ✅ **Performance Fix** — O(N²) → O(N) completed
- 🔄 **Phase 2: LLM System** — Coming after core is validated
- 📋 **Phase 3: Reproducibility Bundle** — Automated bundling + analysis notebook

## Dependencies

This project uses:
- Python 3.10+
- PyTorch (neural networks)
- NumPy, Pandas, SciPy (numerics)
- scikit-learn (clustering, metrics)
- Matplotlib (plots)
- Ollama (for LLM experiments — optional)

All managed via `uv` — one command installs everything.

## Want to Dive Deeper?

- `project-context.md` — The full technical specification for the core neural observer system
- `llm-sim.md` — The full specification for the LLM observer experiment
- `src/tracereality/config.py` — All configurable parameters documented

---
**TraceReality** — Testing whether probability, learning, and evolution produce stable interfaces.


uv run python -m tracereality.experiments.char_llm --model qwen2.5:1.5b --n-states 40 --n-clusters 4 --n-variants 6 --generations 8 --n-eval-fragments 3