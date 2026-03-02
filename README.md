# Game of Life Bench

`game-of-life-bench` is a Python benchmark runner for testing whether LLMs can design long-lived initial boards for Conway's Game of Life.

## What It Measures

- Grid: `8 x 8`
- Topology: toroidal
- Rule: Conway's Game of Life (`B3/S23`)
- Initial live-cell cap: `32` cells
- Score: steps until the first repeated global state
- Default rollout cap: `1000` steps

Because the board is finite and deterministic, every run eventually repeats. The benchmark therefore measures how long a model can delay the first repeated state.

## Install

Install from a local checkout:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e .
```

To use the local web UI:

```bash
python -m pip install -e ".[web]"
```

## CLI Quickstart

Run a benchmark batch:

```bash
game-of-life-bench benchmark --models openai/gpt-5.2 anthropic/claude-sonnet-4.6 --trials 10
```

This writes:

- per-trial artifacts to `runs/`
- benchmark summaries to `benchmarks/`

Export a leaderboard snapshot from saved benchmark files:

```bash
game-of-life-bench leaderboard --json --out leaderboard.json
```

Serve the local web app:

```bash
game-of-life-bench serve
```

## Python API

```python
from game_of_life_bench import evaluate_board

board = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
]

result = evaluate_board(
    board=board,
    rows=4,
    cols=4,
    rule="B3/S23",
    topology="toroidal",
    max_steps=100,
    max_live_fraction=0.5,
)

print(result.score)
print(result.simulation.period)
```

Useful entry points:

- `game_of_life_bench.BenchmarkRunner`
- `game_of_life_bench.LifeSimulator`
- `game_of_life_bench.evaluate_board`
- `game_of_life_bench.build_leaderboard`
- `game_of_life_bench.build_leaderboard_payload`

## Web UI

The optional web app is a local interface for exploring boards and inspecting saved results. It uses the same `runs/` and `benchmarks/` artifacts as the CLI.

Start it with:

```bash
game-of-life-bench serve
```

Then open `http://127.0.0.1:8000`.

Pages:

- `/`: simulator and manual board runner
- `/benchmark`: local leaderboard view backed by saved benchmark JSON

How it works:

- the browser talks to the local FastAPI app
- model-backed runs go through the server, which calls OpenRouter using `OPENROUTER_API_KEY`
- completed runs are written to `runs/`
- benchmark batches are written to `benchmarks/`
- the leaderboard page reads saved benchmark outputs rather than recomputing scores in the browser

## Configuration

The benchmark runner uses `OPENROUTER_API_KEY` for model-backed runs.

Example:

```bash
export OPENROUTER_API_KEY=...
game-of-life-bench benchmark --models openai/gpt-5.2 --trials 10
```
