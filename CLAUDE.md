# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal (solo) project building a super-human agent for the board game Quoridor, using an
AlphaZero-style MCTS+neural-network agent. It originated as a team "Advanced Machine Learning"
course project (see [README.md](README.md), [CHECKPOINT.md](CHECKPOINT.md)) that also explored
tabular RL baselines (Q-learning/SARSA) — that RL track (`rl_models.py`, `rl_train_evaluate.py`)
was course-scoped and is no longer being developed; don't spend effort maintaining or extending
it. There's also a Flask web app and a static GitHub Pages frontend (`docs/`) for playing against
exported models in-browser via ONNX.

**Read [docs/cpp_selfplay_notes.md](docs/cpp_selfplay_notes.md) before touching `cpp/engine.hpp`,
`cpp/selfplay.hpp`, `cpp/tournament.hpp`, `selfplay_cpp.py`, `tournament_cpp.py`, `train.py`, or
`cpp_train_loop.py`** — it has detailed benchmark history and design rationale not repeated here.
[TARGET_MACHINE_BENCHMARK_PLAN.md](TARGET_MACHINE_BENCHMARK_PLAN.md) is the re-benchmarking
checklist for the CUDA target machine (all numbers measured so far are from an M1 MacBook Air/MPS
and are NOT valid throughput baselines for CUDA — see below).

## Environment

- Python via `.venv` (3.13). Run scripts with `.venv/Scripts/python` (Windows) — there is no
  system-wide install of the dependencies.
- Install deps: `pip install -r requirements.txt` (flask, waitress, numpy, torch, tqdm, pandas,
  joblib, pybind11).
- No test framework/pytest config — verification scripts are plain argparse CLIs run directly
  (see below), not `pytest`-discovered tests.
- Development/benchmarking has mostly happened on an M1 MacBook Air (MPS). Target production
  machine is a Ryzen 8c/8t + RTX 5070 Ti, 32GB (CUDA). **MPS and CUDA throughput numbers are not
  comparable** — treat any M1-measured games/hour, evals/s, or tuned batch sizes as "the pipeline
  works," not as a value to reuse on CUDA. Re-sweep `max_batch`/`parallel`/TT caps on whichever
  machine you're actually using.

## Build the C++ extension

```bash
python setup_cpp.py build_ext --inplace
```

Produces `quoridor_cpp*.pyd`/`.so` from `cpp/bindings.cpp` (pybind11, C++17). `-O3` is used on
non-Windows; MSVC gets `/O2` (MSVC doesn't understand `-O3`).

**Build gotcha:** `build_ext --inplace` does not reliably detect header-only changes in
`engine.hpp`/`selfplay.hpp`. After editing any `cpp/*.hpp`, delete `build/` and the compiled
extension before rebuilding, or you'll silently test stale code:

```bash
rm -rf build && rm -f quoridor_cpp*.pyd quoridor_cpp*.so
python setup_cpp.py build_ext --inplace
```

## Common commands

```bash
# Verify the C++ engine matches the Python reference engine (run after any engine change)
python test_cpp_parity.py --games 50

# Full AlphaZero training loop (C++ self-play + train, alternated as subprocesses)
python cpp_train_loop.py --cycles 20 --games 500 --sims 400

# Self-play data generation only (writes cycle_NNNN.npz, does not touch best.pt)
python selfplay_cpp.py --model models_9x9/best.pt --out-dir data_9x9 --games 1024 --sims 800

# Train on existing data only (updates best.pt/checkpoint, generates no new data)
python train.py --resume --train-only --cycles 1

# C++ round-robin Elo tournament between checkpoints
python tournament_cpp.py --dir models_9x9/checkpoints --games 100

# Play against the agent in a browser
python app.py --port 5000
```

`selfplay_cpp.py` and `train.py` each tee stdout/stderr to `logs/selfplay_<timestamp>.log` /
`logs/train_<timestamp>.log` (smoke-test runs of `train.py` skip file logging).

## Architecture

### Two independent engines kept in parity

`game.py` (`State`, game.py:99) is the canonical pure-Python rules engine — board/wall state,
legal-move generation, BFS pathfinding, action encoding (`action_to_index`/`index_to_action`).
`cpp/engine.hpp` + `cpp/bindings.cpp` compile to `quoridor_cpp`, a **second, from-scratch
implementation** of the same rules (not a wrapper around `game.py`) used for fast batched
self-play/tournaments. The two are kept behaviorally identical by `test_cpp_parity.py`, which
plays identical action sequences through both and asserts matching legal moves/results across
board sizes. Any change to game rules must be made in both places and re-verified with
`test_cpp_parity.py`.

### AlphaZero pipeline (two parallel implementations: pure-Python and C++-accelerated)

- **Pure Python**: `train.py` does the full self-play → train → checkpoint loop in-process using
  `game.py` + `mcts.py` + `dual_network.py`. No promotion gate — each cycle unconditionally
  overwrites `best.pt` after training (no win-rate check against the previous best); periodic
  full checkpoints go to `<MODEL_DIR>/checkpoints/cycle_NNNN.pt`.
- **C++-accelerated**: `selfplay_cpp.py` (games generation only, via `quoridor_cpp.SelfPlayManager`,
  writes `train.py`-compatible `.npz`) and `train.py --train-only` (training only, no self-play)
  are separate halves of the same loop. `cpp_train_loop.py` alternates them as subprocesses for N
  cycles — this is the actual production training entry point, not `train.py` alone.
- `mcts.py` (`MCTSAgent`) implements PUCT search against an `Evaluator` interface (NN or rollout),
  used by the pure-Python paths (`train.py`, `tournament.py`).
- `dual_network.py`'s `DualNetwork` is the policy+value ResNet (optional KataGo-style global
  pooling block). `load_model`/`_infer_arch` auto-detects filters/blocks/boardsize/gpool from a
  checkpoint's `state_dict`, so checkpoints are self-describing — you don't need to know a
  checkpoint's architecture to load it.

### Model/data directory lineages

Each board-size lineage is a matched `models_<N>x<N>` / `data_<N>x<N>` pair (checkpoints and
self-play data are architecture/board-size-specific and not interchangeable across lineages).
`models_9x9`/`data_9x9` is the currently active lineage in `train.py`, `cpp_train_loop.py`, and
`app.py`'s serving defaults. `models_7x7`/`data_7x7` is an older lineage, still used by
`tournament_cpp.py`'s default `--dir`. When changing which lineage a script targets, its
model-dir and data-dir constants must be changed together — they're paired, not independent.
Watch for stale hardcoded `best.pt` paths in eval-opponent lists (e.g. `train.py`'s
`EVAL_OPPONENTS`) after changing a script's active lineage — a hardcoded opponent path equal to
the current model dir's `best.pt` silently turns evaluation into a self-comparison.

### RL baseline track (legacy, not actively developed)

`rl_models.py` defines tabular agents (`QLearningAgent`, `SarsaAgent`, `ExpectedSarsaAgent`, and
Double- variants) hardcoded to a small `BOARDSIZE = 3` — this track targets tiny boards, not the
7x7/9x9 boards the AlphaZero track uses. `rl_train_evaluate.py` is its self-play/eval/promote
driver, writing to its own `BEST_DIR/*.pkl`, unrelated to `models_*`/`data_*`. This was a
course-requirement comparison point and is no longer part of the project's direction — treat it
as frozen/legacy, not something to extend.

### Evaluation / benchmarking

`benchmark_agents.py` defines shared baseline opponents (`RandomAgent`, `MinimaxAgent`,
`GreedyDistanceAgent`) used by `benchmark_selfplay.py`, `eval_vs_minimax.py`, `eval_exploit.py`.
`tournament.py` (pure Python) and `tournament_cpp.py` (`quoridor_cpp.TournamentManager`) both run
round-robin Elo tournaments between checkpoints — `tournament_cpp.py` deliberately processes
pairs one at a time (not concurrently) due to GIL contention across multiple loaded models; only
worth using for large `--games` per pair (100s), not small smoke scans. Minimax agents currently
only exist in Python (`eval_vs_minimax.py`); no C++ port yet.

### Serving and browser frontend

`app.py` is a Flask server (default 9x9, `models_9x9`) for playing against a served PyTorch model.
`export_onnx.py` converts trained `.pt` checkpoints to ONNX for `docs/models*/`; `export_simple_rl_json.py`
dumps the tabular RL agents to JSON. `docs/` is a separate static GitHub Pages frontend —
`docs/game.js` and `docs/mcts_worker.js` are a from-scratch **JavaScript** reimplementation of the
engine/MCTS (a third engine implementation, running the exported ONNX model client-side), not a
consumer of `game.py`/`quoridor_cpp`. Its board-variant default (7x7) does not necessarily match
`app.py`'s default (9x9) — check both if changing default board size anywhere.

## Working notes

- Files prefixed `_` at the repo root (e.g. `_analyze_illegal_policy.py`, `_bench_nn.py`,
  `_replay_game.py`) are one-off analysis/debugging scripts, not part of the maintained pipeline.
- `data_7x7/`, `data_9x9/`, `models_7x7/`, `models_9x9/` contain real training artifacts
  (`.npz` self-play data, `.pt` checkpoints) — large binary files, not source; don't read them
  as code and be careful about what you stage if committing near them.
- `BREAKTHROUGH.md` is a running log of milestone results (e.g. cycle N beating a reference bot)
  — append-only progress notes, not documentation to keep polished.
