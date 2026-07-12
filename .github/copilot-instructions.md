# SigmaQuoridor — key facts

This file is auto-loaded into every Copilot chat request in this repo, so keep it short/high-signal.
Detailed benchmark history, TT design rationale, and past experiment write-ups live in
[docs/cpp_selfplay_notes.md](../docs/cpp_selfplay_notes.md) — read that file on demand when working
on cpp/engine.hpp, cpp/selfplay.hpp, cpp/tournament.hpp, selfplay_cpp.py, tournament_cpp.py, train.py,
or cpp_train_loop.py; don't assume its contents are already in context.

## Environment
- Development/benchmarking so far has been done on an M1 MacBook Air (Apple Silicon, MPS backend) — an exploration/testing machine, not the production target.
- Target production machine: Ryzen 8c/8t + RTX 5070 Ti, 32GB (CUDA).
- MPS and CUDA throughput numbers are **NOT comparable** — treat all M1-measured absolute numbers (games/hour, evals/s, max_batch sweet spots) as validation that the pipeline is correct and stable at scale, not as a performance baseline for the target machine. Re-benchmark tuning parameters (max_batch, TT caps, etc.) on whichever machine is actually being used before trusting a specific value.
- Python: `.venv` (3.14), run via `.venv/bin/python`.
- Build ext: `.venv/bin/python setup_cpp.py build_ext --inplace` (universal2 clang build).

## C++ self-play pipeline
- cpp/engine.hpp — rules/BFS/Zobrist/NN planes, parity-verified vs game.py (test_cpp_parity.py, 5x5/7x7/9x9 pass).
- cpp/selfplay.hpp — SelfPlayManager: worker threads never touch GIL; Python thread drives get_batch/put_results (masked softmax done in C++). No tree reuse (fresh arena per move).
- cpp/bindings.cpp → module `quoridor_cpp` (State, SelfPlayManager, TournamentManager, random_playouts).
- selfplay_cpp.py — inference driver; saves train.py-compatible cycle_NNNN.npz with LR-flip augmentation.

## Build gotcha (important)
- `setup_cpp.py build_ext --inplace` does NOT reliably detect header-only changes (engine.hpp/selfplay.hpp) for incremental rebuilds. Always `rm -rf build && rm -f quoridor_cpp*.so` before rebuilding when only headers changed, or you'll silently test stale code.

## Model/data lineage: models_7x7 / data_7x7 is the active lineage
- `models_7x7_v2`/`data_7x7_v2` was a separate, abandoned experimental lineage (own independent cycle_0001..0146 self-play data/checkpoints) — NOT a continuation of `models_7x7`/`data_7x7` (which continues from cycle 141 onward). The v2 branch was abandoned; `models_7x7`/`data_7x7` is the one actively trained/used going forward.
- `train.py`: `MODEL_DIR = "models_7x7"`, `DATA_DIR = "data_7x7"` — these two must always be changed together since they're a matched pair per lineage. `ENABLE_HOLDOUT_EVAL = False` (holdout is slow); holdout print message distinguishes "disabled" vs "no data found".
- **Watch out**: `EVAL_OPPONENTS` in `train.py` must never hardcode a path that equals the current `MODEL_DIR`'s `best.pt` — that would silently turn eval into a self-comparison. Any `"old-best Ns"`-style fixed-checkpoint opponents referencing `models_7x7/best.pt` are commented out while `MODEL_DIR == "models_7x7"`. Grep `EVAL_OPPONENTS` for hardcoded model paths if `MODEL_DIR` is ever changed again.
- `cpp_train_loop.py`'s `MODEL_PATH`/`DATA_DIR` constants and `app.py`'s serving `MODEL_DIR` for boardsize 7 must match this same lineage (`models_7x7/best.pt`, `data_7x7`).

## Target production self-play parameters
- Real training cycles run close to: `--sims 800 --parallel 1024 --leaf-batch 1 --max-batch 256 --threads 8 --boardsize 7 --walls 5 --tt-max-depth -1 --tt-max-entries 20000000`. Don't draw conclusions from smaller smoke-test configs (e.g. sims=64) and assume they transfer to this scale — see docs/cpp_selfplay_notes.md for why.
- `temp_threshold` default is 20 (raised from 14) across `cpp/selfplay.hpp`, `cpp/bindings.cpp`, `selfplay_cpp.py`, `train.py` — keep these four in sync if changed again.

## Full C++ self-play + train cycle loop
- Neither `selfplay_cpp.py` (generates cycle_NNNN.npz via C++ engine, never trains/touches best.pt) nor `train.py --train-only` (trains on whatever's on disk + updates best.pt/checkpoint, never generates new data) does the full AlphaZero loop alone.
- `cpp_train_loop.py` alternates the two as subprocesses for N cycles: `selfplay_cpp.py --model models_7x7/best.pt --out-dir data_7x7 ...` then `train.py --resume --train-only --cycles 1`. Usage: `python cpp_train_loop.py --cycles 20 --games 500 --sims 400`.
- `train.py`'s model/data dir paths (`MODEL_DIR="models_7x7"`, `DATA_DIR="data_7x7"`) are hardcoded constants, not CLI flags.

## Console log bookkeeping
- `train.py` and `selfplay_cpp.py` each tee stdout+stderr to `logs/train_<timestamp>.log` / `logs/selfplay_<timestamp>.log` via their own `_Tee`/`_start_run_log()`. Smoke-test runs of `train.py` skip file logging.

## C++ tournament tool (`cpp/tournament.hpp` + `TournamentManager` + `tournament_cpp.py`)
- Ports `tournament.py`'s round-robin Elo tournament to C++. Processes pairs ONE AT A TIME sequentially (see docs/cpp_selfplay_notes.md for why — GIL contention across many concurrent models otherwise).
- Only pays off at large `--games` per pair (100s); don't use for small games-per-pair smoke scans.

## Next steps (deferred by user)
- Gumbel MCTS root-only modification — after infra proves out on the target PC.
- Consider re-measuring TT hit-rate/depth cutoff (`_measure_opening_overlap.py`) periodically across training cycles since the useful depth grows as the network converges.
- Port the existing minimax agents (currently Python-only, see `eval_vs_minimax.py`) to C++ — deferred, not yet started.
