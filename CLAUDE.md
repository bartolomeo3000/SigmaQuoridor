# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal (solo) project building a super-human agent for the board game Quoridor, using an
AlphaZero-style MCTS+neural-network agent. It originated as a team "Advanced Machine Learning"
course project (see [README.md](README.md)) that also explored tabular RL baselines
(Q-learning/SARSA). **That RL track was deleted on 2026-07-25** — `rl_models.py`,
`rl_train_evaluate.py`, `export_simple_rl_json.py`, the Flask/JS agent options, and
`docs/models/simple_rl/` are all gone. Don't reintroduce it. There's also a Flask web app and a
static GitHub Pages frontend (`docs/`) for playing against exported models in-browser via ONNX.

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
- No test framework/pytest config — the verification scripts in `tests/` are plain argparse CLIs
  run directly (`python tests/test_cpp_parity.py --games 50`), not `pytest`-discovered tests.
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
python tests/test_cpp_parity.py --games 50

# Full AlphaZero training loop (C++ self-play + train, alternated as subprocesses)
python cpp_train_loop.py --cycles 20 --games 500 --sims 400

# Self-play data generation only (writes cycle_NNNN.npz, does not touch best.pt)
python selfplay_cpp.py --model runs/models_9x9/best.pt --out-dir runs/data_9x9 --games 1024 --sims 800

# Train on existing data only (updates best.pt/checkpoint, generates no new data)
python train.py --resume --train-only --cycles 1

# C++ round-robin Elo tournament between checkpoints
python tournament_cpp.py --dir runs/models_9x9/checkpoints --games 100

# Play against the agent in a browser
python app.py --port 5000
```

`selfplay_cpp.py` and `train.py` each tee stdout/stderr to `runs/logs/selfplay_<timestamp>.log` /
`runs/logs/train_<timestamp>.log` (smoke-test runs of `train.py` skip file logging).

## Architecture

### Two independent engines kept in parity

`game.py` (`State`, game.py:99) is the canonical pure-Python rules engine — board/wall state,
legal-move generation, BFS pathfinding, action encoding (`action_to_index`/`index_to_action`).
`cpp/engine.hpp` + `cpp/bindings.cpp` compile to `quoridor_cpp`, a **second, from-scratch
implementation** of the same rules (not a wrapper around `game.py`) used for fast batched
self-play/tournaments. The two are kept behaviorally identical by `tests/test_cpp_parity.py`, which
plays identical action sequences through both and asserts matching legal moves/results across
board sizes. Any change to game rules must be made in both places and re-verified with
`tests/test_cpp_parity.py`.

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

**All generated artifacts live under `runs/`** — model lineages, self-play data, tournament
results (`runs/tournaments/`), and logs (`runs/logs/`, the only gitignored part). Source code
stays at the repo root; `docs/` is pinned there too because GitHub Pages serves it. Paths are
resolved relative to the current working directory, so run everything **from the repo root**.

Each board-size lineage is a matched `runs/models_<N>x<N>` / `runs/data_<N>x<N>` pair (checkpoints
and self-play data are architecture/board-size-specific and not interchangeable across lineages).
`models_9x9`/`data_9x9` is the currently active lineage in `train.py`, `cpp_train_loop.py`, and
`app.py`'s serving defaults. `models_7x7`/`data_7x7` is an older lineage, still used by
`tournament_cpp.py`'s default `--dir`. When changing which lineage a script targets, its
model-dir and data-dir constants must be changed together — they're paired, not independent.
Watch for stale hardcoded `best.pt` paths in eval-opponent lists (e.g. `train.py`'s
`EVAL_OPPONENTS`) after changing a script's active lineage — a hardcoded opponent path equal to
the current model dir's `best.pt` silently turns evaluation into a self-comparison.

### Evaluation / benchmarking

`benchmark_agents.py` defines shared baseline opponents (`RandomAgent`, `MinimaxAgent`,
`GreedyDistanceAgent`) used by `benchmark_selfplay.py`, `eval_vs_minimax.py`, `eval_exploit.py`.
`tournament.py` (pure Python) and `tournament_cpp.py` (`quoridor_cpp.TournamentManager`) both run
round-robin Elo tournaments between checkpoints — `tournament_cpp.py` deliberately processes
pairs one at a time (not concurrently) due to GIL contention across multiple loaded models; only
worth using for large `--games` per pair (100s), not small smoke scans. Minimax agents currently
only exist in Python (`eval_vs_minimax.py`); no C++ port yet.

Tournament output lives in `runs/tournaments/<series>/vN.csv` (+ `vN_matchups.csv` and a
`vN_matchups.csv.meta.json` config sidecar); ad-hoc runs default to `runs/tournaments/adhoc/`.
A *series* is a chain of runs that each reuse the previous one's games via `--baseline`, so
adding a checkpoint only plays the new pairings. `tournament_cpp.py --series <name> --add
<model.pt>` derives all of it from the latest `vN` — output path, baseline, roster (from the
`model` column), games/pair, and the rules config (from the sidecar) — so no paths need
hand-writing. Explicit flags override what's inherited, but changing a rules flag makes the
baseline incompatible and silently forces a **full** replay of every pair. The active series
is `scratch_vs_heads` (fresh `models_9x9_scratch` run vs. the old `models_9x9_heads/best.pt`);
see the `tournament-add-cycle` skill. Don't delete a checkpoint that appears in a series
roster — reuse matches on model path, so its pairs become unreplayable.

### Serving and browser frontend

`app.py` is a Flask server (default 9x9, `runs/models_9x9_heads`) for playing against a served
PyTorch model. `export_onnx.py` converts trained `.pt` checkpoints to ONNX for `docs/models*/`.
`docs/` is a separate static GitHub Pages frontend —
`docs/game.js` and `docs/mcts_worker.js` are a from-scratch **JavaScript** reimplementation of the
engine/MCTS (a third engine implementation, running the exported ONNX model client-side), not a
consumer of `game.py`/`quoridor_cpp`. Both its board-variant default and `app.py`'s are 9x9, but
they're set independently (`currentBoardVariant` in `docs/index.html` vs `app.py`'s args) — check
both if changing default board size anywhere.

The whole frontend app is inline in `docs/index.html` (there is no separate app.js/css). Things
worth knowing before editing it:

- **Four modes** (`gameMode`: `hvh`/`hva`/`avh`/`ava`) with **per-side agent configuration** —
  `sideConfig[1|2]` carries each side's agent, sims, minimax depth, temperature and checkpoint,
  and `MODE_AI_SIDES` maps the mode to which sides the AI plays. Anything that used to be a
  single global (agent id, sim count, model path) is now per-side.
- **Temperature** is applied in the worker's `pickFromVisits`, same convention as `mcts.py`
  (0 = argmax, else sample ∝ `visits^(1/T)`). It divides by the max visit count before the power;
  that cancels in the normalisation but keeps `5000^100` from overflowing to Infinity.
- **No status card.** Whose turn it is and which agent is playing are the Players card (active
  row + name); "is it searching" is that row's pulsing `.pawn-dot`; search progress is the
  `.sim-progress` bar at the bottom of that card (numbers in its `title`); the ply counter is the
  timeline label (`viewIdx` *is* the ply); the game result is drawn onto the board by
  `drawGameOver()`.
- **One play worker per distinct checkpoint** (`_playWorkers`, keyed by model path) so `ava` can
  run two different nets without re-initialising an ONNX session every ply. Each worker derives
  its own `modelFullCanonical` from its `?model=` query, which is what keeps the P2 policy frame
  correct per checkpoint — don't collapse this into one worker holding several sessions without
  making that flag per-session. `releaseUnusedPlayWorkers()` reaps the rest.
- **Game history is a timeline, not an undo stack**: `timeline[]` (State snapshots) + `moves[]`
  (`{action,label}`) + `viewIdx`, with `timeline.length === moves.length + 1`. Moving from a
  rewound position truncates the future. `seekTo()` is user navigation (sets `paused` by the
  tip rule); `gotoIndex()` is the mechanical jump used by the replay ticker and must not touch
  `paused`.
- `applyState()` is still the single choke point for "state changed → redraw → decide what's
  next", and `maybeAdvance()` is the only thing that starts a search or a replay step. One timer
  (`_advanceTimer`), always cleared first.
- `moveDelayMs` (the "Playback delay" slider) is a *minimum* display time, not a fixed sleep, and
  it only paces the two cases where you're spectating: `ava` live moves (applied in the worker's
  `move` reply, gated on `gameMode === 'ava'`) and timeline replay (in `maybeAdvance`). It must
  never delay the AI's reply in `hva`/`avh` — the human is waiting on that move.
- The analysis panel deliberately keeps its **own** worker and `analysisModelPath`, separate from
  the per-side play workers.

## Working notes

- `tools/` holds one-off analysis/debugging/setup scripts (e.g. `tools/_analyze_illegal_policy.py`,
  `tools/_bench_nn.py`, `tools/reset_heads.py`, `tools/init_head_redesign.py`) — not part of the
  maintained pipeline. The `_`-prefixed ones keep that prefix for continuity; scripts moved in
  later (`reset_heads.py`, `init_head_redesign.py`) don't have it, so the prefix no longer means
  anything beyond history. `tests/` holds the verification CLIs and follows the same rules.
  Run both from the repo root (`python tools/_bench_nn.py`, `python tests/test_cpp_parity.py`): they
  import their directory's `_bootstrap.py` first (each dir has its own copy), which puts the repo
  root on `sys.path` so `game`, `dual_network`, `mcts`, `train` and the compiled `quoridor_cpp`
  resolve from one level up. `_bootstrap` fixes imports only, not the working directory — several
  still use cwd-relative data paths (`models_7x7/best.pt` etc.), so don't `cd` into the directory
  to run them. A new script in either dir that imports a project module needs
  `import _bootstrap  # noqa: F401` above that import, or it will fail with ModuleNotFoundError.
- `data_7x7/`, `data_9x9/`, `models_7x7/`, `models_9x9/` contain real training artifacts
  (`.npz` self-play data, `.pt` checkpoints) — large binary files, not source; don't read them
  as code and be careful about what you stage if committing near them.
- `BREAKTHROUGH.md` is a running log of milestone results (e.g. cycle N beating a reference bot)
  — append-only progress notes, not documentation to keep polished.
