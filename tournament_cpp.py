"""C++-backed round-robin tournament between model checkpoints (and/or the
same checkpoint at different simulation budgets).

Reimplements what tournament.py / tournament_simcounts.py do, but on top of
the C++ self-play engine (quoridor_cpp.TournamentManager): worker threads
run leaf-parallel MCTS for many concurrent cross-play games (GIL-free).

Pairs are played SEQUENTIALLY, one pair at a time, each with its own
TournamentManager that only knows about that pair's 1-2 distinct models.
This keeps only 1-2 models "hot" (own inference thread + resident GPU
weights) at any moment -- mirroring self-play's single-model-big-batch
regime -- instead of fanning out one inference thread per distinct model
in the whole roster, which was measured to cause GIL/scheduling contention
that made a big (~10 model) round-robin *slower* than the multiprocess
Python tournament.py. This design pays off specifically in the "several
models x hundreds of games per pair" regime, since parallel_games then
concentrates entirely on 1-2 models per pair, giving large batches.

Two modes, matching the two existing python scripts:

  Checkpoint round-robin (like tournament.py):
    python tournament_cpp.py --dir models_7x7/checkpoints --sims 800 --games 4

  Same model, multiple sim budgets (like tournament_simcounts.py):
    python tournament_cpp.py --model models_7x7/best.pt \
        --sim 200 --sim 800 --sim 2000 --games 2 --temp 0

Elo is computed via tournament.py's Bradley-Terry MLE (reused, not
reimplemented).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import quoridor_cpp
from dual_network import load_model
from tournament import BOARDSIZE, WALLS_PER_PLAYER, compute_elo

LOG_DIR = "runs/logs"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Console logging (same convention as selfplay_cpp.py / train.py) ────────

class _Tee:
    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._log_file.write(data)
        self._log_file.flush()
        try:
            return self._stream.write(data)
        except UnicodeEncodeError:
            # Console encoding (e.g. Windows cp1252) can't encode the box-drawing
            # chars in the results table; degrade gracefully instead of crashing
            # the whole run after all games are already played.
            encoding = getattr(self._stream, "encoding", None) or "ascii"
            safe = data.encode(encoding, errors="replace").decode(encoding)
            return self._stream.write(safe)

    def flush(self) -> None:
        self._log_file.flush()
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


def _start_run_log(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"tournament_{time.strftime('%Y%m%d_%H%M%S')}.log")
    # utf-8 explicitly: default encoding on Windows is cp1252, which can't
    # encode the box-drawing characters used in the results table below.
    log_file = open(path, "a", encoding="utf-8")
    import sys
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return path


# ── Inference thread: one per distinct loaded model ─────────────────────────

def _inference_loop(mgr, model_id: int, model: torch.nn.Module, device: torch.device,
                    max_batch: int, flush_us: int) -> None:
    with torch.inference_mode():
        while True:
            batch_id, states = mgr.get_batch(model_id, max_batch, flush_us)
            b = states.shape[0]
            if b == 0:
                break
            x = torch.from_numpy(states).to(device, non_blocking=True)
            logits, value = model(x)
            mgr.put_results(
                model_id, batch_id,
                logits.float().cpu().numpy(),
                value.float().view(-1).cpu().numpy(),
            )


# ── Roster / matchup construction ───────────────────────────────────────────

def build_roster(args) -> tuple[list[str], list[str], list[int]]:
    """Returns (agent_names, agent_model_paths, agent_sims) — one entry per
    registered agent (agents may share a model_path with different sims, or
    share sims with different model_paths)."""
    if args.sim:
        model_path = args.model
        if not model_path:
            raise SystemExit("--sim requires --model")
        sim_counts = sorted(set(args.sim))
        names = [f"{Path(model_path).stem}_sim{s}" for s in sim_counts]
        paths = [model_path] * len(sim_counts)
        return names, paths, sim_counts

    ckpt_dir = Path(args.dir)
    paths_p = sorted(ckpt_dir.glob("cycle_*.pt"))
    for extra in args.extra:
        ep = Path(extra)
        if ep.exists():
            paths_p.append(ep)
        else:
            print(f"Warning: extra path not found: {extra}")
    names = [p.stem if p.parent == ckpt_dir else str(p) for p in paths_p]
    paths = [str(p) for p in paths_p]
    sims = [args.sims] * len(paths)
    return names, paths, sims


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default="runs/models_7x7/checkpoints",
                   help="Directory containing cycle_*.pt checkpoint files")
    p.add_argument("--extra", action="append", default=[], metavar="PATH",
                   help="Extra model file(s) to include (repeatable)")
    p.add_argument("--model", type=str, default=None,
                   help="Single model to test at multiple --sim budgets")
    p.add_argument("--sim", action="append", type=int, default=[], metavar="N",
                   help="Simulation budget to include with --model (repeatable); "
                        "activates simcounts mode instead of checkpoint round-robin")
    # These eight are "inheritable": under --series they default to whatever the
    # previous version of the series used (so a follow-up run is config-identical
    # by construction, which is what makes --baseline reuse work). Passing one
    # explicitly always wins. default=None is how "not passed" is detected --
    # the real fallbacks live in _INHERITABLE_DEFAULTS.
    p.add_argument("--sims", type=int, default=None,
                   help="Sims/move for checkpoint round-robin mode (default: 800, "
                        "or inherited under --series)")
    p.add_argument("--games", type=int, default=None, metavar="N",
                   help="Games per pair (even; split equally by colour) "
                        "(default: 4, or inherited under --series)")
    p.add_argument("--temp", type=float, default=None,
                   help="Sampling temperature (0 = deterministic argmax) "
                        "(default: 1.0, or inherited under --series)")
    p.add_argument("--c-puct", type=float, default=None)
    p.add_argument("--fpu", type=float, default=None)
    p.add_argument("--boardsize", type=int, default=None)
    p.add_argument("--walls", type=int, default=None)
    p.add_argument("--max-moves", type=int, default=None)
    p.add_argument("--threads", type=int, default=8, help="MCTS worker threads")
    p.add_argument("--parallel", type=int, default=128, help="concurrent games")
    p.add_argument("--max-batch", type=int, default=256)
    p.add_argument("--flush-us", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="Optional CSV path for summary results")
    p.add_argument("--baseline", type=str, default="", metavar="PATH",
                   help="Prior run's *_matchups.csv. Pairs already fully played there "
                        "(matched by model path, same --games count) are reused instead "
                        "of resimulated -- only pairs touching a genuinely new model get "
                        "played. Every run also writes a small sidecar next to its own "
                        "matchups CSV recording sims/temp/c-puct/fpu/boardsize/walls/"
                        "max-moves, so a later --baseline with an incompatible config is "
                        "detected and ignored (with a warning) rather than silently mixed "
                        "into one Elo table.")
    p.add_argument("--series", type=str, default="", metavar="NAME",
                   help="Run as the next version of a tournament series kept in "
                        "<--series-dir>/NAME (e.g. 'scratch_vs_heads'). Finds the latest "
                        "vN there and derives everything from it: --out becomes v(N+1).csv, "
                        "--baseline becomes vN_matchups.csv, the roster is inherited from "
                        "vN.csv's model column, and --games plus the rules config are "
                        "inherited from vN's sidecar -- so an extension run needs no "
                        "hand-written paths, just --add. An empty/new series starts at v1.")
    p.add_argument("--add", action="append", default=[], metavar="PATH",
                   help="Model(s) to append to the roster inherited by --series "
                        "(repeatable). This is the normal way to extend a series: every "
                        "prior model is carried over and reused from the baseline, so only "
                        "pairs involving the added model(s) are actually played.")
    p.add_argument("--series-dir", type=str, default="runs/tournaments", metavar="DIR",
                   help="Root directory holding tournament series (default: tournaments)")

    args = p.parse_args()
    if args.series:
        _apply_series(args)
    # Fill in any inheritable option the user didn't pass and --series didn't supply.
    for name, default in _INHERITABLE_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    return args


# ── Tournament series (tournaments/<name>/vN.csv) ───────────────────────────
#
# A "series" is just a directory of versioned runs that each reuse the previous
# one via --baseline. Everything a follow-up run needs is already recorded in
# the previous version's files, so --series reads it back instead of making the
# caller retype it:
#
#   vN.csv                        -> the roster (its `model` column) + games/pair
#   vN_matchups.csv               -> the --baseline to reuse
#   vN_matchups.csv.meta.json     -> the rules config (sims/temp/c_puct/...)
#
# Deriving the roster from the CSV (rather than a separate manifest) keeps a
# single source of truth: the names stored there are the exact strings
# --baseline matches pairs on, so inheritance can't drift out of sync with it.

_INHERITABLE_DEFAULTS = {
    "sims": 800, "games": 4, "temp": 1.0, "c_puct": 1.0, "fpu": 0.1,
    "boardsize": BOARDSIZE, "walls": WALLS_PER_PLAYER, "max_moves": 100,
}

# Sidecar key -> args attribute. Mirrors _config_signature()'s schema.
_META_TO_ARG = {
    "sims": "sims", "temp": "temp", "c_puct": "c_puct", "fpu": "fpu",
    "boardsize": "boardsize", "walls": "walls", "max_moves": "max_moves",
}


def _series_versions(d: Path) -> list[int]:
    """Version numbers of completed runs in a series dir (v3.csv -> 3).
    Only bare vN.csv counts -- vN_matchups.csv is a companion, not a version."""
    out = []
    for p in d.glob("v*.csv"):
        m = re.fullmatch(r"v(\d+)", p.stem)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _apply_series(args) -> None:
    """Resolve --series into concrete --out/--baseline/--extra/config values."""
    d = Path(args.series_dir) / args.series
    d.mkdir(parents=True, exist_ok=True)
    versions = _series_versions(d)
    inherited: list[str] = []

    if versions:
        n = versions[-1]
        prev_csv, prev_matchups = d / f"v{n}.csv", d / f"v{n}_matchups.csv"
        with open(prev_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        # The `model` column holds the roster names, which for --extra-supplied
        # models are their full paths -- exactly what --baseline matches on.
        inherited = [r["model"] for r in rows]
        # Each model played (k-1) pairs, --games each.
        if args.games is None and len(rows) > 1 and rows[0].get("games"):
            args.games = int(rows[0]["games"]) // (len(rows) - 1)
        meta_p = Path(_meta_path(str(prev_matchups)))
        if meta_p.exists():
            with open(meta_p) as f:
                meta = json.load(f)
            for key, attr in _META_TO_ARG.items():
                if getattr(args, attr) is None and key in meta:
                    setattr(args, attr, meta[key])
        if not args.baseline and prev_matchups.exists():
            args.baseline = str(prev_matchups)
        new_version = n + 1
        print(f"Series {args.series!r}: extending v{n} -> v{new_version}  "
              f"({len(inherited)} model(s) inherited, {len(args.add)} added)")
    else:
        new_version = 1
        print(f"Series {args.series!r}: no existing versions in {d} -- starting at v1")

    if not args.out:
        args.out = str(d / f"v{new_version}.csv")

    # Roster = inherited + --add + any explicit --extra, de-duplicated in order.
    # Compare on the normalised path, not the raw string: the inherited names use
    # OS separators ('a\b.pt') while a hand-typed --add usually has forward
    # slashes ('a/b.pt'). Those are the same file, but as bare strings they look
    # distinct -- which would silently register one model twice as two separate
    # agents (extra pairs played, and its Elo fitted against a copy of itself).
    # Store the normalised form too, so roster names match what --baseline
    # recorded for the same model.
    seen, roster = set(), []
    for m in inherited + list(args.add) + list(args.extra):
        norm = os.path.normpath(m)
        key = os.path.normcase(norm)
        if key not in seen:
            seen.add(key)
            roster.append(norm)
    args.extra = roster
    # build_roster() also globs --dir for cycle_*.pt; point it at the series dir
    # (which has none) so the roster is exactly what we assembled here.
    args.dir = str(d)


def play_pair(args, i: int, j: int, agent_model_ids: list[int],
             agent_num_simulations: list[int], agent_leaf_batch: list[int],
             agent_c_puct: list[float], agent_fpu_reduction: list[float],
             agent_temperature: list[float], models: list[torch.nn.Module],
             device: torch.device, games_per_pair: int,
             progress_cb=None) -> list[tuple[bool, int]]:
    """Play all games for one pair (i, j) with a dedicated TournamentManager
    that only knows about the (1 or 2) distinct models this pair needs.

    Keeping only this pair's model(s) "hot" (their own small inference
    thread(s), a queue each) avoids the GIL/scheduling contention that comes
    from running all-pairs-at-once with one thread per distinct model in the
    whole roster -- and lets parallel_games concentrate entirely on this
    pair, so batches are as large as in single-model self-play.

    Returns a list of (i_is_p1, winner) tuples, one per game, in the order
    P1=i games first, then P1=j games (matches tournament.py's split).
    """
    global_ids = [agent_model_ids[i], agent_model_ids[j]]
    uniq = sorted(set(global_ids))
    remap = {g: local for local, g in enumerate(uniq)}
    local_a, local_b = remap[global_ids[0]], remap[global_ids[1]]

    parallel = max(1, min(args.parallel, games_per_pair))
    mgr = quoridor_cpp.TournamentManager(
        boardsize=args.boardsize, walls=args.walls, max_moves=args.max_moves,
        seed=args.seed, num_threads=args.threads, parallel_games=parallel,
        agent_model_ids=[local_a, local_b],
        agent_num_simulations=[agent_num_simulations[i], agent_num_simulations[j]],
        agent_leaf_batch=[agent_leaf_batch[i], agent_leaf_batch[j]],
        agent_c_puct=[agent_c_puct[i], agent_c_puct[j]],
        agent_fpu_reduction=[agent_fpu_reduction[i], agent_fpu_reduction[j]],
        agent_temperature=[agent_temperature[i], agent_temperature[j]],
    )

    n_per_color = games_per_pair // 2
    p1_agents = [0] * n_per_color + [1] * n_per_color
    p2_agents = [1] * n_per_color + [0] * n_per_color

    threads = [
        threading.Thread(target=_inference_loop,
                         args=(mgr, local, models[g], device, args.max_batch, args.flush_us),
                         daemon=True)
        for g, local in remap.items()
    ]
    mgr.start(p1_agents, p2_agents)
    for t in threads:
        t.start()

    results: list[tuple[bool, int]] = []

    def _drain() -> None:
        res = mgr.get_results()
        for match_idx, winner, plies in zip(res["match_index"], res["winner"], res["plies"]):
            i_is_p1 = match_idx < n_per_color
            results.append((i_is_p1, int(winner), int(plies)))

    while not mgr.is_done():
        time.sleep(0.2)
        _drain()
        if progress_cb:
            progress_cb(mgr.games_finished())
    for t in threads:
        t.join()
    _drain()
    return results


# ── Partial-result checkpointing (survive a mid-run crash) ──────────────────
#
# Pairs are played sequentially, so we snapshot accumulated results after each
# completed pair. A crash then loses at most the one in-flight pair instead of
# the whole run. On restart the snapshot is reloaded and completed pairs are
# skipped — but only if the roster/config signature matches, so a reused --out
# with a different roster can't silently resume the wrong run.

def _progress_path(out_path: str) -> str:
    return str(Path(out_path).with_suffix(".progress.json"))


def _run_signature(names, model_paths, agent_sims, args) -> dict:
    return {
        "roster": [[n, mp, int(s)] for n, mp, s in zip(names, model_paths, agent_sims)],
        "games": args.games, "temp": args.temp, "c_puct": args.c_puct,
        "fpu": args.fpu, "boardsize": args.boardsize, "walls": args.walls,
        "max_moves": args.max_moves, "seed": args.seed,
    }


def _save_progress(path, signature, completed_pairs, game_results, game_log, elapsed) -> None:
    # write-to-tmp + atomic replace, so a crash mid-write can't corrupt the file
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "signature": signature,
            "completed_pairs": [list(p) for p in sorted(completed_pairs)],
            "game_results": [[int(i), int(j), float(s)] for i, j, s in game_results],
            "game_log": game_log,
            "elapsed": elapsed,
        }, f)
    os.replace(tmp, path)


def _load_progress(path, signature):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("signature") != signature:
        return "incompatible"
    return data


# ── Baseline reuse (skip pairs a prior run already fully played) ────────────
#
# --baseline points at a prior run's *_matchups.csv. Pairs are matched by
# MODEL PATH (not roster label -- a checkpoint can be named differently across
# separate invocations), and only reused if the baseline has exactly --games
# games recorded for that pair. A sidecar next to every matchups CSV records
# the per-game rules config (sims/temp/c_puct/fpu/boardsize/walls/max_moves);
# a mismatch there means the baseline games weren't played under comparable
# conditions (e.g. different sims/move), so it's disabled with a warning
# rather than silently merged into one Elo table.

def _config_signature(args) -> dict:
    return {
        "sims": args.sims, "temp": args.temp, "c_puct": args.c_puct,
        "fpu": args.fpu, "boardsize": args.boardsize, "walls": args.walls,
        "max_moves": args.max_moves,
    }


def _meta_path(matchup_path: str) -> str:
    return matchup_path + ".meta.json"


def _write_run_meta(matchup_path: str, args) -> None:
    with open(_meta_path(matchup_path), "w") as f:
        json.dump(_config_signature(args), f)


def _load_baseline(path: str, args) -> dict[frozenset, list[dict]] | None:
    """Returns {frozenset({path_a, path_b}): [matchup rows]} grouped by pair,
    or None if the baseline should be ignored entirely (incompatible config)."""
    if not os.path.exists(path):
        raise SystemExit(f"--baseline file not found: {path}")

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"agent_a", "agent_b", "player1", "player2", "result", "score_for_a", "plies"}
    if rows and not required.issubset(rows[0].keys()):
        raise SystemExit(
            f"--baseline file {path} doesn't look like a *_matchups.csv "
            f"(missing columns: {required - rows[0].keys()})"
        )

    meta_path = _meta_path(path)
    current_cfg = _config_signature(args)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            baseline_cfg = json.load(f)
        if baseline_cfg != current_cfg:
            print(f"WARNING: --baseline {path} was run under a different config "
                  f"({baseline_cfg}) than this run ({current_cfg}) -- ignoring "
                  f"baseline, all pairs will be replayed.\n")
            return None
    else:
        print(f"WARNING: --baseline {path} has no config sidecar ({meta_path} not "
              f"found -- predates this feature). Can't verify it was played under "
              f"the same sims/temp/rules as this run ({current_cfg}); using it "
              f"anyway on trust.\n")

    pairs: dict[frozenset, list[dict]] = {}
    for row in rows:
        pairs.setdefault(frozenset({row["agent_a"], row["agent_b"]}), []).append(row)
    return pairs


def _reuse_baseline_pair(baseline_rows, i, j, names, model_paths):
    """Translate a baseline pair's rows (labeled by model path) into this
    run's (i, j, score_for_i) + game_log dict format. Returns None if the row
    labels don't resolve cleanly onto (model_paths[i], model_paths[j])."""
    path_i, path_j = model_paths[i], model_paths[j]
    out_results = []
    out_logs = []
    for row in baseline_rows:
        a_path, b_path = row["agent_a"], row["agent_b"]
        if a_path == path_i and b_path == path_j:
            score_i = float(row["score_for_a"])
        elif a_path == path_j and b_path == path_i:
            score_i = 1.0 - float(row["score_for_a"])
        else:
            return None  # shouldn't happen -- grouping key already matched these paths
        p1_idx = i if row["player1"] == path_i else j
        p2_idx = i if row["player2"] == path_i else j
        out_results.append((i, j, score_i))
        out_logs.append({
            "agent_a": names[i], "agent_b": names[j],
            "player1": names[p1_idx], "player2": names[p2_idx],
            "result": row["result"], "score_for_a": score_i, "plies": row["plies"],
        })
    return out_results, out_logs


def main() -> None:
    args = parse_args()
    log_path = _start_run_log(LOG_DIR)
    print(f"Logging console output to {log_path}")

    names, model_paths, agent_sims = build_roster(args)
    k = len(names)
    if k < 2:
        print(f"Need at least 2 agents; found {k} (check --dir/--extra or --model/--sim).")
        return
    if args.games % 2 != 0:
        print(f"--games must be even (got {args.games}); rounding up to {args.games + 1}.")
        args.games += 1

    # Deduplicate model paths -> model_id, load each checkpoint once.
    device = pick_device()
    path_to_model_id: dict[str, int] = {}
    models: list[torch.nn.Module] = []
    agent_model_ids: list[int] = []
    for path in model_paths:
        if path not in path_to_model_id:
            path_to_model_id[path] = len(models)
            print(f"Loading {path} ...")
            m = load_model(path, device=device)
            m.eval()
            models.append(m)
        agent_model_ids.append(path_to_model_id[path])

    agent_num_simulations = list(agent_sims)
    agent_leaf_batch = [1] * k
    agent_c_puct = [args.c_puct] * k
    agent_fpu_reduction = [args.fpu] * k
    agent_temperature = [args.temp] * k

    pairs = list(combinations(range(k), 2))
    n_pairs = len(pairs)
    total_games = n_pairs * args.games

    print(f"Agents       : {k}  ({names[0]} … {names[-1]})")
    print(f"Distinct nets: {len(models)}")
    print(f"Pairs        : {n_pairs}  (played sequentially, 1-2 models 'hot' at a time)")
    print(f"Games/pair   : {args.games}  (×{n_pairs} = {total_games} total)")
    print(f"Temperature  : {args.temp}")
    print(f"Threads/parallel: {args.threads}/{args.parallel}")
    print()

    t0 = time.time()
    out_path = args.out or os.path.join("runs/tournaments", "adhoc", "results.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    progress_path = _progress_path(out_path)
    signature = _run_signature(names, model_paths, agent_sims, args)

    game_results: list[tuple[int, int, float]] = []
    game_log: list[dict] = []
    completed_pairs: set[tuple[int, int]] = set()
    elapsed_offset = 0.0

    resumed = _load_progress(progress_path, signature)
    if resumed == "incompatible":
        print(f"Ignoring incompatible progress file {progress_path} "
              f"(roster/config changed); starting fresh.\n")
    elif resumed is not None:
        game_results = [tuple(r) for r in resumed["game_results"]]
        game_log = resumed["game_log"]
        completed_pairs = {tuple(p) for p in resumed["completed_pairs"]}
        elapsed_offset = float(resumed.get("elapsed", 0.0))
        print(f"Resuming from {progress_path}: "
              f"{len(completed_pairs)}/{n_pairs} pairs already complete "
              f"({len(game_results)} games); replaying the rest.\n")

    if args.baseline:
        baseline_pairs = _load_baseline(args.baseline, args)
        if baseline_pairs is not None:
            n_reused_pairs = 0
            for i, j in pairs:
                if (i, j) in completed_pairs:
                    continue
                key = frozenset({model_paths[i], model_paths[j]})
                rows = baseline_pairs.get(key)
                if rows is None or len(rows) != args.games:
                    continue
                reused = _reuse_baseline_pair(rows, i, j, names, model_paths)
                if reused is None:
                    continue
                pair_results, pair_logs = reused
                game_results.extend(pair_results)
                for log_row in pair_logs:
                    log_row["game"] = len(game_log) + 1
                    game_log.append(log_row)
                completed_pairs.add((i, j))
                n_reused_pairs += 1
            if n_reused_pairs:
                print(f"Baseline {args.baseline}: reused {n_reused_pairs} pair(s) "
                      f"({n_reused_pairs * args.games} games) — "
                      f"{n_pairs - len(completed_pairs)} pair(s) left to simulate.\n")
                _save_progress(progress_path, signature, completed_pairs,
                               game_results, game_log, elapsed_offset + (time.time() - t0))
            else:
                print(f"Baseline {args.baseline}: no fully-matching pairs found "
                      f"(new roster shares no complete {args.games}-game pair with it) "
                      f"— simulating everything.\n")

    games_done_so_far = len(game_results)

    def _progress(pair_done: int) -> None:
        done = games_done_so_far + pair_done
        elapsed = elapsed_offset + (time.time() - t0)
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total_games - done) / rate if rate > 0 else 0.0
        print(f"\r  {done:>{len(str(total_games))}}/{total_games}  "
              f"{elapsed:>6.0f}s elapsed  ETA {eta:>5.0f}s", end="", flush=True)

    for i, j in pairs:
        if (i, j) in completed_pairs:
            continue
        pair_results = play_pair(
            args, i, j, agent_model_ids, agent_num_simulations, agent_leaf_batch,
            agent_c_puct, agent_fpu_reduction, agent_temperature, models, device,
            args.games, progress_cb=_progress,
        )
        for i_is_p1, winner, plies in pair_results:
            if winner == 0:
                score = 0.5
            else:
                i_won = (winner == 1 and i_is_p1) or (winner == 2 and not i_is_p1)
                score = 1.0 if i_won else 0.0
            game_results.append((i, j, score))
            p1_idx, p2_idx = (i, j) if i_is_p1 else (j, i)
            result = "draw" if winner == 0 else ("p1_win" if winner == 1 else "p2_win")
            game_log.append({
                "game": len(game_results),
                "agent_a": names[i], "agent_b": names[j],
                "player1": names[p1_idx], "player2": names[p2_idx],
                "result": result, "score_for_a": score, "plies": plies,
            })
        games_done_so_far += len(pair_results)
        completed_pairs.add((i, j))
        _save_progress(progress_path, signature, completed_pairs,
                       game_results, game_log, elapsed_offset + (time.time() - t0))
        _progress(0)

    elapsed = elapsed_offset + (time.time() - t0)
    print(f"\n  All {total_games} games done in {elapsed:.1f}s "
          f"({elapsed / max(total_games, 1):.1f}s/game)\n")

    elo = compute_elo(game_results, k)

    wins = np.zeros(k, dtype=int)
    draws = np.zeros(k, dtype=int)
    losses = np.zeros(k, dtype=int)
    for i, j, score in game_results:
        if score == 1.0:
            wins[i] += 1; losses[j] += 1
        elif score == 0.5:
            draws[i] += 1; draws[j] += 1
        else:
            wins[j] += 1; losses[i] += 1

    total_played = wins + draws + losses
    score_pct = (wins + 0.5 * draws) / np.maximum(total_played, 1) * 100

    order = np.argsort(elo)[::-1]
    col_w = max(len(nm) for nm in names) + 2
    sep = "─" * (col_w + 46)
    hdr = (f"  {'Rank':>4}  {'Model':<{col_w}}  {'Elo':>7}  "
           f"{'W':>5}  {'D':>5}  {'L':>5}  {'Score%':>7}")
    print(sep); print(hdr); print(sep)
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:>4}  {names[idx]:<{col_w}}  {elo[idx]:>7.1f}  "
              f"{wins[idx]:>5}  {draws[idx]:>5}  {losses[idx]:>5}  "
              f"{score_pct[idx]:>6.1f}%")
    print(sep)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "model", "elo", "wins", "draws", "losses",
                         "games", "score_pct"])
        for rank, idx in enumerate(order, 1):
            writer.writerow([rank, names[idx], f"{elo[idx]:.2f}",
                             wins[idx], draws[idx], losses[idx],
                             total_played[idx], f"{score_pct[idx]:.2f}"])
    print(f"\nResults saved to {out_path}")

    matchup_path = out_path.replace(".csv", "_matchups.csv")
    with open(matchup_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "game", "agent_a", "agent_b", "player1", "player2",
            "result", "score_for_a", "plies",
        ])
        writer.writeheader()
        writer.writerows(game_log)
    print(f"Matchup details saved to {matchup_path}")

    # So this run's own output is itself usable as a future --baseline.
    _write_run_meta(matchup_path, args)

    # Run finished cleanly — drop the resume snapshot so a later run with the
    # same --out starts fresh instead of thinking this one is still in progress.
    if os.path.exists(progress_path):
        os.remove(progress_path)


if __name__ == "__main__":
    main()
