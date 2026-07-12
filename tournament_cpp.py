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
import os
import threading
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import quoridor_cpp
from dual_network import load_model
from tournament import BOARDSIZE, WALLS_PER_PLAYER, compute_elo

LOG_DIR = "logs"


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
        return self._stream.write(data)

    def flush(self) -> None:
        self._log_file.flush()
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


def _start_run_log(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"tournament_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(path, "a")
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
    p.add_argument("--dir", default="models_7x7/checkpoints",
                   help="Directory containing cycle_*.pt checkpoint files")
    p.add_argument("--extra", action="append", default=[], metavar="PATH",
                   help="Extra model file(s) to include (repeatable)")
    p.add_argument("--model", type=str, default=None,
                   help="Single model to test at multiple --sim budgets")
    p.add_argument("--sim", action="append", type=int, default=[], metavar="N",
                   help="Simulation budget to include with --model (repeatable); "
                        "activates simcounts mode instead of checkpoint round-robin")
    p.add_argument("--sims", type=int, default=800,
                   help="Sims/move for checkpoint round-robin mode")
    p.add_argument("--games", type=int, default=4, metavar="N",
                   help="Games per pair (even; split equally by colour)")
    p.add_argument("--temp", type=float, default=1.0,
                   help="Sampling temperature (0 = deterministic argmax)")
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--fpu", type=float, default=0.1)
    p.add_argument("--boardsize", type=int, default=BOARDSIZE)
    p.add_argument("--walls", type=int, default=WALLS_PER_PLAYER)
    p.add_argument("--max-moves", type=int, default=200)
    p.add_argument("--threads", type=int, default=8, help="MCTS worker threads")
    p.add_argument("--parallel", type=int, default=128, help="concurrent games")
    p.add_argument("--max-batch", type=int, default=256)
    p.add_argument("--flush-us", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="Optional CSV path for summary results")
    return p.parse_args()


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
    game_results: list[tuple[int, int, float]] = []
    game_log: list[dict] = []
    games_done_so_far = 0

    def _progress(pair_done: int) -> None:
        done = games_done_so_far + pair_done
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total_games - done) / rate if rate > 0 else 0.0
        print(f"\r  {done:>{len(str(total_games))}}/{total_games}  "
              f"{elapsed:>6.0f}s elapsed  ETA {eta:>5.0f}s", end="", flush=True)

    for i, j in pairs:
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
        _progress(0)

    elapsed = time.time() - t0
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

    out_path = args.out or "tournament_cpp_results.csv"
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


if __name__ == "__main__":
    main()
