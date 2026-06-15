"""Round-robin tournament for one model played with different MCTS simulation budgets.

This is meant to answer a narrow question: how much stronger does the same
network become when given a larger search budget at inference time?

By default the tournament is deterministic (temperature=0) and each pair plays
2 games, one with each colour. That is usually the right setup here: if the
policy is deterministic, more games would just repeat the same outcomes.

Example
-------
  python tournament_simcounts.py \
      --model models_7x7/supervised_extended.pt \
      --sim 200 --sim 800 --sim 2000 --sim 5000 --sim 10000
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from dual_network import NNEvaluator, load_model
from game import State
from mcts import MCTSAgent
from tournament import BOARDSIZE, WALLS_PER_PLAYER, compute_elo

DEFAULT_GAMES = 2
DEFAULT_TEMP = 0.0
DEFAULT_WORKERS = os.cpu_count() or 1


def _game_worker(args: tuple) -> float:
    """Play one game between the same model using two different sim budgets.

    Returns score from agent-A's perspective:
      1.0 = A wins, 0.5 = draw, 0.0 = B wins.
    """
    (
        model_path,
        sims_a,
        sims_b,
        boardsize,
        walls,
        temperature,
        a_is_p1,
        n_workers,
    ) = args

    cpu_count = os.cpu_count() or 1
    n_threads = max(1, cpu_count // n_workers)
    torch.set_num_threads(n_threads)

    device = torch.device("cpu")
    model = load_model(model_path, device=device)
    model.eval()
    evaluator = NNEvaluator(model, device=device)

    agent_a = MCTSAgent(
        evaluator=evaluator,
        num_simulations=sims_a,
        training=False,
        temperature=temperature,
    )
    agent_b = MCTSAgent(
        evaluator=evaluator,
        num_simulations=sims_b,
        training=False,
        temperature=temperature,
    )

    p1, p2 = (agent_a, agent_b) if a_is_p1 else (agent_b, agent_a)

    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
    while not state.is_finished():
        action = (p1 if state.is_player1_turn() else p2).select_action(state)
        state = state.next(action)

    winner = state.winner()
    if winner == 0:
        return 0.5
    a_won = (winner == 1 and a_is_p1) or (winner == 2 and not a_is_p1)
    return 1.0 if a_won else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tournament across MCTS simulation budgets")
    p.add_argument("--model", required=True, metavar="PATH",
                   help="Path to the model checkpoint to test")
    p.add_argument("--sim", action="append", type=int, required=True, metavar="N",
                   help="Simulation budget to include (repeatable)")
    p.add_argument("--games", type=int, default=DEFAULT_GAMES, metavar="N",
                   help="Games per pair (even; split equally by colour). Default: 2")
    p.add_argument("--temp", type=float, default=DEFAULT_TEMP, metavar="T",
                   help="Temperature for action sampling. Default: 0")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N")
    p.add_argument("--out", default="",
                   help="Optional CSV path for saving summary results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    sim_counts = sorted(set(args.sim))
    if len(sim_counts) < 2:
        raise ValueError("Provide at least two distinct --sim values")

    if args.games % 2 != 0:
        print(f"--games must be even (got {args.games}); rounding up to {args.games + 1}.")
        args.games += 1

    names = [f"{model_path.name} @ {s} sims" for s in sim_counts]
    n = len(sim_counts)
    n_pairs = n * (n - 1) // 2
    total_games = n_pairs * args.games
    n_workers = min(args.workers, total_games)

    print(f"Model        : {model_path}")
    print(f"Variants     : {n}  ({sim_counts})")
    print(f"Pairs        : {n_pairs}")
    print(f"Games/pair   : {args.games}  (x{n_pairs} = {total_games} total)")
    print(f"Temperature  : {args.temp}")
    print(f"Workers      : {n_workers}")
    print()

    tasks: list[tuple] = []
    task_meta: list[tuple[int, int, bool]] = []
    n_per_color = args.games // 2

    for i, j in combinations(range(n), 2):
        for _ in range(n_per_color):
            tasks.append((
                str(model_path), sim_counts[i], sim_counts[j],
                BOARDSIZE, WALLS_PER_PLAYER, args.temp, True, n_workers,
            ))
            task_meta.append((i, j, True))
        for _ in range(n_per_color):
            tasks.append((
                str(model_path), sim_counts[i], sim_counts[j],
                BOARDSIZE, WALLS_PER_PLAYER, args.temp, False, n_workers,
            ))
            task_meta.append((i, j, False))

    game_results: list[tuple[int, int, float]] = []
    game_log: list[dict] = []
    t0 = time.perf_counter()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for k, (score, (i, j, a_is_p1)) in enumerate(zip(pool.imap(_game_worker, tasks), task_meta)):
            game_results.append((i, j, score))
            p1_idx, p2_idx = (i, j) if a_is_p1 else (j, i)
            if score == 0.5:
                result = "draw"
            elif (score == 1.0 and a_is_p1) or (score == 0.0 and not a_is_p1):
                result = "p1_win"
            else:
                result = "p2_win"
            game_log.append({
                "game": k + 1,
                "agent_a": names[i],
                "agent_b": names[j],
                "player1": names[p1_idx],
                "player2": names[p2_idx],
                "result": result,
                "score_for_a": score,
                "sims_a": sim_counts[i],
                "sims_b": sim_counts[j],
            })
            elapsed = time.perf_counter() - t0
            rate = (k + 1) / elapsed
            eta = (total_games - k - 1) / rate if rate > 0 else 0.0
            print(
                f"\r  {k+1:>{len(str(total_games))}}/{total_games}  "
                f"{elapsed:>6.0f}s elapsed  ETA {eta:>5.0f}s",
                end="", flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(f"\n  All {total_games} games done in {elapsed:.1f}s ({elapsed/total_games:.1f}s/game)\n")

    elo = compute_elo(game_results, n)

    wins = np.zeros(n, dtype=int)
    draws = np.zeros(n, dtype=int)
    losses = np.zeros(n, dtype=int)
    for i, j, score in game_results:
        if score == 1.0:
            wins[i] += 1
            losses[j] += 1
        elif score == 0.5:
            draws[i] += 1
            draws[j] += 1
        else:
            wins[j] += 1
            losses[i] += 1

    total_played = wins + draws + losses
    score_pct = (wins + 0.5 * draws) / np.maximum(total_played, 1) * 100
    order = np.argsort(elo)[::-1]

    col_w = max(len(nm) for nm in names) + 2
    sep = "─" * (col_w + 46)
    hdr = (f"  {'Rank':>4}  {'Variant':<{col_w}}  {'Elo':>7}  "
           f"{'W':>5}  {'D':>5}  {'L':>5}  {'Score%':>7}")
    print(sep)
    print(hdr)
    print(sep)
    for rank, idx in enumerate(order, 1):
        print(
            f"  {rank:>4}  {names[idx]:<{col_w}}  {elo[idx]:>7.1f}  "
            f"{wins[idx]:>5}  {draws[idx]:>5}  {losses[idx]:>5}  "
            f"{score_pct[idx]:>6.1f}%"
        )
    print(sep)

    out_path = args.out or str(model_path.with_name(model_path.stem + "_simcount_tournament.csv"))
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "variant", "elo", "wins", "draws", "losses",
                         "games", "score_pct", "temperature"])
        for rank, idx in enumerate(order, 1):
            writer.writerow([
                rank, names[idx], f"{elo[idx]:.2f}",
                wins[idx], draws[idx], losses[idx], total_played[idx],
                f"{score_pct[idx]:.2f}", args.temp,
            ])
    print(f"\nResults saved to {out_path}")

    matchup_path = out_path.replace(".csv", "_matchups.csv")
    with open(matchup_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "game", "agent_a", "agent_b", "player1", "player2",
            "result", "score_for_a", "sims_a", "sims_b",
        ])
        writer.writeheader()
        writer.writerows(game_log)
    print(f"Matchup details saved to {matchup_path}")


if __name__ == "__main__":
    main()
