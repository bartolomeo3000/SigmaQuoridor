"""
Round-robin Elo tournament between model checkpoints.

Each pair plays --games games (split evenly between colors).  Games use
non-zero temperature so every game is an independent stochastic sample,
which is required for meaningful Elo estimation.

Elo is computed via the Bradley-Terry maximum-likelihood model
(iterative fixed-point, converges reliably for full round-robins).

Usage
-----
    # Benchmark-only tournament (no checkpoints unless --dir is provided)
    python tournament.py --raw models_7x7/best.pt --minimax 1 --minimax 2

  # Custom directory, 6 games per pair, 400 sims
  python tournament.py --dir models_7x7_v2/checkpoints --games 6 --sims 400

  # Include an extra reference model (e.g. best.pt)
  python tournament.py --extra models_7x7/best.pt --extra models_7x7_v2/best.pt
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

from benchmark_agents import GreedyDistanceAgent, MinimaxAgent, RandomAgent, RawPolicyAgent
from dual_network import NNEvaluator, load_model
from game import State
from mcts import MCTSAgent

# ── Defaults ─────────────────────────────────────────────────────────────────

BOARDSIZE        = 7
WALLS_PER_PLAYER = 5
DEFAULT_DIR      = ""
DEFAULT_SIMS     = 800
DEFAULT_GAMES    = 4     # per pair; must be even (split P1/P2)
DEFAULT_TEMP     = 1.0   # stochastic sampling — essential for Elo estimation
DEFAULT_WORKERS  = os.cpu_count() or 1
DEFAULT_SIGMA_MODEL = "models_7x7/supervised_extended.pt"


# ── Worker ────────────────────────────────────────────────────────────────────

def _game_worker(args: tuple) -> float:
    """
    Play one game between two models identified by file paths.

    Returns the score from model-A's perspective:
      1.0 = A wins, 0.5 = draw, 0.0 = B wins.
    """
    (
        path_a, path_b,
        boardsize, walls, sims, temperature,
        a_is_p1, n_workers,
    ) = args

    # Limit per-process threads to avoid CPU over-subscription.
    cpu_count = os.cpu_count() or 1
    n_threads = max(1, cpu_count // n_workers)
    torch.set_num_threads(n_threads)

    device = torch.device("cpu")

    def make_agent(spec: str):
        if spec == "random":
            return RandomAgent()
        if spec == "greedy":
            return GreedyDistanceAgent()
        if spec.startswith("minimax:"):
            depth = int(spec.split(":", 1)[1])
            return MinimaxAgent(depth=depth)
        if spec.startswith("raw:"):
            model = load_model(spec.split(":", 1)[1], device=device)
            model.eval()
            return RawPolicyAgent(NNEvaluator(model, device=device))
        if spec.startswith("mcts:"):
            _, model_path, sim_count = spec.split(":", 2)
            model = load_model(model_path, device=device)
            model.eval()
            return MCTSAgent(
                evaluator       = NNEvaluator(model, device=device),
                num_simulations = int(sim_count),
                training        = False,
                temperature     = temperature,
            )
        model = load_model(spec, device=device)
        model.eval()
        return MCTSAgent(
            evaluator       = NNEvaluator(model, device=device),
            num_simulations = sims,
            training        = False,
            temperature     = temperature,
        )

    agent_a = make_agent(path_a)
    agent_b = make_agent(path_b)

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


# ── Elo via Bradley-Terry MLE ─────────────────────────────────────────────────

def compute_elo(
    games:    list[tuple[int, int, float]],
    n_models: int,
    n_iter:   int = 5_000,
    base_elo: float = 1500.0,
) -> np.ndarray:
    """
    Bradley-Terry maximum-likelihood Elo.

    Fixed-point iteration:  r_i  ←  W_i / Σ_j  n_ij / (r_i + r_j)

    where W_i = total score of model i, n_ij = games played between i and j.
    Converges to the unique MLE for connected round-robins.

    Returns an array of Elo values with mean = base_elo.
    """
    W = np.zeros(n_models)          # total score per model
    N = np.zeros((n_models, n_models), dtype=np.float64)  # games between i and j

    for i, j, score in games:
        W[i] += score
        W[j] += 1.0 - score
        N[i][j] += 1.0
        N[j][i] += 1.0

    r = np.ones(n_models)

    for iteration in range(n_iter):
        r_old = r.copy()
        for i in range(n_models):
            denom = sum(N[i][j] / (r_old[i] + r_old[j])
                        for j in range(n_models) if N[i][j] > 0)
            if denom > 0 and W[i] > 0:
                r[i] = W[i] / denom
            # else: r[i] unchanged (model with 0 wins or no games)

        # Normalize to prevent drift.
        r /= r.mean()

        if np.max(np.abs(r - r_old)) < 1e-9:
            break

    # Convert to Elo scale, anchored at base_elo.
    log_r = 400.0 * np.log10(np.maximum(r, 1e-12))
    elo = log_r - log_r.mean() + base_elo
    return elo


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round-robin Elo tournament")
    p.add_argument("--dir",     default=DEFAULT_DIR,
                   help="Optional directory containing cycle_*.pt checkpoint files")
    p.add_argument("--extra",   action="append", default=[],  metavar="PATH",
                   help="Extra model file(s) to include (repeatable)")
    p.add_argument("--raw",     action="append", default=[],  metavar="PATH",
                   help="Include a raw NN policy argmax agent for this model path (repeatable)")
    p.add_argument("--sigma",   action="append", default=[], type=int, metavar="SIMS",
                   help=f"Include SigmaQuoridor ({DEFAULT_SIGMA_MODEL}) at this MCTS sim count (repeatable)")
    p.add_argument("--sims",    type=int,   default=DEFAULT_SIMS,   metavar="N")
    p.add_argument("--games",   type=int,   default=DEFAULT_GAMES,  metavar="N",
                   help="Games per pair (even; split equally by colour)")
    p.add_argument("--temp",    type=float, default=DEFAULT_TEMP,   metavar="T")
    p.add_argument("--workers", type=int,   default=DEFAULT_WORKERS, metavar="N")
    p.add_argument("--out",     default="",
                   help="Optional CSV path for saving full results")
    p.add_argument("--minimax", action="append", default=[], type=int, metavar="DEPTH",
                   help="Include a MinimaxAgent at the given depth (repeatable)")
    p.add_argument("--random",  action="store_true",
                   help="Include a RandomAgent")
    p.add_argument("--greedy",  action="store_true",
                   help="Include a GreedyDistanceAgent")
    return p.parse_args()


def name_to_spec(name: str, ckpt_dir: Path | None) -> str:
    """Convert a matchup CSV display name back to a worker agent spec."""
    if name.startswith("minimax-"):
        return f"minimax:{name.split('-', 1)[1]}"
    if name.startswith("raw-"):
        stem = name.split("-", 1)[1]
        candidates = []
        if ckpt_dir is not None:
            candidates.extend([ckpt_dir / f"{stem}.pt", ckpt_dir.parent / f"{stem}.pt"])
        candidates.extend([Path("models_7x7") / f"{stem}.pt", Path("models_7x7_v2") / f"{stem}.pt"])
        for candidate in candidates:
            if candidate.exists():
                return f"raw:{candidate}"
        raise FileNotFoundError(f"Could not resolve raw policy agent '{name}' to a .pt file")
    if name.startswith("sigmaquoridor-"):
        return f"mcts:{DEFAULT_SIGMA_MODEL}:{name.split('-', 1)[1]}"
    if name in {"random", "greedy"}:
        return name
    if "\\" in name or "/" in name:
        return name
    if ckpt_dir is None:
        raise FileNotFoundError(f"Could not resolve checkpoint stem '{name}' without --dir")
    return str(ckpt_dir / f"{name}.pt")


def main() -> None:
    args = parse_args()

    # ── Collect model paths ────────────────────────────────────────────────
    ckpt_dir = Path(args.dir) if args.dir else None
    if ckpt_dir is not None:
        if not ckpt_dir.is_dir():
            print(f"Checkpoint directory not found: {ckpt_dir}")
            return
        paths = sorted(ckpt_dir.glob("cycle_*.pt"))
    else:
        paths = []
    for extra in args.extra:
        ep = Path(extra)
        if ep.exists():
            paths.append(ep)
        else:
            print(f"Warning: extra path not found: {extra}")

    names = [p.stem if ckpt_dir is not None and p.parent == ckpt_dir else str(p) for p in paths]
    paths_str = [str(p) for p in paths]
    specs_by_name = dict(zip(names, paths_str))

    # Add benchmark agents (identified by a spec string, not a file path)
    for depth in args.minimax:
        paths_str.append(f"minimax:{depth}")
        names.append(f"minimax-{depth}")
    for raw_path in args.raw:
        rp = Path(raw_path)
        if rp.exists():
            paths_str.append(f"raw:{raw_path}")
            names.append(f"raw-{rp.stem}")
            specs_by_name[f"raw-{rp.stem}"] = f"raw:{raw_path}"
        else:
            print(f"Warning: raw path not found: {raw_path}")
    sigma_model = Path(DEFAULT_SIGMA_MODEL)
    for sim_count in args.sigma:
        if sim_count < 1:
            print(f"Warning: sigma sim count must be positive: {sim_count}")
            continue
        if sigma_model.exists():
            name = f"sigmaquoridor-{sim_count}"
            spec = f"mcts:{DEFAULT_SIGMA_MODEL}:{sim_count}"
            paths_str.append(spec)
            names.append(name)
            specs_by_name[name] = spec
        else:
            print(f"Warning: SigmaQuoridor model not found: {DEFAULT_SIGMA_MODEL}")
    if args.random:
        paths_str.append("random")
        names.append("random")
        specs_by_name["random"] = "random"
    if args.greedy:
        paths_str.append("greedy")
        names.append("greedy")
        specs_by_name["greedy"] = "greedy"

    for name, spec in zip(names, paths_str):
        specs_by_name[name] = spec

    out_path = args.out or str((ckpt_dir or Path(".")) / "tournament_results.csv")
    out_parent = Path(out_path).parent
    if str(out_parent):
        out_parent.mkdir(parents=True, exist_ok=True)
    matchup_path = out_path.replace(".csv", "_matchups.csv")

    existing_rows: list[dict] = []
    if Path(matchup_path).exists():
        with open(matchup_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))

        # Preserve existing tournament agents and add requested agents afterward.
        seen_names = set(names)
        existing_names: list[str] = []
        for row in existing_rows:
            for col in ("agent_a", "agent_b"):
                name = row[col]
                if name not in existing_names:
                    existing_names.append(name)
                if name not in seen_names:
                    names.append(name)
                    seen_names.add(name)
                    try:
                        specs_by_name[name] = name_to_spec(name, ckpt_dir)
                    except FileNotFoundError as exc:
                        print(exc)
                        return
        names = existing_names + [name for name in names if name not in existing_names]
        paths_str = [specs_by_name[name] for name in names]

    n = len(paths_str)

    if n < 2:
        print(f"Need at least 2 agents; found {n} (check --dir, --extra, --raw, --minimax, --random, --greedy).")
        return

    if args.games % 2 != 0:
        print(f"--games must be even (got {args.games}); rounding up to {args.games + 1}.")
        args.games += 1

    n_pairs = n * (n - 1) // 2
    target_games = n_pairs * args.games

    print(f"Models       : {n}  ({names[0]} … {names[-1]})")
    print(f"Pairs        : {n_pairs}")
    print(f"Games/pair   : {args.games}  (×{n_pairs} = {target_games} target)")
    print(f"Sims/move    : {args.sims}   Temperature: {args.temp}")
    if existing_rows:
        print(f"Existing rows : {len(existing_rows)} from {matchup_path}")
    print()

    # ── Build task list ────────────────────────────────────────────────────
    tasks:     list[tuple] = []
    task_meta: list[tuple[int, int, bool]] = []   # (idx_a, idx_b, a_is_p1) for each task
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    existing_counts: dict[tuple[int, int, bool], int] = {}
    game_results: list[tuple[int, int, float]] = []
    game_log: list[dict] = list(existing_rows)

    for row in existing_rows:
        if row["agent_a"] not in name_to_idx or row["agent_b"] not in name_to_idx:
            continue
        row_i = name_to_idx[row["agent_a"]]
        row_j = name_to_idx[row["agent_b"]]
        score = float(row["score_for_a"])
        if row_i < row_j:
            i, j = row_i, row_j
            score_for_i = score
        else:
            i, j = row_j, row_i
            score_for_i = 1.0 - score
        game_results.append((i, j, score_for_i))
        a_is_p1 = row["player1"] == names[i]
        key = (i, j, a_is_p1)
        existing_counts[key] = existing_counts.get(key, 0) + 1

    n_per_color = args.games // 2
    for i, j in combinations(range(n), 2):
        pa, pb = paths_str[i], paths_str[j]
        missing_a_p1 = max(0, n_per_color - existing_counts.get((i, j, True), 0))
        missing_a_p2 = max(0, n_per_color - existing_counts.get((i, j, False), 0))
        for _ in range(missing_a_p1):
            tasks.append((pa, pb, BOARDSIZE, WALLS_PER_PLAYER,
                          args.sims, args.temp, True,  1))
            task_meta.append((i, j, True))
        for _ in range(missing_a_p2):
            tasks.append((pa, pb, BOARDSIZE, WALLS_PER_PLAYER,
                          args.sims, args.temp, False, 1))
            task_meta.append((i, j, False))

    total_games = len(tasks)
    n_workers = min(args.workers, total_games) if total_games else 0
    tasks = [task[:-1] + (n_workers or 1,) for task in tasks]
    print(f"Scheduled    : {total_games} new games")
    if n_workers:
        print(f"Workers      : {n_workers}")
    print()

    # ── Run games ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if tasks:
        game_offset = len(game_log)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            for k, (score, (i, j, a_is_p1)) in enumerate(
                zip(pool.imap(_game_worker, tasks), task_meta)
            ):
                game_results.append((i, j, score))
                p1_idx, p2_idx = (i, j) if a_is_p1 else (j, i)
                if score == 0.5:
                    result = "draw"
                elif (score == 1.0 and a_is_p1) or (score == 0.0 and not a_is_p1):
                    result = "p1_win"
                else:
                    result = "p2_win"
                game_log.append({
                    "game":    game_offset + k + 1,
                    "agent_a": names[i],
                    "agent_b": names[j],
                    "player1": names[p1_idx],
                    "player2": names[p2_idx],
                    "result":  result,
                    "score_for_a": score,
                })
                elapsed = time.perf_counter() - t0
                rate    = (k + 1) / elapsed
                eta     = (total_games - k - 1) / rate if rate > 0 else 0.0
                print(
                    f"\r  {k+1:>{len(str(total_games))}}/{total_games}  "
                    f"{elapsed:>6.0f}s elapsed  ETA {eta:>5.0f}s",
                    end="", flush=True,
                )

        elapsed = time.perf_counter() - t0
        print(f"\n  Added {total_games} games in {elapsed:.1f}s "
              f"({elapsed/total_games:.1f}s/game)\n")
    else:
        print("No missing games; recomputing summary from existing matchups.\n")

    # ── Compute Elo ────────────────────────────────────────────────────────
    elo = compute_elo(game_results, n)

    # Per-model win/draw/loss tallies.
    wins   = np.zeros(n, dtype=int)
    draws  = np.zeros(n, dtype=int)
    losses = np.zeros(n, dtype=int)
    for i, j, score in game_results:
        if score == 1.0:
            wins[i]   += 1
            losses[j] += 1
        elif score == 0.5:
            draws[i]  += 1
            draws[j]  += 1
        else:
            wins[j]   += 1
            losses[i] += 1

    total_played = wins + draws + losses
    score_pct    = (wins + 0.5 * draws) / np.maximum(total_played, 1) * 100

    # ── Print table ────────────────────────────────────────────────────────
    order   = np.argsort(elo)[::-1]
    col_w   = max(len(nm) for nm in names) + 2

    sep  = "─" * (col_w + 46)
    hdr  = (f"  {'Rank':>4}  {'Model':<{col_w}}  {'Elo':>7}  "
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

    # ── Save CSV ───────────────────────────────────────────────────────────
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "model", "elo", "wins", "draws", "losses",
                         "games", "score_pct", "sims", "temperature"])
        for rank, idx in enumerate(order, 1):
            writer.writerow([
                rank, names[idx], f"{elo[idx]:.2f}",
                wins[idx], draws[idx], losses[idx], total_played[idx],
                f"{score_pct[idx]:.2f}", args.sims, args.temp,
            ])
    print(f"\nResults saved to {out_path}")

    # ── Save detailed matchup CSV ──────────────────────────────────────────
    with open(matchup_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "game", "agent_a", "agent_b", "player1", "player2",
            "result", "score_for_a",
        ])
        writer.writeheader()
        writer.writerows(game_log)
    print(f"Matchup details saved to {matchup_path}")


if __name__ == "__main__":
    main()
