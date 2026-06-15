"""
Run games for one new agent against every agent already recorded in an
existing tournament_results_matchups.csv, append the results, then
recompute and overwrite tournament_results.csv.

Usage
-----
  python _run_supplement.py \
      --new   models_7x7/supervised_extended.pt \
      --matchups models_7x7/checkpoints/tournament_results_matchups.csv \
      [--summary  models_7x7/checkpoints/tournament_results.csv] \
      [--games 30] [--sims 800] [--temp 0.5] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from tournament import BOARDSIZE, WALLS_PER_PLAYER, _game_worker, compute_elo


# ── Name ↔ spec helpers ───────────────────────────────────────────────────────

def name_to_spec(name: str, ckpt_dir: Path) -> str:
    """Convert a display name (as stored in the CSV) back to the agent spec
    accepted by _game_worker / make_agent."""
    if name.startswith("minimax-"):
        return f"minimax:{name.split('-', 1)[1]}"
    if name == "random":
        return "random"
    if name == "greedy":
        return "greedy"
    # Names that contain a path separator are file paths stored as-is.
    if "\\" in name or "/" in name:
        return name
    # Short stem → look for the .pt file in the checkpoint directory.
    return str(ckpt_dir / f"{name}.pt")


def path_to_name(new_path: Path, ckpt_dir: Path) -> str:
    """Mirror the naming logic in tournament.py's main()."""
    if new_path.parent == ckpt_dir:
        return new_path.stem
    return str(new_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supplement an existing tournament with one new agent."
    )
    p.add_argument("--new",      required=True, metavar="PATH",
                   help="Path to the new agent's .pt file")
    p.add_argument("--matchups", required=True, metavar="CSV",
                   help="Path to existing tournament_results_matchups.csv")
    p.add_argument("--summary",  default="",
                   help="Path to tournament_results.csv (default: auto-derived)")
    p.add_argument("--games",    type=int,   default=30,                metavar="N")
    p.add_argument("--sims",     type=int,   default=800,               metavar="N")
    p.add_argument("--temp",     type=float, default=0.5,               metavar="T")
    p.add_argument("--workers",  type=int,   default=os.cpu_count() or 1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    matchups_path = Path(args.matchups)
    ckpt_dir      = matchups_path.parent
    summary_path  = (Path(args.summary) if args.summary
                     else ckpt_dir / "tournament_results.csv")

    # ── Read existing matchups ─────────────────────────────────────────────
    with open(matchups_path, newline="") as f:
        existing_rows = list(csv.DictReader(f))

    # Discover existing agents in first-appearance order.
    seen: set[str] = set()
    existing_agents: list[str] = []
    for row in existing_rows:
        for col in ("agent_a", "agent_b"):
            nm = row[col]
            if nm not in seen:
                seen.add(nm)
                existing_agents.append(nm)

    # ── Resolve the new agent ──────────────────────────────────────────────
    new_path = Path(args.new)
    new_name = path_to_name(new_path, ckpt_dir)
    new_spec = str(new_path)

    if new_name in seen:
        print(f"Agent '{new_name}' already present in {matchups_path}. Nothing to do.")
        return

    all_agents = existing_agents + [new_name]
    n         = len(all_agents)
    new_idx   = n - 1

    if args.games % 2 != 0:
        print(f"--games must be even; rounding up to {args.games + 1}.")
        args.games += 1

    n_per_color  = args.games // 2
    total_new    = len(existing_agents) * args.games
    n_workers    = min(args.workers, total_new)

    print(f"Existing agents : {len(existing_agents)}")
    print(f"New agent       : {new_name}")
    print(f"Games per pair  : {args.games}  (×{len(existing_agents)} = {total_new} total)")
    print(f"Sims/move       : {args.sims}   Temperature: {args.temp}")
    print(f"Workers         : {n_workers}")
    print()

    # ── Build task list ────────────────────────────────────────────────────
    tasks:     list[tuple] = []
    task_meta: list[tuple[int, int, bool]] = []  # (new_idx, existing_idx, new_is_a)

    for ei, existing_name in enumerate(existing_agents):
        ex_spec = name_to_spec(existing_name, ckpt_dir)
        # new agent as "a" (P1 half)
        for _ in range(n_per_color):
            tasks.append((new_spec, ex_spec, BOARDSIZE, WALLS_PER_PLAYER,
                          args.sims, args.temp, True, n_workers))
            task_meta.append((new_idx, ei, True))
        # new agent as "a" (P2 half)
        for _ in range(n_per_color):
            tasks.append((new_spec, ex_spec, BOARDSIZE, WALLS_PER_PLAYER,
                          args.sims, args.temp, False, n_workers))
            task_meta.append((new_idx, ei, False))

    # ── Run games ─────────────────────────────────────────────────────────
    new_rows: list[dict] = []
    game_offset = len(existing_rows)
    t0 = time.perf_counter()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for k, (score, (i, j, a_is_p1)) in enumerate(
            zip(pool.imap(_game_worker, tasks), task_meta)
        ):
            p1_idx, p2_idx = (i, j) if a_is_p1 else (j, i)
            if score == 0.5:
                result = "draw"
            elif (score == 1.0 and a_is_p1) or (score == 0.0 and not a_is_p1):
                result = "p1_win"
            else:
                result = "p2_win"
            new_rows.append({
                "game":        game_offset + k + 1,
                "agent_a":     all_agents[i],
                "agent_b":     all_agents[j],
                "player1":     all_agents[p1_idx],
                "player2":     all_agents[p2_idx],
                "result":      result,
                "score_for_a": score,
            })
            elapsed = time.perf_counter() - t0
            rate    = (k + 1) / elapsed
            eta     = (total_new - k - 1) / rate if rate > 0 else 0.0
            print(
                f"\r  {k+1:>{len(str(total_new))}}/{total_new}  "
                f"{elapsed:>6.0f}s elapsed  ETA {eta:>5.0f}s",
                end="", flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/total_new:.1f}s/game)\n")

    # ── Append new rows to matchups CSV ────────────────────────────────────
    with open(matchups_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "game", "agent_a", "agent_b", "player1", "player2",
            "result", "score_for_a",
        ])
        writer.writerows(new_rows)
    print(f"Appended {total_new} rows to {matchups_path}")

    # ── Recompute Elo from all matchup data ────────────────────────────────
    name_to_idx = {nm: idx for idx, nm in enumerate(all_agents)}
    game_results: list[tuple[int, int, float]] = []
    for row in existing_rows + new_rows:
        i     = name_to_idx[row["agent_a"]]
        j     = name_to_idx[row["agent_b"]]
        score = float(row["score_for_a"])
        game_results.append((i, j, score))

    elo = compute_elo(game_results, n)

    wins   = np.zeros(n, dtype=int)
    draws  = np.zeros(n, dtype=int)
    losses = np.zeros(n, dtype=int)
    for i, j, score in game_results:
        if score == 1.0:
            wins[i] += 1;  losses[j] += 1
        elif score == 0.5:
            draws[i] += 1; draws[j]  += 1
        else:
            wins[j] += 1;  losses[i] += 1

    total_played = wins + draws + losses
    score_pct    = (wins + 0.5 * draws) / np.maximum(total_played, 1) * 100
    order        = np.argsort(elo)[::-1]

    # ── Print table ────────────────────────────────────────────────────────
    col_w = max(len(nm) for nm in all_agents) + 2
    sep   = "─" * (col_w + 46)
    hdr   = (f"  {'Rank':>4}  {'Model':<{col_w}}  {'Elo':>7}  "
             f"{'W':>5}  {'D':>5}  {'L':>5}  {'Score%':>7}")
    print(sep)
    print(hdr)
    print(sep)
    for rank, idx in enumerate(order, 1):
        print(
            f"  {rank:>4}  {all_agents[idx]:<{col_w}}  {elo[idx]:>7.1f}  "
            f"{wins[idx]:>5}  {draws[idx]:>5}  {losses[idx]:>5}  "
            f"{score_pct[idx]:>6.1f}%"
        )
    print(sep)

    # ── Save updated summary CSV ───────────────────────────────────────────
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "model", "elo", "wins", "draws", "losses",
                         "games", "score_pct", "sims", "temperature"])
        for rank, idx in enumerate(order, 1):
            writer.writerow([
                rank, all_agents[idx], f"{elo[idx]:.2f}",
                wins[idx], draws[idx], losses[idx], total_played[idx],
                f"{score_pct[idx]:.2f}", args.sims, args.temp,
            ])
    print(f"\nUpdated summary saved to {summary_path}")


if __name__ == "__main__":
    main()
