"""Compare per-move search time of the no-NN random-rollout MCTS baseline:
pure-Python (mcts.py's MCTSAgent with the default rollout_evaluator) vs the
new C++ port (cpp/rollout_mcts.hpp, quoridor_cpp.State.rollout_action).

Each implementation plays one full self-play game against itself (no cross
-play, no action-index translation needed) at the same simulation count;
we just record wall-clock time per move for each.
"""

import argparse
import time

import _bootstrap  # noqa: F401  (puts the repo root on sys.path)

import quoridor_cpp as qc
from game import State
from mcts import MCTSAgent


def play_python_game(sims: int, c_puct: float, boardsize: int, walls: int) -> list[float]:
    agent = MCTSAgent(num_simulations=sims, c_puct=c_puct, training=False)
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
    times = []
    while not state.is_finished():
        t0 = time.perf_counter()
        action = agent.select_action(state)
        times.append(time.perf_counter() - t0)
        state = state.next(action)
    print(f"  python game: winner={state.winner()} plies={state.depth}")
    return times


def play_cpp_game(sims: int, c_puct: float, boardsize: int, walls: int, seed: int,
                   dist_bonus_weight: float = 0.0) -> list[float]:
    state = qc.State(boardsize=boardsize, walls=walls)
    times = []
    ply = 0
    while not state.is_finished():
        t0 = time.perf_counter()
        action = state.rollout_action(num_simulations=sims, c_puct=c_puct,
                                       seed=seed + ply, dist_bonus_weight=dist_bonus_weight)
        times.append(time.perf_counter() - t0)
        state.apply(action)
        ply += 1
    print(f"  c++ game: winner={state.winner()} plies={state.depth()}")
    return times


def summarize(name: str, times: list[float]) -> None:
    total = sum(times)
    avg = total / len(times) if times else 0.0
    print(f"{name:<8}: {len(times)} moves, total {total:.2f}s, avg {avg*1000:.1f}ms/move, "
          f"min {min(times)*1000:.1f}ms, max {max(times)*1000:.1f}ms")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--boardsize", type=int, default=7)
    p.add_argument("--walls", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dist-bonus-weight", type=float, default=0.0,
                   help="C++ rollout agent only: nudge PUCT with a "
                        "(my_dist_to_goal - opp_dist_to_goal) heuristic bonus "
                        "(0.0 = pure rollout MCTS, no nudge)")
    args = p.parse_args()

    print(f"Settings: sims={args.sims} c_puct={args.c_puct} "
          f"boardsize={args.boardsize} walls={args.walls}\n")

    print("Playing Python self-play game...")
    py_times = play_python_game(args.sims, args.c_puct, args.boardsize, args.walls)

    print("Playing C++ self-play game...")
    cpp_times = play_cpp_game(args.sims, args.c_puct, args.boardsize, args.walls,
                               args.seed, args.dist_bonus_weight)

    print()
    summarize("python", py_times)
    summarize("c++", cpp_times)
    if py_times and cpp_times:
        py_avg = sum(py_times) / len(py_times)
        cpp_avg = sum(cpp_times) / len(cpp_times)
        print(f"\nC++ speedup: {py_avg / cpp_avg:.1f}x faster per move")


if __name__ == "__main__":
    main()
