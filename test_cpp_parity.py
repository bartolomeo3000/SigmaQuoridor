"""Parity test: C++ engine vs the reference Python engine in game.py.

Plays random games stepped in lockstep on both engines and asserts that
legal-action sets, NN input planes, terminal status and winners agree.

Usage:
    python setup_cpp.py build_ext --inplace
    python test_cpp_parity.py [--games 30] [--seed 0]
"""

import argparse
import random

import numpy as np

import quoridor_cpp
from game import State, action_to_index, index_to_action


def run_game(boardsize: int, walls: int, seed: int) -> int:
    rng = random.Random(seed)
    py = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
    cp = quoridor_cpp.State(boardsize=boardsize, walls=walls)
    moves = 0

    while True:
        py_legal = sorted(action_to_index(a, boardsize)
                          for a in py.get_legal_actions())
        cp_legal = sorted(cp.legal_actions())
        assert py_legal == cp_legal, (
            f"seed={seed} move={moves}: legal mismatch\n"
            f"  py-only: {sorted(set(py_legal) - set(cp_legal))}\n"
            f"  cpp-only: {sorted(set(cp_legal) - set(py_legal))}")

        py_nn = np.asarray(py.to_nn_input(), dtype=np.float32)
        cp_nn = cp.nn_input()
        assert np.allclose(py_nn, cp_nn, atol=1e-5), (
            f"seed={seed} move={moves}: nn_input mismatch, "
            f"planes differing: "
            f"{[i for i in range(8) if not np.allclose(py_nn[i], cp_nn[i], atol=1e-5)]}")

        py_fin = py.is_finished()
        cp_fin = cp.is_finished()
        assert py_fin == cp_fin, (
            f"seed={seed} move={moves}: is_finished {py_fin} vs {cp_fin}")
        if py_fin:
            assert py.winner() == cp.winner(), (
                f"seed={seed} move={moves}: winner {py.winner()} vs {cp.winner()}")
            return moves

        idx = rng.choice(py_legal)
        py = py.next(index_to_action(idx, boardsize))
        cp.apply(idx)
        moves += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    total = 0
    for boardsize, walls in ((7, 5), (9, 10), (5, 2)):
        for i in range(args.games):
            moves = run_game(boardsize, walls, args.seed * 100003 + i)
            total += moves
        print(f"OK boardsize={boardsize} walls={walls}: "
              f"{args.games} games in lockstep")
    print(f"all parity checks passed ({total} moves compared)")


if __name__ == "__main__":
    main()
