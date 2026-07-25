"""
Replay (visually, in console) a specific rollout-MCTS-vs-AlphaZero game from
a saved _rollout_cpp_vs_alphazero.py log, using the shared action-index
sequence it printed (action_idx is in quoridor_cpp's index space, translated
via game.py's index_to_action -- same convention the live script uses).

Board rendering reuses draw_board()/fmt_action() from _replay_game.py.

Usage:
    python _replay_rollout_vs_az.py
"""

import argparse

import _bootstrap  # noqa: F401  (puts the repo root on sys.path)

from game import State, index_to_action
from _replay_game import draw_board, fmt_action

BOARDSIZE        = 7
WALLS_PER_PLAYER = 5
ROLLOUT_PLAYER   = 2  # matches --rollout-player 2 in the logged run

# (player, tag, action_idx) for every ply, taken verbatim from the pasted log.
MOVES = [
    (1, "alphazero", 0), (2, "rollout", 1), (1, "alphazero", 0), (2, "rollout", 40),
    (1, "alphazero", 0), (2, "rollout", 38), (1, "alphazero", 3), (2, "rollout", 42),
    (1, "alphazero", 0), (2, "rollout", 72), (1, "alphazero", 0), (2, "rollout", 60),
    (1, "alphazero", 1), (2, "rollout", 2), (1, "alphazero", 11), (2, "rollout", 1),
    (1, "alphazero", 1), (2, "rollout", 2), (1, "alphazero", 1), (2, "rollout", 1),
    (1, "alphazero", 15), (2, "rollout", 0), (1, "alphazero", 1), (2, "rollout", 3),
    (1, "alphazero", 3), (2, "rollout", 3), (1, "alphazero", 1), (2, "rollout", 2),
    (1, "alphazero", 0), (2, "rollout", 2), (1, "alphazero", 1), (2, "rollout", 3),
    (1, "alphazero", 3), (2, "rollout", 1), (1, "alphazero", 0), (2, "rollout", 0),
    (1, "alphazero", 1), (2, "rollout", 2), (1, "alphazero", 0), (2, "rollout", 3),
    (1, "alphazero", 0), (2, "rollout", 3), (1, "alphazero", 1), (2, "rollout", 1),
    (1, "alphazero", 1), (2, "rollout", 2), (1, "alphazero", 0), (2, "rollout", 3),
    (1, "alphazero", 1), (2, "rollout", 0), (1, "alphazero", 0), (2, "rollout", 3),
    (1, "alphazero", 0), (2, "rollout", 2), (1, "alphazero", 0), (2, "rollout", 1),
    (1, "alphazero", 1), (2, "rollout", 2), (1, "alphazero", 0), (2, "rollout", 0),
    (1, "alphazero", 1), (2, "rollout", 1), (1, "alphazero", 0), (2, "rollout", 0),
    (1, "alphazero", 0), (2, "rollout", 3), (1, "alphazero", 1), (2, "rollout", 1),
    (1, "alphazero", 1), (2, "rollout", 1), (1, "alphazero", 1), (2, "rollout", 2),
    (1, "alphazero", 0), (2, "rollout", 3), (1, "alphazero", 0), (2, "rollout", 2),
    (1, "alphazero", 0), (2, "rollout", 3), (1, "alphazero", 1), (2, "rollout", 3),
    (1, "alphazero", 1), (2, "rollout", 0), (1, "alphazero", 1), (2, "rollout", 0),
    (1, "alphazero", 0), (2, "rollout", 1), (1, "alphazero", 1), (2, "rollout", 1),
    (1, "alphazero", 17), (2, "rollout", 0), (1, "alphazero", 0), (2, "rollout", 2),
    (1, "alphazero", 0), (2, "rollout", 2), (1, "alphazero", 0), (2, "rollout", 2),
    (1, "alphazero", 56), (2, "rollout", 0), (1, "alphazero", 8), (2, "rollout", 3),
    (1, "alphazero", 0), (2, "rollout", 0), (1, "alphazero", 0),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-plies", type=int, default=None,
                    help="stop after this many plies (default: play the full game)")
    args = p.parse_args()

    moves = MOVES if args.max_plies is None else MOVES[:args.max_plies]

    state = State(boardsize=BOARDSIZE, walls_p1=WALLS_PER_PLAYER, walls_p2=WALLS_PER_PLAYER)
    print(draw_board(state))
    print()

    for ply, (player, tag, idx) in enumerate(moves, start=1):
        action = index_to_action(idx, BOARDSIZE)
        notation = fmt_action(action, state)
        state = state.next(action)
        print(f"ply {ply:3d}  player {player}  {tag:<9}  action_idx={idx:3d}  {notation}")
        print(draw_board(state))
        print()

    if args.max_plies is not None and args.max_plies < len(MOVES):
        print(f"(stopped after {len(moves)} of {len(MOVES)} plies; game not finished)")
    else:
        w = state.winner()
        winner_tag = "draw" if w == 0 else ("rollout" if w == ROLLOUT_PLAYER else "alphazero")
        print(f"Winner: player {w} ({winner_tag}), plies={state.depth}")


if __name__ == "__main__":
    main()
