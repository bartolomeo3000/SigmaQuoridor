"""Play one game between the C++ no-NN random-rollout MCTS
(cpp/rollout_mcts.hpp, quoridor_cpp.State.rollout_action) and the
AlphaZero-style NN-guided MCTS (mcts.py's MCTSAgent + a trained model,
e.g. models_7x7/best.pt), at independently chosen simulation counts.

Two mirrored states are kept in sync (game.py's State for the NN agent,
quoridor_cpp.State for the rollout agent) via action-index translation
(action_to_index/index_to_action), since the engines use different action
representations but are parity-verified to agree (test_cpp_parity.py).
"""

import argparse
import time

import torch

import quoridor_cpp as qc
from dual_network import NNEvaluator, load_model
from game import State, action_to_index, index_to_action
from mcts import MCTSAgent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="models_7x7/best.pt")
    p.add_argument("--rollout-sims", type=int, default=10_000)
    p.add_argument("--nn-sims", type=int, default=400)
    p.add_argument("--rollout-c-puct", type=float, default=1.0)
    p.add_argument("--nn-c-puct", type=float, default=1.0)
    p.add_argument("--rollout-dist-bonus-weight", type=float, default=0.0,
                   help="nudge the C++ rollout agent's PUCT with a "
                        "(my_dist_to_goal - opp_dist_to_goal) heuristic bonus "
                        "(0.0 = pure rollout MCTS, no nudge)")
    p.add_argument("--boardsize", type=int, default=7)
    p.add_argument("--walls", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout-player", type=int, default=1, choices=(1, 2),
                   help="which player (1 or 2) is played by the C++ rollout "
                        "MCTS; the other is the NN-guided MCTS")
    args = p.parse_args()

    print(f"Rollout (C++, no NN): player {args.rollout_player}, "
          f"{args.rollout_sims} sims, c_puct={args.rollout_c_puct}")
    other = 2 if args.rollout_player == 1 else 1
    print(f"AlphaZero ({args.model}): player {other}, "
          f"{args.nn_sims} sims, c_puct={args.nn_c_puct}\n")

    model = load_model(args.model, device=torch.device("cpu"))
    model.eval()
    nn_agent = MCTSAgent(
        evaluator=NNEvaluator(model, device=torch.device("cpu")),
        num_simulations=args.nn_sims,
        c_puct=args.nn_c_puct,
        training=False,
    )

    py_state = State(boardsize=args.boardsize, walls_p1=args.walls, walls_p2=args.walls)
    cpp_state = qc.State(boardsize=args.boardsize, walls=args.walls)

    rollout_times: list[float] = []
    nn_times: list[float] = []
    ply = 0

    while not py_state.is_finished():
        cur = py_state.get_current_player()
        if cur == args.rollout_player:
            t0 = time.perf_counter()
            idx = cpp_state.rollout_action(
                num_simulations=args.rollout_sims,
                c_puct=args.rollout_c_puct,
                seed=args.seed + ply,
                dist_bonus_weight=args.rollout_dist_bonus_weight,
            )
            dt = time.perf_counter() - t0
            rollout_times.append(dt)
            action = index_to_action(idx, args.boardsize)
            tag = "rollout"
        else:
            t0 = time.perf_counter()
            action = nn_agent.select_action(py_state)
            dt = time.perf_counter() - t0
            nn_times.append(dt)
            idx = action_to_index(action, args.boardsize)
            tag = "alphazero"

        py_state = py_state.next(action)
        cpp_state.apply(idx)
        ply += 1
        print(f"ply {ply:3d}  player {cur}  {tag:<9}  action_idx={idx:3d}  time={dt:6.2f}s")

    w = py_state.winner()
    winner_tag = "draw" if w == 0 else ("rollout" if w == args.rollout_player else "alphazero")
    print(f"\nWinner: player {w} ({winner_tag}), plies={py_state.depth}\n")

    def summarize(name: str, times: list[float]) -> None:
        if not times:
            print(f"{name:<9}: no moves")
            return
        total = sum(times)
        avg = total / len(times)
        print(f"{name:<9}: {len(times)} moves, total {total:.2f}s, avg {avg:.3f}s/move, "
              f"min {min(times):.3f}s, max {max(times):.3f}s")

    summarize("rollout", rollout_times)
    summarize("alphazero", nn_times)


if __name__ == "__main__":
    main()
