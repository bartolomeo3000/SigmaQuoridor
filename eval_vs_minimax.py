"""
Quick evaluation: best model from models_7x7_v2 vs MinimaxAgent.

Plays N_GAMES with the model as P1 and N_GAMES with the model as P2.
Uses deterministic play (argmax policy) for both agents.

Usage:
    python eval_vs_minimax.py [--depth D] [--sims S] [--games N] [--model PATH]
"""

import argparse
import torch
from pathlib import Path
from game import State
from mcts import MCTSAgent
from benchmark_agents import MinimaxAgent
from dual_network import DEVICE, NNEvaluator, load_model

# ---------------------------------------------------------------------------
BOARDSIZE        = 7
WALLS_PER_PLAYER = 5
N_GAMES          = 1        # games per side (total = 2 × N_GAMES)
DEFAULT_MODEL    = "models_7x7/best.pt"
DEFAULT_SIMS     = 800
DEFAULT_DEPTH    = 3
# ---------------------------------------------------------------------------


def play_game(agent1, agent2) -> int:
    """Deterministic game; returns 0 (draw), 1 (P1 win), 2 (P2 win)."""
    state = State(boardsize=BOARDSIZE, walls_p1=WALLS_PER_PLAYER, walls_p2=WALLS_PER_PLAYER)
    while not state.is_finished():
        agent  = agent1 if state.is_player1_turn() else agent2
        action = max(agent.get_policy(state), key=lambda x: x[1])[0]
        state  = state.next(action)
    return state.winner()


def result_str(winner: int, model_is_p1: bool) -> str:
    if winner == 0:
        return "Draw"
    model_won = (winner == 1 and model_is_p1) or (winner == 2 and not model_is_p1)
    return "Model wins" if model_won else "Minimax wins"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--sims",  type=int, default=DEFAULT_SIMS)
    parser.add_argument("--games", type=int, default=N_GAMES)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.depth < 1:
        parser.error("--depth must be a positive integer")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    net = load_model(str(model_path), device=torch.device("cpu"))
    net.eval()
    evaluator = NNEvaluator(net, device=torch.device("cpu"))

    def make_model_agent():
        return MCTSAgent(evaluator=evaluator, num_simulations=args.sims,
                         training=False)

    minimax = MinimaxAgent(depth=args.depth)

    print(f"Model ({args.sims} sims)  vs  MinimaxAgent(depth={args.depth})")
    print(f"Board: {BOARDSIZE}×{BOARDSIZE}, {WALLS_PER_PLAYER} walls/player")
    print(f"{args.games} games as P1  +  {args.games} games as P2\n")

    model_wins = draws = minimax_wins = 0

    # Model as P1
    for i in range(1, args.games + 1):
        w = play_game(make_model_agent(), minimax)
        label = result_str(w, model_is_p1=True)
        if w == 0:
            draws += 1
        elif w == 1:
            model_wins += 1
        else:
            minimax_wins += 1
        print(f"  P1=Model  game {i}: {label}")

    print()

    # Model as P2
    for i in range(1, args.games + 1):
        w = play_game(minimax, make_model_agent())
        label = result_str(w, model_is_p1=False)
        if w == 0:
            draws += 1
        elif w == 2:
            model_wins += 1
        else:
            minimax_wins += 1
        print(f"  P1=Minimax  game {i}: {label}")

    total = args.games * 2
    print(f"\n{'─'*40}")
    print(f"  Total:  Model {model_wins}  /  Draw {draws}  /  Minimax {minimax_wins}  "
          f"({model_wins}/{total} = {model_wins/total*100:.0f}%)")


if __name__ == "__main__":
    main()
