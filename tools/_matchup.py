import argparse
import torch
import _bootstrap  # noqa: F401  (puts the repo root on sys.path)

from dual_network import NNEvaluator, load_model
from game import State
from mcts import MCTSAgent

WALLS = 5

def play(model_p1, model_p2, name_p1, name_p2, sims):
    cpu = torch.device("cpu")
    a1 = MCTSAgent(evaluator=NNEvaluator(model_p1, device=cpu), num_simulations=sims, training=False)
    a2 = MCTSAgent(evaluator=NNEvaluator(model_p2, device=cpu), num_simulations=sims, training=False)
    state = State(boardsize=7, walls_p1=WALLS, walls_p2=WALLS)
    while not state.is_finished():
        agent = a1 if state.get_current_player() == 1 else a2
        action = max(agent.get_policy(state), key=lambda x: x[1])[0]
        state  = state.next(action)
    w = state.winner()
    winner_name = "draw" if w == 0 else (name_p1 if w == 1 else name_p2)
    return winner_name, state.depth

parser = argparse.ArgumentParser()
parser.add_argument("--model-a", default="models/best.pt")
parser.add_argument("--model-b", default="models/checkpoints/cycle_0119.pt")
parser.add_argument("--name-a",  default=None)
parser.add_argument("--name-b",  default=None)
parser.add_argument("--sims-list", default="800,1000,2000,4000,8000")
args = parser.parse_args()

name_a = args.name_a or args.model_a.split("/")[-1].replace(".pt", "")
name_b = args.name_b or args.model_b.split("/")[-1].replace(".pt", "")
sims_list = [int(s) for s in args.sims_list.split(",")]

print("Loading models...")
ma = load_model(args.model_a, device=torch.device("cpu")); ma.eval()
mb = load_model(args.model_b, device=torch.device("cpu")); mb.eval()

col = max(len(name_a), len(name_b), 10)
header = f"  {'Sims':>6}  {'Game A: '+name_a+'(P1) vs '+name_b+'(P2)':<{col+28}}  {'Game B: '+name_b+'(P1) vs '+name_a+'(P2)'}"
print(f"\n{header}")
print("  " + "-" * (len(header) - 2))

for sims in sims_list:
    w_a, d_a = play(ma, mb, name_a, name_b, sims)
    w_b, d_b = play(mb, ma, name_b, name_a, sims)
    ga = f"{w_a} wins ({d_a}p)"
    gb = f"{w_b} wins ({d_b}p)"
    print(f"  {sims:>6}  {ga:<{col+28}}  {gb}")
