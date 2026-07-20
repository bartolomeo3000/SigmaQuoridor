"""Quick, safe (CPU) diagnostic: compare network outputs of three checkpoints
on a handful of positions. Looks for degenerate value/policy that would signal
a real regression rather than a subtle strength difference."""
import math
import torch
import numpy as np
from game import State, WallAction, PawnAction, index_to_action, action_to_index
from dual_network import load_model, NNEvaluator

CPU = torch.device("cpu")
CKPTS = {
    "legacy-359": "models_9x9/checkpoints/cycle_0359.pt",
    "heads-56":   "models_9x9_heads/checkpoints/cycle_0056.pt",
    "heads-164":  "models_9x9_heads/checkpoints/cycle_0164.pt",
}

def make_positions():
    pos = {}
    # 1) initial position, P1 to move
    pos["start (P1)"] = State(boardsize=9, walls_p1=10, walls_p2=10)

    # 2) a few pawn advances so it's P2 to move (tests canonical/P2 path)
    s = State(boardsize=9, walls_p1=10, walls_p2=10)
    s = s.next(PawnAction(direction=(0, 1)))   # P1 up
    pos["after 1 ply (P2)"] = s

    # 3) deeper: both advanced + a wall, mid-ish, P1 to move
    s2 = State(boardsize=9, walls_p1=10, walls_p2=10)
    for _ in range(3):
        s2 = s2.next(PawnAction(direction=(0, 1)))    # P1 up
        s2 = s2.next(PawnAction(direction=(0, -1)))   # P2 down (toward its goal)
    pos["mid, symmetric (P2)"] = s2
    return pos

def summarize(name, ev, state):
    legal = state.get_legal_actions()
    priors, value = ev(state, legal)
    p = np.asarray(priors, dtype=np.float64)
    p = p / p.sum()
    order = np.argsort(-p)
    ent = -np.sum(p * np.log(p + 1e-12))
    top = []
    for k in order[:3]:
        a = legal[k]
        top.append(f"{_lbl(a)}={p[k]:.2f}")
    print(f"    {name:11s}  value={value:+.3f}  entropy={ent:4.2f}  "
          f"top: {', '.join(top)}   (nlegal={len(legal)})")

def _lbl(a):
    if isinstance(a, WallAction):
        return f"W{a.orientation}({a.x},{a.y})"
    dx, dy = a.direction
    return f"P({dx:+d},{dy:+d})"

def main():
    positions = make_positions()
    models = {n: load_model(p, device=CPU) for n, p in CKPTS.items()}
    evs = {n: NNEvaluator(m, device=CPU) for n, m in models.items()}
    for pname, state in positions.items():
        print(f"\n  === {pname}  (turn: {'P1' if state.is_player1_turn() else 'P2'}) ===")
        for n in CKPTS:
            summarize(n, evs[n], state)

if __name__ == "__main__":
    main()
