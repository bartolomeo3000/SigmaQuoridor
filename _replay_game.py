"""
Replay a single deterministic game and print a human-readable move log
plus a board diagram after every move.

Usage:
    python _replay_game.py

Hard-coded: cycle_0071 (P1) vs supervised_extended (P2), temp=0.
"""

import torch
from dual_network import NNEvaluator, load_model
from game import State, PawnAction, WallAction
from mcts import MCTSAgent

BOARDSIZE        = 7
WALLS_PER_PLAYER = 5
SIMS             = 10000
DEVICE           = torch.device("cpu")

def fmt_action(action, state_before):
    """Return a concise notation for the action (all numeric, 0-based)."""
    if isinstance(action, PawnAction):
        dx, dy = action.direction
        if state_before.is_player1_turn():
            x, y = state_before.player1pos
        else:
            x, y = state_before.player2pos
        nx, ny = x + dx, y + dy
        return f"({nx},{ny})"
    elif isinstance(action, WallAction):
        return f"({action.x},{action.y}){action.orientation}"
    return str(action)

def draw_board(state):
    """Print a 7x7 ASCII board. Rows printed top to bottom = y=6 down to y=0.

    hwall anchor (x, y) blocks movement between row y and row y+1
    for columns x and x+1.
    vwall anchor (x, y) blocks movement between col x and col x+1
    for rows y and y+1.
    """
    s = state
    B = s.boardsize
    lines = []
    lines.append("    " + "   ".join(str(x) for x in range(B)))
    for y in range(B - 1, -1, -1):
        # Cell row
        row_str = f" {y} "
        for x in range(B):
            if (x, y) == s.player1pos:
                cell = "1"
            elif (x, y) == s.player2pos:
                cell = "2"
            else:
                cell = "."
            row_str += f" {cell} "
            # Vertical wall to the right of cell (x, y)?
            # vwall anchor (x, y) or (x, y-1) blocks the passage at x+0.5
            if x < B - 1:
                v_blocked = (x, y) in s.vwall_anchors or (x, y - 1) in s.vwall_anchors
                row_str += "|" if v_blocked else " "
        lines.append(row_str)
        # Horizontal wall row between y and y-1
        if y > 0:
            hw_str = "   "
            for x in range(B):
                # hwall anchor (x, y-1) or (x-1, y-1) blocks passage at y-0.5
                h_blocked = (x, y - 1) in s.hwall_anchors or (x - 1, y - 1) in s.hwall_anchors
                hw_str += "---" if h_blocked else "   "
                hw_str += "+" if x < B - 1 else ""
            lines.append(hw_str)
    lines.append(f"   P1 walls left: {s.walls_p1}  |  P2 walls left: {s.walls_p2}")
    return "\n".join(lines)


def make_agent(path, sims, device):
    if path.startswith("minimax:"):
        from benchmark_agents import MinimaxAgent
        return MinimaxAgent(depth=int(path.split(":")[1]))
    model = load_model(path, device=device)
    model.eval()
    return MCTSAgent(
        evaluator=NNEvaluator(model, device=device),
        num_simulations=sims,
        training=False,
        temperature=0.0,
    )


def replay(path_p1, path_p2, out_file=None):
    out = open(out_file, "w", encoding="utf-8") if out_file else None
    def p(*args, **kwargs):
        print(*args, **kwargs)
        if out:
            kwargs.pop("flush", None)
            print(*args, **kwargs, file=out)

    agent_p1 = make_agent(path_p1, SIMS, DEVICE)
    agent_p2 = make_agent(path_p2, SIMS, DEVICE)

    state = State(boardsize=BOARDSIZE, walls_p1=WALLS_PER_PLAYER, walls_p2=WALLS_PER_PLAYER)
    move_num = 1

    p("=" * 50)
    p(f"P1: {path_p1}")
    p(f"P2: {path_p2}")
    p("=" * 50)
    p("\nInitial position:")
    p(draw_board(state))
    p()

    while not state.is_finished():
        agent = agent_p1 if state.is_player1_turn() else agent_p2
        player_label = "P1" if state.is_player1_turn() else "P2"
        action = agent.select_action(state)
        notation = fmt_action(action, state)
        p(f"Move {move_num:>3}  [{player_label}]  {notation}")
        state = state.next(action)
        p(draw_board(state))
        p()
        move_num += 1

    winner = state.winner()
    p("=" * 50)
    if winner == 0:
        p("Result: DRAW")
    else:
        p(f"Result: Player {winner} wins!")
    p("=" * 50)
    if out:
        out.close()


if __name__ == "__main__":
    replay(
        path_p1="models_7x7/supervised_extended.pt",
        path_p2="models_7x7/supervised_extended.pt",
        out_file="replay_sup_vs_sup.txt",
    )
    print("Saved to replay_sup_vs_sup.txt")
