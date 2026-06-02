"""
benchmark_selfplay.py — time breakdown of the AlphaZero self-play pipeline.

Measures wall-clock time across every major phase so you can see exactly
where cycles are being spent:

  NN calls   — to_nn_input · tensor prep · forward pass · index/softmax
  Game logic — get_legal_actions · state.next() · is_finished()
  MCTS ops   — selection · backup
  Overhead   — anything not in the above buckets

Usage
-----
  python benchmark_selfplay.py                   # 5 games, 200 sims, auto device
  python benchmark_selfplay.py --games 3 --sims 50 --device cpu
  python benchmark_selfplay.py --games 10 --sims 400 --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from game import State, Action, action_to_index, action_space_size
from dual_network import DualNetwork, NNEvaluator, DEVICE, load_model
from mcts import MCTSAgent, MCTSNode

# ---------------------------------------------------------------------------
# Timing accumulators
# ---------------------------------------------------------------------------

_KEYS = [
    "nn.to_nn_input",
    "nn.tensor_prep",
    "nn.forward",
    "nn.index_softmax",
    "nn.total",
    "game.get_legal_actions",
    "game.state_next",
    "game.is_finished",
    "mcts.select",
    "mcts.backup",
    "mcts.simulation",
    "selfplay.total",
]

T: dict[str, float] = {k: 0.0 for k in _KEYS}   # accumulated seconds
N: dict[str, int]   = {k: 0   for k in _KEYS}   # call counts

def _t(key: str) -> float:
    return time.perf_counter()

def _rec(key: str, t0: float) -> None:
    T[key] += time.perf_counter() - t0
    N[key] += 1

def reset_stats() -> None:
    for k in T:
        T[k] = 0.0
        N[k] = 0


# ---------------------------------------------------------------------------
# Timing NNEvaluator
# ---------------------------------------------------------------------------

class TimingNNEvaluator(NNEvaluator):
    """Drop-in replacement that records per-phase timing to global T/N."""

    def __call__(
        self,
        state:         State,
        legal_actions: list[Action],
    ) -> tuple[list[float], float]:
        t_total = time.perf_counter()

        # Phase 1 — pure-Python / NumPy: build the (8,N,N) array
        t1 = time.perf_counter()
        arr = state.to_nn_input()
        _rec("nn.to_nn_input", t1)

        with torch.no_grad():
            # Phase 2 — tensor creation + host→device copy
            t2 = time.perf_counter()
            x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            _rec("nn.tensor_prep", t2)

            # Phase 3 — actual forward pass through the dual network
            t3 = time.perf_counter()
            policy_logits, value = self.model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()          # ensure GPU work is done
            _rec("nn.forward", t3)

            # Phase 4 — gather logits for legal actions + masked softmax
            t4 = time.perf_counter()
            policy_logits = policy_logits[0]      # (A,)
            indices = torch.tensor(
                [action_to_index(a, state.boardsize) for a in legal_actions],
                dtype=torch.long,
                device=self.device,
            )
            legal_logits = policy_logits[indices]
            priors = torch.softmax(legal_logits, dim=0).cpu().tolist()
            _rec("nn.index_softmax", t4)

        _rec("nn.total", t_total)
        return priors, float(value[0, 0])


# ---------------------------------------------------------------------------
# Timing MCTSAgent
# ---------------------------------------------------------------------------

class TimingMCTSAgent(MCTSAgent):
    """MCTSAgent subclass that instruments select / expand / backup / simulate."""

    # -- _select: traverse tree until unexpanded or terminal ----------------
    def _select(self, root: MCTSNode) -> MCTSNode:
        t_sel = time.perf_counter()
        node = root
        while True:
            node.ensure_state()
            t1 = time.perf_counter()
            is_fin = node.state.is_finished()
            _rec("game.is_finished", t1)
            if not node.is_expanded or is_fin:
                break
            node = node.best_child(self.c_puct)
        _rec("mcts.select", t_sel)
        return node

    # -- _expand: create children, call evaluator ---------------------------
    def _expand(self, node: MCTSNode) -> float:
        # Legal actions
        t1 = time.perf_counter()
        legal_actions = node.state.get_legal_actions()
        _rec("game.get_legal_actions", t1)

        # Evaluator (NN call — timed inside TimingNNEvaluator)
        priors, value = self.evaluator(node.state, legal_actions)

        # Lazy child creation — state.next() deferred to ensure_state() on first visit
        for action, prior in zip(legal_actions, priors):
            node.children.append(
                MCTSNode(
                    state=None,
                    parent=node,
                    action=action,
                    prior=prior,
                    parent_state=node.state,
                )
            )

        node.is_expanded = True
        return value

    # -- _backup: negamax walk to root --------------------------------------
    def _backup(self, node: MCTSNode, value: float) -> None:
        t1 = time.perf_counter()
        n = node
        while n is not None:
            n.visit_count += 1
            n.value_sum   += value
            value = -value
            n = n.parent
        _rec("mcts.backup", t1)

    # -- _run_simulation: one full select→expand→backup cycle ---------------
    def _run_simulation(self, root: MCTSNode) -> None:
        t1 = time.perf_counter()
        leaf = self._select(root)
        # Time the lazy state materialisation (no-op on re-visited terminal nodes)
        if leaf.state is None:
            t_next = time.perf_counter()
            leaf.ensure_state()
            _rec("game.state_next", t_next)
        else:
            leaf.ensure_state()  # no-op, no timing needed
        if leaf.state.is_finished():
            value = self._terminal_value(leaf.state)
        else:
            value = self._expand(leaf)
        self._backup(leaf, value)
        _rec("mcts.simulation", t1)


# ---------------------------------------------------------------------------
# Self-play driver (mirrors train.py's self_play_game, no augmentation)
# ---------------------------------------------------------------------------

def _play_one_game(
    agent:          TimingMCTSAgent,
    boardsize:      int,
    walls:          int,
    temp_threshold: int = 30,
) -> int:
    """Play one game to completion, return number of plies."""
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls, walls_initial=walls)
    ply   = 0
    while not state.is_finished():
        agent.temperature = 1.0 if ply < temp_threshold else 0.0
        action = agent.select_action(state)
        state  = state.next(action)
        ply   += 1
    return ply


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    num_games:   int,
    num_sims:    int,
    boardsize:   int,
    walls:       int,
    device_str:  str,
    model_path:  str,
    filters:     int = 64,
    num_residual: int = 6,
) -> None:
    device = torch.device(device_str)

    # Load or create model
    if Path(model_path).exists():
        model = load_model(model_path, device=device)
        print(f"Loaded model from {model_path}")
    else:
        model = DualNetwork(boardsize=boardsize, filters=filters, num_residual=num_residual).to(device)
        print(f"No model found at {model_path} — using random weights")

    model.eval()
    evaluator = TimingNNEvaluator(model, device=device)
    agent = TimingMCTSAgent(
        evaluator=evaluator,
        num_simulations=num_sims,
        training=False,
    )

    action_sz = action_space_size(boardsize)
    print(f"\nWarmup (1 game, {num_sims} sims, device={device_str})…", end=" ", flush=True)
    reset_stats()
    _play_one_game(agent, boardsize, walls)
    print("done")

    # --- real measurement --------------------------------------------------
    reset_stats()
    print(f"Benchmarking {num_games} games…", end=" ", flush=True)

    t_wall = time.perf_counter()
    total_plies = 0
    for g in range(num_games):
        t_game = time.perf_counter()
        plies = _play_one_game(agent, boardsize, walls)
        T["selfplay.total"] += time.perf_counter() - t_game
        N["selfplay.total"] += 1
        total_plies += plies
        print(f"{g+1}", end=" ", flush=True)
    wall_elapsed = time.perf_counter() - t_wall
    print()

    # Number of expansions = N["game.get_legal_actions"]
    n_expansions = N["game.get_legal_actions"]
    n_simulations = N["mcts.simulation"]
    n_nn_calls = N["nn.total"]

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------

    bar  = "─" * 68
    dbar = "═" * 68

    def pct(t: float) -> str:
        return f"{100 * t / wall_elapsed:5.1f}%"

    def ms(t: float) -> str:
        return f"{t * 1e3:8.1f} ms"

    def us_avg(t: float, c: int) -> str:
        if c == 0:
            return "       —    "
        return f"{t / c * 1e6:8.1f} µs/call"

    print()
    print(dbar)
    print(f"  Self-play benchmark   {num_games} games · {num_sims} sims/move · device={device_str}")
    print(dbar)
    print(f"  Total wall-clock      {wall_elapsed:7.2f} s")
    print(f"  Games                 {num_games}")
    print(f"  Total plies           {total_plies}   ({total_plies/num_games:.1f} avg/game)")
    print(f"  NN evaluations        {n_nn_calls}   ({n_nn_calls/num_games:.0f}/game · {n_nn_calls/total_plies:.1f}/ply)")
    print(f"  MCTS simulations      {n_simulations}   ({n_simulations/total_plies:.0f}/ply)")
    print(f"  State expansions      {n_expansions}   (one NN call each)")
    n_state_next = N["game.state_next"]
    print(f"  state.next() calls    {n_state_next}   ({n_state_next/n_simulations*100:.0f}% of sims materialize a new node)")
    print(bar)

    # --- NN breakdown ---
    print(f"  {'PHASE':<28}  {'TOTAL':>10}  {'% WALL':>7}  {'AVG / CALL':>18}")
    print(bar)

    nn_rows = [
        ("NN calls (total)",           "nn.total",         N["nn.total"]),
        ("  to_nn_input (numpy)",      "nn.to_nn_input",   N["nn.to_nn_input"]),
        ("  tensor prep + H2D",        "nn.tensor_prep",   N["nn.tensor_prep"]),
        ("  forward pass",             "nn.forward",       N["nn.forward"]),
        ("  index gather + softmax",   "nn.index_softmax", N["nn.index_softmax"]),
    ]
    for label, key, cnt in nn_rows:
        print(f"  {label:<28}  {ms(T[key]):>10}  {pct(T[key]):>7}  {us_avg(T[key], cnt):>18}")

    print(bar)

    # --- Game logic breakdown ---
    game_rows = [
        ("get_legal_actions",         "game.get_legal_actions", N["game.get_legal_actions"]),
        ("state.next() (lazy/sim)",   "game.state_next",        N["game.state_next"]),
        ("is_finished() (in select)", "game.is_finished",       N["game.is_finished"]),
    ]
    for label, key, cnt in game_rows:
        t_val = T[key]
        print(f"  {label:<28}  {ms(t_val):>10}  {pct(t_val):>7}  {us_avg(t_val, cnt):>18}")

    game_total = T["game.get_legal_actions"] + T["game.state_next"] + T["game.is_finished"]
    print(f"  {'Game logic (total)':<28}  {ms(game_total):>10}  {pct(game_total):>7}")
    print(bar)

    # --- MCTS tree ops ---
    mcts_rows = [
        ("MCTS select (excl. logic)", "mcts.select",  N["mcts.select"]),
        ("MCTS backup",               "mcts.backup",  N["mcts.backup"]),
    ]
    for label, key, cnt in mcts_rows:
        print(f"  {label:<28}  {ms(T[key]):>10}  {pct(T[key]):>7}  {us_avg(T[key], cnt):>18}")

    mcts_total = T["mcts.select"] + T["mcts.backup"]
    print(f"  {'MCTS tree ops (total)':<28}  {ms(mcts_total):>10}  {pct(mcts_total):>7}")
    print(bar)

    # --- unaccounted ---
    accounted = T["nn.total"] + game_total + mcts_total
    unaccounted = wall_elapsed - accounted
    print(f"  {'Unaccounted (Python etc.)':<28}  {ms(unaccounted):>10}  {pct(unaccounted):>7}")
    print(dbar)

    # --- Per-simulation summary ---
    print(f"\n  Per-simulation averages (over {n_simulations} sims):")
    sim_total_t = T["mcts.simulation"]          # includes select+expand+backup
    avg_sim_us = sim_total_t / n_simulations * 1e6 if n_simulations else 0
    nn_frac = n_nn_calls / n_simulations        # fraction of sims that expand
    print(f"    avg simulation time    {avg_sim_us:8.1f} µs")
    print(f"    avg NN call time       {T['nn.total']/n_nn_calls*1e6 if n_nn_calls else 0:8.1f} µs   ({nn_frac*100:.1f}% of sims expand)")
    print(f"    avg get_legal_actions  {T['game.get_legal_actions']/N['game.get_legal_actions']*1e6 if N['game.get_legal_actions'] else 0:8.1f} µs")
    print(f"    avg state.next()       {T['game.state_next']/N['game.state_next']*1e6 if N['game.state_next'] else 0:8.1f} µs  (per new node materialised)")
    print(f"    avg is_finished()      {T['game.is_finished']/N['game.is_finished']*1e6 if N['game.is_finished'] else 0:8.1f} µs")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AlphaZero self-play pipeline"
    )
    parser.add_argument("--games",    type=int, default=5,     help="number of games to play")
    parser.add_argument("--sims",     type=int, default=200,   help="MCTS simulations per move")
    parser.add_argument("--boardsize",type=int, default=7,     help="board side length")
    parser.add_argument("--walls",    type=int, default=5,     help="walls per player")
    parser.add_argument("--device",   type=str, default=None,  help="cpu or cuda (default: auto)")
    parser.add_argument("--model",    type=str, default="models/best.pt", help="path to model weights")
    parser.add_argument("--filters",  type=int, default=64,    help="conv filters (for random init)")
    parser.add_argument("--residual", type=int, default=6,     help="residual blocks (for random init)")
    args = parser.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_benchmark(
        num_games    = args.games,
        num_sims     = args.sims,
        boardsize    = args.boardsize,
        walls        = args.walls,
        device_str   = dev,
        model_path   = args.model,
        filters      = args.filters,
        num_residual = args.residual,
    )


if __name__ == "__main__":
    main()
