"""
Flask server for SigmaQuoridor.

API
---
GET  /api/state          -> full game state + legal actions (JSON)
GET  /api/agents         -> list of registered AI agents and their availability
POST /api/move           -> apply a move, return new state + legal actions
POST /api/agent_move     -> let the active agent make one move
POST /api/reset          -> reset the game to initial state

Move body (JSON):
  Pawn move:  {"type": "pawn", "direction": [dx, dy]}
  Wall place: {"type": "wall", "x": int, "y": int, "orientation": "h"|"v"}

Reset body (JSON, all optional):
  {"boardsize": 7, "walls": 5, "num_simulations": 800, "agent_id": "alphazero"}

The /api/state response is the single source of truth for the frontend
and for AI agents. An agent only needs to POST to /api/move.
"""

import time
from pathlib import Path
import torch
from flask import Flask, jsonify, request, send_from_directory

# ── Board / model configuration ───────────────────────────────────────────────
# Change these two lines when switching to a different board variant.
DEFAULT_BOARDSIZE    = 7
DEFAULT_WALLS        = 5
MODEL_DIR            = "models_9x9" if DEFAULT_BOARDSIZE == 9 else "models_7x7"
import re
from game import State, PawnAction, WallAction
from mcts import MCTSAgent
from dual_network import make_nn_evaluator
from rl_models import (
    QLearningAgent, SarsaAgent, ExpectedSarsaAgent,
    DoubleQLearningAgent, DoubleSarsaAgent, DoubleExpectedSarsaAgent,
    SimpleRLAgentWrapper,
)
from benchmark_agents import RandomAgent, GreedyDistanceAgent

CPU_DEVICE = torch.device("cpu")

_PAWN_ARROW: dict = {
    (0,  1): '↑', (0, -1): '↓', (-1, 0): '←', (1,  0): '→',
    (-1, 1): '↖', (1,  1): '↗', (-1,-1): '↙', (1, -1): '↘',
}


def _action_label(state: State, action) -> str:
    """Short human-readable label for an action, including pawn landing cell."""
    if isinstance(action, PawnAction):
        dx, dy = action.direction
        arrow  = _PAWN_ARROW.get((dx, dy), f'({dx},{dy})')
        is_p1  = state.is_player1_turn()
        cx, cy = state.player1pos if is_p1 else state.player2pos
        opp    = state.player2pos if is_p1 else state.player1pos
        if dx == 0 or dy == 0:
            tx, ty = cx + dx, cy + dy
            lx, ly = (cx + 2*dx, cy + 2*dy) if (tx, ty) == opp else (tx, ty)
        else:
            lx, ly = cx + dx, cy + dy
        return f'{arrow}({lx},{ly})'
    return ('H' if action.orientation == 'h' else 'V') + f'({action.x},{action.y})'


app = Flask(__name__, static_folder="static", template_folder="static")

# ── Agent registry ────────────────────────────────────────────────────────────
# To add a new agent, append an entry here.  Keys:
#   name        – display name shown in the frontend dropdown
#   description – one-line subtitle shown below the selector
#   available   – callable() -> bool; checked at request time so model
#                 files can be hot-loaded without restarting the server
#   factory     – callable(num_simulations: int) -> MCTSAgent
#
# The first available agent is used as the default on /api/reset.

def _always() -> bool:
    return True

_AGENT_REGISTRY: dict[str, dict] = {
    "alphazero": {
        "name": "SigmaQuoridor",
        "description": "MCTS guided by a trained neural network",
        "available": lambda: Path(MODEL_DIR, "best.pt").exists(),
        "factory": lambda n: MCTSAgent(
            evaluator=make_nn_evaluator(str(Path(MODEL_DIR, "best.pt")), device=CPU_DEVICE),
            num_simulations=n,
        ),
    },
    "mcts_rollout": {
        "name": "MCTS · Rollout",
        "description": "Pure MCTS with random rollouts",
        "available": _always,
        "factory": lambda n: MCTSAgent(num_simulations=n),
    },
    "supervised": {
        "name": "Supervised",
        "description": "Network trained purely by supervised learning on self-play data",
        "available": lambda: Path("models_7x7", "supervised.pt").exists(),
        "factory": lambda n: MCTSAgent(
            evaluator=make_nn_evaluator(str(Path("models_7x7", "supervised.pt")), device=CPU_DEVICE),
            num_simulations=n,
        ),
    },
    "greedy_distance": {
        "name": "Greedy Distance",
        "description": "Always picks the move maximising (opp dist − my dist) to goal",
        "available": _always,
        "factory": lambda _n: GreedyDistanceAgent(),
    },
    "random": {
        "name": "Random",
        "description": "Picks a legal move uniformly at random",
        "available": _always,
        "factory": lambda _n: RandomAgent(),
    },
}

# ── Simple RL agent registry ───────────────────────────────────────────────────
_RL_AGENT_CLASSES: dict[str, type] = {
    "QLearningAgent":           QLearningAgent,
    "SarsaAgent":               SarsaAgent,
    "ExpectedSarsaAgent":       ExpectedSarsaAgent,
    "DoubleQLearningAgent":     DoubleQLearningAgent,
    "DoubleSarsaAgent":         DoubleSarsaAgent,
    "DoubleExpectedSarsaAgent": DoubleExpectedSarsaAgent,
}
_RL_AGENT_DISPLAY_NAMES: dict[str, str] = {
    "QLearningAgent":           "Q-Learning",
    "SarsaAgent":               "SARSA",
    "ExpectedSarsaAgent":       "Expected SARSA",
    "DoubleQLearningAgent":     "Double Q-Learning",
    "DoubleSarsaAgent":         "Double SARSA",
    "DoubleExpectedSarsaAgent": "Double Expected SARSA",
}
_RL_DIR_PATTERN = re.compile(r"models_(\d+)x\d+_with_(\d+)_walls")

def _register_rl_models() -> None:
    """Scan for .pkl RL model files and register them in _AGENT_REGISTRY."""
    search_roots = [Path("."), Path("simple_rl_models")]
    for root in search_roots:
        if not root.exists():
            continue
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            m = _RL_DIR_PATTERN.search(model_dir.name)
            if m is None:
                continue
            boardsize = int(m.group(1))
            walls     = int(m.group(2))
            best_dir  = model_dir / "best"
            if not best_dir.is_dir():
                continue
            for pkl_file in sorted(best_dir.glob("*.pkl")):
                # Strip optional "_compressed" suffix to get the class name
                class_name = pkl_file.stem
                if class_name.endswith("_compressed"):
                    class_name = class_name[: -len("_compressed")]
                if class_name not in _RL_AGENT_CLASSES:
                    continue
                agent_cls = _RL_AGENT_CLASSES[class_name]
                display   = _RL_AGENT_DISPLAY_NAMES.get(class_name, class_name)
                suffix    = (
                    ""
                    if boardsize == DEFAULT_BOARDSIZE and walls == DEFAULT_WALLS
                    else f" ({boardsize}\u00d7{boardsize}, {walls}W)"
                )
                agent_id  = f"rl_{class_name}_{boardsize}_{walls}"
                if agent_id in _AGENT_REGISTRY:
                    continue  # already registered (prefer earlier path)

                def _make_factory(cls=agent_cls, path=pkl_file, bs=boardsize):
                    def factory(_num_sims):
                        inst = cls(boardsize=bs)
                        inst.load(path)
                        return SimpleRLAgentWrapper(inst)
                    return factory

                _AGENT_REGISTRY[agent_id] = {
                    "name":        display + suffix,
                    "description": f"Tabular RL agent ({boardsize}\u00d7{boardsize}, {walls} walls per player)",
                    "available":   (
                        lambda p=pkl_file, bs=boardsize, ws=walls:
                        p.exists() and bs == DEFAULT_BOARDSIZE and ws == DEFAULT_WALLS
                    ),
                    "factory":     _make_factory(),
                }

_register_rl_models()


def _default_agent_id() -> str:
    for agent_id, info in _AGENT_REGISTRY.items():
        if info["available"]():
            return agent_id
    raise RuntimeError("No agents available")

# Global mutable game state and active agent (single-session)
_state: State = State(boardsize=DEFAULT_BOARDSIZE, walls_p1=DEFAULT_WALLS, walls_p2=DEFAULT_WALLS)
_agent = _AGENT_REGISTRY[_default_agent_id()]["factory"](800)
_history: list[State] = []   # stack of states before each move, for undo


def _serialize_state(state: State) -> dict:
    """Convert a State object to a JSON-serialisable dict."""
    legal = state.get_legal_actions()

    pawn_moves = []
    for action in legal:
        if isinstance(action, PawnAction):
            pawn_moves.append(list(action.direction))

    wall_moves = []
    for action in legal:
        if isinstance(action, WallAction):
            wall_moves.append({
                "x": action.x,
                "y": action.y,
                "orientation": action.orientation,
            })

    return {
        "boardsize": state.boardsize,
        "player1pos": list(state.player1pos),
        "player2pos": list(state.player2pos),
        "hwalls": list(state.hwalls),    # boardsize x boardsize 0/1 matrix
        "vwalls": list(state.vwalls),
        "walls_p1": state.walls_p1,
        "walls_p2": state.walls_p2,
        # Anchor lists let the frontend draw each wall as one solid 2-cell span
        "hwall_anchors": [list(a) for a in state.hwall_anchors],
        "vwall_anchors": [list(a) for a in state.vwall_anchors],
        "current_player": state.get_current_player(),
        "depth": state.depth,
        "is_finished": state.is_finished(),
        "winner": state.winner(),
        "legal_pawn_moves": pawn_moves,  # list of [dx, dy]
        "legal_wall_moves": wall_moves,  # list of {x, y, orientation}
    }


@app.get("/api/checkpoints")
def get_checkpoints():
    checkpoint_dir = Path(MODEL_DIR, "checkpoints")
    if not checkpoint_dir.is_dir():
        return jsonify([])
    files = sorted(checkpoint_dir.glob("*.pt"))
    return jsonify([
        {"id": f.name, "path": str(Path(MODEL_DIR, "checkpoints", f.name)), "label": f.stem}
        for f in files
    ])


@app.get("/api/agents")
def get_agents():
    return jsonify([
        {
            "id":          agent_id,
            "name":        info["name"],
            "description": info["description"],
            "available":   info["available"](),
        }
        for agent_id, info in _AGENT_REGISTRY.items()
    ])


@app.get("/api/state")
def get_state():
    return jsonify(_serialize_state(_state))


@app.get("/api/nn_input")
def get_nn_input():
    N = _state.boardsize
    wi = _state.walls_initial if _state.walls_initial > 0 else 1
    md = N * N - 1
    planes = [ch.tolist() for ch in _state.to_nn_input()]
    names = [
        "My pawn",
        "Opp. pawn",
        "H-walls",
        "V-walls",
        "My walls left",
        "Opp. walls left",
        "My BFS dist",
        "Opp. BFS dist",
    ]
    # scale[i]: multiply normalised value by this to recover the original unit
    scales = [1, 1, 1, 1, wi, wi, md, md]
    return jsonify({"channels": planes, "names": names, "scales": scales,
                    "current_player": _state.get_current_player()})


@app.post("/api/move")
def post_move():
    global _state
    t0 = time.perf_counter()
    data = request.get_json(force=True)

    if data.get("type") == "pawn":
        dx, dy = data["direction"]
        action = PawnAction(direction=(dx, dy))
    elif data.get("type") == "wall":
        action = WallAction(x=data["x"], y=data["y"], orientation=data["orientation"])
    else:
        return jsonify({"error": "Unknown move type"}), 400

    if not _state.is_action_legal(action):
        return jsonify({"error": "Illegal action"}), 422

    _history.append(_state)
    _state = _state.next(action, check_legal=False)
    t1 = time.perf_counter()
    result = _serialize_state(_state)
    t2 = time.perf_counter()
    print(f"[move] next={1000*(t1-t0):.1f}ms  serialize={1000*(t2-t1):.1f}ms  total={1000*(t2-t0):.1f}ms", flush=True)
    return jsonify(result)


@app.post("/api/agent_move")
def post_agent_move():
    global _state
    if _state.is_finished():
        return jsonify({"error": "Game is already finished"}), 400
    data = request.get_json(force=True, silent=True) or {}
    if "num_simulations" in data:
        _agent.num_simulations = int(data["num_simulations"])
    t0 = time.perf_counter()
    action = _agent.select_action(_state)
    t1 = time.perf_counter()
    _history.append(_state)
    _state = _state.next(action, check_legal=False)
    t2 = time.perf_counter()
    result = _serialize_state(_state)
    print(f"[agent] think={1000*(t1-t0):.0f}ms  next={1000*(t2-t1):.1f}ms", flush=True)
    return jsonify(result)


@app.post("/api/undo")
def post_undo():
    global _state
    data = request.get_json(force=True, silent=True) or {}
    count = max(1, int(data.get("count", 1)))
    if len(_history) < count:
        if not _history:
            return jsonify({"error": "Nothing to undo"}), 400
        count = len(_history)  # undo as many as we have
    for _ in range(count):
        _state = _history.pop()
    return jsonify(_serialize_state(_state))


@app.post("/api/set_agent")
def post_set_agent():
    global _agent
    data = request.get_json(force=True, silent=True) or {}
    agent_id = data.get("agent_id", _default_agent_id())
    num_sims = data.get("num_simulations", 800)
    model_path = data.get("model_path")  # optional checkpoint override

    if agent_id not in _AGENT_REGISTRY:
        return jsonify({"error": f"Unknown agent: {agent_id}"}), 400
    info = _AGENT_REGISTRY[agent_id]

    if model_path and agent_id == "alphazero":
        # Validate path to prevent directory traversal
        resolved = Path(model_path).resolve()
        allowed = Path(MODEL_DIR).resolve()
        if not str(resolved).startswith(str(allowed)) or resolved.suffix != ".pt":
            return jsonify({"error": "Invalid model path"}), 400
        if not resolved.exists():
            return jsonify({"error": "Model file not found"}), 404
        _agent = MCTSAgent(evaluator=make_nn_evaluator(str(resolved)), num_simulations=num_sims)
    else:
        if not info["available"]():
            return jsonify({"error": f"Agent '{agent_id}' is not available"}), 400
        _agent = info["factory"](num_sims)
    return jsonify({"ok": True})


@app.post("/api/reset")
def post_reset():
    global _state, _agent
    data = request.get_json(force=True, silent=True) or {}
    size       = data.get("boardsize", DEFAULT_BOARDSIZE)
    walls      = data.get("walls", DEFAULT_WALLS)
    num_sims   = data.get("num_simulations", 800)
    agent_id   = data.get("agent_id", _default_agent_id())
    model_path = data.get("model_path")  # optional checkpoint override

    if agent_id not in _AGENT_REGISTRY:
        return jsonify({"error": f"Unknown agent: {agent_id}"}), 400
    info = _AGENT_REGISTRY[agent_id]

    if model_path and agent_id == "alphazero":
        resolved = Path(model_path).resolve()
        allowed  = Path(MODEL_DIR).resolve()
        if not str(resolved).startswith(str(allowed)) or resolved.suffix != ".pt":
            return jsonify({"error": "Invalid model path"}), 400
        if not resolved.exists():
            return jsonify({"error": "Model file not found"}), 404
        agent_instance = MCTSAgent(evaluator=make_nn_evaluator(str(resolved)), num_simulations=num_sims)
    else:
        if not info["available"]():
            return jsonify({"error": f"Agent '{agent_id}' is not available"}), 400
        agent_instance = info["factory"](num_sims)

    _history.clear()
    _state = State(boardsize=size, walls_p1=walls, walls_p2=walls)
    _agent = agent_instance
    return jsonify(_serialize_state(_state))


@app.get("/api/nn_analysis")
def get_nn_analysis():
    if _state.is_finished():
        return jsonify({"error": "Game is finished"}), 400
    legal = _state.get_legal_actions()
    if not legal:
        return jsonify({"error": "No legal actions"}), 400
    if not callable(getattr(_agent, "evaluator", None)):
        return jsonify({"error": "Active agent has no evaluator"}), 400
    priors, value = _agent.evaluator(_state, legal)
    moves = sorted(
        [{"label": _action_label(_state, a), "prob": float(p)}
         for a, p in zip(legal, priors)],
        key=lambda m: m["prob"], reverse=True,
    )
    return jsonify({
        "value":          float(value),
        "moves":          moves[:30],
        "current_player": _state.get_current_player(),
    })


@app.post("/api/mcts_analysis")
def post_mcts_analysis():
    if _state.is_finished():
        return jsonify({"error": "Game is finished"}), 400
    data     = request.get_json(force=True, silent=True) or {}
    num_sims = max(1, min(10000, int(data.get("num_simulations", 400))))
    # Fresh agent with independent tree — don't pollute the play agent's tree.
    analysis_agent = MCTSAgent(
        evaluator          = _agent.evaluator,
        num_simulations    = num_sims,
        training           = False,
        temperature        = 1.0,
        dist_bonus_weight  = 3.0,
        sim_batch_size     = 1,
    )
    policy = analysis_agent.get_policy(_state)
    root_q = analysis_agent._root.q_value if analysis_agent._root else 0.0
    moves  = [{"label": _action_label(_state, a), "prob": float(p)} for a, p in policy]
    return jsonify({
        "value":          float(root_q),
        "moves":          moves[:30],
        "current_player": _state.get_current_player(),
    })


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    import argparse
    from waitress import serve
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    print(f" * Running on http://127.0.0.1:{args.port}", flush=True)
    serve(app, host='127.0.0.1', port=args.port)
