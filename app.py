"""
Flask server for SigmaQuoridor.

API
---
GET  /api/state          -> full game state + legal actions (JSON)
POST /api/move           -> apply a move, return new state + legal actions
POST /api/reset          -> reset the game to initial state

Move body (JSON):
  Pawn move:  {"type": "pawn", "direction": [dx, dy]}
  Wall place: {"type": "wall", "x": int, "y": int, "orientation": "h"|"v"}

The /api/state response is the single source of truth for the frontend
and for AI agents. An agent only needs to POST to /api/move.
"""

import time
from flask import Flask, jsonify, request, send_from_directory
from game import State, PawnAction, WallAction

app = Flask(__name__, static_folder="static", template_folder="static")

# Global mutable game state (single-session, single-game)
_state: State = State()


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

    _state = _state.next(action, check_legal=False)
    t1 = time.perf_counter()
    result = _serialize_state(_state)
    t2 = time.perf_counter()
    print(f"[move] next={1000*(t1-t0):.1f}ms  serialize={1000*(t2-t1):.1f}ms  total={1000*(t2-t0):.1f}ms", flush=True)
    return jsonify(result)


@app.post("/api/reset")
def post_reset():
    global _state
    boardsize = request.get_json(force=True, silent=True) or {}
    size = boardsize.get("boardsize", 7)
    walls = boardsize.get("walls", 5)
    _state = State(boardsize=size, walls_p1=walls, walls_p2=walls)
    return jsonify(_serialize_state(_state))


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    from waitress import serve
    print(" * Running on http://127.0.0.1:5000", flush=True)
    serve(app, host='127.0.0.1', port=5000)
