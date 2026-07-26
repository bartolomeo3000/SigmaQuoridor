'use strict';
// ============================================================
// game.js — JavaScript port of game.py (State class + helpers)
// Runs in both main thread (index.html) and Web Worker (mcts_worker.js).
// ============================================================

// Pawn direction order MUST match Python ALL_PAWN_DIRECTIONS exactly
// so that actionToIndex() agrees with the ONNX model's policy head.
const ALL_PAWN_DIRECTIONS = [
  [ 0,  1], [ 0, -1], [-1,  0], [ 1,  0],  // up, down, left, right
  [-1,  1], [ 1,  1], [-1, -1], [ 1, -1],  // up-left, up-right, down-left, down-right
];

function actionSpaceSize(N) {
  return 8 + 2 * (N - 1) * (N - 1);
}

// Maps an action to the policy-head flat index (mirrors Python action_to_index).
function actionToIndex(action, N) {
  if (action.type === 'pawn') {
    const [dx, dy] = action.direction;
    for (let i = 0; i < 8; i++) {
      if (ALL_PAWN_DIRECTIONS[i][0] === dx && ALL_PAWN_DIRECTIONS[i][1] === dy) return i;
    }
    throw new Error('Unknown pawn direction: ' + JSON.stringify(action.direction));
  }
  const W = N - 1;
  if (action.orientation === 'h') return 8 + action.y * W + action.x;
  return 8 + W * W + action.y * W + action.x;
}

// Vertical policy-frame permutation — mirrors Python vert_policy_permutation()
// in game.py and vflip_action() in cpp/engine.hpp. The network sees a board
// canonicalised to the current player's POV (vertically flipped for P2), so its
// policy output is in that flipped frame. To read the logit for a REAL-board
// action when it is P2's turn, look it up at perm[realIndex]. perm is an
// involution (self-inverse). Pawn dirs: up<->down (and the up/down diagonal
// pairs); wall anchors: y -> N-2-y (x unchanged).
const _VERT_PAWN_FLIP = [1, 0, 2, 3, 6, 7, 4, 5];
const _vpermCache = new Map();
function vertPolicyPermutation(N) {
  if (_vpermCache.has(N)) return _vpermCache.get(N);
  const W = N - 1;
  const perm = new Int32Array(actionSpaceSize(N));
  for (let i = 0; i < 8; i++) perm[i] = _VERT_PAWN_FLIP[i];
  for (let y = 0; y < W; y++) {
    for (let x = 0; x < W; x++) {
      perm[8 + y * W + x]         = 8 + (W - 1 - y) * W + x;          // H-walls
      perm[8 + W * W + y * W + x] = 8 + W * W + (W - 1 - y) * W + x;  // V-walls
    }
  }
  _vpermCache.set(N, perm);
  return perm;
}

// ── Action label (matches Python _action_label in app.py) ────────────────────
// Lives here rather than in mcts_worker.js so the main thread can label plies
// in the game timeline with the same strings the analysis panel shows.
const PAWN_ARROW = {
  '0,1': '↑', '0,-1': '↓', '-1,0': '←', '1,0': '→',
  '-1,1': '↖', '1,1': '↗', '-1,-1': '↙', '1,-1': '↘',
};

function actionLabel(state, action) {
  if (action.type === 'pawn') {
    const [dx, dy] = action.direction;
    const arrow     = PAWN_ARROW[`${dx},${dy}`] || `(${dx},${dy})`;
    const [cx, cy]  = state.isPlayer1Turn() ? state.player1pos : state.player2pos;
    const [ox, oy]  = state.isPlayer1Turn() ? state.player2pos : state.player1pos;
    let lx, ly;
    if (dx === 0 || dy === 0) {
      const tx = cx + dx, ty = cy + dy;
      if (tx === ox && ty === oy) { lx = cx + 2 * dx; ly = cy + 2 * dy; }
      else                        { lx = tx; ly = ty; }
    } else { lx = cx + dx; ly = cy + dy; }
    return `${arrow}(${lx},${ly})`;
  }
  return (action.orientation === 'h' ? 'H' : 'V') + `(${action.x},${action.y})`;
}

// ============================================================
class State {
  /**
   * @param {object} opts
   * @param {number}   [opts.boardsize=7]
   * @param {number}   [opts.depth=0]
   * @param {number[]} [opts.player1pos]   [x,y]
   * @param {number[]} [opts.player2pos]   [x,y]
   * @param {Uint8Array}[opts.hwalls]
   * @param {Uint8Array}[opts.vwalls]
   * @param {number}   [opts.walls_p1=5]
   * @param {number}   [opts.walls_p2=5]
   * @param {Set}      [opts.hwall_anchors]  Set<"x,y">
   * @param {Set}      [opts.vwall_anchors]  Set<"x,y">
   * @param {number}   [opts.walls_initial]
   * @param {Map|Iterable}[opts.position_history]  Map<string,int>
   * @param {Int32Array}[opts.p1_dist]   flat N*N dist grid — skip recompute
   * @param {Int32Array}[opts.p2_dist]
   * @param {Set}      [opts.p1_path_edges]
   * @param {Set}      [opts.p2_path_edges]
   */
  constructor({
    boardsize        = 7,
    depth            = 0,
    player1pos       = null,
    player2pos       = null,
    hwalls           = null,
    vwalls           = null,
    walls_p1         = 5,
    walls_p2         = 5,
    hwall_anchors    = null,
    vwall_anchors    = null,
    walls_initial    = null,
    position_history = null,
    p1_dist          = null,
    p2_dist          = null,
    p1_path_edges    = null,
    p2_path_edges    = null,
  } = {}) {
    if (boardsize % 2 === 0) throw new Error('Board size must be odd');
    this.boardsize    = boardsize;
    this.depth        = depth;
    this.player1pos   = player1pos !== null ? player1pos  : [Math.floor(boardsize / 2), 0];
    this.player2pos   = player2pos !== null ? player2pos  : [Math.floor(boardsize / 2), boardsize - 1];
    this.hwalls       = hwalls     !== null ? hwalls      : new Uint8Array(boardsize * boardsize);
    this.vwalls       = vwalls     !== null ? vwalls      : new Uint8Array(boardsize * boardsize);
    this.walls_p1     = walls_p1;
    this.walls_p2     = walls_p2;
    this.walls_initial = walls_initial !== null ? walls_initial : walls_p1;

    // Anchor sets: Set<"x,y" strings> — O(1) overlap checks
    this.hwall_anchors = hwall_anchors !== null ? hwall_anchors : new Set();
    this.vwall_anchors = vwall_anchors !== null ? vwall_anchors : new Set();

    // Distance grids (flat Int32Array of N*N, indexed dist[y*N+x]).
    // When provided from copy(), skip recomputation.
    if (p1_dist !== null) {
      this.p1_dist        = p1_dist;
      this.p2_dist        = p2_dist;
      this.p1_path_edges  = p1_path_edges;
      this.p2_path_edges  = p2_path_edges;
    } else {
      this._recomputeDists();
    }

    // Position history: Map<string, count>
    // Accepts a Map directly, or an iterable of [key,count] pairs.
    if (position_history !== null) {
      this.position_history = position_history instanceof Map
        ? position_history
        : new Map(position_history);
    } else {
      this.position_history = new Map();
      this._recordPosition();
    }

    this._legal_actions_cache = null;
  }

  // ── Position key ────────────────────────────────────────────────────────────
  // Mirrors Python _position_key(): (p1pos, p2pos, depth%2, frozenset(h), frozenset(v))
  _positionKey() {
    const sortedH = [...this.hwall_anchors].sort().join(';');
    const sortedV = [...this.vwall_anchors].sort().join(';');
    const [p1x, p1y] = this.player1pos;
    const [p2x, p2y] = this.player2pos;
    return `${p1x},${p1y}|${p2x},${p2y}|${this.depth % 2}|${sortedH}|${sortedV}`;
  }

  _recordPosition() {
    const key = this._positionKey();
    this.position_history.set(key, (this.position_history.get(key) || 0) + 1);
  }

  // ── Board helpers ────────────────────────────────────────────────────────────
  getCurrentPlayer() { return this.depth % 2 === 0 ? 1 : 2; }
  isPlayer1Turn()    { return this.depth % 2 === 0; }
  _inBounds(x, y)    { return x >= 0 && x < this.boardsize && y >= 0 && y < this.boardsize; }

  // O(1) wall-blocked check — mirrors Python _is_edge_blocked().
  _isEdgeBlocked(x, y, dx, dy) {
    const N = this.boardsize;
    if (dy ===  1) return this.hwalls[y       * N + x    ] !== 0;
    if (dy === -1) return this.hwalls[(y - 1) * N + x    ] !== 0;
    if (dx ===  1) return this.vwalls[y       * N + x    ] !== 0;
    if (dx === -1) return this.vwalls[y       * N + x - 1] !== 0;
    return false;
  }

  // ── BFS (reachability) ──────────────────────────────────────────────────────
  // Returns true if start=[x,y] can reach goal_row.
  BFS(start, goal_row) {
    const [sx, sy] = start;
    if (sy === goal_row) return true;
    const N   = this.boardsize;
    const vis = new Uint8Array(N * N);
    vis[sy * N + sx] = 1;
    const q  = [sx, sy];
    let head = 0;
    const hw = this.hwalls, vw = this.vwalls;
    while (head < q.length) {
      const x = q[head++], y = q[head++];
      const yi = y * N;
      if (y + 1 < N && !vis[yi + N + x] && !hw[yi + x]) {
        if (y + 1 === goal_row) return true;
        vis[yi + N + x] = 1; q.push(x, y + 1);
      }
      if (y > 0 && !vis[yi - N + x] && !hw[yi - N + x]) {
        if (y - 1 === goal_row) return true;
        vis[yi - N + x] = 1; q.push(x, y - 1);
      }
      if (x + 1 < N && !vis[yi + x + 1] && !vw[yi + x]) {
        if (y === goal_row) return true;
        vis[yi + x + 1] = 1; q.push(x + 1, y);
      }
      if (x > 0 && !vis[yi + x - 1] && !vw[yi + x - 1]) {
        if (y === goal_row) return true;
        vis[yi + x - 1] = 1; q.push(x - 1, y);
      }
    }
    return false;
  }

  // Multi-source BFS from all cells in goal_row.
  // Returns flat Int32Array of N*N; unreachable = N*N (INF).
  static _bfsDistGrid(hwalls, vwalls, goal_row, N) {
    const INF  = N * N;
    const dist = new Int32Array(N * N).fill(INF);
    const q    = [];
    let head   = 0;
    for (let x = 0; x < N; x++) { dist[goal_row * N + x] = 0; q.push(x, goal_row); }
    while (head < q.length) {
      const x = q[head++], y = q[head++];
      const d = dist[y * N + x], yi = y * N;
      if (y + 1 < N && dist[yi + N + x] === INF && !hwalls[yi + x])
        { dist[yi + N + x] = d + 1; q.push(x, y + 1); }
      if (y > 0   && dist[yi - N + x] === INF && !hwalls[yi - N + x])
        { dist[yi - N + x] = d + 1; q.push(x, y - 1); }
      if (x + 1 < N && dist[yi + x + 1] === INF && !vwalls[yi + x])
        { dist[yi + x + 1] = d + 1; q.push(x + 1, y); }
      if (x > 0   && dist[yi + x - 1] === INF && !vwalls[yi + x - 1])
        { dist[yi + x - 1] = d + 1; q.push(x - 1, y); }
    }
    return dist;
  }

  // Greedy shortest-path edge set — mirrors Python _shortest_path_edge_set().
  // Returns Set<"h,y,x"|"v,y,x"> or null if player is unreachable.
  _shortestPathEdgeSet(start, dist_grid) {
    const N   = this.boardsize;
    const INF = N * N;
    const [sx, sy] = start;
    let d = dist_grid[sy * N + sx];
    if (d === INF) return null;
    const edges = new Set();
    let x = sx, y = sy;
    while (d > 0) {
      let moved = false;
      for (const [dx, dy] of [[0, 1], [0, -1], [1, 0], [-1, 0]]) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
        if (dist_grid[ny * N + nx] !== d - 1) continue;
        if (this._isEdgeBlocked(x, y, dx, dy)) continue;
        if      (dy ===  1) edges.add(`h,${y},${x}`);
        else if (dy === -1) edges.add(`h,${y - 1},${x}`);
        else if (dx ===  1) edges.add(`v,${y},${x}`);
        else                edges.add(`v,${y},${x - 1}`);
        x = nx; y = ny; d--;
        moved = true;
        break;
      }
      if (!moved) break;
    }
    return edges;
  }

  _recomputePathEdges() {
    this.p1_path_edges = this._shortestPathEdgeSet(this.player1pos, this.p1_dist);
    this.p2_path_edges = this._shortestPathEdgeSet(this.player2pos, this.p2_dist);
  }

  _recomputeDists() {
    const N = this.boardsize;
    this.p1_dist = State._bfsDistGrid(this.hwalls, this.vwalls, N - 1, N);
    this.p2_dist = State._bfsDistGrid(this.hwalls, this.vwalls, 0,     N);
    this._recomputePathEdges();
  }

  // ── Game state ───────────────────────────────────────────────────────────────
  winner() {
    if (this.player1pos[1] === this.boardsize - 1) return 1;
    if (this.player2pos[1] === 0)                  return 2;
    return 0;
  }

  isDrawn() {
    if (this.depth >= 200) return true;
    if (this.getLegalActions().length === 0) return true;
    return false;
  }

  isFinished() { return this.winner() !== 0 || this.isDrawn(); }

  // ── Pawn moves ───────────────────────────────────────────────────────────────
  // Returns the landing cell [x,y] for a (legal) pawn action, without state mutation.
  _pawnDest(action) {
    const [dx, dy] = action.direction;
    const [cx, cy] = this.isPlayer1Turn() ? this.player1pos : this.player2pos;
    const [ox, oy] = this.isPlayer1Turn() ? this.player2pos : this.player1pos;
    if (dx === 0 || dy === 0) {
      const tx = cx + dx, ty = cy + dy;
      return (tx === ox && ty === oy) ? [cx + 2 * dx, cy + 2 * dy] : [tx, ty];
    }
    return [cx + dx, cy + dy];
  }

  computeMoveAdvantages(legalActions) {
    let myPos, oppPos, myDistGrid, oppDistGrid, myPathEdges, oppPathEdges;
    if (this.isPlayer1Turn()) {
      myPos        = this.player1pos;
      oppPos       = this.player2pos;
      myDistGrid   = this.p1_dist;
      oppDistGrid  = this.p2_dist;
      myPathEdges  = this.p1_path_edges;
      oppPathEdges = this.p2_path_edges;
    } else {
      myPos        = this.player2pos;
      oppPos       = this.player1pos;
      myDistGrid   = this.p2_dist;
      oppDistGrid  = this.p1_dist;
      myPathEdges  = this.p2_path_edges;
      oppPathEdges = this.p1_path_edges;
    }

    const N          = this.boardsize;
    const myDistNow  = myDistGrid[myPos[1] * N + myPos[0]];
    const oppDistNow = oppDistGrid[oppPos[1] * N + oppPos[0]];
    const baseline   = oppDistNow - myDistNow;

    return legalActions.map(action => {
      let myDistAfter, oppDistAfter;
      if (action.type === 'pawn') {
        const [destX, destY] = this._pawnDest(action);
        myDistAfter  = myDistGrid[destY * N + destX];
        oppDistAfter = oppDistNow;
      } else {
        const { x, y, orientation } = action;
        const segA = orientation === 'h' ? `h,${y},${x}`     : `v,${y},${x}`;
        const segB = orientation === 'h' ? `h,${y},${x + 1}` : `v,${y + 1},${x}`;
        const oppDelta = oppPathEdges !== null && (oppPathEdges.has(segA) || oppPathEdges.has(segB)) ? 1 : 0;
        const myDelta  = myPathEdges  !== null && (myPathEdges.has(segA)  || myPathEdges.has(segB))  ? 1 : 0;
        myDistAfter  = myDistNow + myDelta;
        oppDistAfter = oppDistNow + oppDelta;
      }
      return (oppDistAfter - myDistAfter) - baseline;
    });
  }

  isPawnMoveLegal(action) {
    const [dx, dy] = action.direction;
    const [cx, cy] = this.isPlayer1Turn() ? this.player1pos : this.player2pos;
    const [ox, oy] = this.isPlayer1Turn() ? this.player2pos : this.player1pos;

    if (dx === 0 || dy === 0) {
      // Orthogonal
      const tx = cx + dx, ty = cy + dy;
      if (!this._inBounds(tx, ty) || this._isEdgeBlocked(cx, cy, dx, dy)) return false;
      if (tx !== ox || ty !== oy) return true;
      // Opponent occupies target: straight jump
      return this._inBounds(tx + dx, ty + dy) && !this._isEdgeBlocked(tx, ty, dx, dy);
    }

    // Diagonal go-around: try both axes as the jump axis
    for (const [[sdx, sdy], [ldx, ldy]] of [[[dx, 0], [0, dy]], [[0, dy], [dx, 0]]]) {
      if (cx + sdx !== ox || cy + sdy !== oy) continue;           // opponent not this way
      if (this._isEdgeBlocked(cx, cy, sdx, sdy)) continue;        // wall to opponent
      const px = ox + sdx, py = oy + sdy;
      if (this._inBounds(px, py) && !this._isEdgeBlocked(ox, oy, sdx, sdy)) continue; // straight open
      const lx = ox + ldx, ly = oy + ldy;
      if (!this._inBounds(lx, ly) || this._isEdgeBlocked(ox, oy, ldx, ldy)) continue;
      return true;
    }
    return false;
  }

  _getLegalPawnActions() {
    const actions = [];
    for (const dir of ALL_PAWN_DIRECTIONS) {
      const a = { type: 'pawn', direction: dir };
      if (this.isPawnMoveLegal(a)) actions.push(a);
    }
    // Filter out moves that would create a third repetition
    const hist        = this.position_history;
    const sortedH     = [...this.hwall_anchors].sort().join(';');
    const sortedV     = [...this.vwall_anchors].sort().join(';');
    const nextParity  = (this.depth + 1) % 2;
    const [p1x, p1y]  = this.player1pos;
    const [p2x, p2y]  = this.player2pos;
    const isP1        = this.isPlayer1Turn();

    return actions.filter(a => {
      const [dx, dy]  = a.direction;
      const dest       = this._pawnDest(a);
      const p1k = isP1 ? `${dest[0]},${dest[1]}` : `${p1x},${p1y}`;
      const p2k = isP1 ? `${p2x},${p2y}`         : `${dest[0]},${dest[1]}`;
      const key = `${p1k}|${p2k}|${nextParity}|${sortedH}|${sortedV}`;
      return (hist.get(key) || 0) < 2;
    });
  }

  // ── Wall moves ───────────────────────────────────────────────────────────────
  // Returns (h_illegal, v_illegal) as Set<"x,y"> — mirrors _build_overlap_sets().
  _buildOverlapSets() {
    const h_ill = new Set(), v_ill = new Set();
    for (const key of this.hwall_anchors) {
      const [ax, ay] = key.split(',').map(Number);
      h_ill.add(`${ax - 1},${ay}`); h_ill.add(`${ax},${ay}`); h_ill.add(`${ax + 1},${ay}`);
      v_ill.add(`${ax},${ay}`);
    }
    for (const key of this.vwall_anchors) {
      const [ax, ay] = key.split(',').map(Number);
      v_ill.add(`${ax},${ay - 1}`); v_ill.add(`${ax},${ay}`); v_ill.add(`${ax},${ay + 1}`);
      h_ill.add(`${ax},${ay}`);
    }
    return { h_ill, v_ill };
  }

  _getLegalWallActions() {
    const player = this.getCurrentPlayer();
    if ((player === 1 ? this.walls_p1 : this.walls_p2) === 0) return [];

    const N              = this.boardsize;
    const { h_ill, v_ill } = this._buildOverlapSets();
    const pe1            = this.p1_path_edges;
    const pe2            = this.p2_path_edges;
    const hw             = this.hwalls;
    const vw             = this.vwalls;
    const goal1          = N - 1;
    const p1pos          = this.player1pos;
    const p2pos          = this.player2pos;
    const actions        = [];

    for (let y = 0; y < N - 1; y++) {
      for (let x = 0; x < N - 1; x++) {
        // ── H-wall ──
        if (!h_ill.has(`${x},${y}`)) {
          const sA = `h,${y},${x}`, sB = `h,${y},${x + 1}`;
          const p1n = pe1 === null || pe1.has(sA) || pe1.has(sB);
          const p2n = pe2 === null || pe2.has(sA) || pe2.has(sB);
          if (!p1n && !p2n) {
            actions.push({ type: 'wall', x, y, orientation: 'h' });
          } else {
            const idx = y * N + x;
            hw[idx] = 1; hw[idx + 1] = 1;
            const ok = (!p1n || this.BFS(p1pos, goal1)) && (!p2n || this.BFS(p2pos, 0));
            hw[idx] = 0; hw[idx + 1] = 0;
            if (ok) actions.push({ type: 'wall', x, y, orientation: 'h' });
          }
        }
        // ── V-wall ──
        if (!v_ill.has(`${x},${y}`)) {
          const sA = `v,${y},${x}`, sB = `v,${y + 1},${x}`;
          const p1n = pe1 === null || pe1.has(sA) || pe1.has(sB);
          const p2n = pe2 === null || pe2.has(sA) || pe2.has(sB);
          if (!p1n && !p2n) {
            actions.push({ type: 'wall', x, y, orientation: 'v' });
          } else {
            const idx = y * N + x;
            vw[idx] = 1; vw[idx + N] = 1;
            const ok = (!p1n || this.BFS(p1pos, goal1)) && (!p2n || this.BFS(p2pos, 0));
            vw[idx] = 0; vw[idx + N] = 0;
            if (ok) actions.push({ type: 'wall', x, y, orientation: 'v' });
          }
        }
      }
    }
    return actions;
  }

  getLegalActions() {
    if (this._legal_actions_cache === null)
      this._legal_actions_cache = [
        ...this._getLegalPawnActions(),
        ...this._getLegalWallActions(),
      ];
    return this._legal_actions_cache;
  }

  isActionLegal(action) {
    if (action.type === 'pawn') return this.isPawnMoveLegal(action);
    if (action.type === 'wall') {
      const { x, y, orientation } = action;
      const lim = this.boardsize - 2;
      if (x < 0 || x > lim || y < 0 || y > lim) return false;
      if ((this.getCurrentPlayer() === 1 ? this.walls_p1 : this.walls_p2) === 0) return false;
      if (orientation === 'h') {
        if (this.hwall_anchors.has(`${x - 1},${y}`) || this.hwall_anchors.has(`${x},${y}`) ||
            this.hwall_anchors.has(`${x + 1},${y}`) || this.vwall_anchors.has(`${x},${y}`))
          return false;
      } else {
        if (this.vwall_anchors.has(`${x},${y - 1}`) || this.vwall_anchors.has(`${x},${y}`) ||
            this.vwall_anchors.has(`${x},${y + 1}`) || this.hwall_anchors.has(`${x},${y}`))
          return false;
      }
      // Temporarily apply bits for BFS check
      const N = this.boardsize;
      if (orientation === 'h') { this.hwalls[y * N + x] = 1; this.hwalls[y * N + x + 1] = 1; }
      else                     { this.vwalls[y * N + x] = 1; this.vwalls[(y + 1) * N + x] = 1; }
      const ok = this.BFS(this.player1pos, N - 1) && this.BFS(this.player2pos, 0);
      if (orientation === 'h') { this.hwalls[y * N + x] = 0; this.hwalls[y * N + x + 1] = 0; }
      else                     { this.vwalls[y * N + x] = 0; this.vwalls[(y + 1) * N + x] = 0; }
      return ok;
    }
    return false;
  }

  // ── State transitions ─────────────────────────────────────────────────────────
  copy() {
    return new State({
      boardsize:        this.boardsize,
      depth:            this.depth,
      player1pos:       this.player1pos.slice(),
      player2pos:       this.player2pos.slice(),
      hwalls:           new Uint8Array(this.hwalls),
      vwalls:           new Uint8Array(this.vwalls),
      walls_p1:         this.walls_p1,
      walls_p2:         this.walls_p2,
      hwall_anchors:    new Set(this.hwall_anchors),
      vwall_anchors:    new Set(this.vwall_anchors),
      walls_initial:    this.walls_initial,
      position_history: new Map(this.position_history),
      // Share dist/path_edge references — only replaced wholesale, never mutated
      p1_dist:          this.p1_dist,
      p2_dist:          this.p2_dist,
      p1_path_edges:    this.p1_path_edges,
      p2_path_edges:    this.p2_path_edges,
    });
  }

  next(action) {
    const s          = this.copy();
    const [cx, cy]   = s.isPlayer1Turn() ? s.player1pos : s.player2pos;
    const [ox, oy]   = s.isPlayer1Turn() ? s.player2pos : s.player1pos;

    if (action.type === 'pawn') {
      const [dx, dy] = action.direction;
      let np;
      if (dx === 0 || dy === 0) {
        const tx = cx + dx, ty = cy + dy;
        np = (tx === ox && ty === oy) ? [cx + 2 * dx, cy + 2 * dy] : [tx, ty];
      } else {
        np = [cx + dx, cy + dy];
      }
      if (s.isPlayer1Turn()) s.player1pos = np; else s.player2pos = np;
      s._recomputePathEdges();

    } else {
      const { x, y, orientation } = action;
      const N = s.boardsize;
      if (orientation === 'h') {
        s.hwalls[y * N + x] = 1; s.hwalls[y * N + x + 1] = 1;
        s.hwall_anchors.add(`${x},${y}`);
      } else {
        s.vwalls[y * N + x] = 1; s.vwalls[(y + 1) * N + x] = 1;
        s.vwall_anchors.add(`${x},${y}`);
      }
      if (s.getCurrentPlayer() === 1) s.walls_p1--; else s.walls_p2--;
      s._recomputeDists();
    }

    s.depth++;
    s._recordPosition();
    return s;
  }

  // ── Neural-network input (mirrors Python to_nn_input()) ──────────────────────
  // Returns Float32Array of shape [8 * N * N], channels first.
  // Board is flipped vertically for P2 so the current player always moves toward row N-1.
  toNNInput() {
    const N       = this.boardsize;
    const INF     = N * N;
    const maxDist = INF - 1;
    const norm    = this.walls_initial > 0 ? this.walls_initial : 1;
    const out     = new Float32Array(8 * N * N);

    let myPos, oppPos, myWalls, oppWalls, hw, vw, myDist, oppDist;

    if (this.isPlayer1Turn()) {
      myPos    = this.player1pos;
      oppPos   = this.player2pos;
      myWalls  = this.walls_p1;
      oppWalls = this.walls_p2;
      hw       = this.hwalls;
      vw       = this.vwalls;
      myDist   = this.p1_dist;
      oppDist  = this.p2_dist;
    } else {
      // Flip vertically: new_r = N-1-old_r
      // hwalls: rows 0..N-2 only (row N-1 stays 0); flipped_hw[r] = hw[N-2-r]
      const fhw = new Float32Array(N * N);
      for (let r = 0; r < N - 1; r++) {
        const src = (N - 2 - r) * N;
        for (let c = 0; c < N; c++) fhw[r * N + c] = this.hwalls[src + c];
      }
      // vwalls: all rows flipped; flipped_vw[r] = vw[N-1-r]
      const fvw = new Float32Array(N * N);
      for (let r = 0; r < N; r++) {
        const src = (N - 1 - r) * N;
        for (let c = 0; c < N; c++) fvw[r * N + c] = this.vwalls[src + c];
      }
      // Dist grids: my = p2_dist reversed, opp = p1_dist reversed
      const fMyD = new Int32Array(N * N), fOppD = new Int32Array(N * N);
      for (let r = 0; r < N; r++) {
        const src = (N - 1 - r) * N;
        for (let c = 0; c < N; c++) {
          fMyD[r * N + c]  = this.p2_dist[src + c];
          fOppD[r * N + c] = this.p1_dist[src + c];
        }
      }
      myPos    = [this.player2pos[0], N - 1 - this.player2pos[1]];
      oppPos   = [this.player1pos[0], N - 1 - this.player1pos[1]];
      myWalls  = this.walls_p2;
      oppWalls = this.walls_p1;
      hw       = fhw;
      vw       = fvw;
      myDist   = fMyD;
      oppDist  = fOppD;
    }

    // Ch 0: my pawn (1-hot)
    out[0 * N * N + myPos[1]  * N + myPos[0]]  = 1;
    // Ch 1: opp pawn (1-hot)
    out[1 * N * N + oppPos[1] * N + oppPos[0]] = 1;
    // Ch 2: h-walls, Ch 3: v-walls
    for (let i = 0; i < N * N; i++) { out[2 * N * N + i] = hw[i]; out[3 * N * N + i] = vw[i]; }
    // Ch 4-5: wall counts (scalar planes, normalised)
    const mwn = myWalls / norm, own = oppWalls / norm;
    for (let i = 0; i < N * N; i++) { out[4 * N * N + i] = mwn; out[5 * N * N + i] = own; }
    // Ch 6-7: BFS distance planes
    for (let i = 0; i < N * N; i++) {
      out[6 * N * N + i] = myDist[i]  < INF ? myDist[i]  / maxDist : 1;
      out[7 * N * N + i] = oppDist[i] < INF ? oppDist[i] / maxDist : 1;
    }
    return out;
  }

  // ── Serialization (matches Flask /api/state response format exactly) ─────────
  serialize() {
    const legal = this.getLegalActions();
    const w     = this.winner();
    return {
      boardsize:        this.boardsize,
      player1pos:       this.player1pos.slice(),
      player2pos:       this.player2pos.slice(),
      hwalls:           Array.from(this.hwalls),
      vwalls:           Array.from(this.vwalls),
      walls_p1:         this.walls_p1,
      walls_p2:         this.walls_p2,
      // hwall_anchors / vwall_anchors: array of [x,y] number pairs
      hwall_anchors:    [...this.hwall_anchors].map(s => s.split(',').map(Number)),
      vwall_anchors:    [...this.vwall_anchors].map(s => s.split(',').map(Number)),
      current_player:   this.getCurrentPlayer(),
      depth:            this.depth,
      is_finished:      this.isFinished(),
      winner:           w,
      legal_pawn_moves: legal.filter(a => a.type === 'pawn').map(a => a.direction),
      legal_wall_moves: legal.filter(a => a.type === 'wall').map(a =>
        ({ x: a.x, y: a.y, orientation: a.orientation })),
    };
  }
}
