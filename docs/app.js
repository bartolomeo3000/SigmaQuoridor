// ── Constants ────────────────────────────────────────────────────────────────
// Board variants: swapping variant changes boardsize/walls and the available
// model list (7x7 and 9x9 nets are architecturally incompatible — different
// action-space size — so they're never interchangeable).
const BOARD_VARIANTS = {
  '7x7': {
    boardsize: 7, walls: 5,
    defaultModel: './models/supervised_extended.onnx',
  },
  '9x9': {
    boardsize: 9, walls: 10,
    defaultModel: './models_9x9/best.onnx',
  },
};
let currentBoardVariant = '9x9';
function currentVariant() { return BOARD_VARIANTS[currentBoardVariant]; }

// ── Layout ───────────────────────────────────────────────────────────────────
const CELL   = 62;
const GAP    = 10;
const STEP   = CELL + GAP;
const CELL_R = 5;
const LABEL  = 20;

// ── Colours ───────────────────────────────────────────────────────────────────
const CLR = {
  cell:             '#7a6a58',
  cellGoal1:        '#3a6e3a',
  cellGoal2:        '#6e3030',
  cellLegal:        '#2a4a7a',
  cellLegalRing:    '#7ab8ff',
  wall:             '#b5611a',
  wallGlow:         'rgba(181,97,26,0.55)',
  wallHoverOk:      'rgba(220,140,40,0.62)',
  wallHoverOkGlow:  'rgba(220,140,40,0.7)',
  wallHoverBad:     'rgba(210,55,55,0.50)',
  wallHoverBadGlow: 'rgba(210,55,55,0.6)',
  pawn1:            '#3fb950',
  pawn2:            '#f85149',
  pawnStroke:       'rgba(255,255,255,0.7)',
  // last move played — tint on the cell it left, ring on the one it reached
  lastFrom1:        'rgba(63,185,80,0.20)',
  lastRing1:        'rgba(99,220,116,0.65)',
  lastFrom2:        'rgba(248,81,73,0.20)',
  lastRing2:        'rgba(255,130,124,0.65)',
  lastWall:         'rgba(255,196,120,0.95)',
  lastWallGlow:     'rgba(255,170,60,0.85)',
};

// Cosmetic transition for the move just played: the pawn slides to its new cell,
// a freshly placed wall fades in. Paint-time only — `state` is already the
// post-move position, so logic and hit-testing don't see any of it.
const ANIM_MS = 170;
let _anim    = null;   // {kind:'pawn', player, from, to, t0} | {kind:'wall', t0}
let _animRAF = null;

// ── Game state ───────────────────────────────────────────────────────────────
// `gameState` is the canonical State instance for the position on screen.
// `state` is a plain serialized object (same format as Flask /api/state) used
//   by all rendering / interaction code — never modified except by applyState().
//
// The game is kept as a TIMELINE rather than a pop-only undo stack, so history
// navigation is symmetric (back/forward/scrub) and a rewound position can be
// replayed forward again:
//     timeline[i] is the position BEFORE moves[i]
//  => timeline.length === moves.length + 1, and gameState === timeline[viewIdx]
// State.next() is non-mutating, so every entry is an independent snapshot.
let gameState = null;
let timeline  = [];
let moves     = [];   // { action, label } — moves[i] takes timeline[i] -> timeline[i+1]
let viewIdx   = 0;

let state          = null;
let landing        = [];
let hoverWall      = null;
let _hoverWallTimer = null;
let _pendingWall   = null;
let hoverCell      = null;
let analysisHoverMove = null;
let legalWallSet   = new Set();
let gameMode       = 'hva';
let agentThinking   = false;
let _searchGen      = 0;
let _thinkingSide   = 0;   // which player the search we're waiting on belongs to
let _thinkStart     = 0;   // performance.now() when that search was posted
let _thinkDone      = 0;   // simulations finished in the search we're waiting on
let _thinkTotal     = 0;   // 0 = no countable progress (minimax), so no bar
let _analysisDone   = 0;   // same, for the analysis panel's own MCTS run
let _analysisTotal  = 0;
let _analysisNnAvailable   = false;
let _analysisNnLoadFailed  = false;
let analysisModelPath      = './models_9x9/best.onnx';
let _analysisModelLoading  = false;

// ── Playback ─────────────────────────────────────────────────────────────────
// `paused` gates every kind of auto-advance: agent searches at the live
// position, and replay of already-recorded moves when parked behind it.
// `moveDelayMs` paces the two cases where a game plays out on its own and you
// are only watching: AI vs AI, and replaying recorded moves. There it is a
// MINIMUM display time, so a fast search doesn't flash past (a slow one is
// never delayed further). It deliberately does NOT apply to the AI's reply in
// H vs AI / AI vs H — you're waiting on that move, so it shows immediately.
let paused        = false;
let moveDelayMs   = 750;
let _stepOnce     = false;   // one-shot: advance a single move, then re-pause
let _advanceTimer = null;

// ── Per-side agent configuration ──────────────────────────────────────────────
// Each side carries its own agent, search budget and checkpoint, which is what
// makes AI vs AI worth watching (e.g. the current net against its predecessor).
// `manual` records whether the user has picked this side's agent, so async ONNX
// loading may only auto-select 'alphazero' for sides they haven't touched.
const SIDES = [1, 2];
function _defaultSideConfig() {
  return {
    agentId:      'alphazero',
    numSims:      100,
    minimaxDepth: 3,
    // 0 = always play the most-visited move; above 0 samples from the visit
    // counts, so repeated games diverge.
    temperature:  0,
    modelPath:    currentVariant().defaultModel,
    manual:       false,
  };
}
const sideConfig = { 1: _defaultSideConfig(), 2: _defaultSideConfig() };

const MODE_AI_SIDES = { hvh: [], hva: [2], avh: [1], ava: [1, 2] };
function isAgentSide(p) { return MODE_AI_SIDES[gameMode].includes(p); }

// ── Web Workers ──────────────────────────────────────────────────────────────
// One play worker per distinct model path. Each worker owns exactly one ONNX
// session, so two NN agents on different checkpoints can face each other
// without a 'load_model' round trip (re-initialising ~12MB) on every ply. Each
// worker also derives its own `modelFullCanonical` from its `?model=` query, so
// the P2 policy frame stays correct per checkpoint. Both sides default to the
// same path, so the common case is still a single worker.
const _playWorkers = new Map();   // path -> { worker, ready, failed }
let _analysisWorker = null;

function getPlayWorker(path) {
  let entry = _playWorkers.get(path);
  if (!entry) {
    const worker = new Worker(`mcts_worker.js?model=${encodeURIComponent(path)}`);
    entry = { worker, ready: false, failed: false };
    worker.onmessage = e => handlePlayWorkerMessage(e, path);
    worker.onerror   = e => console.error('[worker]', path, e);
    _playWorkers.set(path, entry);
  }
  return entry;
}

function nnReady(path)   { const e = _playWorkers.get(path); return !!(e && e.ready); }
function nnFailed(path)  { const e = _playWorkers.get(path); return !!(e && e.failed); }
function nnLoading(path) { return !nnReady(path) && !nnFailed(path); }

function ensurePlayWorkers() {
  for (const side of SIDES) getPlayWorker(sideConfig[side].modelPath);
}

// A worker per checkpoint the user browses through would pile up fast (the 7x7
// picker alone has 17), so drop any that neither side is configured for.
function releaseUnusedPlayWorkers() {
  const keep = new Set(SIDES.map(s => sideConfig[s].modelPath));
  for (const [path, entry] of [..._playWorkers]) {
    if (keep.has(path)) continue;
    entry.worker.terminate();
    _playWorkers.delete(path);
  }
}

function getAnalysisWorker() {
  if (!_analysisWorker) {
    _analysisWorker = new Worker(`mcts_worker.js?model=${encodeURIComponent(analysisModelPath)}`);
    _analysisWorker.onmessage = handleAnalysisWorkerMessage;
    _analysisWorker.onerror   = e => console.error('[analysis-worker]', e);
  }
  return _analysisWorker;
}

function workerPayload(gs) {
  return {
    boardsize:        gs.boardsize,
    depth:            gs.depth,
    player1pos:       gs.player1pos,
    player2pos:       gs.player2pos,
    hwalls:           Array.from(gs.hwalls),
    vwalls:           Array.from(gs.vwalls),
    walls_p1:         gs.walls_p1,
    walls_p2:         gs.walls_p2,
    walls_initial:    gs.walls_initial,
    hwall_anchors:    [...gs.hwall_anchors],
    vwall_anchors:    [...gs.vwall_anchors],
    position_history: [...gs.position_history.entries()],
  };
}

function handlePlayWorkerMessage(e, path) {
  const d     = e.data;
  const entry = _playWorkers.get(path);

  if (d.type === 'model_loaded') {
    if (entry) { entry.ready = true; entry.failed = false; }
    // Only sides actually configured for this checkpoint — and whose agent the
    // user hasn't picked themselves — snap to the net agent once it's ready.
    for (const side of SIDES) {
      if (sideConfig[side].modelPath === path && !sideConfig[side].manual)
        sideConfig[side].agentId = 'alphazero';
    }
    initAgentList();
    return;
  }

  if (d.type === 'model_failed') {
    console.error('[main] ONNX model failed to load:', path, d.reason);
    if (entry) { entry.ready = false; entry.failed = true; }
    initAgentList();
    return;
  }

  if (d.type === 'progress') {
    // Stale searches keep reporting until the worker notices the cancel, so
    // gate on the generation the same way the 'move' reply does.
    if (d.gen !== _searchGen || !agentThinking) return;
    _thinkDone  = d.done;
    _thinkTotal = d.total;
    renderThinkProgress();
    return;
  }

  if (d.type === 'move') {
    // Gate on the generation BEFORE touching agentThinking: a stale reply must
    // not clear the flag belonging to the search that superseded it.
    if (d.gen !== _searchGen) return;
    if (!gameState || gameState.isFinished()) { agentThinking = false; return; }
    // In AI vs AI, hold the move until moveDelayMs has passed since the search
    // started, so a fast agent stays watchable. agentThinking stays true through
    // the wait, which keeps the board locked. Never applied in hva/avh: the
    // human is waiting on that reply.
    const gen  = d.gen;
    const wait = gameMode === 'ava'
      ? Math.max(0, moveDelayMs - (performance.now() - _thinkStart))
      : 0;
    clearAdvance();
    _advanceTimer = setTimeout(() => {
      _advanceTimer = null;
      if (gen !== _searchGen) return;
      agentThinking = false;
      _thinkDone = 0; _thinkTotal = 0;
      if (!gameState || gameState.isFinished()) return;
      postMove(d.action);
    }, wait);
    return;
  }
}

function handleAnalysisWorkerMessage(e) {
  const d = e.data;

  if (d.type === 'model_loaded') {
    _analysisNnAvailable = true;
    _analysisNnLoadFailed = false;
    _analysisModelLoading = false;
    if (!_analysisAgentManuallyChanged) analysisAgentId = 'alphazero';
    initAgentList();
    if (document.getElementById('analysis-panel').classList.contains('open')) {
      refreshNnAnalysis();
      if (_analysisMctsAuto && state && !state.is_finished && !agentThinking) runMctsAnalysis();
    }
    return;
  }

  if (d.type === 'model_failed') {
    console.error('[analysis-worker] ONNX model failed to load:', d.reason);
    _analysisNnLoadFailed = true;
    _analysisModelLoading = false;
    initAgentList();
    return;
  }

  if (d.type === 'nn_result') {
    if (d.gen !== _analysisGen) return;
    _analysisNnData = d;
    renderAnalysis();
    document.getElementById('btn-run-mcts').disabled = false;
    return;
  }

  if (d.type === 'nn_unavailable') {
    if (d.gen !== _analysisGen) return;
    _analysisNnData = null;
    renderAnalysis();
    return;
  }

  if (d.type === 'progress') {
    if (d.gen !== _analysisGen || !_analysisMctsRunning) return;
    _analysisDone  = d.done;
    _analysisTotal = d.total;
    renderAnalysisProgress();
    return;
  }

  if (d.type === 'mcts_result') {
    if (d.gen !== _analysisGen) return;
    if (d.nn) _analysisNnData = d.nn;
    _analysisMctsData = d;
    _analysisMctsRunning = false;
    document.getElementById('btn-run-mcts').classList.remove('running');
    renderAnalysisProgress();
    renderAnalysis();
    return;
  }
}

// ── Coordinate helpers ────────────────────────────────────────────────────────
// boardFlipped is a pure DISPLAY preference (never touches game state/logic):
// when on, row 0 (Player 1's home row) renders at the top instead of the
// bottom, which is what you want when playing as Player 2 and prefer your
// own pawn to start at the bottom of the screen. Both helpers below are
// self-inverse (applying them twice returns the input), so the same
// function serves as both the forward (game -> screen) and inverse
// (screen -> game) row mapping.
let boardFlipped = false;
function flipRow(gy) {
  return boardFlipped ? gy : (state.boardsize - 1 - gy);
}
function flipWallRow(ay) {
  return boardFlipped ? ay : (state.boardsize - 2 - ay);
}
function cellOrigin(gx, gy) {
  return { x: gx * STEP, y: flipRow(gy) * STEP };
}
function canvasSize() { return state.boardsize * STEP - GAP; }

function pixelToCell(px, py) {
  const col    = Math.floor(px / STEP);
  const row    = Math.floor(py / STEP);
  const localX = px - col * STEP;
  const localY = py - row * STEP;
  if (localX >= CELL || localY >= CELL) return null;
  const gy = flipRow(row);
  if (col < 0 || col >= state.boardsize || gy < 0 || gy >= state.boardsize) return null;
  return { gx: col, gy };
}

function computeLanding() {
  // Highlights mean "you can click here", so only for a side the human controls.
  if (!state || state.is_finished || isAgentTurn()) return [];
  const [cx, cy] = state.current_player === 1 ? state.player1pos : state.player2pos;
  const [ox, oy] = state.current_player === 1 ? state.player2pos : state.player1pos;
  return state.legal_pawn_moves.map(([dx, dy]) => {
    let gx, gy;
    if (dx === 0 || dy === 0) {
      gx = (cx + dx === ox && cy + dy === oy) ? cx + 2 * dx : cx + dx;
      gy = (cx + dx === ox && cy + dy === oy) ? cy + 2 * dy : cy + dy;
    } else { gx = cx + dx; gy = cy + dy; }
    return { gx, gy, dir: [dx, dy] };
  });
}

function computeLegalWallSet() {
  if (!state || state.is_finished) return new Set();
  return new Set(state.legal_wall_moves.map(w => `${w.orientation}:${w.x}:${w.y}`));
}

function pixelToWall(px, py) {
  if (!state) return null;
  const bs   = state.boardsize;
  const modX = ((px % STEP) + STEP) % STEP;
  const modY = ((py % STEP) + STEP) % STEP;
  const inHGap = modY >= CELL;
  const inVGap = modX >= CELL;
  if (!inHGap && !inVGap) return null;
  const gapRowIdx = Math.floor((py - CELL) / STEP);
  const gapColIdx = Math.floor((px - CELL) / STEP);
  let hCand = null, vCand = null;
  if (inHGap) {
    const ay = flipWallRow(gapRowIdx);
    if (ay >= 0 && ay <= bs - 2) {
      const ax = Math.max(0, Math.min(bs - 2, Math.round((px - CELL - GAP / 2) / STEP)));
      if (px >= ax * STEP && px <= ax * STEP + 2 * CELL + GAP)
        hCand = { ax, ay, orientation: 'h' };
    }
  }
  if (inVGap) {
    const ax = gapColIdx;
    if (ax >= 0 && ax <= bs - 2) {
      const ay = Math.max(0, Math.min(bs - 2,
        Math.round(flipWallRow((py - CELL - GAP / 2) / STEP))));
      const wallTop = flipWallRow(ay) * STEP;
      if (py >= wallTop && py <= wallTop + 2 * CELL + GAP)
        vCand = { ax, ay, orientation: 'v' };
    }
  }
  if (!hCand && !vCand) return null;
  if (hCand && !vCand) return hCand;
  if (!hCand && vCand) return vCand;
  const hCenterY = flipWallRow(hCand.ay) * STEP + CELL + GAP / 2;
  const vCenterX = vCand.ax * STEP + CELL + GAP / 2;
  return Math.abs(px - vCenterX) < Math.abs(py - hCenterY) - GAP / 2 ? vCand : hCand;
}

// ── roundRect polyfill ────────────────────────────────────────────────────────
function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); }
  else {
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y,     x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x,     y + h, r);
    ctx.arcTo(x,     y + h, x,     y,     r);
    ctx.arcTo(x,     y,     x + w, y,     r);
    ctx.closePath();
  }
}

// ── Draw ──────────────────────────────────────────────────────────────────────
function draw(ctx) {
  const W = canvasSize() + LABEL;
  const H = canvasSize() + LABEL;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#1e1b18';
  ctx.fillRect(0, 0, W, H);
  ctx.save();
  ctx.translate(LABEL, 0);
  drawCells(ctx);
  drawLastMoveCell(ctx);   // under the walls
  drawWalls(ctx);
  drawLastMoveWall(ctx);   // over the wall it highlights
  drawWallPreview(ctx);
  drawAnalysisHover(ctx);
  drawPawns(ctx);
  drawGameOver(ctx);
  ctx.restore();
  drawLabels(ctx);
}

// Result of a finished game, over the board itself. Scrubbing back to an
// earlier position clears it, since that position isn't terminal.
function drawGameOver(ctx) {
  if (!state.is_finished) return;
  const size = canvasSize();
  // Fade in with the winning move rather than slamming on mid-slide.
  ctx.globalAlpha = _anim ? animProgress() : 1;

  ctx.fillStyle = 'rgba(18,16,14,0.72)';
  ctx.fillRect(0, 0, size, size);

  // Redraw the pawns at full strength on top of the scrim, so the final
  // position stays readable underneath the verdict.
  drawPawns(ctx);

  const win  = state.winner;
  const mid  = size / 2;
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  ctx.shadowColor  = 'rgba(0,0,0,0.85)';
  ctx.shadowBlur   = 12;
  ctx.fillStyle    = win === 1 ? CLR.pawn1 : win === 2 ? CLR.pawn2 : '#d0c8bc';
  ctx.font         = `700 ${Math.round(size * 0.082)}px 'Segoe UI', system-ui, sans-serif`;
  ctx.fillText(win ? `Player ${win} wins` : 'Draw', mid, mid - size * 0.035);

  ctx.fillStyle = 'rgba(230,225,218,0.82)';
  ctx.font      = `500 ${Math.round(size * 0.032)}px 'Segoe UI', system-ui, sans-serif`;
  // Name the agent only when there is one — for a human side the headline
  // already said "Player N".
  const who = win && isAgentSide(win) ? `${agentName(win)} · ` : '';
  const sub = win ? `${who}${state.depth} plies`
                  : `Max depth reached · ${state.depth} plies`;
  ctx.fillText(sub, mid, mid + size * 0.042);

  ctx.shadowColor = 'transparent';
  ctx.shadowBlur  = 0;
  ctx.globalAlpha = 1;
}

// ── Last move ────────────────────────────────────────────────────────────────
// The move that produced the position on screen, plus the position it came from
// (needed for a pawn's origin cell).
function lastMovePlayed() {
  if (viewIdx <= 0 || !moves[viewIdx - 1]) return null;
  return { action: moves[viewIdx - 1].action, prev: timeline[viewIdx - 1] };
}

function drawLastMoveCell(ctx) {
  const lm = lastMovePlayed();
  if (!lm || lm.action.type !== 'pawn') return;
  const mover = lm.prev.getCurrentPlayer();
  const from  = mover === 1 ? lm.prev.player1pos : lm.prev.player2pos;
  const to    = mover === 1 ? state.player1pos   : state.player2pos;
  const tint  = mover === 1 ? CLR.lastFrom1 : CLR.lastFrom2;
  const ring  = mover === 1 ? CLR.lastRing1 : CLR.lastRing2;

  const o = cellOrigin(from[0], from[1]);
  rrect(ctx, o.x, o.y, CELL, CELL, CELL_R);
  ctx.fillStyle = tint;
  ctx.fill();

  // Inset so the ring clears the pawn sitting in the middle of the cell.
  const d = cellOrigin(to[0], to[1]);
  rrect(ctx, d.x + 2.5, d.y + 2.5, CELL - 5, CELL - 5, CELL_R);
  ctx.strokeStyle = ring;
  ctx.lineWidth   = 2;
  ctx.stroke();
}

function drawLastMoveWall(ctx) {
  const lm = lastMovePlayed();
  if (!lm || lm.action.type !== 'wall') return;
  const { x, y, orientation } = lm.action;
  ctx.fillStyle   = CLR.lastWall;
  ctx.shadowColor = CLR.lastWallGlow;
  ctx.shadowBlur  = 12;
  // A wall fades in over its first frames, then stays highlighted.
  ctx.globalAlpha = _anim && _anim.kind === 'wall' ? animProgress() : 1;
  drawOneWall(ctx, x, y, orientation);
  ctx.globalAlpha = 1;
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur  = 0;
}

// ── Move animation ───────────────────────────────────────────────────────────
function animProgress() {
  if (!_anim) return 1;
  const t = (performance.now() - _anim.t0) / ANIM_MS;
  return t >= 1 ? 1 : 1 - Math.pow(1 - t, 3);   // easeOutCubic
}

// Where to actually paint a pawn this frame: its real cell, unless it's the one
// mid-slide.
function animPawnPos(player, pos) {
  if (!_anim || _anim.kind !== 'pawn' || _anim.player !== player) return pos;
  const p = animProgress();
  return [
    _anim.from[0] + (_anim.to[0] - _anim.from[0]) * p,
    _anim.from[1] + (_anim.to[1] - _anim.from[1]) * p,
  ];
}

function animTick() {
  _animRAF = null;
  if (!_anim || !state) return;
  const done = animProgress() >= 1;
  draw(document.getElementById('board').getContext('2d'));
  if (done) { _anim = null; return; }
  _animRAF = requestAnimationFrame(animTick);
}

// Called from applyState with the position we were showing a moment ago. Only a
// forward single-ply transition animates; timeline jumps and new games cut.
function startMoveAnim(prev, next) {
  _anim = null;
  if (_animRAF !== null) { cancelAnimationFrame(_animRAF); _animRAF = null; }
  if (!prev || next.boardsize !== prev.boardsize || next.depth !== prev.depth + 1) return;
  const mover = prev.current_player;
  const before = mover === 1 ? prev.player1pos : prev.player2pos;
  const after  = mover === 1 ? next.player1pos : next.player2pos;
  if (before[0] !== after[0] || before[1] !== after[1])
    _anim = { kind: 'pawn', player: mover, from: before, to: after, t0: performance.now() };
  else
    _anim = { kind: 'wall', t0: performance.now() };
  _animRAF = requestAnimationFrame(animTick);
}

function drawLabels(ctx) {
  const bs = state.boardsize;
  ctx.fillStyle = 'rgba(200,180,160,0.5)';
  ctx.font = `${Math.round(LABEL * 0.6)}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let gy = 0; gy < bs; gy++) {
    ctx.fillText(String(gy), LABEL / 2, flipRow(gy) * STEP + CELL / 2);
  }
  for (let gx = 0; gx < bs; gx++) {
    ctx.fillText(String(gx), LABEL + gx * STEP + CELL / 2, canvasSize() + LABEL / 2);
  }
}

function drawCells(ctx) {
  const bs       = state.boardsize;
  const legalSet = new Set(landing.map(l => `${l.gx},${l.gy}`));
  for (let gy = 0; gy < bs; gy++) {
    for (let gx = 0; gx < bs; gx++) {
      const { x, y }  = cellOrigin(gx, gy);
      const isLegal    = legalSet.has(`${gx},${gy}`);
      const fill       = gy === bs - 1 ? CLR.cellGoal1 : gy === 0 ? CLR.cellGoal2 : CLR.cell;
      rrect(ctx, x, y, CELL, CELL, CELL_R);
      ctx.fillStyle = fill; ctx.fill();
      if (isLegal) {
        ctx.fillStyle = 'rgba(255,255,255,0.18)'; ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.45)'; ctx.lineWidth = 1.5; ctx.stroke();
      }
      if (isLegal && hoverCell && hoverCell.gx === gx && hoverCell.gy === gy) {
        rrect(ctx, x, y, CELL, CELL, CELL_R);
        ctx.fillStyle = 'rgba(255,255,255,0.22)'; ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.65)'; ctx.lineWidth = 2; ctx.stroke();
      }
    }
  }
}

function drawOneWall(ctx, ax, ay, orientation) {
  const bs = state.boardsize;
  if (orientation === 'h') {
    rrect(ctx, ax * STEP, flipWallRow(ay) * STEP + CELL, 2 * CELL + GAP, GAP, 3);
  } else {
    rrect(ctx, ax * STEP + CELL, flipWallRow(ay) * STEP, GAP, 2 * CELL + GAP, 3);
  }
  ctx.fill();
}

function drawWalls(ctx) {
  ctx.shadowBlur = 6;
  ctx.fillStyle   = CLR.wall;
  ctx.shadowColor = CLR.wallGlow;
  for (const [ax, ay] of state.hwall_anchors) drawOneWall(ctx, ax, ay, 'h');
  for (const [ax, ay] of state.vwall_anchors) drawOneWall(ctx, ax, ay, 'v');
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur  = 0;
}

function drawWallPreview(ctx) {
  if (!hoverWall || state.is_finished || isAgentTurn()) return;
  const hasWalls = state.current_player === 1 ? state.walls_p1 > 0 : state.walls_p2 > 0;
  if (!hasWalls) return;
  const key   = `${hoverWall.orientation}:${hoverWall.ax}:${hoverWall.ay}`;
  const legal = legalWallSet.has(key);
  ctx.fillStyle   = legal ? CLR.wallHoverOk  : CLR.wallHoverBad;
  ctx.shadowColor = legal ? CLR.wallHoverOkGlow : CLR.wallHoverBadGlow;
  ctx.shadowBlur  = 10;
  drawOneWall(ctx, hoverWall.ax, hoverWall.ay, hoverWall.orientation);
  ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0;
}

function drawPawns(ctx) {
  // Fractional cell coordinates mid-slide; cellOrigin interpolates.
  const p1 = animPawnPos(1, state.player1pos);
  const p2 = animPawnPos(2, state.player2pos);
  // Sliding pawn on top, so it stays visible while jumping over the other one.
  const moverFirst = _anim && _anim.kind === 'pawn' && _anim.player === 1;
  if (moverFirst) {
    drawOnePawn(ctx, p2[0], p2[1], CLR.pawn2, String(state.walls_p2));
    drawOnePawn(ctx, p1[0], p1[1], CLR.pawn1, String(state.walls_p1));
  } else {
    drawOnePawn(ctx, p1[0], p1[1], CLR.pawn1, String(state.walls_p1));
    drawOnePawn(ctx, p2[0], p2[1], CLR.pawn2, String(state.walls_p2));
  }
}

function drawOnePawn(ctx, gx, gy, color, label) {
  const { x, y } = cellOrigin(gx, gy);
  const cx = x + CELL / 2, cy = y + CELL / 2, r = CELL * 0.30;
  ctx.shadowColor = 'rgba(0,0,0,0.6)'; ctx.shadowBlur = 14; ctx.shadowOffsetY = 4;
  const grad = ctx.createRadialGradient(cx - r * 0.28, cy - r * 0.3, r * 0.05, cx, cy, r);
  grad.addColorStop(0, lighten(color, 60)); grad.addColorStop(0.6, color); grad.addColorStop(1, darken(color, 40));
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.fillStyle = grad; ctx.fill();
  ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
  ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.lineWidth = 1.5; ctx.stroke();
  const shine = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.35, 0, cx - r * 0.2, cy - r * 0.25, r * 0.55);
  shine.addColorStop(0, 'rgba(255,255,255,0.28)'); shine.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.fillStyle = shine; ctx.fill();
  ctx.shadowColor = 'rgba(0,0,0,0.6)'; ctx.shadowBlur = 4; ctx.shadowOffsetY = 1;
  ctx.fillStyle = '#fff';
  ctx.font = `700 ${Math.round(CELL * 0.26)}px 'Segoe UI', system-ui, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, cx, cy);
  ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
}

function lighten(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  return `rgb(${Math.min(255,(n>>16)+amt)},${Math.min(255,((n>>8)&0xff)+amt)},${Math.min(255,(n&0xff)+amt)})`;
}
function darken(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  return `rgb(${Math.max(0,(n>>16)-amt)},${Math.max(0,((n>>8)&0xff)-amt)},${Math.max(0,(n&0xff)-amt)})`;
}

function agentDetail(side) {
  const cfg  = sideConfig[side];
  const name = agentName(side);
  if (cfg.agentId === 'minimax') return `${name} · depth ${cfg.minimaxDepth}`;
  if (agentThinking && _thinkingSide === side && _thinkTotal > 0)
    return `${name} · ${_thinkDone} / ${_thinkTotal} simulations`;
  return `${name} · ${cfg.numSims} simulations`;
}

// The side whose search the status card is currently reporting on.
function reportingSide() {
  return _thinkingSide || (state ? state.current_player : 1);
}

// The search we're waiting on, as a bar in the Players card. Kept separate from
// updateSidebar() so a progress tick only touches these two nodes. Only the fill
// changes; the track itself always stays in the layout (see .sim-progress).
function renderThinkProgress() {
  const track = document.getElementById('think-progress');
  const fill  = document.getElementById('think-progress-fill');
  const show  = agentThinking && _thinkTotal > 0;
  if (!show) { fill.style.width = '0%'; track.title = ''; return; }
  fill.style.width = `${Math.min(100, 100 * _thinkDone / _thinkTotal)}%`;
  track.title = agentDetail(reportingSide());
}

// Same, for the analysis panel's MCTS run — this bar replaces what used to be
// a "Running…" label on the button.
function renderAnalysisProgress() {
  const show = _analysisMctsRunning && _analysisTotal > 0;
  document.getElementById('analysis-progress').firstElementChild.style.width =
    show ? `${Math.min(100, 100 * _analysisDone / _analysisTotal)}%` : '0%';
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function updateSidebar() {
  document.getElementById('walls-p1').textContent = `${state.walls_p1} wall${state.walls_p1 !== 1 ? 's' : ''}`;
  document.getElementById('walls-p2').textContent = `${state.walls_p2} wall${state.walls_p2 !== 1 ? 's' : ''}`;
  document.getElementById('name-p1').textContent = isAgentSide(1) ? agentName(1) : 'Player 1';
  document.getElementById('name-p2').textContent = isAgentSide(2) ? agentName(2) : 'Player 2';
  // Winner is shown on the board, so no row is "active" once the game is over.
  const active = state.is_finished ? 0 : state.current_player;
  const pe1 = document.getElementById('pe-p1');
  const pe2 = document.getElementById('pe-p2');
  pe1.className = 'player-entry' + (active === 1 ? ' active-p1' : '')
                + (agentThinking && reportingSide() === 1 ? ' thinking' : '');
  pe2.className = 'player-entry' + (active === 2 ? ' active-p2' : '')
                + (agentThinking && reportingSide() === 2 ? ' thinking' : '');
  renderThinkProgress();
  renderTimeline();
}

// ── Input ─────────────────────────────────────────────────────────────────────
function setupInput(canvas) {
  canvas.addEventListener('click', e => {
    if (!state || state.is_finished || agentThinking || isAgentTurn()) return;
    const { px, py } = canvasPos(e, canvas);
    const cell = pixelToCell(px, py);
    if (cell) {
      const match = landing.find(l => l.gx === cell.gx && l.gy === cell.gy);
      if (match) { postMove({ type: 'pawn', direction: match.dir }); return; }
    }
    const wall = pixelToWall(px, py);
    if (wall) {
      const key      = `${wall.orientation}:${wall.ax}:${wall.ay}`;
      const hasWalls = state.current_player === 1 ? state.walls_p1 > 0 : state.walls_p2 > 0;
      if (hasWalls && legalWallSet.has(key))
        postMove({ type: 'wall', x: wall.ax, y: wall.ay, orientation: wall.orientation });
    }
  });

  canvas.addEventListener('mousemove', e => {
    if (!state || state.is_finished) return;
    const { px, py } = canvasPos(e, canvas);
    if (agentThinking || isAgentTurn()) {
      if (hoverWall || hoverCell) { hoverWall = null; hoverCell = null; draw(canvas.getContext('2d')); }
      clearTimeout(_hoverWallTimer); _hoverWallTimer = null; _pendingWall = null;
      canvas.style.cursor = 'default'; return;
    }
    const cell       = pixelToCell(px, py);
    const onLegalCell = cell && landing.some(l => l.gx === cell.gx && l.gy === cell.gy);
    const newHoverCell = onLegalCell ? cell : null;
    const wall      = pixelToWall(px, py);
    const hasWalls  = state.current_player === 1 ? state.walls_p1 > 0 : state.walls_p2 > 0;
    const newHover  = (wall && hasWalls) ? wall : null;
    const newKey    = newHover ? `${newHover.orientation}:${newHover.ax}:${newHover.ay}` : null;
    const pendingKey = _pendingWall ? `${_pendingWall.orientation}:${_pendingWall.ax}:${_pendingWall.ay}` : null;
    let needRedraw  = false;
    if (newKey !== pendingKey) {
      clearTimeout(_hoverWallTimer); _hoverWallTimer = null;
      _pendingWall = newHover;
      if (hoverWall) { hoverWall = null; needRedraw = true; }
      if (newHover) {
        _hoverWallTimer = setTimeout(() => {
          _hoverWallTimer = null; hoverWall = _pendingWall;
          if (hoverWall) draw(document.getElementById('board').getContext('2d'));
        }, 50);
      }
    }
    const newCellKey = newHoverCell ? `${newHoverCell.gx},${newHoverCell.gy}` : null;
    const curCellKey = hoverCell    ? `${hoverCell.gx},${hoverCell.gy}` : null;
    if (newCellKey !== curCellKey) { hoverCell = newHoverCell; needRedraw = true; }
    if (needRedraw) draw(canvas.getContext('2d'));
    canvas.style.cursor = (onLegalCell || (wall && hasWalls)) ? 'pointer' : 'default';
  });

  canvas.addEventListener('mouseleave', () => {
    clearTimeout(_hoverWallTimer); _hoverWallTimer = null; _pendingWall = null;
    if (hoverWall || hoverCell) {
      hoverWall = null; hoverCell = null;
      draw(document.getElementById('board').getContext('2d'));
    }
  });
}

function canvasPos(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  return { px: e.clientX - rect.left - LABEL, py: e.clientY - rect.top };
}

// ── Game actions ──────────────────────────────────────────────────────────────
function bodyToAction(body) {
  if (body.type === 'pawn') return { type: 'pawn', direction: body.direction };
  return { type: 'wall', x: body.x, y: body.y, orientation: body.orientation };
}

// Is the displayed position the live one (the end of the timeline)?
function atTip() { return viewIdx === timeline.length - 1; }

function clearAdvance() {
  if (_advanceTimer !== null) { clearTimeout(_advanceTimer); _advanceTimer = null; }
}

// Abandon whatever search we're waiting on. The worker keeps computing until it
// notices its generation is stale; its reply is discarded on arrival.
function cancelSearch() {
  _searchGen++;
  agentThinking = false;
  _thinkDone = 0; _thinkTotal = 0;
  clearAdvance();
}

function postMove(body) {
  if (!gameState || gameState.isFinished()) return;
  const action = bodyToAction(body);
  const label  = actionLabel(gameState, action);
  // Moving from a rewound position branches the game: whatever used to follow
  // is no longer reachable from here, so drop it.
  if (!atTip()) { timeline.length = viewIdx + 1; moves.length = viewIdx; }
  moves.push({ action, label });
  gameState = gameState.next(action);
  timeline.push(gameState);
  viewIdx = timeline.length - 1;
  if (_stepOnce) { _stepOnce = false; paused = true; }
  else           { paused = false; }
  applyState(gameState.serialize());
}

function initGame(boardsize, walls) {
  const v = currentVariant();
  if (boardsize === undefined) boardsize = v.boardsize;
  if (walls === undefined) walls = v.walls;
  gameState = new State({ boardsize, walls_p1: walls, walls_p2: walls });
  timeline  = [gameState];
  moves     = [];
  viewIdx   = 0;
  applyState(gameState.serialize());
}

function resetGame() {
  cancelSearch();
  _stepOnce = false;
  // `paused` is deliberately preserved: hitting New Game during a running
  // AI vs AI game should start the next one running too.
  initGame();
}

// ── Timeline navigation ───────────────────────────────────────────────────────
// Mechanical jump. Used by both user navigation and the replay ticker, so it
// must NOT touch `paused` — the ticker relies on it staying false.
function gotoIndex(i) {
  cancelSearch();
  viewIdx   = Math.max(0, Math.min(i, timeline.length - 1));
  gameState = timeline[viewIdx];
  applyState(gameState.serialize());
}

// User navigation (arrow buttons, scrubber, keyboard). Landing behind the live
// position parks the game there; landing exactly on it resumes play.
function seekTo(i) {
  const t           = Math.max(0, Math.min(i, timeline.length - 1));
  const shouldPause = t < timeline.length - 1;
  _stepOnce = false;
  if (t === viewIdx) {
    if (shouldPause === paused) { renderTimeline(); return; }
    paused = shouldPause;
    if (paused) cancelSearch(); else maybeAdvance();
    updateSidebar();
    return;
  }
  paused = shouldPause;
  gotoIndex(t);
}

function stepBack() { seekTo(viewIdx - 1); }

function stepForward() {
  if (!atTip()) { seekTo(viewIdx + 1); return; }
  // At the live position there is nothing recorded to step to, so ▶ means "let
  // this side make one move, then stop again" — the natural single-step for a
  // paused AI vs AI game. A game that is already running needs no nudge.
  if (!canStepForwardLive()) return;
  _stepOnce = true;
  maybeAdvance();
  updateSidebar();
}

// ▶ only does something at the live position if an agent is sitting there
// waiting for permission to move.
function canStepForwardLive() {
  return paused && !state.is_finished && isAgentSide(state.current_player);
}

function togglePause() {
  _stepOnce = false;
  paused    = !paused;
  if (paused) cancelSearch(); else maybeAdvance();
  updateSidebar();
}

function renderTimeline() {
  const last = timeline.length - 1;
  const sl   = document.getElementById('timeline-slider');
  sl.max      = String(last);
  sl.value    = String(viewIdx);
  sl.disabled = last === 0;
  document.getElementById('btn-step-back').disabled = viewIdx === 0;
  document.getElementById('btn-step-fwd').disabled  = atTip() && !canStepForwardLive();
  const play = document.getElementById('btn-play');
  play.innerHTML = paused ? '&#x25B6; Play' : '&#x23F8; Pause';
  play.classList.toggle('toggle-active', !paused);
  const move = viewIdx > 0 ? ` · ${moves[viewIdx - 1].label}` : '';
  const lbl  = document.getElementById('timeline-label');
  lbl.textContent = `${viewIdx} / ${last}${move}`;   // viewIdx IS the ply
  lbl.classList.toggle('reviewing', !atTip());
  lbl.title = atTip() ? `Ply ${viewIdx} of ${last}`
                      : `Reviewing ply ${viewIdx} of ${last} — press ▶ to return to the live game`;
}

// ── Autoplay ──────────────────────────────────────────────────────────────────
// The single place that decides what happens next after a state change: replay
// the next recorded move when parked behind the live position, or kick off the
// agent's search when sitting on it. One timer, always cleared first.
function maybeAdvance() {
  clearAdvance();
  if (paused && !_stepOnce) return;
  if (!atTip()) {
    // Walk forward through the recorded game rather than re-searching.
    _advanceTimer = setTimeout(() => {
      _advanceTimer = null;
      gotoIndex(viewIdx + 1);
    }, Math.max(80, moveDelayMs));
    return;
  }
  if (state.is_finished || !isAgentSide(state.current_player)) { _stepOnce = false; return; }
  agentThinking = true;                      // set before the sidebar renders "Thinking…"
  _thinkingSide = state.current_player;      // ...and before it names the agent
  _advanceTimer = setTimeout(triggerAgentMove, 80);
}

function applyState(s) {
  const prev   = state;
  state        = s;
  landing      = computeLanding();
  legalWallSet = computeLegalWallSet();
  hoverWall    = null;
  const cvs    = document.getElementById('board');
  cvs.width    = canvasSize() + LABEL;
  cvs.height   = canvasSize() + LABEL;
  startMoveAnim(prev, s);
  draw(cvs.getContext('2d'));
  maybeAdvance();
  updateSidebar();
  if (document.getElementById('nn-panel').classList.contains('open')) openNNPanel();
  if (document.getElementById('analysis-panel').classList.contains('open')) {
    cancelAnalysisWork();
    _analysisMctsData = null;
    _analysisNnData = null;
    refreshNnAnalysis();
    // Only a settled position: don't compete with a play search for CPU, and
    // don't fire once per position while replaying history.
    if (_analysisMctsAuto && !state.is_finished && !agentThinking && (paused || atTip()))
      runMctsAnalysis();
  }
}

// ── Agent ─────────────────────────────────────────────────────────────────────
function isAgentTurn() {
  if (!state || state.is_finished) return false;
  return isAgentSide(state.current_player);
}

function triggerAgentMove() {
  _advanceTimer = null;
  if (!state || state.is_finished || !agentThinking) return;
  if (!atTip()) { agentThinking = false; return; }   // never search a hidden position
  const side = state.current_player;
  const cfg  = sideConfig[side];
  const gen  = ++_searchGen;
  _thinkingSide = side;
  _thinkStart   = performance.now();
  _thinkDone    = 0;
  _thinkTotal   = cfg.agentId === 'minimax' ? 0 : cfg.numSims;
  updateSidebar();
  getPlayWorker(cfg.modelPath).worker.postMessage({
    ...workerPayload(gameState),
    type:         'think',
    numSims:      cfg.numSims,
    minimaxDepth: cfg.minimaxDepth,
    temperature:  cfg.temperature,
    agentId:      cfg.agentId,
    gen,
  });
}

// ── Agent selector ────────────────────────────────────────────────────────────
const AGENT_DEFS = [
  { id: 'alphazero',    name: 'SigmaQuoridor',  description: 'MCTS guided by a trained neural network' },
  { id: 'mcts_rollout', name: 'MCTS',           description: 'Pure MCTS with random rollouts' },
  { id: 'minimax',      name: 'Minimax',        description: 'Alpha-beta search with adjustable depth' },
];


// Labels are kept short enough to fit the sidebar select without being
// ellipsised — the full model path is also set as the select's tooltip.
const MODEL_LIST_7X7 = [
  { label: 'extended (latest)', path: './models/supervised_extended.onnx' },
  { label: 'supervised',        path: './models/supervised.onnx' },
  { label: 'Cycle 141', path: './models/checkpoints/cycle_0141.onnx' },
  { label: 'Cycle 131', path: './models/checkpoints/cycle_0131.onnx' },
  { label: 'Cycle 121', path: './models/checkpoints/cycle_0121.onnx' },
  { label: 'Cycle 111', path: './models/checkpoints/cycle_0111.onnx' },
  { label: 'Cycle 101', path: './models/checkpoints/cycle_0101.onnx' },
  { label: 'Cycle 91',  path: './models/checkpoints/cycle_0091.onnx' },
  { label: 'Cycle 81',  path: './models/checkpoints/cycle_0081.onnx' },
  { label: 'Cycle 71',  path: './models/checkpoints/cycle_0071.onnx' },
  { label: 'Cycle 61',  path: './models/checkpoints/cycle_0061.onnx' },
  { label: 'Cycle 51',  path: './models/checkpoints/cycle_0051.onnx' },
  { label: 'Cycle 41',  path: './models/checkpoints/cycle_0041.onnx' },
  { label: 'Cycle 31',  path: './models/checkpoints/cycle_0031.onnx' },
  { label: 'Cycle 21',  path: './models/checkpoints/cycle_0021.onnx' },
  { label: 'Cycle 11',  path: './models/checkpoints/cycle_0011.onnx' },
  { label: 'Cycle 1',   path: './models/checkpoints/cycle_0001.onnx' },
];

// NOTE: only "full-canonical" nets (trained after the 2026-07-19 policy-frame
// fix) belong here — the JS now un-flips P2's policy (see vertPolicyPermutation
// in game.js). The pre-fix legacy checkpoints (≤ cycle 339) used the opposite
// (real-frame) P2 convention, so serving them through the fixed JS would give a
// vertically-scrambled P2 policy. They've been removed from this picker.
// "best" is the from-scratch lineage's cycle 321, which beat the old heads
// cycle 56 head-to-head (61.1% over 500 games) and took rank 1 in the v7
// tournament; heads cycle 56 is kept alongside it as the previous best.
const MODEL_LIST_9X9 = [
  { label: 'best · scratch 321',  path: './models_9x9/best.onnx' },
  { label: 'previous · heads 56', path: './models_9x9/checkpoints/cycle_0056.onnx' },
];

function currentModelList() {
  return currentBoardVariant === '9x9' ? MODEL_LIST_9X9 : MODEL_LIST_7X7;
}

function agentDefById(id) { return AGENT_DEFS.find(a => a.id === id); }

// `modelPath` decides availability: only the net agent can be unavailable, and
// only while ITS checkpoint is still loading (or failed) — the other two need no
// model at all. Availability is therefore per-side, not global.
function _buildAgentSelect(selEl, currentId, setId, modelPath) {
  const list = AGENT_DEFS.map(a => ({
    ...a,
    available: a.id === 'alphazero' ? nnReady(modelPath) : true,
  }));
  selEl.innerHTML = '';
  for (const a of list) {
    const opt       = document.createElement('option');
    opt.value       = a.id;
    // 'alphazero' availability tracks async ONNX model loading (loading… vs
    // failed); every other unavailable agent is unavailable for a static
    // reason — always say so.
    const suffix    = a.available ? ''
      : a.id !== 'alphazero' ? ' (unavailable)'
      : (nnFailed(modelPath) ? ' (unavailable)' : ' (loading…)');
    opt.textContent = a.name + suffix;
    opt.disabled    = !a.available;
    if (a.id === currentId) opt.selected = true;
    selEl.appendChild(opt);
  }
  const current = list.find(a => a.id === currentId && a.available);
  if (!current) {
    const first = list.find(a => a.available);
    if (first) { setId(first.id); selEl.value = first.id; }
  }
}

function _buildAnalysisAgentSelect() {
  const sel = document.getElementById('analysis-agent-select');
  sel.innerHTML = '';
  const available = _analysisNnAvailable;
  const suffix = available ? '' : (_analysisNnLoadFailed ? ' (unavailable)' : ' (loading…)');
  const opt = document.createElement('option');
  opt.value = 'alphazero';
  opt.textContent = 'SigmaQuoridor' + suffix;
  opt.disabled = !available;
  opt.selected = true;
  sel.appendChild(opt);
}

// One template for both sides, so their controls can't drift apart.
function agentCardHTML(side) {
  return `
    <div class="card-label" style="display:flex;align-items:center;gap:6px">
      <span class="pawn-dot ${side === 1 ? 'blue' : 'red'}"></span>Player ${side} agent
    </div>
    <select id="agent-select-p${side}" class="agent-select"></select>
    <div id="agent-desc-p${side}" style="font-size:11px;color:var(--muted);margin-bottom:8px"></div>
    <div id="model-picker-row-p${side}" style="display:none">
      <span style="font-size:11px;color:var(--muted)">Checkpoint</span>
      <select id="model-select-p${side}" class="agent-select" style="margin-top:4px;margin-bottom:8px"></select>
    </div>
    <div id="minimax-depth-row-p${side}" style="display:none">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span style="font-size:11px;color:var(--muted);flex:1">Depth</span>
        <span id="minimax-depth-display-p${side}" style="font-size:12px;font-weight:600">3</span>
      </div>
      <input id="minimax-depth-slider-p${side}" type="range" min="2" max="8" step="1" value="3"
             style="width:100%;accent-color:#3fb950;cursor:pointer;margin-bottom:8px">
    </div>
    <div id="sims-row-p${side}" style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
      <span style="font-size:11px;color:var(--muted);flex:1">Simulations</span>
      <span id="sims-display-p${side}" style="font-size:12px;font-weight:600">100</span>
    </div>
    <input id="sims-slider-p${side}" type="range" min="0" max="39" step="1" value="26"
           style="width:100%;accent-color:#3fb950;cursor:pointer;margin-bottom:8px">
    <div id="temp-row-p${side}">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span style="font-size:11px;color:var(--muted);flex:1"
              title="0 plays the most-visited move every time; above 0 samples from the visit counts">Temperature</span>
        <span id="temp-display-p${side}" style="font-size:12px;font-weight:600">argmax</span>
      </div>
      <input id="temp-slider-p${side}" type="range" min="0" max="14" step="1" value="0"
             style="width:100%;accent-color:#3fb950;cursor:pointer">
    </div>`;
}

function buildAgentCards() {
  for (const side of SIDES) {
    document.getElementById(`agent-card-p${side}`).innerHTML = agentCardHTML(side);

    document.getElementById(`agent-select-p${side}`).addEventListener('change', e => {
      sideConfig[side].manual  = true;
      sideConfig[side].agentId = e.target.value;
      _updateAgentDesc(side);
      updateAgentRowVisibility(side);
      // Take effect on the move in progress, not the next one.
      cancelSearch();
      maybeAdvance();
      updateSidebar();
    });

    document.getElementById(`sims-slider-p${side}`).addEventListener('input', e => {
      sideConfig[side].numSims = simsFromIndex(Number(e.target.value));
      document.getElementById(`sims-display-p${side}`).textContent = sideConfig[side].numSims;
    });

    document.getElementById(`minimax-depth-slider-p${side}`).addEventListener('input', e => {
      sideConfig[side].minimaxDepth = Number(e.target.value);
      document.getElementById(`minimax-depth-display-p${side}`).textContent = sideConfig[side].minimaxDepth;
    });

    document.getElementById(`temp-slider-p${side}`).addEventListener('input', e => {
      sideConfig[side].temperature = tempFromIndex(Number(e.target.value));
      document.getElementById(`temp-display-p${side}`).textContent =
        tempLabel(sideConfig[side].temperature);
    });

    document.getElementById(`model-select-p${side}`).addEventListener('change', e => {
      sideConfig[side].modelPath = e.target.value;
      cancelSearch();
      getPlayWorker(sideConfig[side].modelPath);   // starts loading in the background
      releaseUnusedPlayWorkers();
      initAgentList();
      maybeAdvance();
      updateSidebar();
    });
  }
}

// Which agent cards are relevant in the current mode. The playback-delay row
// stays put in every mode — replaying the timeline works even in H vs H.
function updateAgentCardVisibility() {
  for (const side of SIDES)
    document.getElementById(`agent-card-p${side}`).style.display = isAgentSide(side) ? '' : 'none';
  // Nothing ever searches in H vs H, so the track would sit empty forever.
  document.getElementById('think-progress').style.display = gameMode === 'hvh' ? 'none' : '';
}

function initAgentList() {
  for (const side of SIDES) {
    _buildAgentSelect(
      document.getElementById(`agent-select-p${side}`),
      sideConfig[side].agentId,
      id => { sideConfig[side].agentId = id; },
      sideConfig[side].modelPath,
    );
    _updateAgentDesc(side);
    buildModelSelect(side);
    updateAgentRowVisibility(side);
  }
  _buildAnalysisAgentSelect();
  _updateAnalysisAgentDesc();
  buildAnalysisModelSelect();
  // The Players card shows the agent's name, and only updateSidebar() writes it.
  // Rebuilding the agent list can change which agent is selected (async ONNX
  // loading flips alphazero's availability, and _buildAgentSelect falls back to
  // the first available agent while it loads), so refresh the card here too or
  // it keeps showing the superseded name until the next move. Guarded because
  // the first call happens before initGame() creates `state`.
  if (state) updateSidebar();
}

function _updateAgentDesc(side) {
  const a = agentDefById(sideConfig[side].agentId);
  document.getElementById(`agent-desc-p${side}`).textContent = a ? a.description : '';
}

function _updateAnalysisAgentDesc() {
  const a = agentDefById(analysisAgentId);
  document.getElementById('analysis-agent-desc').textContent = a ? a.description : '';
}

function buildModelSelect(side) {
  const path = sideConfig[side].modelPath;
  const sel  = document.getElementById(`model-select-p${side}`);
  sel.innerHTML = '';
  for (const m of currentModelList()) {
    const opt = document.createElement('option');
    opt.value = m.path;
    opt.textContent = (m.path === path && nnLoading(path)) ? m.label + ' (loading…)' : m.label;
    opt.title = m.path;
    if (m.path === path) { opt.selected = true; sel.title = m.path; }
    sel.appendChild(opt);
  }
}

function buildAnalysisModelSelect() {
  const sel = document.getElementById('analysis-model-select');
  sel.innerHTML = '';
  sel.disabled = _analysisModelLoading;
  for (const m of currentModelList()) {
    const opt = document.createElement('option');
    opt.value = m.path;
    opt.textContent = (_analysisModelLoading && m.path === analysisModelPath)
      ? m.label + ' (loading…)'
      : m.label;
    opt.title = m.path;
    if (m.path === analysisModelPath) { opt.selected = true; sel.title = m.path; }
    sel.appendChild(opt);
  }
}

// Checkpoint picker is only meaningful for the net agent; sims vs depth depends
// on which search the agent runs, and temperature needs visit counts to sample
// from — alpha-beta has none, so minimax gets neither.
function updateAgentRowVisibility(side) {
  const id      = sideConfig[side].agentId;
  const isMcts  = id !== 'minimax';
  document.getElementById(`model-picker-row-p${side}`).style.display  = id === 'alphazero' ? '' : 'none';
  document.getElementById(`minimax-depth-row-p${side}`).style.display = id === 'minimax' ? '' : 'none';
  document.getElementById(`sims-row-p${side}`).style.display          = isMcts ? 'flex' : 'none';
  document.getElementById(`sims-slider-p${side}`).style.display       = isMcts ? '' : 'none';
  document.getElementById(`temp-row-p${side}`).style.display          = isMcts ? '' : 'none';
}

// './models_9x9/best.onnx' -> 'best';
// './models_9x9/checkpoints/cycle_0056.onnx' -> 'cycle 56'
function modelShortLabel(path) {
  const base = String(path).split('/').pop().replace(/\.onnx$/, '');
  const m    = base.match(/^cycle_0*(\d+)$/);
  return m ? `cycle ${m[1]}` : base;
}

// True when both sides are the net on DIFFERENT checkpoints — the headline
// AI vs AI matchup, and the one case where the agent name alone identifies
// neither side.
function isCheckpointDuel() {
  return SIDES.every(s => isAgentSide(s) && sideConfig[s].agentId === 'alphazero')
      && sideConfig[1].modelPath !== sideConfig[2].modelPath;
}

function agentName(side) {
  const a = agentDefById(sideConfig[side].agentId);
  const name = a ? a.name : 'AI';
  return isCheckpointDuel() ? `${name} · ${modelShortLabel(sideConfig[side].modelPath)}` : name;
}

// ── NN channel viewer ─────────────────────────────────────────────────────────
function renderChannel(canvas, matrix, showNumbers) {
  const N   = matrix.length;
  const S   = showNumbers ? Math.max(28, Math.floor(200 / N)) : Math.max(4, Math.floor(200 / N));
  const dpr = showNumbers ? (window.devicePixelRatio || 1) : 1;
  const logW = N * S;
  canvas.width  = logW * dpr; canvas.height = logW * dpr;
  canvas.style.width  = logW + 'px'; canvas.style.height = logW + 'px';
  canvas.style.imageRendering = showNumbers ? 'auto' : 'pixelated';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const v = matrix[N - 1 - y][x];
      const c = Math.round(v * 255);
      ctx.fillStyle = `rgb(${c},${c},${c})`;
      ctx.fillRect(x * S, y * S, S, S);
      if (showNumbers) {
        const raw = Math.round(v * showNumbers * 1e4) / 1e4;
        ctx.fillStyle = v > 0.5 ? '#000' : '#fff';
        ctx.font = `bold ${Math.round(S * 0.35)}px monospace`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(String(Math.round(raw)), x * S + S / 2, y * S + S / 2);
      }
    }
  }
}

function openNNPanel() {
  if (!gameState) return;
  const N      = gameState.boardsize;
  const flat   = gameState.toNNInput();  // Float32Array [8*N*N]
  const names  = ['My pawn','Opp. pawn','H-walls','V-walls','My walls left','Opp. walls left','My BFS dist','Opp. BFS dist'];
  const wi     = gameState.walls_initial > 0 ? gameState.walls_initial : 1;
  const md     = N * N - 1;
  const scales = [1, 1, 1, 1, wi, wi, md, md];

  document.getElementById('nn-note').textContent =
    `Player ${gameState.getCurrentPlayer()}'s turn (board shown from their perspective — moving up)`;
  const grid = document.getElementById('nn-grid');
  grid.innerHTML = '';
  for (let ch = 0; ch < 8; ch++) {
    // Convert flat channel to 2D matrix[row][col] for renderChannel
    const matrix = [];
    for (let y = 0; y < N; y++) {
      const row = [];
      for (let x = 0; x < N; x++) row.push(flat[ch * N * N + y * N + x]);
      matrix.push(row);
    }
    const div   = document.createElement('div');   div.className = 'nn-channel';
    const lbl   = document.createElement('div');   lbl.className = 'nn-channel-name';
    lbl.textContent = `ch${ch}: ${names[ch]}`;
    const cvs   = document.createElement('canvas');
    div.appendChild(lbl); div.appendChild(cvs);
    grid.appendChild(div);
    renderChannel(cvs, matrix, ch >= 4 ? scales[ch] : 0);
  }
  document.getElementById('nn-panel').classList.add('open');
  positionNNPanel();
}

// The analysis panel lives in the rail to the left of the board. That rail only
// exists on a wide enough window; when it doesn't, put the panel under the board
// rather than off the edge of the screen.
function positionAnalysisPanel() {
  const panel = document.getElementById('analysis-panel');
  if (!panel.classList.contains('open')) return;
  // #layout spans the board column only (both side panels are out of flow), so
  // its left edge is the room available — and it doesn't move when we restack.
  const room = document.getElementById('layout').getBoundingClientRect().left;
  panel.classList.toggle('stacked', room < 340 + 24 + 16);
}

// Both panels are positioned against each other, and the NN one has to go last
// because it sits below whatever the analysis panel ended up being.
function repositionPanels() {
  positionAnalysisPanel();
  positionNNPanel();
}

// The side panels are absolutely positioned, so #layout's height only covers the
// board column — either of them can hang lower than it. Push the NN panel below
// the lowest of the three so it can never clip a corner of one.
function positionNNPanel() {
  const panel = document.getElementById('nn-panel');
  if (!panel.classList.contains('open')) return;
  const layout = document.getElementById('layout').getBoundingClientRect();
  let bottom   = layout.height;
  for (const id of ['sidebar', 'analysis-panel']) {
    // A hidden panel measures as all-zeros, which loses the max on its own.
    const r = document.getElementById(id).getBoundingClientRect();
    bottom = Math.max(bottom, r.bottom - layout.top);
  }
  panel.style.top = `${Math.round(bottom) + 16}px`;
  // It grows symmetrically from the board's centre, so its room is twice the
  // smaller side gap — past that, scroll it internally rather than letting it
  // overflow the page and reintroduce the sideways shove.
  const centre = layout.left + layout.width / 2;
  const room   = 2 * Math.min(centre - 24, window.innerWidth - centre - 24);
  panel.style.maxWidth = `${Math.max(320, Math.round(room))}px`;
}

// ── Analysis panel ────────────────────────────────────────────────────────────
let analysisNumSims      = 100;
let analysisAgentId      = 'alphazero';
let _analysisAgentManuallyChanged = false;
let _analysisMctsRunning = false;
let _analysisMctsAuto    = false;
let _analysisNnData      = null;
let _analysisMctsData    = null;
let _analysisGen         = 0;

function openAnalysisPanel() {
  document.getElementById('analysis-panel').classList.add('open');
  positionAnalysisPanel();
  _analysisMctsData = null;
  refreshNnAnalysis();
}

function refreshNnAnalysis() {
  if (!gameState || gameState.isFinished()) return;
  if (!_analysisNnAvailable) {
    _analysisNnData = null;
    renderAnalysis();
    return;
  }
  const gen = ++_analysisGen;
  const w   = getAnalysisWorker();
  w.postMessage({
    ...workerPayload(gameState),
    type:    'nn_eval',
    numSims: analysisNumSims,
    agentId: analysisAgentId,
    gen,
  });
}

function renderAnalysis() {
  const nn   = _analysisNnData;
  const mcts = _analysisMctsData;

  const p1Name = isAgentSide(1) ? agentName(1) : 'P1';
  const p2Name = isAgentSide(2) ? agentName(2) : 'P2';

  if (nn) {
    const curP   = nn.current_player;
    const p1Win  = Math.round((curP === 1 ? (nn.value + 1) / 2 : (1 - nn.value) / 2) * 100);
    document.getElementById('analysis-win-bar-p1').style.width = p1Win + '%';
    document.getElementById('analysis-win-p1-label').textContent = `${p1Name}: ${p1Win}%`;
    document.getElementById('analysis-win-p2-label').textContent = `${p2Name}: ${100 - p1Win}%`;
  }

  const mctsValRow = document.getElementById('mcts-value-row');
  if (mcts) {
    const m1Win = Math.round((mcts.current_player === 1 ? (mcts.value+1)/2 : (1-mcts.value)/2) * 100);
    document.getElementById('analysis-mcts-bar-p1').style.width = m1Win + '%';
    document.getElementById('analysis-mcts-p1-label').textContent = `${p1Name}: ${m1Win}%`;
    document.getElementById('analysis-mcts-p2-label').textContent = `${p2Name}: ${100-m1Win}%`;
    mctsValRow.style.display = '';
  } else {
    mctsValRow.style.display = 'none';
  }

  document.getElementById('analysis-moves-title').textContent =
    mcts ? 'Move Probabilities  (green NN↑  red MCTS↓, sorted by MCTS)' : 'NN Policy (top 20)';

  let top;
  if (mcts && nn) {
    const merged = new Map();
    for (const m of (nn.moves || []))   merged.set(m.label, { label: m.label, nnProb: m.prob, mctsProb: 0 });
    for (const m of (mcts.moves || [])) {
      if (merged.has(m.label)) merged.get(m.label).mctsProb = m.prob;
      else merged.set(m.label, { label: m.label, nnProb: 0, mctsProb: m.prob });
    }
    top = [...merged.values()].sort((a, b) => b.mctsProb - a.mctsProb).slice(0, 20);
  } else if (nn) {
    top = (nn.moves || []).slice(0, 20).map(m => ({ label: m.label, nnProb: m.prob, mctsProb: 0 }));
  } else if (mcts) {
    top = (mcts.moves || []).slice(0, 20).map(m => ({ label: m.label, nnProb: 0, mctsProb: m.prob }));
  } else {
    return;
  }

  const maxP      = top.length ? Math.max(mcts ? top[0].mctsProb : top[0].nnProb, 1e-6) : 1;
  const container = document.getElementById('analysis-move-bars');
  container.innerHTML = '';
  const cvs = document.getElementById('board');

  for (const move of top) {
    const isPawn = move.label[0] !== 'H' && move.label[0] !== 'V';
    const nnW    = Math.min(100, move.nnProb   / maxP * 100).toFixed(1);
    const mctsW  = Math.min(100, move.mctsProb / maxP * 100).toFixed(1);
    const row    = document.createElement('div');  row.className = 'move-bar-row';
    const lbl    = document.createElement('span'); lbl.className = 'move-bar-label ' + (isPawn ? 'pawn' : 'wall');
    lbl.textContent = move.label; lbl.title = move.label;
    const track  = document.createElement('div'); track.className = 'move-bar-track';
    const bNN    = document.createElement('div'); bNN.className = 'move-bar-nn'; bNN.style.width = nnW + '%';
    track.appendChild(bNN);
    if (mcts) {
      const bM = document.createElement('div'); bM.className = 'move-bar-mcts'; bM.style.width = mctsW + '%';
      track.appendChild(bM);
    }
    const pct = document.createElement('span'); pct.className = 'move-bar-pct';
    pct.textContent = mcts
      ? `${(move.nnProb*100).toFixed(1)}|${(move.mctsProb*100).toFixed(1)}`
      : `${(move.nnProb*100).toFixed(1)}%`;
    row.appendChild(lbl); row.appendChild(track); row.appendChild(pct);
    row.addEventListener('click', () => playAnalysisMove(move.label));
    const _hov = parseAnalysisHover(move.label);
    if (_hov) {
      row.addEventListener('mouseenter', () => { analysisHoverMove = _hov; draw(cvs.getContext('2d')); });
      row.addEventListener('mouseleave', () => { analysisHoverMove = null; draw(cvs.getContext('2d')); });
    }
    container.appendChild(row);
  }
  // Rows just added/removed change this panel's height, which the NN panel
  // below is positioned against.
  repositionPanels();
}

function drawAnalysisHover(ctx) {
  if (!analysisHoverMove || !state) return;
  if (analysisHoverMove.type === 'wall') {
    const { ax, ay, orientation } = analysisHoverMove;
    ctx.fillStyle = CLR.wallHoverOk; ctx.shadowColor = CLR.wallHoverOkGlow; ctx.shadowBlur = 10;
    drawOneWall(ctx, ax, ay, orientation);
    ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0;
  } else {
    const { gx, gy } = analysisHoverMove;
    const { x, y }   = cellOrigin(gx, gy);
    rrect(ctx, x, y, CELL, CELL, CELL_R);
    ctx.fillStyle = 'rgba(255,255,255,0.22)'; ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1.5; ctx.stroke();
  }
}

const _ARROW_DIR = {
  '\u2191':[0,1],'\u2193':[0,-1],'\u2190':[-1,0],'\u2192':[1,0],
  '\u2196':[-1,1],'\u2197':[1,1],'\u2199':[-1,-1],'\u2198':[1,-1],
};

function parseAnalysisHover(label) {
  const wm = label.match(/^([HV])\((\d+),(\d+)\)$/);
  if (wm) return { type: 'wall', orientation: wm[1]==='H'?'h':'v', ax: parseInt(wm[2]), ay: parseInt(wm[3]) };
  const pm = label.match(/^.\((\d+),(\d+)\)$/);
  if (pm) return { type: 'pawn', gx: parseInt(pm[1]), gy: parseInt(pm[2]) };
  return null;
}

function playAnalysisMove(label) {
  if (state.is_finished) return;
  const m = label.match(/^([HV])\((\d+),(\d+)\)$/);
  if (m) { postMove({ type:'wall', orientation: m[1]==='H'?'h':'v', x: parseInt(m[2]), y: parseInt(m[3]) }); return; }
  const pm = label.match(/^(.)\((\d+),(\d+)\)$/);
  if (pm) { const dir = _ARROW_DIR[pm[1]]; if (dir) postMove({ type:'pawn', direction: dir }); }
}

function cancelAnalysisWork() {
  ++_analysisGen;
  document.getElementById('btn-run-mcts').classList.remove('running');
  const wasRunning = _analysisMctsRunning;
  _analysisMctsRunning = false;
  renderAnalysisProgress();
  if (!wasRunning) return;
  _analysisNnAvailable = false;
  _analysisNnLoadFailed = false;
  _analysisModelLoading = true;
  if (_analysisWorker) {
    _analysisWorker.terminate();
    _analysisWorker = null;
  }
  initAgentList();
  getAnalysisWorker();
}

function runMctsAnalysis() {
  if (_analysisMctsRunning || !gameState || gameState.isFinished() || !_analysisNnAvailable) return;
  _analysisMctsRunning = true;
  document.getElementById('btn-run-mcts').classList.add('running');
  _analysisDone  = 0;
  _analysisTotal = analysisNumSims;
  renderAnalysisProgress();
  const gen = ++_analysisGen;
  const w   = getAnalysisWorker();
  w.postMessage({
    ...workerPayload(gameState),
    type:    'mcts_analysis',
    numSims: analysisNumSims,
    agentId: analysisAgentId,
    gen,
  });
}

// ── Board variant switching ──────────────────────────────────────────────────
// 7x7 and 9x9 nets are architecturally incompatible (different action-space
// size), so switching variant always: starts a fresh game at the new
// boardsize/walls, and points both sides plus the analysis panel at the new
// variant's default model.
function switchBoardVariant(id) {
  if (id === currentBoardVariant || !BOARD_VARIANTS[id]) return;
  currentBoardVariant = id;
  const v = currentVariant();

  document.querySelectorAll('.board-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.board === id));

  // Cancel any in-flight play search / analysis work.
  cancelSearch();
  _analysisGen++;
  _analysisMctsRunning = false;
  document.getElementById('btn-run-mcts').classList.remove('running');
  renderAnalysisProgress();
  _analysisMctsData = null;
  _analysisNnData = null;

  // Every loaded play session is now for the wrong board size, so drop them all
  // and rebuild on the new variant's default net.
  for (const entry of _playWorkers.values()) entry.worker.terminate();
  _playWorkers.clear();
  for (const side of SIDES) {
    sideConfig[side].modelPath = v.defaultModel;
    sideConfig[side].manual    = false;
  }
  ensurePlayWorkers();

  // The analysis panel keeps its own worker and checkpoint — swapping in place
  // via 'load_model' is enough there.
  analysisModelPath = v.defaultModel;
  _analysisNnAvailable = false;
  _analysisNnLoadFailed = false;
  _analysisModelLoading = true;
  _analysisAgentManuallyChanged = false;
  getAnalysisWorker().postMessage({ type: 'load_model', path: analysisModelPath });

  initAgentList();
  resetGame();
}

// ── Slider level tables ───────────────────────────────────────────────────────
// Explicit sim-count levels, one per slider position (NOT evenly spaced in
// value — fine (1-step) resolution from 1-20 where low-sim play differs move
// to move, coarser above that). An index-based slider (uniform tick spacing,
// non-uniform values via this lookup) is what gives exact integer control at
// the low end while still reaching 5000 without an unusably long slider.
const SIMS_LEVELS = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  40, 50, 60, 70, 80, 90, 100,
  200, 300, 400, 500, 600, 700, 800, 900, 1000,
  2000, 3000, 4000, 5000,
];
function simsFromIndex(i) {
  return SIMS_LEVELS[Math.max(0, Math.min(i, SIMS_LEVELS.length - 1))];
}

// Sampling temperature for the visit-count distribution, same convention as
// Python's MCTSAgent (mcts.py) and tournament_cpp.py's --temp: 0 is argmax,
// ~0.3 is what the tournament uses, 1.0 samples the raw visit proportions.
const TEMP_LEVELS = [
  0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0,
];
function tempFromIndex(i) {
  return TEMP_LEVELS[Math.max(0, Math.min(i, TEMP_LEVELS.length - 1))];
}
// 0 gets a word rather than a number — "argmax" says what actually happens.
function tempLabel(t) { return t === 0 ? 'argmax' : String(t); }

// Same idea for the minimum time a move stays on screen: dense at the low end
// where the difference between "instant" and "followable" lives.
const DELAY_LEVELS = [0, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000];
function delayFromIndex(i) {
  return DELAY_LEVELS[Math.max(0, Math.min(i, DELAY_LEVELS.length - 1))];
}
function delayLabel(ms) { return ms === 0 ? 'none' : `${ms / 1000}s`; }

// ── Boot ──────────────────────────────────────────────────────────────────────
const canvas = document.getElementById('board');
setupInput(canvas);
buildAgentCards();

document.querySelectorAll('.board-btn').forEach(btn => {
  btn.addEventListener('click', () => switchBoardVariant(btn.dataset.board));
});

document.getElementById('btn-reset').addEventListener('click', resetGame);
document.getElementById('btn-flip').addEventListener('click', () => {
  boardFlipped = !boardFlipped;
  document.getElementById('btn-flip').classList.toggle('toggle-active', boardFlipped);
  hoverWall = null;
  if (state) draw(document.getElementById('board').getContext('2d'));
});
document.getElementById('btn-nn').addEventListener('click', () => {
  const panel = document.getElementById('nn-panel');
  if (panel.classList.contains('open')) panel.classList.remove('open');
  else openNNPanel();
});

// ── Timeline controls ─────────────────────────────────────────────────────────
document.getElementById('btn-step-back').addEventListener('click', stepBack);
document.getElementById('btn-step-fwd').addEventListener('click', stepForward);
document.getElementById('btn-play').addEventListener('click', togglePause);
document.getElementById('timeline-slider').addEventListener('input', e => seekTo(Number(e.target.value)));
document.addEventListener('keydown', e => {
  // A focused slider or select owns the arrow keys — there are four sliders on
  // this page (sims, depth, delay, timeline), so never steal them.
  if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'ArrowLeft')       { e.preventDefault(); stepBack(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); stepForward(); }
  else if (e.key === ' ')          { e.preventDefault(); togglePause(); }
});

document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (btn.dataset.mode === gameMode) return;
    gameMode = btn.dataset.mode;
    cancelSearch();
    _stepOnce = false;
    // AI vs AI starts parked so both agents can be configured before the game
    // runs off on its own. Every other mode resumes, so that leaving a paused
    // AI vs AI game doesn't silently freeze the agent in H vs AI. Reviewing
    // history stays put either way.
    paused = gameMode === 'ava' || !atTip();
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    updateAgentCardVisibility();
    hoverWall = null;
    // Whose turn it is didn't change, but who controls it did: the destination
    // highlights and wall preview are human-only affordances.
    landing = computeLanding();
    draw(document.getElementById('board').getContext('2d'));
    maybeAdvance();
    updateSidebar();
  });
});
document.getElementById('delay-slider').addEventListener('input', e => {
  moveDelayMs = delayFromIndex(Number(e.target.value));
  document.getElementById('delay-display').textContent = delayLabel(moveDelayMs);
});
document.getElementById('btn-analysis').addEventListener('click', () => {
  const panel = document.getElementById('analysis-panel');
  if (panel.classList.contains('open')) panel.classList.remove('open');
  else openAnalysisPanel();
  repositionPanels();   // this panel's height is part of where the NN panel sits
});
window.addEventListener('resize', repositionPanels);
document.getElementById('analysis-sims-slider').addEventListener('input', e => {
  analysisNumSims = simsFromIndex(Number(e.target.value));
  document.getElementById('analysis-sims-display').textContent = analysisNumSims;
});
document.getElementById('analysis-agent-select').addEventListener('change', e => {
  _analysisAgentManuallyChanged = true;
  analysisAgentId = e.target.value;
  _updateAnalysisAgentDesc();
  _analysisMctsData = null;
  refreshNnAnalysis();
});
document.getElementById('analysis-model-select').addEventListener('change', e => {
  analysisModelPath = e.target.value;
  _analysisNnAvailable = false;
  _analysisModelLoading = true;
  _analysisMctsData = null;
  _analysisNnData = null;
  _analysisMctsRunning = false;
  ++_analysisGen;
  if (_analysisWorker) {
    _analysisWorker.terminate();
    _analysisWorker = null;
  }
  initAgentList();
  getAnalysisWorker();
});
document.getElementById('btn-run-mcts').addEventListener('click', runMctsAnalysis);
document.getElementById('chk-auto-mcts').addEventListener('change', e => {
  _analysisMctsAuto = e.target.checked;
  if (_analysisMctsAuto && document.getElementById('analysis-panel').classList.contains('open')
      && state && !state.is_finished && !agentThinking)
    runMctsAnalysis();
});

// Kick off the workers early so ONNX loads in the background (both sides start
// on the same checkpoint, so this is one play worker plus the analysis one)
ensurePlayWorkers();
getAnalysisWorker();
// Initial agent list (may show alphazero as loading until the model loads)
updateAgentCardVisibility();
initAgentList();
// Start the game
initGame();
