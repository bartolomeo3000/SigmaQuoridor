'use strict';
// ============================================================
// mcts_worker.js — Web Worker: MCTS engine + ONNX Runtime Web
// ============================================================

// Load game engine and ORT at TOP LEVEL (synchronous) so they are
// guaranteed available before any async code or message handling.
// Both files are same-origin to avoid CORS / COEP issues in the Worker.
importScripts('game.js');
importScripts('ort.min.js');   // local copy of onnxruntime-web

// ── ONNX Runtime Web ────────────────────────────────────────────────────────
let ortSession = null;
let _activeTaskGen = -1;  // incremented on each new task; used to cancel stale loops
const DEFAULT_MODEL_PATH = './models/supervised_extended.onnx';
const START_MODEL_PATH = (() => {
  try { return new URL(self.location.href).searchParams.get('model') || DEFAULT_MODEL_PATH; }
  catch (_) { return DEFAULT_MODEL_PATH; }
})();

// Whether the loaded net is "full-canonical" (trained after the 2026-07-19
// policy-frame fix). Only those need P2's policy un-flipped; pre-fix
// "half-canonical" nets (all 7x7 models, and the retired 9x9 legacy
// checkpoints) output P2 policy already in the real-board frame and must NOT
// be un-flipped. The ONNX can't self-describe this, so we key on the model
// directory: the active 9x9 lineage (models_9x9/) is full-canonical, and the
// half-canonical 9x9 legacy checkpoints have been removed from the picker.
function isFullCanonicalModel(path) {
  return /models_9x9\b/.test(path || '');
}
let modelFullCanonical = isFullCanonicalModel(START_MODEL_PATH);

// Configure ORT immediately after importScripts while ort is guaranteed defined.
try {
  // WASM binaries are fetched from the CDN; credentials are not needed.
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';
  // Multi-threaded WASM (intra-op parallelism for the conv/matmul work in
  // each inference call) needs SharedArrayBuffer, which needs cross-origin
  // isolation — already set up via COOP/COEP (serve.py locally,
  // coi-serviceworker.js on GitHub Pages), so this is safe to enable. Falls
  // back to single-threaded automatically if crossOriginIsolated is false
  // for any reason. Capped rather than using the full core count to avoid
  // hogging low-end visitor machines.
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 8))
    : 1;
  // We are already inside a Worker — do not proxy back to main thread.
  ort.env.wasm.proxy = false;
} catch (e) {
  console.error('[worker] ORT env setup failed:', e);
}

(async function loadONNX() {
  try {
    // Force the WASM execution provider so WebGPU/WebGL detection is skipped.
    ortSession = await ort.InferenceSession.create(START_MODEL_PATH, {
      executionProviders: ['wasm'],
    });
    console.log('[worker] ONNX model loaded OK');
    postMessage({ type: 'model_loaded' });
  } catch (e) {
    console.error('[worker] ONNX model load failed:', e);
    postMessage({ type: 'model_failed', reason: String(e.message || e) });
  }
})();

// ── Action label (matches Python _action_label in app.py) ───────────────────
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

// ── Evaluators ───────────────────────────────────────────────────────────────
async function nnEvaluator(state, legalActions) {
  if (!ortSession || legalActions.length === 0)
    return rolloutEvaluator(state, legalActions);
  try {
    const N     = state.boardsize;
    const flat  = state.toNNInput();
    const tensor = new ort.Tensor('float32', flat, [1, 8, N, N]);
    const res    = await ortSession.run({ input: tensor });
    const logits = res.policy_logits.data;
    const value  = res.value.data[0];   // model already applies tanh

    // Softmax over legal-action logits. The net outputs policy in the current
    // player's canonical (P2-flipped) frame, so for P2 we must read each real
    // action's logit at its vertically-permuted index (matches Python
    // NNEvaluator._canon_indices / C++ vflip_action). Without this, P2's whole
    // policy is vertically scrambled — pawn up<->down and walls on the wrong
    // rows — which makes any full-canonical net play nonsense as P2.
    const flip    = modelFullCanonical && !state.isPlayer1Turn();
    const vperm   = flip ? vertPolicyPermutation(N) : null;
    const indices = legalActions.map(a => {
      const idx = actionToIndex(a, N);
      return flip ? vperm[idx] : idx;
    });
    let maxL = -Infinity;
    for (const i of indices) if (logits[i] > maxL) maxL = logits[i];
    let sumE = 0;
    const exps = indices.map(i => { const e = Math.exp(logits[i] - maxL); sumE += e; return e; });
    const priors = exps.map(e => e / sumE);
    return [priors, value];
  } catch (e) {
    return rolloutEvaluator(state, legalActions);
  }
}

function rolloutEvaluator(state, legalActions) {
  const n      = legalActions.length;
  const priors = n > 0 ? new Array(n).fill(1 / n) : [];
  const value  = n > 0 ? randomRollout(state) : 0;
  return [priors, value];
}

function randomRollout(rootState) {
  const rootPlayer = rootState.getCurrentPlayer();
  let s = rootState, steps = 0;
  while (!s.isFinished() && steps < 300) {
    const acts = s.getLegalActions();
    if (acts.length === 0) break;
    s = s.next(acts[Math.floor(Math.random() * acts.length)]);
    steps++;
  }
  const w = s.winner();
  if (w === 0) return 0;
  return w === rootPlayer ? 1 : -1;
}

// ── Minimax ─────────────────────────────────────────────────────────────────
function minimaxHeuristic(state, maximizingPlayer) {
  const N = state.boardsize;
  const [p1x, p1y] = state.player1pos;
  const [p2x, p2y] = state.player2pos;
  const p1Dist = state.p1_dist[p1y * N + p1x];
  const p2Dist = state.p2_dist[p2y * N + p2x];
  return maximizingPlayer === 1 ? p2Dist - p1Dist : p1Dist - p2Dist;
}

function minimaxTerminalValue(state, maximizingPlayer) {
  const winner = state.winner();
  if (winner === maximizingPlayer) return 1e9;
  if (winner !== 0) return -1e9;
  if (state.isDrawn()) return 0;
  return null;
}

function orderedMinimaxActions(state) {
  const legal = state.getLegalActions();
  const advantages = state.computeMoveAdvantages(legal);
  return legal
    .map((action, i) => ({ action, advantage: advantages[i] }))
    .sort((a, b) => b.advantage - a.advantage)
    .map(item => item.action);
}

function alphabeta(state, depth, alpha, beta, maximizing, maximizingPlayer) {
  const terminal = minimaxTerminalValue(state, maximizingPlayer);
  if (terminal !== null) return terminal;
  if (depth === 0) return minimaxHeuristic(state, maximizingPlayer);

  const legal = orderedMinimaxActions(state);
  if (legal.length === 0) return minimaxHeuristic(state, maximizingPlayer);

  if (maximizing) {
    let value = -1e18;
    for (const action of legal) {
      value = Math.max(value, alphabeta(state.next(action), depth - 1, alpha, beta, false, maximizingPlayer));
      alpha = Math.max(alpha, value);
      if (alpha >= beta) break;
    }
    return value;
  }

  let value = 1e18;
  for (const action of legal) {
    value = Math.min(value, alphabeta(state.next(action), depth - 1, alpha, beta, true, maximizingPlayer));
    beta = Math.min(beta, value);
    if (alpha >= beta) break;
  }
  return value;
}

function selectMinimaxAction(state, depth) {
  const maximizingPlayer = state.getCurrentPlayer();
  const legal = orderedMinimaxActions(state);
  let bestValue = -1e18;
  let bestActions = [];
  for (const action of legal) {
    const value = alphabeta(state.next(action), depth - 1, bestValue, 1e18, false, maximizingPlayer);
    if (value > bestValue) {
      bestValue = value;
      bestActions = [action];
    } else if (value === bestValue) {
      bestActions.push(action);
    }
  }
  return bestActions.length ? bestActions[Math.floor(Math.random() * bestActions.length)] : null;
}

// ── MCTS ──────────────────────────────────────────────────────────────────────
class MCTSNode {
  constructor(state, parent = null, action = null, prior = 1.0, parentState = null) {
    this.state        = state;
    this._parentState = parentState;
    this.parent       = parent;
    this.action       = action;
    this.prior        = prior;
    this.basePrior    = prior;
    this.children     = [];
    this.visitCount   = 0;
    this.valueSum     = 0;
    this.isExpanded   = false;
  }

  ensureState() {
    if (this.state === null) {
      this.state        = this._parentState.next(this.action);
      this._parentState = null;
    }
  }

  get qValue() { return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount; }

  bestChild(cPuct = 1.0, fpuReduction = 0.2) {
    const pq      = this.qValue;
    const sqrtN   = Math.sqrt(this.visitCount);
    let visitedPriorSum = 0;
    for (const c of this.children) if (c.visitCount > 0) visitedPriorSum += c.basePrior;
    let best = null, bestScore = -Infinity;
    for (const c of this.children) {
      const u     = cPuct * c.prior * sqrtN / (1 + c.visitCount);
      const q     = c.visitCount === 0
        ? pq - fpuReduction * Math.sqrt(visitedPriorSum)
        : -c.qValue;
      const score = q + u;
      if (score > bestScore) { bestScore = score; best = c; }
    }
    return best;
  }
}

const C_PUCT       = 1.0;
const FPU_REDUCTION = 0.2;

async function expandNode(node, evaluator) {
  const legal = node.state.getLegalActions();
  let priors, value;
  if (legal.length === 0) {
    priors = []; value = 0;
  } else {
    [priors, value] = await evaluator(node.state, legal);
  }
  for (let i = 0; i < legal.length; i++) {
    node.children.push(new MCTSNode(null, node, legal[i], priors[i], node.state));
  }
  node.isExpanded = true;
  return value;
}

function backup(node, value) {
  let n = node;
  while (n !== null) { n.visitCount++; n.valueSum += value; value = -value; n = n.parent; }
}

function selectLeaf(root) {
  let node = root;
  while (true) {
    node.ensureState();
    if (!node.isExpanded || node.state.isFinished()) break;
    node = node.bestChild(C_PUCT, FPU_REDUCTION);
  }
  return node;
}

async function runMCTS(state, numSims, evaluator, cancelToken, onProgress = null) {
  const root     = new MCTSNode(state);
  const initVal  = await expandNode(root, evaluator);
  if (cancelToken !== _activeTaskGen) return null;
  backup(root, initVal);
  // Report at most ~40 times per search regardless of numSims: enough for a
  // smooth bar, few enough that postMessage never becomes part of the cost.
  const step = Math.max(1, Math.ceil(numSims / 40));
  for (let i = 0; i < numSims; i++) {
    if (cancelToken !== _activeTaskGen) return null;
    const leaf = selectLeaf(root);
    leaf.ensureState();
    let value;
    if (leaf.state.isFinished()) {
      value = leaf.state.winner() !== 0 ? -1 : 0;
    } else {
      value = await expandNode(leaf, evaluator);
    }
    if (cancelToken !== _activeTaskGen) return null;
    backup(leaf, value);
    if (onProgress && ((i + 1) % step === 0 || i + 1 === numSims)) onProgress(i + 1, numSims);
  }
  return root;
}

async function selectAction(state, numSims, evaluator, cancelToken, onProgress = null) {
  const root = await runMCTS(state, numSims, evaluator, cancelToken, onProgress);
  if (!root) return null;
  let best = null, bestCount = -1;
  for (const c of root.children) {
    if (c.visitCount > bestCount) { bestCount = c.visitCount; best = c; }
  }
  return best ? best.action : (state.getLegalActions()[0] || null);
}

async function getPolicy(state, numSims, evaluator, cancelToken, onProgress = null) {
  const root = await runMCTS(state, numSims, evaluator, cancelToken, onProgress);
  if (!root) return null;
  const total = root.children.reduce((s, c) => s + c.visitCount, 0);
  const rootQ = root.qValue;
  const policy = root.children
    .map(c => ({ action: c.action, prob: total > 0 ? c.visitCount / total : 0 }))
    .sort((a, b) => b.prob - a.prob);
  return { policy, rootQ };
}

// ── Reconstruct State from worker-message payload ────────────────────────────
function stateFromMsg(d) {
  return new State({
    boardsize:        d.boardsize,
    depth:            d.depth,
    player1pos:       d.player1pos,
    player2pos:       d.player2pos,
    hwalls:           new Uint8Array(d.hwalls),
    vwalls:           new Uint8Array(d.vwalls),
    walls_p1:         d.walls_p1,
    walls_p2:         d.walls_p2,
    walls_initial:    d.walls_initial,
    hwall_anchors:    new Set(d.hwall_anchors),
    vwall_anchors:    new Set(d.vwall_anchors),
    position_history: new Map(d.position_history),
    // p1_dist etc. not sent — recomputed from walls in constructor
  });
}

// ── Message handler ───────────────────────────────────────────────────────────
onmessage = async function (e) {
  const d   = e.data;
  const gen = d.gen;

  if (d.type === 'cancel') {
    _activeTaskGen = gen;
    return;
  }

  if (d.type === 'load_model') {
    try {
      if (ortSession) { try { await ortSession.release(); } catch (_) {} ortSession = null; }
      ortSession = await ort.InferenceSession.create(d.path, { executionProviders: ['wasm'] });
      modelFullCanonical = isFullCanonicalModel(d.path);
      postMessage({ type: 'model_loaded' });
    } catch (err) {
      postMessage({ type: 'model_failed', reason: String(err.message || err) });
    }
    return;
  }

  // Synchronously update active gen before any await — cancels any running loop
  _activeTaskGen = gen;

  // Pick evaluator based on requested agent + NN availability
  const useNN   = (d.agentId === 'alphazero' || d.agentId === 'supervised') && ortSession !== null;
  const evalFn  = useNN ? nnEvaluator : rolloutEvaluator;
  const numSims = Math.max(1, d.numSims || 200);

  if (d.type === 'think') {
    const s      = stateFromMsg(d);
    let action;
    if (d.agentId === 'minimax') {
      // Alpha-beta runs synchronously with no natural checkpoints, so it has no
      // simulation count to report — the main thread shows no bar for it.
      action = selectMinimaxAction(s, Math.max(2, Math.min(8, d.minimaxDepth || 3)));
    } else {
      action = await selectAction(s, numSims, evalFn, gen, (done, total) => {
        if (gen === _activeTaskGen) postMessage({ type: 'progress', done, total, gen });
      });
    }
    if (gen !== _activeTaskGen) return;  // cancelled
    if (action === null) return;  // cancelled
    postMessage({ type: 'move', action, gen });

  } else if (d.type === 'nn_eval') {
    if (!ortSession) { postMessage({ type: 'nn_unavailable', gen }); return; }
    const s      = stateFromMsg(d);
    const legal  = s.getLegalActions();
    const [priors, value] = await nnEvaluator(s, legal);
    if (gen !== _activeTaskGen) return;  // cancelled
    const moves = legal
      .map((a, i) => ({ label: actionLabel(s, a), prob: priors[i] }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 30);
    postMessage({ type: 'nn_result', value, moves, current_player: s.getCurrentPlayer(), gen });

  } else if (d.type === 'mcts_analysis') {
    const s      = stateFromMsg(d);
    let nn = null;
    if (ortSession) {
      const legal = s.getLegalActions();
      const [priors, value] = await nnEvaluator(s, legal);
      if (gen !== _activeTaskGen) return;  // cancelled
      const moves = legal
        .map((a, i) => ({ label: actionLabel(s, a), prob: priors[i] }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 30);
      nn = { value, moves, current_player: s.getCurrentPlayer() };
    }
    const result = await getPolicy(s, numSims, evalFn, gen, (done, total) => {
      if (gen === _activeTaskGen) postMessage({ type: 'progress', done, total, gen });
    });
    if (!result) return;  // cancelled
    const moves = result.policy
      .map(({ action, prob }) => ({ label: actionLabel(s, action), prob }))
      .slice(0, 30);
    postMessage({ type: 'mcts_result', value: result.rootQ, moves, current_player: s.getCurrentPlayer(), nn, gen });
  }
};
