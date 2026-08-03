"""AlphaZero-style self-play training loop for SigmaQuoridor.

Cycle structure
---------------
  1. Self-play  — the current model plays GAMES_PER_CYCLE games against
                  itself using MCTS.  Each position's MCTS visit distribution
                  is stored as the policy training target.
  2. Buffer     — new positions are appended to the replay buffer; positions
                  older than BUFFER_CYCLES cycles are discarded (FIFO per
                  cycle, so the most recent BUFFER_CYCLES cycles are kept).
  3. Training   — TRAIN_STEPS mini-batches are sampled uniformly from the
                  full buffer and used to update the network weights.
  4. Checkpoint — updated weights are saved; cycle repeats.

Losses
------
  policy : cross-entropy between MCTS visit distribution and NN policy
           logits.  Illegal actions are masked with -∞ before log-softmax so
           the softmax normalises only over legal actions.
  value  : MSE between the predicted value scalar and the game outcome
           (+1 win / -1 loss / 0 draw) from the current player's POV.
  total  : policy_loss + value_loss  (equal weighting, as in AlphaZero).

Usage
-----
  # Fresh training run:
  python train.py

  # Resume from the latest checkpoint:
  python train.py --resume

  # Quick smoke-test (fast settings):
  python train.py --games 4 --sims 20 --cycles 2 --steps 50
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from benchmark_agents import GreedyDistanceAgent, MinimaxAgent, RandomAgent
from dual_network import DEVICE, DualNetwork, NNEvaluator, load_model, save_model
from game import State, WallAction, action_space_size, action_to_index, flip_nn_input_lr, flip_policy_lr, flip_policy_vert
from mcts import MCTSAgent

# ── Default hyper-parameters ────────────────────────────────────────────────
# Override any of these via CLI flags (see parse_args()).

# Board
BOARDSIZE        = 9
WALLS_PER_PLAYER = 10

# Self-play
GAMES_PER_CYCLE   = 2048    # self-play games played per cycle
MCTS_SIMS         = 800   # MCTS simulations per move during self-play
NUM_WORKERS       = os.cpu_count() or 1  # parallel self-play processes
C_PUCT            = 1.0
DIRICHLET_ALPHA   = 0.3
DIRICHLET_EPSILON = 0.25
DIST_BONUS_WEIGHT_MAX = 0.5  # each game samples w1, w2 ~ Uniform[0, MAX] independently per side
FPU_REDUCTION     = 0.1    # First Play Urgency: unvisited child Q estimate = parent_Q - FPU
# Move-selection temperature schedule (KataGo-style): decays exponentially
# from TEMP_EARLY at ply 0 toward TEMP_FINAL with halflife TEMP_HALFLIFE
# plies. A small nonzero TEMP_FINAL keeps trajectories diverging for the
# whole game (fixes the hard-argmax lock-in effect; see docs). Moves with
# <= TEMP_PRUNE_VISITS visits are never sampled (argmax always eligible).
# Must be kept in sync with cpp/selfplay.hpp / cpp/bindings.cpp /
# selfplay_cpp.py defaults.
TEMP_EARLY        = 1.0
TEMP_FINAL        = 0.2
TEMP_HALFLIFE     = 10.0
TEMP_PRUNE_VISITS = 4
FAST_PLAY_PROB      = 0.0  # fraction of moves that use fast MCTS; remainder use full search
MCTS_SIMS_FAST      = 128   # simulations for fast plies (2 NN batches); not saved unless surprising
FAST_KL_THRESHOLD   = 0.7   # KL(visit_dist ∥ prior) nats — fast positions above this are saved
MCTS_SIM_BATCH_SIZE = 1     # legacy sequential mode: one simulation/eval at a time

# Random wall pre-fill — each player places this many walls randomly before
# MCTS self-play begins, leaving them with 0 walls for actual play.
# Applied to RANDOM_WALL_FRACTION of games; the other (1 - RANDOM_WALL_FRACTION) start normally with WALLS_PER_PLAYER.
# Set to 0 to disable entirely.
RANDOM_WALL_PLIES = 1      # walls placed per player in pre-filled games
RANDOM_WALL_FRACTION = 0.05  # fraction of games to apply random wall pre-fill to

# Replay buffer. 60 rather than 30 since playout cap randomization records ~4x
# fewer positions per cycle; see markdown_notes/cpp_selfplay_notes.md.
BUFFER_CYCLES = 60         # keep positions from this many recent cycles

# Training
BATCH_SIZE                 = 1024   # was 256; GPU has headroom, and BatchNorm/value-target
                                    # noise both benefit from the larger batch (see
                                    # docs/cpp_selfplay_notes.md for the LR sweep this was
                                    # paired with -- LR was scaled ~sqrt(4x) alongside this)
TRAIN_POSITIONS_PER_CYCLE  = 1_024_000  # gradient updates = this // BATCH_SIZE per cycle
BUFFER_RECENCY_DECAY       = 0.99     # per-cycle weight decay; 1.0 = uniform, lower = more recency. BUFFER_RECENCY_DECAY^CYCLE_AGE = relative weight of positions from a cycle CYCLE_AGE cycles ago when sampling training batches. 0.9^5 = 0.59, 0.9^10 = 0.35, 0.9^20 = 0.12, 0.9^40 = 0.01
MIN_BUFFER_SIZE            = BATCH_SIZE

# Optimizer
LEARNING_RATE     = 3e-4  # was 1e-4 at batch 256; sqrt-scaled for the 4x batch increase
                          # (Adam heuristic -- see chat history). Not yet empirically swept;
WEIGHT_DECAY      = 1e-4
VALUE_LOSS_WEIGHT = 1.0         # multiply value MSE loss (KataGo uses ~1.5)
LR_MILESTONES  = [800, 1600]    # cycle numbers at which to multiply LR by LR_DECAY
LR_DECAY       = 0.1

# Network
FILTERS      = 128
NUM_RESIDUAL = 10
GPOOL_EVERY  = 3   # every k-th residual block is a KataGo-style global-pooling
                   # block (0 = none); see dual_network.GPoolResidualBlock

# Evaluation
EVAL_EVERY = 0    # run evaluation every N cycles; 0 = never
EVAL_SIMS  = 800   # MCTS simulations for the challenger during evaluation

# Holdout validation (fixed set sampled once from a previous run's data)
ENABLE_HOLDOUT_EVAL = False   # holdout loss is slow (HOLDOUT_SIZE positions/cycle); off by default
HOLDOUT_DIR    = "runs/data_9x9_scratch"   # source directory for holdout positions
HOLDOUT_CYCLES = 10            # how many most-recent cycles to draw from
HOLDOUT_SIZE   = 4096*100         # positions to evaluate per cycle (fixed sample)

# Each entry: (label, num_games, spec)
# spec: {"type": "random"} | {"type": "greedy"} |
#        {"type": "mcts", "path": str, "sims": int} |
#        {"type": "minimax", "depth": int}
# num_games is split: ceil(n/2) games as P1, floor(n/2) games as P2.
EVAL_OPPONENTS: list[tuple[str, int, dict]] = [
    ("random",          2, {"type": "random"}),
    ("greedy",          2, {"type": "greedy"}),
    # Disabled: models_7x7/best.pt is now the same file MODEL_DIR trains and
    # overwrites every cycle, so these would compare the model against itself.
    # ("old-best   1s",   2, {"type": "mcts", "path": "runs/models_7x7/best.pt", "sims":   1}),
    # ("old-best  50s",   2, {"type": "mcts", "path": "runs/models_7x7/best.pt", "sims":  50}),
    # ("old-best 100s",   2, {"type": "mcts", "path": "runs/models_7x7/best.pt", "sims": 100}),
    # ("old-best 200s",   2, {"type": "mcts", "path": "runs/models_7x7/best.pt", "sims": 200}),
    # ("old-best 400s",   2, {"type": "mcts", "path": "runs/models_7x7/best.pt", "sims": 400}),
    # ("old-best 800s",   2, {"type": "mcts", "path": "runs/models_7x7/best.pt", "sims": 800}),
    ("minimax d2",       2, {"type": "minimax", "depth": 2}),
    ("minimax d3",       2, {"type": "minimax", "depth": 3}),
    # ("minimax d4",       2, {"type": "minimax", "depth": 4}),
]

# Paths
MODEL_DIR      = "runs/models_9x9_scratch"                       # model checkpoints and best.pt weights
MODEL_PATH     = os.path.join(MODEL_DIR, "best.pt")         # inference weights only
LOG_DIR        = "runs/logs"                                     # per-run console transcripts
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")     # full training state
DATA_DIR       = "runs/data_9x9_scratch"                          # persisted self-play cycles

# Run
NUM_CYCLES        = 100
CHECKPOINT_EVERY  = 1   # save a full checkpoint every N cycles


# ── Replay buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores positions from the last ``maxcycles`` self-play cycles in a
    single pre-allocated, growable buffer.

    Rationale (memory): the previous implementation stored each cycle as a
    separate array and rebuilt a full concatenated copy on every call to
    ``as_arrays()`` (i.e. every training phase / cycle). That meant the
    *entire* buffer was resident twice simultaneously during each rebuild
    (originals + fresh concatenation) — a ~2x transient memory spike on top
    of an already-large steady-state footprint, which is what made buffers
    beyond ~20 cycles fail to fit in RAM and start swapping. This version
    keeps ONE persistent buffer: new cycles are appended in place (growing
    the underlying array only when actually needed, amortized), oldest
    cycles are evicted by shifting the remaining valid region down to index
    0 (an in-place overlapping slice assignment — safe and memmove-like in
    NumPy), and ``as_arrays()`` returns zero-copy VIEWS into the live
    buffer instead of a fresh copy.

    Also stores ``states``/``policies`` as float16 (state is by far the
    largest per-position array) to roughly halve memory further; values
    stay float32 (already negligible size). Callers must upcast the
    sampled mini-batch back to float32 before feeding the network.

    Per-position arrays
    -------------------
    state  : (8, N, N) float16  — ``State.to_nn_input()`` output, downcast
    policy : (A,)      float16  — full action-space MCTS visit distribution;
                                  0.0 for illegal actions, downcast
    value  : float32            — game outcome from the current player's POV
                                  (+1 win, -1 loss, 0 draw)
    """

    _GROWTH_FACTOR = 1.5   # amortized-doubling-style growth when capacity is exceeded

    def __init__(self, maxcycles: int) -> None:
        self._maxcycles = maxcycles
        self._cycle_lens: deque[int] = deque()
        self._end      = 0        # number of valid positions (index of first free slot)
        self._capacity  = 0        # allocated capacity (>= self._end)
        self._states:   np.ndarray | None = None   # (capacity, 8, N, N) float16
        self._policies: np.ndarray | None = None   # (capacity, A)       float16
        self._values:   np.ndarray | None = None   # (capacity,)         float32

    def _ensure_capacity(self, extra: int) -> None:
        need = self._end + extra
        if self._capacity >= need:
            return
        new_capacity = max(need, int(self._capacity * self._GROWTH_FACTOR))
        for attr, dtype in (("_states", np.float16), ("_policies", np.float16),
                           ("_values", np.float32)):
            old = getattr(self, attr)
            new_shape = (new_capacity,) + old.shape[1:]
            new_arr = np.empty(new_shape, dtype=dtype)
            new_arr[: self._end] = old[: self._end]
            setattr(self, attr, new_arr)
        self._capacity = new_capacity

    def add_cycle(
        self,
        states:   np.ndarray,   # (M, 8, N, N)
        policies: np.ndarray,   # (M, A)
        values:   np.ndarray,   # (M,)
    ) -> None:
        n = states.shape[0]
        if self._states is None:
            # First-ever cycle: allocate zero-length arrays with the right
            # trailing shape so _ensure_capacity's np.empty(new_shape,...)
            # below has something to match.
            self._states   = np.empty((0,) + states.shape[1:],   dtype=np.float16)
            self._policies = np.empty((0,) + policies.shape[1:], dtype=np.float16)
            self._values   = np.empty((0,),                      dtype=np.float32)

        # Evict oldest cycle(s) if already at the cycle-count limit, by
        # shifting the remaining valid region down to index 0 in place
        # (no extra allocation — safe overlapping slice assignment).
        while len(self._cycle_lens) >= self._maxcycles and self._cycle_lens:
            evict = self._cycle_lens.popleft()
            remaining = self._end - evict
            self._states[:remaining]   = self._states[evict:self._end]
            self._policies[:remaining] = self._policies[evict:self._end]
            self._values[:remaining]   = self._values[evict:self._end]
            self._end = remaining

        self._ensure_capacity(n)
        self._states[self._end:self._end + n]   = states.astype(np.float16, copy=False)
        self._policies[self._end:self._end + n] = policies.astype(np.float16, copy=False)
        self._values[self._end:self._end + n]   = values.astype(np.float32, copy=False)
        self._end += n
        self._cycle_lens.append(n)

    def size(self) -> int:
        return self._end

    def num_cycles(self) -> int:
        return len(self._cycle_lens)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zero-copy views into the live buffer (valid until the next add_cycle())."""
        return self._states[:self._end], self._policies[:self._end], self._values[:self._end]

    def sampling_weights(self, recency_decay: float = 0.9) -> np.ndarray:
        """
        Per-position sampling probabilities that favour more recent cycles.

        Cycle at index i (0 = oldest, n-1 = newest) receives unnormalized
        weight ``recency_decay ** (n - 1 - i)``.  All positions within a
        cycle share the same weight.  The returned array is normalized to
        sum to 1 and is aligned with the arrays from ``as_arrays()``.

        ``recency_decay = 1.0`` reproduces uniform sampling.
        """
        n = len(self._cycle_lens)
        weights = np.empty(self._end, dtype=np.float64)
        offset = 0
        for i, clen in enumerate(self._cycle_lens):
            weights[offset: offset + clen] = recency_decay ** (n - 1 - i)
            offset += clen
        weights /= weights.sum()
        return weights



# ── Self-play ────────────────────────────────────────────────────────────────

def _place_random_walls(state: State, walls_per_player: int) -> State:
    """
    Alternately place ``walls_per_player`` walls for each player at random,
    choosing uniformly from the legal wall actions at each step.
    Pawn moves are skipped — only WallActions are sampled.
    Returns the resulting state (original is unchanged).
    If a player runs out of legal wall placements early, that player stops.
    """
    for _ in range(walls_per_player * 2):   # alternate: P1 wall, P2 wall, …
        legal = [a for a in state.get_legal_actions() if isinstance(a, WallAction)]
        if not legal:
            break
        state = state.next(random.choice(legal))
    return state


def selection_temperature(depth: int) -> float:
    """KataGo-style decaying selection temperature at a given game depth."""
    return TEMP_FINAL + (TEMP_EARLY - TEMP_FINAL) * 0.5 ** (depth / TEMP_HALFLIFE)


def self_play_game(
    agent1:         MCTSAgent,
    agent2:         MCTSAgent,
    boardsize:      int,
    walls:          int,
    fast_play_prob: float = FAST_PLAY_PROB,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.float32]], int]:
    """
    Play one full self-play game and return collected positions + winner.

    Temperature schedule
    --------------------
    Selection temperature decays exponentially with depth from TEMP_EARLY
    to TEMP_FINAL (halflife TEMP_HALFLIFE plies); moves are sampled with
    probability proportional to visits^(1/τ), with moves at
    <= TEMP_PRUNE_VISITS visits excluded from sampling. The recorded policy
    TARGET is always the raw visit distribution regardless of τ.

    Playout cap randomization
    -------------------------
    Each ply is independently tagged as "fast" (prob = fast_play_prob) or
    "full".  Fast plies skip MCTS entirely — the evaluator is called once
    and the raw NN prior is used to pick the move, but the position is NOT
    recorded as training data.  Fast plies exist solely to advance the game
    to diverse board states so that full-MCTS plies see varied positions.
    Full plies run the normal MCTS search and ARE recorded.  This matches
    KataGo's playout cap randomization design: only full-search positions
    enter the training buffer.

    Value targets (game outcome) are assigned retroactively for full plies
    only.  Policy targets come from MCTS visit counts.

    Returns
    -------
    positions : list of (state_tensor, policy_vector, value_target)
                value_target is assigned retroactively after the game ends:
                +1.0 if the player to move at that state won, -1.0 if lost,
                0.0 for a draw.
    winner    : 0 (draw), 1 (Player 1), or 2 (Player 2)
    """
    if RANDOM_WALL_PLIES > 0 and random.random() < RANDOM_WALL_FRACTION:
        # Pre-filled game: start with a full set of walls, place them all randomly,
        # then MCTS plays with 0 walls left — pure labyrinth navigation.
        state = _place_random_walls(
            State(boardsize=boardsize, walls_p1=RANDOM_WALL_PLIES, walls_p2=RANDOM_WALL_PLIES),
            RANDOM_WALL_PLIES,
        )
    else:
        state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
    A     = action_space_size(boardsize)
    history: list[tuple[np.ndarray, np.ndarray, int]] = []

    while not state.is_finished():
        agent = agent1 if state.is_player1_turn() else agent2
        agent.temperature = selection_temperature(state.depth)

        if fast_play_prob > 0.0 and random.random() < fast_play_prob:
            # ── Fast ply: reduced MCTS (MCTS_SIMS_FAST sims) ─────────────
            # Matches KataGo's playout cap randomization: fast plies run a
            # real (but smaller) search.  The position is saved only if the
            # search was "surprising" — i.e. KL(visit_dist ∥ NN prior) exceeds
            # FAST_KL_THRESHOLD nats.  This avoids circular self-distillation
            # while still capturing positions where the search genuinely
            # discovered something the raw prior missed.
            # base_prior on each child holds the clean NN prior (pre-Dirichlet),
            # so no second evaluator call is needed after search().
            root_fast = agent.search(state, num_sims=MCTS_SIMS_FAST)
            children  = root_fast.children

            # Build visit distribution Q and prior distribution P
            total_visits = sum(c.visit_count for c in children)
            total_prior  = sum(c.base_prior  for c in children) or 1.0
            kl = 0.0
            if total_visits > 0:
                for c in children:
                    q = c.visit_count / total_visits
                    p = c.base_prior  / total_prior
                    if q > 1e-10 and p > 1e-10:
                        kl += q * math.log(q / p)

            if kl > FAST_KL_THRESHOLD:
                # Training target: raw visit distribution (same convention as
                # full plies — temperature affects selection only).
                counts_t = np.array([c.visit_count for c in children], dtype=np.float64)
                tot = counts_t.sum()
                target = counts_t / tot if tot > 0 else np.full(len(children), 1.0 / len(children))
                policy_vec = np.zeros(A, dtype=np.float32)
                for child, prob in zip(children, target):
                    policy_vec[action_to_index(child.action, boardsize)] = prob
                history.append((
                    state.to_nn_input(),
                    policy_vec,
                    state.get_current_player(),
                ))

            # Move selection from the fast-MCTS visit distribution
            counts_arr = np.array([c.visit_count for c in children], dtype=np.float64)
            if agent.temperature == 0.0:
                chosen = int(np.argmax(counts_arr))
            else:
                inv_t  = 1.0 / agent.temperature
                raw    = counts_arr ** inv_t
                raw   /= raw.sum()
                chosen = int(np.random.choice(len(children), p=raw))
            state = state.next(children[chosen].action)
            # Discard fast-ply tree; it had too few sims to be worth reusing.
            agent._root = None

        else:
            # ── Full ply: normal MCTS search ──────────────────────────────
            # Run MCTS once; derive BOTH the training target and the move
            # selection from the root's visit counts.
            root     = agent.search(state)
            children = root.children
            counts   = np.array([c.visit_count for c in children], dtype=np.float64)
            total    = counts.sum()

            # Training target: RAW visit distribution at every ply
            # (LC0/KataGo convention) — temperature only affects move
            # selection, never the recorded target.
            target = counts / total if total > 0 else np.full(len(children), 1.0 / len(children))

            # Build a full-size policy vector (zeros at illegal action indices).
            policy_vec = np.zeros(A, dtype=np.float32)
            for child, prob in zip(children, target):
                policy_vec[action_to_index(child.action, boardsize)] = prob
            # Record the target in the current player's canonical frame (matches
            # the canonical to_nn_input): vertically flip P2's policy indices.
            if state.get_current_player() == 2:
                policy_vec = flip_policy_vert(policy_vec, boardsize)

            history.append((
                state.to_nn_input(),         # (8, N, N) float32
                policy_vec,                  # (A,)      float32
                state.get_current_player(),  # 1 or 2
            ))

            # Move selection: temperature-applied.
            if agent.temperature == 0.0:
                chosen = int(np.argmax(counts))
            else:
                inv_t = 1.0 / agent.temperature
                raw   = counts ** inv_t
                raw  /= raw.sum()
                chosen = int(np.random.choice(len(children), p=raw))
            state = state.next(children[chosen].action)

    # Retroactively assign value targets; store original AND mirror each position.
    winner       = state.winner()
    walls_placed = 2 * state.walls_initial - state.walls_p1 - state.walls_p2
    positions = []
    for state_tensor, policy_vec, player in history:
        if winner == 0:
            value = np.float32(0.0)
        else:
            value = np.float32(1.0 if winner == player else -1.0)
        positions.append((state_tensor, policy_vec, value))
        # Left-right flip is a free symmetry: produces a valid equivalent position.
        positions.append((
            flip_nn_input_lr(state_tensor),
            flip_policy_lr(policy_vec, state.boardsize),
            value,
        ))

    return positions, winner, walls_placed


# ── Symmetric evaluator ───────────────────────────────────────────────────────

class SymmetricEvaluator:
    """
    Wraps any Evaluator and randomly applies a left-right board flip 50% of
    the time before querying the network, then unflips the returned priors.

    This prevents the MCTS search from exploiting any residual left-right bias
    that remains in the network weights, and provides an additional source of
    stochasticity during self-play that is independent of Dirichlet noise.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __call__(self, state, legal_actions):
        if np.random.random() < 0.5:
            return self._inner(state, legal_actions)

        # Flip: temporarily swap to a mirrored evaluator call.
        # We don't actually flip the State object — instead we monkey-patch
        # to_nn_input to return the flipped tensor, then unflip the priors.
        boardsize = state.boardsize
        orig_to_nn = state.to_nn_input

        def flipped_to_nn():
            return flip_nn_input_lr(orig_to_nn())

        state.to_nn_input = flipped_to_nn

        # Mirror the legal actions so the network sees the right indices.
        from game import WallAction, PawnAction, ALL_PAWN_DIRECTIONS, _LR_PAWN_FLIP
        W = boardsize - 1

        def mirror_action(a):
            if isinstance(a, PawnAction):
                dx, dy = a.direction
                return PawnAction(direction=(-dx, dy))
            return WallAction(x=W - 1 - a.x, y=a.y, orientation=a.orientation)

        mirrored_legal = [mirror_action(a) for a in legal_actions]
        priors, value = self._inner(state, mirrored_legal)

        # Restore original method.
        state.to_nn_input = orig_to_nn

        # priors[i] corresponds to mirrored_legal[i] which maps to legal_actions[i]
        # — order is preserved by mirror_action, so no reordering needed.
        return priors, value

    def batch_call(
        self,
        states:             list,
        legal_actions_list: list,
    ) -> list:
        """
        Batched variant of ``__call__``.  Applies a per-item random LR flip
        (50 % probability each) then delegates to ``inner.batch_call_raw``
        with the pre-flipped numpy arrays so the inner evaluator issues one
        GPU/CPU forward pass for the entire batch.

        Falls back to sequential ``__call__`` when the inner evaluator has no
        ``batch_call_raw`` (e.g. rollout evaluator).
        """
        if not hasattr(self._inner, "batch_call_raw"):
            return [self(s, la) for s, la in zip(states, legal_actions_list)]

        from game import WallAction, PawnAction
        boardsize = states[0].boardsize
        W = boardsize - 1

        def mirror_action(a):
            if isinstance(a, PawnAction):
                dx, dy = a.direction
                return PawnAction(direction=(-dx, dy))
            return WallAction(x=W - 1 - a.x, y=a.y, orientation=a.orientation)

        arrays    = []
        out_legal = []
        flips     = []
        for state, la in zip(states, legal_actions_list):
            # Vertical canonical-frame flip depends only on side-to-move, which
            # the LR mirror below does not change.
            flips.append(not state.is_player1_turn())
            if np.random.random() < 0.5:
                arrays.append(state.to_nn_input())
                out_legal.append(la)
            else:
                arrays.append(flip_nn_input_lr(state.to_nn_input()))
                out_legal.append([mirror_action(a) for a in la])

        return self._inner.batch_call_raw(arrays, out_legal, flips)


def _worker_play_game(args: tuple):
    """
    Top-level worker entry-point for multiprocessing.Pool.

    Runs entirely on CPU — each spawned process owns a private copy of the
    model weights so there is no inter-process GPU contention.  The GPU is
    left free for the training phase in the main process.
    """
    (
        weights, boardsize, filters, num_residual, gpool_every,
        value_head, pawn_head,
        walls, num_sims,
        c_puct, dirichlet_alpha, dirichlet_epsilon,
        dist_bonus_weight_max, n_workers,
    ) = args

    # Limit PyTorch OpenMP threads so n_workers processes × n_threads = cpu_count.
    # Without this every process tries to use all cores → severe contention.
    cpu_count = os.cpu_count() or 1
    n_threads = max(1, cpu_count // n_workers)
    torch.set_num_threads(n_threads)

    device = torch.device("cpu")
    model = DualNetwork(boardsize=boardsize, filters=filters, num_residual=num_residual,
                        gpool_every=gpool_every, value_head=value_head, pawn_head=pawn_head)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # Wrap with SymmetricEvaluator to randomly mirror 50% of queries,
    # preventing the network from developing left-right positional bias.
    evaluator = SymmetricEvaluator(NNEvaluator(model, device=device))
    w1 = random.uniform(0.0, dist_bonus_weight_max)
    w2 = random.uniform(0.0, dist_bonus_weight_max)
    def _make_agent(w: float) -> MCTSAgent:
        return MCTSAgent(
            evaluator         = evaluator,
            num_simulations   = num_sims,
            c_puct            = c_puct,
            training          = True,
            dirichlet_alpha   = dirichlet_alpha,
            dirichlet_epsilon = dirichlet_epsilon,
            dist_bonus_weight = w,
            fpu_reduction     = FPU_REDUCTION,
            sim_batch_size    = MCTS_SIM_BATCH_SIZE,
        )
    agent1 = _make_agent(w1)
    agent2 = _make_agent(w2)
    return self_play_game(agent1, agent2, boardsize, walls, FAST_PLAY_PROB)


def collect_cycle_data(
    model:             DualNetwork,
    num_games:         int,
    boardsize:         int,
    walls:             int,
    num_sims:          int,
    c_puct:            float,
    dirichlet_alpha:   float,
    dirichlet_epsilon: float,
    num_workers:       int = 0,   # 0 → use NUM_WORKERS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Play ``num_games`` self-play games in parallel and return positions as
    stacked numpy arrays plus a stats dict.

    Each worker process runs on CPU with its own copy of the current model
    weights.  Games are dispatched via ``Pool.imap_unordered`` so progress
    is reported as games complete rather than waiting for the whole batch.
    NUM_WORKERS defaults to cpu_count//2 to avoid RAM-swapping when each
    spawned process loads a full Python heap + model copy.
    """
    model.eval()
    # Move weights to CPU before pickling so workers never see CUDA tensors.
    weights = {k: v.cpu() for k, v in model.state_dict().items()}

    n_workers = num_workers if num_workers > 0 else NUM_WORKERS
    task_args = [
        (
            weights, boardsize, model.filters, model.num_residual, model.gpool_every,
            model.value_head, model.pawn_head,
            walls, num_sims,
            c_puct, dirichlet_alpha, dirichlet_epsilon,
            DIST_BONUS_WEIGHT_MAX, n_workers,
        )
        for _ in range(num_games)
    ]

    all_positions: list[tuple] = []
    outcomes     = {0: 0, 1: 0, 2: 0}
    game_lengths: list[int] = []
    walls_per_game: list[int] = []

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for g, (positions, winner, walls_placed) in enumerate(
            pool.imap_unordered(_worker_play_game, task_args)
        ):
            all_positions.extend(positions)
            outcomes[winner] += 1
            plies = len(positions) // 2  # positions includes LR-flip augmentation (2× per ply)
            game_lengths.append(plies)
            walls_per_game.append(walls_placed)

            end = "\n" if g == num_games - 1 else "\r"
            print(
                f"  game {g+1:>4}/{num_games}  "
                f"positions={len(all_positions):>7}  "
                f"P1={outcomes[1]}  P2={outcomes[2]}  draws={outcomes[0]}  "
                f"last={plies:>3} plies",
                end=end, flush=True,
            )
    states   = np.stack([p[0] for p in all_positions])   # (M, 8, N, N)
    policies = np.stack([p[1] for p in all_positions])   # (M, A)
    values   = np.array([p[2] for p in all_positions], dtype=np.float32)  # (M,)

    # Policy entropy: H(π) = -Σ p·log(p) over legal actions (0-prob entries contribute 0)
    # Replace 0s with 1.0 before log so numpy doesn't warn about log(0);
    # those slots are multiplied by 0 in the entropy sum so the result is identical.
    log_p        = np.log(np.where(policies > 0, policies, 1.0))
    mean_entropy = float(-(policies * log_p).sum(axis=1).mean())

    stats = {
        "n_positions":        len(all_positions),
        "mean_length":        float(np.mean(game_lengths)),
        "min_length":         int(np.min(game_lengths)),
        "max_length":         int(np.max(game_lengths)),
        "mean_walls_placed":  float(np.mean(walls_per_game)),
        "mean_policy_entropy": mean_entropy,
        "p1_wins":            outcomes[1],
        "p2_wins":            outcomes[2],
        "draws":              outcomes[0],
        "value_mean":         float(values.mean()),
        "value_std":          float(values.std()),
    }
    return states, policies, values, stats


# ── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(
    model:     DualNetwork,
    states:    torch.Tensor,   # (B, 8, N, N)
    target_pi: torch.Tensor,   # (B, A)  float32; 0 for illegal actions
    target_z:  torch.Tensor,   # (B,)    float32
    use_bf16:  bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined policy + value loss.

    Returns
    -------
    total_loss, policy_loss (detached), value_loss (detached)

    Policy loss
    -----------
    Cross-entropy between the MCTS visit distribution π and the NN's policy
    distribution, computed only over the legal actions for each position.

    Illegal action logits are set to -∞ before log-softmax so that the
    softmax normalises exclusively over legal moves — matching how priors
    are used in ``NNEvaluator.__call__``.

    ``nan_to_num`` guards against 0 x (-inf) = NaN at illegal-action positions
    where the target probability is exactly 0.

    Value loss
    ----------
    MSE between the predicted value (tanh output) and the actual game outcome.

    ``use_bf16``: wraps the forward pass + loss computation in
    ``torch.autocast(dtype=torch.bfloat16)`` (CUDA only; a no-op elsewhere).
    bf16 shares fp32's exponent range (no under/overflow risk like fp16), so
    no loss scaling is needed; model parameters and the optimizer's Adam
    state stay fp32 the whole time (autocast only affects op execution
    dtype, not stored tensors) — this is the standard mixed-precision
    training recipe, not a full fp16 switch.
    """
    device_type = states.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                        enabled=use_bf16 and device_type == "cuda"):
        policy_logits, value = model(states)   # (B, A), (B, 1)

        # ── Policy loss ──────────────────────────────────────────────────────────
        legal_mask    = target_pi > 0                                      # (B, A)
        masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
        log_probs     = F.log_softmax(masked_logits, dim=1)                # (B, A)
        # Guard: 0 * (-inf) = NaN  →  replace with 0 (no contribution)
        loss_p = -(target_pi * log_probs).nan_to_num(0.0).sum(dim=1).mean()

        # ── Value loss ───────────────────────────────────────────────────────────
        loss_v = F.mse_loss(value.squeeze(1), target_z)

        total = loss_p + VALUE_LOSS_WEIGHT * loss_v

    return total, loss_p.detach(), loss_v.detach(), value.detach().squeeze(1)


# ── Training phase ────────────────────────────────────────────────────────────

def run_training_phase(
    model:      DualNetwork,
    optimizer:  torch.optim.Optimizer,
    buffer:     ReplayBuffer,
    steps:      int,
    batch_size: int,
    device:     torch.device,
    weights:    np.ndarray | None = None,
    log_every:  int = 200,
    use_bf16:   bool = False,
    loss_window: int = 100,
) -> dict[str, float]:
    """
    Run ``steps`` gradient-update steps on mini-batches sampled from the
    replay buffer.

    When ``weights`` is provided (a normalized probability array aligned with
    ``buffer.as_arrays()``) positions are sampled according to that
    distribution — typically recency-weighted so that recent cycles are
    overrepresented.  Pass ``weights=None`` for uniform sampling.

    The buffer is flattened to numpy arrays once at the start of the phase;
    all subsequent sampling is O(1) index selection into those arrays.
    ``buf_states``/``buf_policies`` are float16 in storage (see ReplayBuffer);
    each sampled mini-batch is upcast to float32 here before the forward pass.

    Returns mean losses over all steps.
    """
    model.train()
    buf_states, buf_policies, buf_values = buffer.as_arrays()
    N = len(buf_states)

    total_sum = policy_sum = value_sum = 0.0
    value_acc_sum   = 0.0
    value_acc_count = 0

    # Trailing windows for the progress line — a rolling mean over the last
    # `loss_window` steps tracks the *current* loss instead of lagging behind
    # it the way a cumulative-from-step-1 average does. The full-phase means
    # returned below still use the running sums.
    total_win  = deque(maxlen=loss_window)
    policy_win = deque(maxlen=loss_window)
    value_win  = deque(maxlen=loss_window)

    for step in range(1, steps + 1):
        if weights is not None:
            idx = np.random.choice(N, size=batch_size, replace=True, p=weights)
        else:
            idx = np.random.randint(0, N, size=batch_size)
        states    = torch.from_numpy(buf_states[idx]).to(device).float()
        target_pi = torch.from_numpy(buf_policies[idx]).to(device).float()
        target_z  = torch.from_numpy(buf_values[idx]).to(device)

        optimizer.zero_grad()
        loss, lp, lv, val_pred = compute_loss(model, states, target_pi, target_z, use_bf16=use_bf16)
        loss.backward()
        optimizer.step()

        loss_val, lp_val, lv_val = loss.item(), lp.item(), lv.item()
        total_sum  += loss_val
        policy_sum += lp_val
        value_sum  += lv_val

        total_win.append(loss_val)
        policy_win.append(lp_val)
        value_win.append(lv_val)

        # Value accuracy: sign(predicted) == sign(target) for non-draw positions
        non_draw = target_z != 0
        if non_draw.any():
            correct = (val_pred[non_draw].sign() == target_z[non_draw].sign()).float().mean().item()
            value_acc_sum   += correct
            value_acc_count += 1

        if log_every > 0 and step % log_every == 0:
            print(
                f"  step {step:>5}/{steps}  "
                f"loss={sum(total_win)/len(total_win):.4f}  "
                f"policy={sum(policy_win)/len(policy_win):.4f}  "
                f"value={sum(value_win)/len(value_win):.4f}"
            )

    return {
        "total":          total_sum  / steps,
        "policy":         policy_sum / steps,
        "value":          value_sum  / steps,
        "value_accuracy": value_acc_sum / value_acc_count if value_acc_count else float("nan"),
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_training_checkpoint(
    path:      str,
    model:     DualNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cycle:     int,
) -> None:
    """Save full training state (model + optimizer + scheduler + cycle index)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "cycle":           cycle,
        },
        path,
    )


def load_training_checkpoint(
    path:      str,
    model:     DualNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    """
    Load full training state in-place.
    Returns the cycle number to resume from (i.e., completed cycles so far).
    """
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return int(ckpt["cycle"])


def _latest_checkpoint(directory: str) -> str | None:
    """Return the path of the highest-numbered model checkpoint file, or None."""
    ckpts = sorted(Path(directory).glob("cycle_*.pt"))
    return str(ckpts[-1]) if ckpts else None


def _latest_data_cycle(data_dir: str = DATA_DIR) -> int:
    """Return the highest cycle number saved to disk (0 if none)."""
    files = sorted(Path(data_dir).glob("cycle_*.npz"))
    if not files:
        return 0
    # Filename is cycle_NNNN.npz; parse the four-digit number.
    return int(files[-1].stem.split("_")[1])


# ── Self-play data persistence ────────────────────────────────────────────────

def save_cycle_data(
    cycle:    int,
    states:   np.ndarray,
    policies: np.ndarray,
    values:   np.ndarray,
    data_dir: str = DATA_DIR,
) -> str:
    """Save one cycle of self-play data to ``data_dir/cycle_NNNN.npz``."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"cycle_{cycle:04d}.npz")
    np.savez_compressed(path, states=states, policies=policies, values=values)
    return path


def load_buffer_from_disk(
    buffer:   ReplayBuffer,
    data_dir: str = DATA_DIR,
) -> int:
    """
    Populate ``buffer`` from the most recent cycle files on disk.

    Loads up to ``buffer._maxcycles`` cycle files (the most recent ones
    by cycle number) so the in-memory buffer exactly mirrors what would have
    been accumulated if training had not been interrupted.

    Returns the number of cycle files loaded.
    """
    files = sorted(Path(data_dir).glob("cycle_*.npz"))
    to_load = files[-buffer._maxcycles:] if buffer._maxcycles else files
    for f in to_load:
        d = np.load(f)
        buffer.add_cycle(d["states"], d["policies"], d["values"])
    return len(to_load)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaZero training for SigmaQuoridor")
    p.add_argument("--resume",  action="store_true",      help="resume from latest checkpoint")
    p.add_argument("--cycles",  type=int, default=NUM_CYCLES,       metavar="N")
    p.add_argument("--games",   type=int, default=GAMES_PER_CYCLE,  metavar="N",
                   help="self-play games per cycle")
    p.add_argument("--sims",    type=int, default=MCTS_SIMS,        metavar="N",
                   help="MCTS simulations per move")
    p.add_argument("--train-positions", type=int, default=TRAIN_POSITIONS_PER_CYCLE,
                   metavar="N",
                   help="total position-updates per training phase (steps = N // batch)")
    p.add_argument("--batch",   type=int, default=BATCH_SIZE,       metavar="N")
    p.add_argument("--lr",      type=float, default=LEARNING_RATE,  metavar="F",
                   help=f"Adam learning rate (default: {LEARNING_RATE})")
    p.add_argument("--filters", type=int, default=FILTERS,          metavar="N")
    p.add_argument("--res",     type=int, default=NUM_RESIDUAL,      metavar="N",
                   help="number of residual blocks")
    p.add_argument("--gpool-every", type=int, default=GPOOL_EVERY,   metavar="N",
                   help="every k-th residual block is a KataGo-style global-"
                        "pooling block (0 = none); only affects fresh model creation")
    p.add_argument("--buffer-cycles", type=int, default=BUFFER_CYCLES, metavar="N",
                   help=f"keep positions from this many recent self-play cycle files "
                        f"(default: {BUFFER_CYCLES})")
    p.add_argument("--recency-decay", type=float, default=BUFFER_RECENCY_DECAY, metavar="F",
                   help=f"per-cycle sampling weight decay; 1.0 = uniform over the whole "
                        f"buffer, lower = more recency-biased (default: {BUFFER_RECENCY_DECAY}). "
                        f"Lower this toward 1.0 when running many training-only cycles over a "
                        f"buffer that isn't growing (e.g. no self-play), since a static buffer "
                        f"means the same recency-favoured subset gets resampled every cycle "
                        f"instead of the favoured window shifting as fresh data arrives.")
    p.add_argument("--value-head", choices=["pooled", "legacy"], default="pooled",
                   help="value head variant; only affects fresh model creation "
                        "(resuming/loading always reconstructs whichever variant "
                        "the checkpoint was saved with — see dual_network._infer_arch)")
    p.add_argument("--pawn-head", choices=["local", "legacy"], default="local",
                   help="pawn policy sub-head variant; only affects fresh model "
                        "creation, same caveat as --value-head")
    p.add_argument("--bf16",   action="store_true",
                   help="bfloat16 autocast for the training forward/backward pass "
                        "(CUDA only; fp32 master weights/optimizer state unaffected)")
    p.add_argument("--workers", type=int, default=NUM_WORKERS,        metavar="N",
                   help="parallel self-play worker processes (default: cpu count)")
    p.add_argument("--train-only", action="store_true",
                   help="skip self-play; train only on the existing buffer")
    p.add_argument("--selfplay-time-s", type=float, default=None, metavar="SECONDS",
                   help="wall-clock seconds the self-play phase took for this cycle, "
                        "recorded into training_stats.csv. Passed by cpp_train_loop.py "
                        "(which runs self-play as a separate subprocess) so cycle_time_s "
                        "reflects self-play + training, not training alone. Ignored when "
                        "self-play runs in-process (full loop measures it directly).")
    p.add_argument("--smoke-test", action="store_true",
                   help="dry-run: exercise the full pipeline but write no files "
                        "(forces --resume; ignores --cycles, runs exactly 1)")
    p.add_argument("--model-dir", type=str, default=None, metavar="DIR",
                   help=f"override MODEL_DIR (default: {MODEL_DIR!r}); also updates "
                        f"MODEL_PATH/CHECKPOINT_DIR")
    p.add_argument("--data-dir", type=str, default=None, metavar="DIR",
                   help=f"override DATA_DIR (default: inferred from --model-dir by "
                        f"swapping its 'models_' prefix for 'data_', or {DATA_DIR!r} "
                        f"if --model-dir is also omitted)")
    return p.parse_args()


# ── Training stats CSV ───────────────────────────────────────────────────────

_STATS_COLUMNS = [
    # Self-play results
    "cycle",
    "cycle_time_s", "selfplay_time_s", "train_time_s", "cumulative_time_s",
    "p1_wins", "p2_wins", "draws",
    "mean_game_length", "min_game_length", "max_game_length",
    "mean_walls_placed",
    "mean_policy_entropy",
    # Training
    "value_accuracy",
    "loss_total", "loss_policy", "loss_value",
    "holdout_loss_total", "holdout_loss_policy", "holdout_loss_value",
    "train_steps", "cumulative_train_steps",
    # Hyperparameter snapshot (useful when constants are tweaked mid-training)
    "lr",
    "boardsize", "walls_per_player",
    "games_per_cycle", "mcts_sims", "mcts_sims_fast",
    "fast_play_prob", "fast_kl_threshold", "temp_threshold",
    "c_puct", "dirichlet_alpha", "dirichlet_epsilon",
    "dist_bonus_weight_max", "fpu_reduction",
    "random_wall_plies", "random_wall_fraction",
    "buffer_cycles",
    "batch_size", "train_positions_per_cycle", "buffer_recency_decay",
    "learning_rate", "weight_decay", "value_loss_weight",
    "lr_milestones", "lr_decay",
    "filters", "num_residual", "gpool_every", "num_workers",
    "value_head", "pawn_head",
]


_EVAL_COLUMNS = [
    "cycle", "opponent", "eval_sims",
    "n_as_p1", "w_as_p1", "d_as_p1", "l_as_p1",
    "n_as_p2", "w_as_p2", "d_as_p2", "l_as_p2",
    "win_pct",
]


def _append_eval_csv(path: str, rows: list[dict]) -> None:
    """Append eval result rows to eval_results.csv, writing the header if new."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EVAL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _prior_cumulative_totals(path: str) -> tuple[float, int]:
    """Sum ``cycle_time_s``/``train_steps`` across every row already in the
    stats CSV, so cumulative totals stay correct even though cpp_train_loop.py
    invokes train.py as a fresh subprocess once per cycle (no in-memory
    accumulator would survive that) and across --resume runs."""
    if not os.path.exists(path):
        return 0.0, 0
    total_time = 0.0
    total_steps = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("cycle_time_s"):
                total_time += float(row["cycle_time_s"])
            if row.get("train_steps"):
                total_steps += int(row["train_steps"])
    return total_time, total_steps


def _append_stats_csv(path: str, row: dict) -> None:
    """Append one row to the training stats CSV, writing the header if new."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_STATS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Evaluation ───────────────────────────────────────────────────────────────

def _eval_game(agent1, agent2, boardsize: int, walls: int) -> int:
    """Play one deterministic game; return 0 (draw), 1 (P1 win), 2 (P2 win)."""
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
    while not state.is_finished():
        agent  = agent1 if state.is_player1_turn() else agent2
        action = max(agent.get_policy(state), key=lambda x: x[1])[0]
        state  = state.next(action)
    return state.winner()


def _eval_worker(args: tuple) -> int:
    """
    Top-level worker entry-point for parallel evaluation.

    Rebuilds the challenger from a serialised CPU state-dict and constructs
    the opponent from its spec (MCTS opponents are loaded from disk; all
    others are stateless and constructed directly).  Returns the game result
    (0 draw, 1 P1 win, 2 P2 win).
    """
    (
        challenger_sd, eval_sims, c_puct,
        opponent_spec,
        boardsize, walls,
        challenger_is_p1,
        n_workers,
    ) = args

    # Limit threads so n_workers processes don't over-subscribe the CPU.
    cpu_count = os.cpu_count() or 1
    n_threads = max(1, cpu_count // n_workers)
    torch.set_num_threads(n_threads)

    device = torch.device("cpu")

    # Build challenger (reconstruct architecture from weight shapes, same as load_model).
    from dual_network import _infer_arch
    _filters, _res, _bs, _gpool_every, _value_head, _pawn_head = _infer_arch(challenger_sd)
    challenger_model = DualNetwork(boardsize=_bs, filters=_filters, num_residual=_res,
                                  gpool_every=_gpool_every, value_head=_value_head,
                                  pawn_head=_pawn_head).to(device)
    challenger_model.load_state_dict(challenger_sd)
    challenger_model.eval()
    challenger = MCTSAgent(
        evaluator       = NNEvaluator(challenger_model, device=device),
        num_simulations = eval_sims,
        training        = False,
        c_puct          = c_puct,
    )

    # Build opponent.
    t = opponent_spec["type"]
    if t == "random":
        opponent = RandomAgent()
    elif t == "greedy":
        opponent = GreedyDistanceAgent()
    elif t == "minimax":
        opponent = MinimaxAgent(depth=opponent_spec["depth"])
    else:  # mcts
        opp_model = load_model(opponent_spec["path"], device=device)
        opp_model.eval()
        opponent = MCTSAgent(
            evaluator       = NNEvaluator(opp_model, device=device),
            num_simulations = opponent_spec["sims"],
            training        = False,
            c_puct          = c_puct,
        )

    if challenger_is_p1:
        return _eval_game(challenger, opponent, boardsize, walls)
    else:
        return _eval_game(opponent, challenger, boardsize, walls)


def run_evaluation(
    model:     DualNetwork,
    cycle:     int,
    boardsize: int,
    walls:     int,
    eval_sims: int,
    opponents: list[tuple[str, int, dict]],
    csv_path:  str | None = None,
) -> None:
    """
    Pit the current model (challenger at eval_sims MCTS sims) against each
    opponent in parallel and print W/D/L/Win% results.

    Each game is dispatched to a worker process via Pool.imap so all games
    across all opponents run concurrently.  Results are printed as a table
    once all games are done.  Missing MCTS model files cause that opponent
    to be skipped.  Results are appended to ``csv_path`` if provided.
    """
    if not opponents:
        return

    model.eval()
    # Serialise challenger weights to CPU (workers never see CUDA tensors).
    challenger_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    # Check which MCTS opponent files actually exist.
    available_mcts: set[str] = {
        spec["path"]
        for _, _, spec in opponents
        if spec["type"] == "mcts" and Path(spec["path"]).exists()
    }

    # Build a flat ordered task list: one entry per game.
    # Each entry is (label, spec, challenger_is_p1).
    game_tasks: list[tuple[str, dict, bool]] = []
    skipped: list[str] = []
    for label, num_games, spec in opponents:
        if spec["type"] == "mcts" and spec.get("path") not in available_mcts:
            skipped.append(label)
            continue
        n_p1 = (num_games + 1) // 2
        n_p2 = num_games // 2
        for _ in range(n_p1):
            game_tasks.append((label, spec, True))
        for _ in range(n_p2):
            game_tasks.append((label, spec, False))

    n_tasks   = len(game_tasks)
    n_workers = max(1, min(NUM_WORKERS, n_tasks))

    worker_args = [
        (challenger_sd, eval_sims, C_PUCT, spec, boardsize, walls, is_p1, n_workers)
        for _, spec, is_p1 in game_tasks
    ]

    # Per-label result accumulators.
    counters: dict[str, list[int]] = {
        label: [0, 0, 0, 0, 0, 0, 0, 0]   # n_p1 w1 d1 l1  n_p2 w2 d2 l2
        for label, _, _ in opponents
        if label not in skipped
    }

    col = max(len(lbl) for lbl, _, _ in opponents) + 2
    sep = "─" * (col + 52)
    print(f"\n  Evaluation — cycle {cycle}  (challenger: {eval_sims} sims, {n_tasks} games, {n_workers} workers)")
    for lbl in skipped:
        print(f"  {lbl:<{col}} — skipped (model not found)")

    t0 = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for result, (label, spec, is_p1) in zip(
            pool.imap(_eval_worker, worker_args), game_tasks
        ):
            c = counters[label]
            if is_p1:
                c[0] += 1                        # n_p1
                if result == 1:   c[1] += 1      # w1
                elif result == 0: c[2] += 1      # d1
                else:             c[3] += 1      # l1
            else:
                c[4] += 1                        # n_p2
                if result == 2:   c[5] += 1      # w2
                elif result == 0: c[6] += 1      # d2
                else:             c[7] += 1      # l2

    # Print results table.
    print(f"  {sep}")
    print(f"  {'Opponent':<{col}} {'as P1':>14}  {'as P2':>14}  {'Win%':>6}")
    print(f"  {'':.<{col}} {'W / D / L':>14}  {'W / D / L':>14}")
    print(f"  {sep}")

    csv_rows: list[dict] = []
    for label, _, _ in opponents:
        if label in skipped:
            continue
        n_p1, w1, d1, l1, n_p2, w2, d2, l2 = counters[label]
        wins    = w1 + w2
        draws   = d1 + d2
        n_total = n_p1 + n_p2
        win_pct = (wins + 0.5 * draws) / n_total * 100 if n_total else 0.0
        print(f"  {label:<{col}} {f'{w1} / {d1} / {l1}':>14}  {f'{w2} / {d2} / {l2}':>14}  {win_pct:>5.1f}%")
        csv_rows.append({
            "cycle":     cycle,
            "opponent":  label.strip(),
            "eval_sims": eval_sims,
            "n_as_p1": n_p1, "w_as_p1": w1, "d_as_p1": d1, "l_as_p1": l1,
            "n_as_p2": n_p2, "w_as_p2": w2, "d_as_p2": d2, "l_as_p2": l2,
            "win_pct": f"{win_pct:.2f}",
        })

    print(f"  {sep}")
    print(f"  done in {time.perf_counter() - t0:.1f}s\n")

    if csv_path and csv_rows:
        _append_eval_csv(csv_path, csv_rows)


# ── Holdout validation ───────────────────────────────────────────────────────

def load_holdout(
    data_dir:    str,
    n_cycles:    int,
    n_positions: int,
    seed:        int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Sample a fixed holdout set from the last ``n_cycles`` files in ``data_dir``.

    Uses a deterministic seed so the exact same positions are selected on
    every resume.  Returns None if no data files exist in ``data_dir``.
    """
    if not data_dir:
        return None
    files = sorted(Path(data_dir).glob("cycle_*.npz"))
    if not files:
        return None
    all_states, all_policies, all_values = [], [], []
    for f in files[-n_cycles:]:
        d = np.load(f)
        all_states.append(d["states"])
        all_policies.append(d["policies"])
        all_values.append(d["values"])
    states   = np.concatenate(all_states)
    policies = np.concatenate(all_policies)
    values   = np.concatenate(all_values)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(states), size=min(n_positions, len(states)), replace=False)
    return states[idx], policies[idx], values[idx]


def compute_holdout_loss(
    model:      DualNetwork,
    states:     np.ndarray,
    policies:   np.ndarray,
    values:     np.ndarray,
    device:     torch.device,
    batch_size: int = 4096,
) -> dict[str, float]:
    """Evaluate loss on the fixed holdout set without updating weights."""
    model.eval()
    total_sum = policy_sum = value_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            s = torch.from_numpy(states[i : i + batch_size]).to(device)
            p = torch.from_numpy(policies[i : i + batch_size]).to(device)
            v = torch.from_numpy(values[i : i + batch_size]).to(device)
            loss, lp, lv, _ = compute_loss(model, s, p, v)
            total_sum  += loss.item()
            policy_sum += lp.item()
            value_sum  += lv.item()
            n_batches  += 1
    return {
        "total":  total_sum  / n_batches,
        "policy": policy_sum / n_batches,
        "value":  value_sum  / n_batches,
    }


# ── Console logging ──────────────────────────────────────────────────────────

class _Tee:
    """File-like object that duplicates writes to an underlying stream and a
    log file, so every print() during a run is also saved to disk."""

    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._log_file.write(data)
        self._log_file.flush()
        try:
            return self._stream.write(data)
        except UnicodeEncodeError:
            # The console stream's encoding (e.g. Windows cp1252) may not
            # support every character some log messages use (e.g. arrows);
            # the UTF-8 log file above already has the exact text, so degrade
            # gracefully on the console instead of crashing the whole run.
            encoding = getattr(self._stream, "encoding", None) or "ascii"
            safe = data.encode(encoding, errors="replace").decode(encoding)
            return self._stream.write(safe)

    def flush(self) -> None:
        self._log_file.flush()
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


def _start_run_log(log_dir: str) -> str:
    """Tee stdout/stderr to a timestamped log file under log_dir; returns its path."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    # utf-8 explicitly: default encoding on Windows is cp1252, which can't
    # encode characters like the arrow used in log messages below.
    log_file = open(path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return path


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    global MODEL_DIR, MODEL_PATH, CHECKPOINT_DIR, DATA_DIR
    args = parse_args()

    if args.model_dir is not None:
        MODEL_DIR = args.model_dir
        MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
        CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
    if args.data_dir is not None:
        DATA_DIR = args.data_dir
    elif args.model_dir is not None:
        base = os.path.basename(os.path.normpath(MODEL_DIR))
        if base.startswith("models_"):
            DATA_DIR = os.path.join(os.path.dirname(MODEL_DIR), "data_" + base[len("models_"):])
        # else: no naming convention to infer from — keep the default DATA_DIR

    if args.smoke_test:
        args.resume = True   # never overwrite best.pt on startup
        args.cycles = 1      # one cycle is enough to exercise everything
        print("*** SMOKE TEST — no files will be written ***")
        print()
    else:
        log_path = _start_run_log(LOG_DIR)
        print(f"Logging console output to {log_path}")

    os.makedirs(MODEL_DIR,      exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,       exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    model = DualNetwork(
        boardsize    = BOARDSIZE,
        filters      = args.filters,
        num_residual = args.res,
        gpool_every  = args.gpool_every,
        value_head   = args.value_head,
        pawn_head    = args.pawn_head,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_DECAY)

    start_cycle = 0

    buffer = ReplayBuffer(maxcycles=args.buffer_cycles)

    if args.resume:
        latest_ckpt = _latest_checkpoint(CHECKPOINT_DIR)
        if latest_ckpt is not None:
            start_cycle = load_training_checkpoint(latest_ckpt, model, optimizer, scheduler)
            print(f"Resumed model checkpoint from {latest_ckpt}  (completed cycles: {start_cycle})")
            # Recreate the scheduler from the current code's constants so that stale
            # milestones baked into the checkpoint don't interfere.  Patching
            # base_lrs alone is not enough: _get_closed_form_lr() re-derives the LR
            # from base_lrs AND milestones on every scheduler.step() call, so a patched
            # base_lrs with the old milestones would silently produce the wrong LR.
            passed = sum(1 for m in LR_MILESTONES if m <= start_cycle)
            resume_lr = args.lr * (LR_DECAY ** passed)
            scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_DECAY)
            scheduler.last_epoch = start_cycle   # fast-forward without firing milestones
            for pg in optimizer.param_groups:
                pg["lr"] = resume_lr
            scheduler._last_lr = [resume_lr]
        else:
            # No model checkpoint yet, but self-play data files might exist
            # (e.g. stopped before the first checkpoint at cycle 10).
            # Use best.pt weights if available, fall back to random init.
            if Path(MODEL_PATH).exists():
                from dual_network import load_model as _load_weights
                _tmp = _load_weights(MODEL_PATH)
                model.load_state_dict(_tmp.state_dict())
                print(f"No training checkpoint found — loaded weights from {MODEL_PATH}")
            else:
                print("No checkpoint or weights found — starting with random weights.")
        start_cycle = max(start_cycle, _latest_data_cycle(DATA_DIR))
        n_loaded = load_buffer_from_disk(buffer, DATA_DIR)
        print(f"Resuming from cycle {start_cycle + 1}  |  "
              f"Loaded {n_loaded} data file(s) → {buffer.size():,} positions in buffer")
    else:
        print(f"Starting fresh — boardsize={BOARDSIZE}, filters={args.filters}, "
              f"res={args.res}, gpool_every={args.gpool_every}")
        save_model(model, MODEL_PATH)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total_params:,}")
    print(f"Device     : {DEVICE}")
    print(f"Action space: {action_space_size(BOARDSIZE)}")
    print(f"Games/cycle: {args.games}  MCTS sims: {args.sims}  "
          f"Train positions/cycle: {args.train_positions:,}  Batch: {args.batch}")
    print(f"Buffer: {args.buffer_cycles} cycles  |  "
          f"Cycles planned: {start_cycle} → {start_cycle + args.cycles}")
    print()

    # ── Holdout set (fixed sample; loaded once before the training loop) ────
    holdout = load_holdout(HOLDOUT_DIR, HOLDOUT_CYCLES, HOLDOUT_SIZE) if ENABLE_HOLDOUT_EVAL else None
    if holdout is not None:
        print(f"Holdout: {len(holdout[0]):,} positions from last {HOLDOUT_CYCLES} cycles of {HOLDOUT_DIR!r}")
    elif not ENABLE_HOLDOUT_EVAL:
        print("Holdout: disabled (ENABLE_HOLDOUT_EVAL = False)")
    else:
        print(f"Holdout: none (no data found in {HOLDOUT_DIR!r})")
    print()

    for cycle in range(start_cycle, start_cycle + args.cycles):
        t_cycle        = time.perf_counter()
        lr_now         = optimizer.param_groups[0]["lr"]
        sp_stats:       dict | None = None
        sp_time:        float | None = None
        losses:         dict | None = None
        holdout_losses: dict | None = None
        train_steps:    int | None = None
        print(f"{'='*66}")
        print(f"Cycle {cycle + 1}    LR={lr_now:.2e}")
        print(f"{'='*66}")

        # ── 1. Self-play ────────────────────────────────────────────────────
        if args.train_only:
            print("Self-play skipped (--train-only)")
            print(f"Buffer: {buffer.size():,} positions across {buffer.num_cycles()} cycle(s)")
        else:
            print(f"Self-play ({args.games} games x {args.sims} sims/move, "
                  f"{args.workers} workers) ...")
            t0 = time.perf_counter()
            states, policies, values, sp_stats = collect_cycle_data(
                model             = model,
                num_games         = args.games,
                boardsize         = BOARDSIZE,
                walls             = WALLS_PER_PLAYER,
                num_sims          = args.sims,
                c_puct            = C_PUCT,
                dirichlet_alpha   = DIRICHLET_ALPHA,
                dirichlet_epsilon = DIRICHLET_EPSILON,
                num_workers       = args.workers,
            )
            sp_time = time.perf_counter() - t0
            print(
                f"  Done {sp_time:.1f}s | "
                f"{sp_stats['n_positions']} positions | "
                f"mean game length {sp_stats['mean_length']:.1f} | "
                f"mean walls placed {sp_stats['mean_walls_placed']:.1f} | "
                f"P1 {sp_stats['p1_wins']} / P2 {sp_stats['p2_wins']} / "
                f"draws {sp_stats['draws']}"
            )

            # ── 2. Save to disk + add to in-memory buffer ───────────────────
            buffer.add_cycle(states, policies, values)
            if args.smoke_test:
                print(
                    f"Buffer: {buffer.size():,} positions "
                    f"across {buffer.num_cycles()} cycle(s)  "
                    f"[SMOKE TEST — data not saved to disk]"
                )
            else:
                data_path = save_cycle_data(cycle + 1, states, policies, values, DATA_DIR)
                print(
                    f"Buffer: {buffer.size():,} positions "
                    f"across {buffer.num_cycles()} cycle(s)  "
                    f"[saved → {data_path}]"
                )

        # ── 3. Training ─────────────────────────────────────────────────────
        if buffer.size() < MIN_BUFFER_SIZE:
            print(f"Buffer too small ({buffer.size()} < {MIN_BUFFER_SIZE}), skipping.")
        else:
            train_steps = max(1, args.train_positions // args.batch)
            buf_weights = buffer.sampling_weights(args.recency_decay)
            print(f"Training ({train_steps} steps, {args.train_positions:,} positions, "
                  f"batch={args.batch}, recency_decay={args.recency_decay}) ...")
            t0 = time.perf_counter()
            losses = run_training_phase(
                model      = model,
                optimizer  = optimizer,
                buffer     = buffer,
                steps      = train_steps,
                batch_size = args.batch,
                device     = DEVICE,
                weights    = buf_weights,
                use_bf16   = args.bf16,
            )
            train_time = time.perf_counter() - t0
            print(
                f"  Done {train_time:.1f}s | "
                f"loss={losses['total']:.4f}  "
                f"policy={losses['policy']:.4f}  "
                f"value={losses['value']:.4f}"
            )
            scheduler.step()

            # ── 3b. Holdout loss ─────────────────────────────────────────────
            if holdout is not None:
                holdout_losses = compute_holdout_loss(model, *holdout, DEVICE)
                print(
                    f"  Holdout  | "
                    f"loss={holdout_losses['total']:.4f}  "
                    f"policy={holdout_losses['policy']:.4f}  "
                    f"value={holdout_losses['value']:.4f}"
                )

        # ── 4. Checkpoint ───────────────────────────────────────────────────
        if args.smoke_test:
            print("[SMOKE TEST — model weights and checkpoint not saved]")
        else:
            # Always overwrite the inference weights for the web app to pick up.
            save_model(model, MODEL_PATH)

            # Full training checkpoint every N cycles.
            if (cycle + 1) % CHECKPOINT_EVERY == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"cycle_{cycle+1:04d}.pt")
                save_training_checkpoint(ckpt_path, model, optimizer, scheduler, cycle + 1)
                print(f"Saved  {MODEL_PATH}  +  checkpoint → {ckpt_path}")
            else:
                print(f"Saved  {MODEL_PATH}")

        proc_time = time.perf_counter() - t_cycle

        # Split the cycle into self-play + training time. In the full in-process
        # loop, proc_time already includes self-play (sp_time), so training is the
        # remainder. Under --train-only, self-play ran in a separate subprocess
        # (cpp_train_loop.py) and its wall time arrives via --selfplay-time-s;
        # proc_time is training-only, so the two are added. cycle_time_s is always
        # the true total so cumulative_time_s (and _prior_cumulative_totals) stay
        # correct regardless of which path produced the row.
        if sp_time is not None:                    # self-play ran in this process
            selfplay_time = sp_time
            train_time    = proc_time - sp_time
        else:                                      # --train-only (self-play elsewhere)
            selfplay_time = args.selfplay_time_s   # may be None if not supplied
            train_time    = proc_time
        cycle_time = train_time + (selfplay_time or 0.0)

        # ── 5. Stats CSV ─────────────────────────────────────────────────────
        if not args.smoke_test:
            stats_path = os.path.join(MODEL_DIR, "training_stats.csv")
            prior_time, prior_steps = _prior_cumulative_totals(stats_path)
            _append_stats_csv(stats_path, {
                # Self-play results
                "cycle":               cycle + 1,
                "cycle_time_s":        f"{cycle_time:.1f}",
                "selfplay_time_s":     f"{selfplay_time:.1f}" if selfplay_time is not None else "",
                "train_time_s":        f"{train_time:.1f}",
                "cumulative_time_s":   f"{prior_time + cycle_time:.1f}",
                "p1_wins":             sp_stats["p1_wins"]            if sp_stats else "",
                "p2_wins":             sp_stats["p2_wins"]            if sp_stats else "",
                "draws":               sp_stats["draws"]              if sp_stats else "",
                "mean_game_length":    f"{sp_stats['mean_length']:.2f}"        if sp_stats else "",
                "min_game_length":     sp_stats["min_length"]                  if sp_stats else "",
                "max_game_length":     sp_stats["max_length"]                  if sp_stats else "",
                "mean_walls_placed":   f"{sp_stats['mean_walls_placed']:.2f}"  if sp_stats else "",
                "mean_policy_entropy": f"{sp_stats['mean_policy_entropy']:.4f}" if sp_stats else "",
                # Training
                "value_accuracy":      f"{losses['value_accuracy']:.4f}" if losses else "",
                "loss_total":          f"{losses['total']:.6f}"         if losses else "",
                "loss_policy":         f"{losses['policy']:.6f}"        if losses else "",
                "loss_value":          f"{losses['value']:.6f}"         if losses else "",
                "holdout_loss_total":  f"{holdout_losses['total']:.6f}"  if holdout_losses else "",
                "holdout_loss_policy": f"{holdout_losses['policy']:.6f}" if holdout_losses else "",
                "holdout_loss_value":  f"{holdout_losses['value']:.6f}"  if holdout_losses else "",
                "train_steps":            train_steps if train_steps else "",
                "cumulative_train_steps": (prior_steps + train_steps) if train_steps else "",
                # Hyperparameter snapshot
                "lr":                        f"{lr_now:.2e}",
                "boardsize":                 BOARDSIZE,
                "walls_per_player":          WALLS_PER_PLAYER,
                "games_per_cycle":           args.games,
                "mcts_sims":                 args.sims,
                "mcts_sims_fast":            MCTS_SIMS_FAST,
                "fast_play_prob":            FAST_PLAY_PROB,
                "fast_kl_threshold":         FAST_KL_THRESHOLD,
                # Column name kept as "temp_threshold" so existing
                # training_stats.csv files stay column-aligned; value now
                # describes the KataGo-style selection-temperature schedule.
                "temp_threshold":            f"sched({TEMP_EARLY}->{TEMP_FINAL},hl={TEMP_HALFLIFE})",
                "c_puct":                    C_PUCT,
                "dirichlet_alpha":           DIRICHLET_ALPHA,
                "dirichlet_epsilon":         DIRICHLET_EPSILON,
                "dist_bonus_weight_max":     DIST_BONUS_WEIGHT_MAX,
                "fpu_reduction":             FPU_REDUCTION,
                "random_wall_plies":         RANDOM_WALL_PLIES,
                "random_wall_fraction":      RANDOM_WALL_FRACTION,
                "buffer_cycles":             args.buffer_cycles,
                "batch_size":               args.batch,
                "train_positions_per_cycle": args.train_positions,
                "buffer_recency_decay":      args.recency_decay,
                "learning_rate":             args.lr,
                "weight_decay":              WEIGHT_DECAY,
                "value_loss_weight":         VALUE_LOSS_WEIGHT,
                "lr_milestones":             str(LR_MILESTONES),
                "lr_decay":                  LR_DECAY,
                "filters":                   args.filters,
                "num_residual":              args.res,
                "gpool_every":               args.gpool_every,
                "num_workers":               args.workers,
                "value_head":                args.value_head,
                "pawn_head":                 args.pawn_head,
            })

        print(f"Cycle {cycle + 1} complete in {cycle_time:.1f}s\n")

        # ── 6. Evaluation ───────────────────────────────────────────────────
        if (
            EVAL_EVERY > 0
            and not args.smoke_test
            and (cycle + 1) % EVAL_EVERY == 0
        ):
            run_evaluation(
                model     = model,
                cycle     = cycle + 1,
                boardsize = BOARDSIZE,
                walls     = WALLS_PER_PLAYER,
                eval_sims = EVAL_SIMS,
                opponents = EVAL_OPPONENTS,
                csv_path  = os.path.join(MODEL_DIR, "eval_results.csv"),
            )


if __name__ == "__main__":
    main()
