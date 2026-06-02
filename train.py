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
import multiprocessing
import os
import random
import time
from collections import deque
from pathlib import Path

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from dual_network import DEVICE, DualNetwork, NNEvaluator, load_model, save_model
from game import State, WallAction, action_space_size, action_to_index, flip_nn_input_lr, flip_policy_lr
from mcts import MCTSAgent

# ── Default hyper-parameters ────────────────────────────────────────────────
# Override any of these via CLI flags (see parse_args()).

# Board
BOARDSIZE        = 9
WALLS_PER_PLAYER = 10

# Self-play
GAMES_PER_CYCLE   = 100     # self-play games played per cycle
MCTS_SIMS         = 800   # MCTS simulations per move during self-play
NUM_WORKERS       = os.cpu_count() or 1  # parallel self-play processes
C_PUCT            = 1.0
DIRICHLET_ALPHA   = 0.3
DIRICHLET_EPSILON = 0.30
DIST_BONUS_WEIGHT_MAX = 2.0  # each game samples w1, w2 ~ Uniform[0, MAX] independently per side
FPU_REDUCTION     = 0.2    # First Play Urgency: unvisited child Q estimate = parent_Q - FPU
TEMP_THRESHOLD    = 20     # plies ≤ this use τ=1.0 (exploration); > uses τ=0.0
FAST_PLAY_PROB      = 0.3  # fraction of moves that use fast MCTS; remainder use full search
MCTS_SIMS_FAST      = 128   # simulations for fast plies (2 NN batches); not saved unless surprising
FAST_KL_THRESHOLD   = 0.7   # KL(visit_dist ∥ prior) nats — fast positions above this are saved
MCTS_SIM_BATCH_SIZE = 1     # legacy sequential mode: one simulation/eval at a time

# Random wall pre-fill — each player places this many walls randomly before
# MCTS self-play begins, leaving them with 0 walls for actual play.
# Applied to RANDOM_WALL_FRACTION of games; the other (1 - RANDOM_WALL_FRACTION) start normally with WALLS_PER_PLAYER.
# Set to 0 to disable entirely.
RANDOM_WALL_PLIES = 10      # walls placed per player in pre-filled games
RANDOM_WALL_FRACTION = 0.0  # fraction of games to apply random wall pre-fill to

# Replay buffer
BUFFER_CYCLES = 40         # keep positions from this many recent cycles

# Training
BATCH_SIZE                 = 256
TRAIN_POSITIONS_PER_CYCLE  = 100_000  # gradient updates = this // BATCH_SIZE per cycle
BUFFER_RECENCY_DECAY       = 0.90     # per-cycle weight decay; 1.0 = uniform, lower = more recency. BUFFER_RECENCY_DECAY^CYCLE_AGE = relative weight of positions from a cycle CYCLE_AGE cycles ago when sampling training batches. 0.9^5 = 0.59, 0.9^10 = 0.35, 0.9^20 = 0.12, 0.9^40 = 0.01
MIN_BUFFER_SIZE            = BATCH_SIZE

# Optimizer
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 1e-4
VALUE_LOSS_WEIGHT = 1.5         # multiply value MSE loss (KataGo uses ~1.5)
LR_MILESTONES  = [800, 1600]    # cycle numbers at which to multiply LR by LR_DECAY
LR_DECAY       = 0.1

# Network
FILTERS      = 128
NUM_RESIDUAL = 10

# Paths
MODEL_DIR      = "models_9x9"                               # model checkpoints and best.pt weights
MODEL_PATH     = os.path.join(MODEL_DIR, "best.pt")         # inference weights only
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")     # full training state
DATA_DIR       = "data_9x9"                                 # persisted self-play cycles

# Run
NUM_CYCLES        = 100
CHECKPOINT_EVERY  = 1   # save a full checkpoint every N cycles


# ── Replay buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores positions from the last ``maxcycles`` self-play cycles.

    Each cycle is stored as three pre-stacked numpy arrays so that
    concatenating the full buffer is a single ``np.concatenate`` call.

    Per-position arrays
    -------------------
    state  : (8, N, N) float32  — ``State.to_nn_input()`` output
    policy : (A,)      float32  — full action-space MCTS visit distribution;
                                  0.0 for illegal actions
    value  : float32            — game outcome from the current player's POV
                                  (+1 win, -1 loss, 0 draw)
    """

    def __init__(self, maxcycles: int) -> None:
        self._cycles: deque[dict] = deque(maxlen=maxcycles)

    def add_cycle(
        self,
        states:   np.ndarray,   # (M, 8, N, N)
        policies: np.ndarray,   # (M, A)
        values:   np.ndarray,   # (M,)
    ) -> None:
        self._cycles.append({"states": states, "policies": policies, "values": values})

    def size(self) -> int:
        return sum(c["states"].shape[0] for c in self._cycles)

    def num_cycles(self) -> int:
        return len(self._cycles)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate all cycles into three flat arrays for batch sampling."""
        states   = np.concatenate([c["states"]   for c in self._cycles])
        policies = np.concatenate([c["policies"] for c in self._cycles])
        values   = np.concatenate([c["values"]   for c in self._cycles])
        return states, policies, values

    def sampling_weights(self, recency_decay: float = 0.9) -> np.ndarray:
        """
        Per-position sampling probabilities that favour more recent cycles.

        Cycle at index i (0 = oldest, n-1 = newest) receives unnormalized
        weight ``recency_decay ** (n - 1 - i)``.  All positions within a
        cycle share the same weight.  The returned array is normalized to
        sum to 1 and is aligned with the arrays from ``as_arrays()``.

        ``recency_decay = 1.0`` reproduces uniform sampling.
        """
        n = len(self._cycles)
        weights: list[np.ndarray] = []
        for i, cycle in enumerate(self._cycles):
            w = recency_decay ** (n - 1 - i)
            weights.append(np.full(cycle["states"].shape[0], w, dtype=np.float64))
        flat = np.concatenate(weights)
        flat /= flat.sum()
        return flat


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


def self_play_game(
    agent1:         MCTSAgent,
    agent2:         MCTSAgent,
    boardsize:      int,
    walls:          int,
    temp_threshold: int,
    fast_play_prob: float = FAST_PLAY_PROB,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.float32]], int]:
    """
    Play one full self-play game and return collected positions + winner.

    Temperature schedule
    --------------------
    Plies ≤ temp_threshold : τ=1.0 — sample proportionally from visit counts
                             (encourages diverse exploration in the opening).
    Plies  > temp_threshold : τ=0.0 — deterministic argmax (sharpens endgame
                             play and produces cleaner value targets).

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
        agent.temperature = 1.0 if state.depth < temp_threshold else 0.0

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
                counts = [c.visit_count for c in children]
                if agent.temperature == 0.0:
                    best  = counts.index(max(counts))
                    probs = [1.0 if i == best else 0.0 for i in range(len(children))]
                else:
                    inv_t = 1.0 / agent.temperature
                    raw   = [c ** inv_t for c in counts]
                    total = sum(raw)
                    probs = [r / total for r in raw] if total > 0 else [1.0 / len(raw)] * len(raw)
                policy_vec = np.zeros(A, dtype=np.float32)
                for child, prob in zip(children, probs):
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
            # Run MCTS.  get_policy() calls search() once and returns a
            # (sorted) list of (Action, probability) pairs.
            mcts_policy = agent.get_policy(state)

            # Build a full-size policy vector (zeros at illegal action indices).
            policy_vec = np.zeros(A, dtype=np.float32)
            for action, prob in mcts_policy:
                policy_vec[action_to_index(action, boardsize)] = prob

            history.append((
                state.to_nn_input(),         # (8, N, N) float32
                policy_vec,                  # (A,)      float32
                state.get_current_player(),  # 1 or 2
            ))

            # Sample the next action from the MCTS distribution.
            actions = [a for a, _ in mcts_policy]
            probs   = np.array([p for _, p in mcts_policy], dtype=np.float64)
            probs  /= probs.sum()       # re-normalise to guard against float error
            chosen  = int(np.random.choice(len(actions), p=probs))
            state   = state.next(actions[chosen])

    # Retroactively assign value targets; store original AND mirror each position.
    winner    = state.winner()
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

    return positions, winner


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
        for state, la in zip(states, legal_actions_list):
            if np.random.random() < 0.5:
                arrays.append(state.to_nn_input())
                out_legal.append(la)
            else:
                arrays.append(flip_nn_input_lr(state.to_nn_input()))
                out_legal.append([mirror_action(a) for a in la])

        return self._inner.batch_call_raw(arrays, out_legal)


def _worker_play_game(args: tuple):
    """
    Top-level worker entry-point for multiprocessing.Pool.

    Runs entirely on CPU — each spawned process owns a private copy of the
    model weights so there is no inter-process GPU contention.  The GPU is
    left free for the training phase in the main process.
    """
    (
        weights, boardsize, filters, num_residual,
        walls, num_sims, temp_threshold,
        c_puct, dirichlet_alpha, dirichlet_epsilon,
        dist_bonus_weight_max, n_workers,
    ) = args

    # Limit PyTorch OpenMP threads so n_workers processes × n_threads = cpu_count.
    # Without this every process tries to use all cores → severe contention.
    cpu_count = os.cpu_count() or 1
    n_threads = max(1, cpu_count // n_workers)
    torch.set_num_threads(n_threads)

    device = torch.device("cpu")
    model = DualNetwork(boardsize=boardsize, filters=filters, num_residual=num_residual)
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
    return self_play_game(agent1, agent2, boardsize, walls, temp_threshold, FAST_PLAY_PROB)


def collect_cycle_data(
    model:             DualNetwork,
    num_games:         int,
    boardsize:         int,
    walls:             int,
    num_sims:          int,
    temp_threshold:    int,
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
            weights, boardsize, model.filters, model.num_residual,
            walls, num_sims, temp_threshold,
            c_puct, dirichlet_alpha, dirichlet_epsilon,
            DIST_BONUS_WEIGHT_MAX, n_workers,
        )
        for _ in range(num_games)
    ]

    all_positions: list[tuple] = []
    outcomes  = {0: 0, 1: 0, 2: 0}
    game_lengths: list[int] = []

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for g, (positions, winner) in enumerate(
            pool.imap_unordered(_worker_play_game, task_args)
        ):
            all_positions.extend(positions)
            outcomes[winner] += 1
            plies = len(positions) // 2  # positions includes LR-flip augmentation (2× per ply)
            game_lengths.append(plies)

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

    stats = {
        "n_positions": len(all_positions),
        "mean_length": float(np.mean(game_lengths)),
        "p1_wins":     outcomes[1],
        "p2_wins":     outcomes[2],
        "draws":       outcomes[0],
        "value_mean":  float(values.mean()),
        "value_std":   float(values.std()),
    }
    return states, policies, values, stats


# ── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(
    model:     DualNetwork,
    states:    torch.Tensor,   # (B, 8, N, N)
    target_pi: torch.Tensor,   # (B, A)  float32; 0 for illegal actions
    target_z:  torch.Tensor,   # (B,)    float32
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
    """
    policy_logits, value = model(states)   # (B, A), (B, 1)

    # ── Policy loss ──────────────────────────────────────────────────────────
    legal_mask    = target_pi > 0                                      # (B, A)
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
    log_probs     = F.log_softmax(masked_logits, dim=1)                # (B, A)
    # Guard: 0 * (-inf) = NaN  →  replace with 0 (no contribution)
    loss_p = -(target_pi * log_probs).nan_to_num(0.0).sum(dim=1).mean()

    # ── Value loss ───────────────────────────────────────────────────────────
    loss_v = F.mse_loss(value.squeeze(1), target_z)

    return loss_p + VALUE_LOSS_WEIGHT * loss_v, loss_p.detach(), loss_v.detach()


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

    Returns mean losses over all steps.
    """
    model.train()
    buf_states, buf_policies, buf_values = buffer.as_arrays()
    N = len(buf_states)

    total_sum = policy_sum = value_sum = 0.0

    for step in range(1, steps + 1):
        if weights is not None:
            idx = np.random.choice(N, size=batch_size, replace=True, p=weights)
        else:
            idx = np.random.randint(0, N, size=batch_size)
        states    = torch.from_numpy(buf_states[idx]).to(device)
        target_pi = torch.from_numpy(buf_policies[idx]).to(device)
        target_z  = torch.from_numpy(buf_values[idx]).to(device)

        optimizer.zero_grad()
        loss, lp, lv = compute_loss(model, states, target_pi, target_z)
        loss.backward()
        optimizer.step()

        total_sum  += loss.item()
        policy_sum += lp.item()
        value_sum  += lv.item()

        if log_every > 0 and step % log_every == 0:
            print(
                f"  step {step:>5}/{steps}  "
                f"loss={total_sum/step:.4f}  "
                f"policy={policy_sum/step:.4f}  "
                f"value={value_sum/step:.4f}"
            )

    return {
        "total":  total_sum  / steps,
        "policy": policy_sum / steps,
        "value":  value_sum  / steps,
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

    Loads up to ``buffer._cycles.maxlen`` cycle files (the most recent ones
    by cycle number) so the in-memory buffer exactly mirrors what would have
    been accumulated if training had not been interrupted.

    Returns the number of cycle files loaded.
    """
    files = sorted(Path(data_dir).glob("cycle_*.npz"))
    to_load = files[-buffer._cycles.maxlen:] if buffer._cycles.maxlen else files
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
    p.add_argument("--filters", type=int, default=FILTERS,          metavar="N")
    p.add_argument("--res",     type=int, default=NUM_RESIDUAL,      metavar="N",
                   help="number of residual blocks")
    p.add_argument("--workers", type=int, default=NUM_WORKERS,        metavar="N",
                   help="parallel self-play worker processes (default: cpu count)")
    p.add_argument("--train-only", action="store_true",
                   help="skip self-play; train only on the existing buffer")
    p.add_argument("--smoke-test", action="store_true",
                   help="dry-run: exercise the full pipeline but write no files "
                        "(forces --resume; ignores --cycles, runs exactly 1)")
    return p.parse_args()


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.smoke_test:
        args.resume = True   # never overwrite best.pt on startup
        args.cycles = 1      # one cycle is enough to exercise everything
        print("*** SMOKE TEST — no files will be written ***")
        print()

    os.makedirs(MODEL_DIR,      exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,       exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    model = DualNetwork(
        boardsize    = BOARDSIZE,
        filters      = args.filters,
        num_residual = args.res,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_DECAY)

    start_cycle = 0

    buffer = ReplayBuffer(maxcycles=BUFFER_CYCLES)

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
            resume_lr = LEARNING_RATE * (LR_DECAY ** passed)
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
        start_cycle = max(start_cycle, _latest_data_cycle())
        n_loaded = load_buffer_from_disk(buffer)
        print(f"Resuming from cycle {start_cycle + 1}  |  "
              f"Loaded {n_loaded} data file(s) → {buffer.size():,} positions in buffer")
    else:
        print(f"Starting fresh — boardsize={BOARDSIZE}, filters={args.filters}, res={args.res}")
        save_model(model, MODEL_PATH)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total_params:,}")
    print(f"Device     : {DEVICE}")
    print(f"Action space: {action_space_size(BOARDSIZE)}")
    print(f"Games/cycle: {args.games}  MCTS sims: {args.sims}  "
          f"Train positions/cycle: {args.train_positions:,}  Batch: {args.batch}")
    print(f"Buffer: {BUFFER_CYCLES} cycles  |  "
          f"Cycles planned: {start_cycle} → {start_cycle + args.cycles}")
    print()

    for cycle in range(start_cycle, start_cycle + args.cycles):
        t_cycle = time.perf_counter()
        lr_now  = optimizer.param_groups[0]["lr"]
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
                temp_threshold    = TEMP_THRESHOLD,
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
                data_path = save_cycle_data(cycle + 1, states, policies, values)
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
            buf_weights = buffer.sampling_weights(BUFFER_RECENCY_DECAY)
            print(f"Training ({train_steps} steps, {args.train_positions:,} positions, "
                  f"batch={args.batch}, recency_decay={BUFFER_RECENCY_DECAY}) ...")
            t0 = time.perf_counter()
            losses = run_training_phase(
                model      = model,
                optimizer  = optimizer,
                buffer     = buffer,
                steps      = train_steps,
                batch_size = args.batch,
                device     = DEVICE,
                weights    = buf_weights,
            )
            train_time = time.perf_counter() - t0
            print(
                f"  Done {train_time:.1f}s | "
                f"loss={losses['total']:.4f}  "
                f"policy={losses['policy']:.4f}  "
                f"value={losses['value']:.4f}"
            )
            scheduler.step()

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

        cycle_time = time.perf_counter() - t_cycle
        print(f"Cycle {cycle + 1} complete in {cycle_time:.1f}s\n")


if __name__ == "__main__":
    main()
