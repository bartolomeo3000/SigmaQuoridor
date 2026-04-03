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
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from dual_network import DEVICE, DualNetwork, NNEvaluator, load_model, save_model
from game import State, action_space_size, action_to_index, flip_nn_input_lr, flip_policy_lr
from mcts import MCTSAgent

# ── Default hyper-parameters ────────────────────────────────────────────────
# Override any of these via CLI flags (see parse_args()).

# Board
BOARDSIZE        = 7
WALLS_PER_PLAYER = 5

# Self-play
GAMES_PER_CYCLE   = 400    # self-play games played per cycle
MCTS_SIMS         = 800    # MCTS simulations per move during self-play
NUM_WORKERS       = os.cpu_count() or 1  # parallel self-play processes
C_PUCT            = 1.0
DIRICHLET_ALPHA   = 0.3
DIRICHLET_EPSILON = 0.25
TEMP_THRESHOLD    = 16     # plies ≤ this use τ=1.0 (exploration); > uses τ=0.0

# Replay buffer
BUFFER_CYCLES = 15         # keep positions from this many recent cycles

# Training
BATCH_SIZE      = 256
TRAIN_EPOCHS    = 2        # passes over the full buffer per cycle
MIN_BUFFER_SIZE = BATCH_SIZE

# Optimizer
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
LR_MILESTONES  = [100, 200]   # cycle numbers at which to multiply LR by LR_DECAY
LR_DECAY       = 0.1

# Network
FILTERS      = 64
NUM_RESIDUAL = 6

# Paths
MODEL_DIR      = "models"
MODEL_PATH     = os.path.join(MODEL_DIR, "best.pt")       # inference weights only
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")   # full training state
DATA_DIR       = "data"                                    # persisted self-play cycles

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


# ── Self-play ────────────────────────────────────────────────────────────────

def self_play_game(
    agent:          MCTSAgent,
    boardsize:      int,
    walls:          int,
    temp_threshold: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.float32]], int]:
    """
    Play one full self-play game and return collected positions + winner.

    Temperature schedule
    --------------------
    Plies ≤ temp_threshold : τ=1.0 — sample proportionally from visit counts
                             (encourages diverse exploration in the opening).
    Plies  > temp_threshold : τ=0.0 — deterministic argmax (sharpens endgame
                             play and produces cleaner value targets).

    Returns
    -------
    positions : list of (state_tensor, policy_vector, value_target)
                value_target is assigned retroactively after the game ends:
                +1.0 if the player to move at that state won, -1.0 if lost,
                0.0 for a draw.
    winner    : 0 (draw), 1 (Player 1), or 2 (Player 2)
    """
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
    A     = action_space_size(boardsize)
    history: list[tuple[np.ndarray, np.ndarray, int]] = []

    while not state.is_finished():
        agent.temperature = 1.0 if state.depth < temp_threshold else 0.0

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
        n_workers,
    ) = args

    # Limit PyTorch OpenMP threads so n_workers processes × n_threads = cpu_count.
    # Without this every process tries to use all cores → severe contention.
    cpu_count = os.cpu_count() or 1
    n_threads = max(1, cpu_count // n_workers)
    torch.set_num_threads(n_threads)

    cpu = torch.device("cpu")
    model = DualNetwork(boardsize=boardsize, filters=filters, num_residual=num_residual)
    model.load_state_dict(weights)
    model.eval()

    # Wrap with SymmetricEvaluator to randomly mirror 50% of queries,
    # preventing the network from developing left-right positional bias.
    evaluator = SymmetricEvaluator(NNEvaluator(model, device=cpu))
    agent = MCTSAgent(
        evaluator         = evaluator,
        num_simulations   = num_sims,
        c_puct            = c_puct,
        training          = True,
        dirichlet_alpha   = dirichlet_alpha,
        dirichlet_epsilon = dirichlet_epsilon,
    )
    return self_play_game(agent, boardsize, walls, temp_threshold)


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
            n_workers,
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

    return loss_p + loss_v, loss_p.detach(), loss_v.detach()


# ── Training phase ────────────────────────────────────────────────────────────

def run_training_phase(
    model:     DualNetwork,
    optimizer: torch.optim.Optimizer,
    buffer:    ReplayBuffer,
    steps:     int,
    batch_size: int,
    device:    torch.device,
    log_every: int = 200,
) -> dict[str, float]:
    """
    Run ``steps`` gradient-update steps on mini-batches sampled uniformly
    from the full replay buffer.

    The buffer is flattened to numpy arrays once at the start of the phase;
    all subsequent sampling is O(1) index selection into those arrays.

    Returns mean losses over all steps.
    """
    model.train()
    buf_states, buf_policies, buf_values = buffer.as_arrays()
    N = len(buf_states)

    total_sum = policy_sum = value_sum = 0.0

    for step in range(1, steps + 1):
        idx       = np.random.randint(0, N, size=batch_size)
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
    p.add_argument("--epochs",  type=float, default=TRAIN_EPOCHS,   metavar="E",
                   help="passes over the full buffer per training phase")
    p.add_argument("--batch",   type=int, default=BATCH_SIZE,       metavar="N")
    p.add_argument("--filters", type=int, default=FILTERS,          metavar="N")
    p.add_argument("--res",     type=int, default=NUM_RESIDUAL,      metavar="N",
                   help="number of residual blocks")
    p.add_argument("--workers", type=int, default=NUM_WORKERS,        metavar="N",
                   help="parallel self-play worker processes (default: cpu count)")
    p.add_argument("--train-only", action="store_true",
                   help="skip self-play; train only on the existing buffer")
    return p.parse_args()


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

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
                _load_weights(MODEL_PATH, model)
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
          f"Train epochs: {args.epochs}  Batch: {args.batch}")
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
            data_path = save_cycle_data(cycle + 1, states, policies, values)
            buffer.add_cycle(states, policies, values)
            print(
                f"Buffer: {buffer.size():,} positions "
                f"across {buffer.num_cycles()} cycle(s)  "
                f"[saved → {data_path}]"
            )

        # ── 3. Training ─────────────────────────────────────────────────────
        if buffer.size() < MIN_BUFFER_SIZE:
            print(f"Buffer too small ({buffer.size()} < {MIN_BUFFER_SIZE}), skipping.")
        else:
            train_steps = max(1, int(args.epochs * buffer.size() / args.batch))
            print(f"Training ({train_steps} steps = {args.epochs} epoch(s) over "
                  f"{buffer.size():,} positions, batch={args.batch}) ...")
            t0 = time.perf_counter()
            losses = run_training_phase(
                model      = model,
                optimizer  = optimizer,
                buffer     = buffer,
                steps      = train_steps,
                batch_size = args.batch,
                device     = DEVICE,
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
