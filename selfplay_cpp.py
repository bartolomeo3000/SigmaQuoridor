"""C++-backed self-play data generation.

Worker threads in the quoridor_cpp extension play many games concurrently
(leaf-parallel MCTS, virtual loss, no GIL). This script runs the single
inference loop: pull large batches of leaf positions, evaluate them on the
GPU, push results back. Output .npz files are compatible with train.py.

Example:
    python setup_cpp.py build_ext --inplace
    python selfplay_cpp.py --games 500 --sims 64 --model models_7x7_v2/best.pt \
        --out-dir data_7x7_v2 --threads 8 --parallel 128 --max-batch 512
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import torch

import quoridor_cpp
from dual_network import DualNetwork, load_model
from game import flip_policy_lr

LOG_DIR = "logs"   # per-run console transcripts (same convention as train.py)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_flip_permutation(boardsize: int) -> np.ndarray:
    """Action-index permutation for a left-right board flip."""
    a = quoridor_cpp.action_space_size(boardsize)
    return flip_policy_lr(np.arange(a, dtype=np.float32), boardsize).astype(np.int64)


def next_cycle_path(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    best = 0
    for name in os.listdir(out_dir):
        m = re.fullmatch(r"cycle_(\d+)\.npz", name)
        if m:
            best = max(best, int(m.group(1)))
    return os.path.join(out_dir, f"cycle_{best + 1:04d}.npz")


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
        return self._stream.write(data)

    def flush(self) -> None:
        self._log_file.flush()
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


def _start_run_log(log_dir: str) -> str:
    """Tee stdout/stderr to a timestamped log file under log_dir; returns its path."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"selfplay_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(path, "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--threads", type=int, default=7)
    p.add_argument("--parallel", type=int, default=1024)
    p.add_argument("--leaf-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=512)
    p.add_argument("--flush-us", type=int, default=100)
    p.add_argument("--model", type=str, default=None,
                   help="checkpoint path; fresh random net if omitted")
    p.add_argument("--boardsize", type=int, default=7)
    p.add_argument("--walls", type=int, default=5)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--res", type=int, default=6)
    p.add_argument("--out-dir", type=str, default=None,
                   help="save cycle_NNNN.npz here (skip saving if omitted)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    p.add_argument("--fpu", type=float, default=0.1)
    p.add_argument("--temp-threshold", type=int, default=20)
    p.add_argument("--dist-bonus-max", type=float, default=0.0)
    p.add_argument("--max-moves", type=int, default=80)
    p.add_argument("--tt-max-depth", type=int, default=-1,
                   help="cache NN outputs for states at depth <= this "
                        "(shared across all parallel games); 0 disables the "
                        "TT entirely; negative means unlimited depth (no ceiling)")
    p.add_argument("--tt-max-entries", type=int, default=2_000_000,
                   help="hard cap on total cached TT entries across all shards "
                        "(bounds worst-case memory); 0 disables the TT entirely; "
                        "negative means unlimited (no cap)")
    p.add_argument("--no-augment", action="store_true",
                   help="skip left-right flip augmentation")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model")
    p.add_argument("--bf16", action="store_true",
                   help="bfloat16 autocast (CUDA only)")
    args = p.parse_args()

    log_path = _start_run_log(LOG_DIR)
    print(f"Logging console output to {log_path}")

    device = pick_device()
    if args.model:
        model = load_model(args.model, device=device)
        print(f"loaded model {args.model}")
    else:
        model = DualNetwork(boardsize=args.boardsize, filters=args.filters,
                            num_residual=args.res).to(device)
        print("using fresh random network")
    model.eval()
    if args.compile:
        model = torch.compile(model)
    use_bf16 = args.bf16 and device.type == "cuda"
    print(f"device={device}  bf16={use_bf16}  compiled={args.compile}")

    mgr = quoridor_cpp.SelfPlayManager(
        boardsize=args.boardsize, walls=args.walls,
        num_simulations=args.sims, leaf_batch=args.leaf_batch,
        num_threads=args.threads, parallel_games=args.parallel,
        c_puct=args.c_puct, dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon, fpu_reduction=args.fpu,
        temp_threshold=args.temp_threshold, max_moves=args.max_moves,
        dist_bonus_max=args.dist_bonus_max, training=True, seed=args.seed,
        tt_max_depth=args.tt_max_depth, tt_max_entries=args.tt_max_entries,
    )
    mgr.start(args.games)

    t0 = time.time()
    n_evals = 0
    n_batches = 0
    win_evals = 0     # evals/batches since the last report, for a rolling
    win_batches = 0   # (last-5s) average instead of a lifetime cumulative one
    last_report = t0
    try:
        with torch.inference_mode():
            while True:
                batch_id, states = mgr.get_batch(args.max_batch, args.flush_us)
                b = states.shape[0]
                if b == 0:
                    break
                x = torch.from_numpy(states).to(device, non_blocking=True)
                if use_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, value = model(x)
                else:
                    logits, value = model(x)
                mgr.put_results(
                    batch_id,
                    logits.float().cpu().numpy(),
                    value.float().view(-1).cpu().numpy(),
                )
                n_evals += b
                n_batches += 1
                win_evals += b
                win_batches += 1
                now = time.time()
                if now - last_report >= 5.0:
                    el = now - t0
                    win_elapsed = now - last_report
                    print(f"[{el:7.1f}s] games {mgr.games_finished():>5}/{args.games}"
                          f"  evals/s {win_evals / win_elapsed:8.0f}"
                          f"  mean batch {win_evals / max(win_batches, 1):6.1f}")
                    last_report = now
                    win_evals = 0
                    win_batches = 0
    finally:
        mgr.stop()

    elapsed = time.time() - t0
    data = mgr.get_data()
    stats = mgr.stats()
    states = data["states"]
    policies = data["policies"]
    values = data["values"]
    print(f"\ndone: {stats['games']} games in {elapsed:.1f}s "
          f"({stats['games'] / elapsed * 3600:.0f} games/hour)")
    print(f"  P1 {stats['p1_wins']}  P2 {stats['p2_wins']}  draws {stats['draws']}")
    print(f"  plies mean {stats['mean_plies']:.1f} "
          f"(min {stats['min_plies']}, max {stats['max_plies']})  "
          f"walls/game {stats['mean_walls']:.1f}")
    print(f"  positions {len(values)}  evals {n_evals} "
          f"({n_evals / elapsed:.0f}/s, mean batch {n_evals / max(n_batches, 1):.1f})")

    if not args.no_augment and len(values) > 0:
        perm = lr_flip_permutation(args.boardsize)
        states = np.concatenate([states, np.flip(states, axis=3)])
        policies = np.concatenate([policies, policies[:, perm]])
        values = np.concatenate([values, values])
        print(f"  after LR augmentation: {len(values)} positions")

    if args.out_dir and len(values) > 0:
        path = next_cycle_path(args.out_dir)
        np.savez_compressed(path, states=states.astype(np.float32),
                            policies=policies.astype(np.float32),
                            values=values.astype(np.float32))
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
