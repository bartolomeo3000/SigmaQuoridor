"""Measure cross-game transposition overlap in the first N real plies.

Runs the actual production self-play pipeline (real MCTS, real Dirichlet
noise, real temperature schedule) for a batch of games capped at a few
plies, then reports how many distinct positions exist at depth 1..4 and
how much reuse (avg games per distinct state) each depth would give a
shared transposition-table cache.

Example:
    .venv/bin/python _measure_opening_overlap.py \
        --model models_7x7_v2/best.pt --games 400 --boardsize 7 --walls 5
"""

import argparse
import time
from collections import Counter

import numpy as np
import torch

import quoridor_cpp
from dual_network import DualNetwork, load_model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--parallel", type=int, default=128)
    p.add_argument("--leaf-batch", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=256)
    p.add_argument("--flush-us", type=int, default=500)
    p.add_argument("--model", type=str, default=None,
                   help="checkpoint path; fresh random net if omitted")
    p.add_argument("--boardsize", type=int, default=7)
    p.add_argument("--walls", type=int, default=5)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--res", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    p.add_argument("--fpu", type=float, default=0.1)
    p.add_argument("--temp-threshold", type=int, default=14)
    p.add_argument("--depth", type=int, default=4, help="max ply to track")
    args = p.parse_args()

    device = pick_device()
    if args.model:
        model = load_model(args.model, device=device)
        print(f"loaded model {args.model}")
    else:
        model = DualNetwork(boardsize=args.boardsize, filters=args.filters,
                            num_residual=args.res).to(device)
        print("using fresh random network")
    model.eval()
    print(f"device={device}")

    # Only the first 4 plies matter for this measurement; cap max_moves so
    # games finish fast (games still play real MCTS at full sims per move,
    # just fewer moves per game).
    mgr = quoridor_cpp.SelfPlayManager(
        boardsize=args.boardsize, walls=args.walls,
        num_simulations=args.sims, leaf_batch=args.leaf_batch,
        num_threads=args.threads, parallel_games=args.parallel,
        c_puct=args.c_puct, dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon, fpu_reduction=args.fpu,
        temp_threshold=args.temp_threshold, max_moves=args.depth,
        dist_bonus_max=0.0, training=True, seed=args.seed,
    )
    mgr.start(args.games)

    t0 = time.time()
    try:
        with torch.inference_mode():
            while True:
                batch_id, states = mgr.get_batch(args.max_batch, args.flush_us)
                b = states.shape[0]
                if b == 0:
                    break
                x = torch.from_numpy(states).to(device, non_blocking=True)
                logits, value = model(x)
                mgr.put_results(
                    batch_id,
                    logits.float().cpu().numpy(),
                    value.float().view(-1).cpu().numpy(),
                )
    finally:
        mgr.stop()

    elapsed = time.time() - t0
    data = mgr.get_openings()
    hashes = data["hashes"]   # (n_games, 4) uint64
    lens = data["lens"]       # (n_games,) int32
    n_games = hashes.shape[0]
    print(f"\n{n_games} games in {elapsed:.1f}s")

    for depth in range(1, args.depth + 1):
        mask = lens >= depth
        n = int(mask.sum())
        if n == 0:
            print(f"depth {depth}: no games reached this ply")
            continue
        # Group by the *prefix* of hashes up to this depth (i.e. the full
        # move sequence so far, not just the hash at this exact ply) --
        # this is what a real TT lookup at that depth would key on.
        prefixes = [tuple(row) for row in hashes[mask, :depth]]
        counts = Counter(prefixes)
        distinct = len(counts)
        reuse = n / distinct
        biggest = counts.most_common(1)[0][1]
        hit_rate = 1.0 - distinct / n  # fraction of games that would hit an
                                       # already-cached entry from an earlier
                                       # game at this depth
        print(f"depth {depth}: {n:4d} games -> {distinct:4d} distinct states  "
              f"(avg reuse {reuse:5.2f}x, hit-rate {hit_rate:5.1%}, "
              f"largest group {biggest})")


if __name__ == "__main__":
    main()
