"""Full AlphaZero cycle loop using the fast C++ self-play engine.

Neither existing script does the whole loop by itself:
  - selfplay_cpp.py generates one cycle_NNNN.npz of self-play data (fast,
    C++ engine) but never trains or touches best.pt.
  - train.py --train-only trains on whatever's already on disk and updates
    models_7x7/best.pt + a checkpoint, but does not generate new data.

This script alternates the two, in separate subprocesses, for a configurable
number of cycles:

    1. selfplay_cpp.py   - self-play with the current best.pt -> new
                           data_7x7_v2/cycle_NNNN.npz
    2. train.py --resume --train-only --cycles 1
                        - trains on the buffer (including the file just
                          produced), overwrites models_7x7/best.pt, saves
                          a checkpoint, appends a training_stats.csv row.

Both scripts already agree on the fixed paths models_7x7/best.pt and
data_7x7/*.npz, so nothing needs to be passed between the two steps
beyond those paths.

Example:
    python cpp_train_loop.py --cycles 20 --games 500 --sims 400
"""

import argparse
import os
import subprocess
import sys

MODEL_PATH = "models_7x7/best.pt"
DATA_DIR = "data_7x7"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cycles", type=int, default=20,
                   help="number of self-play+train cycles to run")

    # Self-play (selfplay_cpp.py) options.
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--threads", type=int, default=7)
    p.add_argument("--parallel", type=int, default=1024)
    p.add_argument("--leaf-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=512)
    p.add_argument("--boardsize", type=int, default=7)
    p.add_argument("--walls", type=int, default=5)
    p.add_argument("--temp-threshold", type=int, default=20)
    p.add_argument("--tt-max-depth", type=int, default=-1)
    p.add_argument("--tt-max-entries", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=0,
                   help="base seed for self-play; incremented once per cycle")

    # Training (train.py) options; omitted flags fall back to train.py's own defaults.
    p.add_argument("--train-positions", type=int, default=None,
                   help="forwarded to train.py --train-positions")
    p.add_argument("--batch", type=int, default=None,
                   help="forwarded to train.py --batch")

    args = p.parse_args()

    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"{MODEL_PATH} does not exist yet. Run `python train.py` once "
            f"(without --train-only) to create an initial checkpoint, or "
            f"`python selfplay_cpp.py` without --model for a fresh random net, "
            f"before using this loop."
        )

    for cycle in range(args.cycles):
        print(f"\n{'#' * 70}\n# Cycle {cycle + 1}/{args.cycles}: self-play\n{'#' * 70}")
        selfplay_cmd = [
            sys.executable, "selfplay_cpp.py",
            "--model", MODEL_PATH,
            "--out-dir", DATA_DIR,
            "--games", str(args.games),
            "--sims", str(args.sims),
            "--threads", str(args.threads),
            "--parallel", str(args.parallel),
            "--leaf-batch", str(args.leaf_batch),
            "--max-batch", str(args.max_batch),
            "--boardsize", str(args.boardsize),
            "--walls", str(args.walls),
            "--temp-threshold", str(args.temp_threshold),
            "--tt-max-depth", str(args.tt_max_depth),
            "--tt-max-entries", str(args.tt_max_entries),
            "--seed", str(args.seed + cycle),
        ]
        subprocess.run(selfplay_cmd, check=True)

        print(f"\n{'#' * 70}\n# Cycle {cycle + 1}/{args.cycles}: train\n{'#' * 70}")
        train_cmd = [sys.executable, "train.py", "--resume", "--train-only", "--cycles", "1"]
        if args.train_positions is not None:
            train_cmd += ["--train-positions", str(args.train_positions)]
        if args.batch is not None:
            train_cmd += ["--batch", str(args.batch)]
        subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
