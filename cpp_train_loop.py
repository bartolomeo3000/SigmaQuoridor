"""Full AlphaZero cycle loop using the fast C++ self-play engine.

Neither existing script does the whole loop by itself:
  - selfplay_cpp.py generates one cycle_NNNN.npz of self-play data (fast,
    C++ engine) but never trains or touches best.pt.
  - train.py --train-only trains on whatever's already on disk and updates
    best.pt + a checkpoint, but does not generate new data.

This script alternates the two, in separate subprocesses, for a configurable
number of cycles:

    1. selfplay_cpp.py   - self-play with the current best.pt -> new
                           cycle_NNNN.npz in the data dir
    2. train.py --resume --train-only --cycles 1
                        - trains on the buffer (including the file just
                          produced), overwrites best.pt, saves a checkpoint,
                          appends a training_stats.csv row.

Both scripts agree on the fixed paths MODEL_PATH (default models_9x9/best.pt)
and DATA_DIR (default data_9x9), overridable via --model-dir/--data-dir, so
nothing needs to be passed between the two steps beyond those paths.

Example:
    python cpp_train_loop.py --cycles 20 --games 500 --sims 400
"""

import argparse
import os
import subprocess
import sys

MODEL_PATH = "models_9x9/best.pt"
DATA_DIR = "data_9x9"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cycles", type=int, default=20,
                   help="number of self-play+train cycles to run")
    p.add_argument("--model-dir", type=str, default=None, metavar="DIR",
                   help=f"override the models dir (default: {os.path.dirname(MODEL_PATH)!r})")
    p.add_argument("--data-dir", type=str, default=None, metavar="DIR",
                   help=f"override the data dir (default: inferred from --model-dir by "
                        f"swapping its 'models_' prefix for 'data_', or {DATA_DIR!r} "
                        f"if --model-dir is also omitted)")

    # Self-play (selfplay_cpp.py) options.
    p.add_argument("--games", type=int, default=2048)
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--threads", type=int, default=7)
    p.add_argument("--parallel", type=int, default=2048)
    p.add_argument("--leaf-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=512)
    p.add_argument("--boardsize", type=int, default=9)
    p.add_argument("--walls", type=int, default=10)
    p.add_argument("--temp-early", type=float, default=0.8)
    p.add_argument("--temp-final", type=float, default=0.15)
    p.add_argument("--temp-halflife", type=float, default=6.0)
    p.add_argument("--temp-prune-visits", type=int, default=4)
    p.add_argument("--max-moves", type=int, default=160)
    p.add_argument("--tt-max-depth", type=int, default=-1)
    p.add_argument("--tt-max-entries", type=int, default=5_000_000)
    p.add_argument("--solver-max-total-walls", type=int, default=2)
    p.add_argument("--solver-node-limit", type=int, default=5_000_000)
    p.add_argument("--solver-time-limit-s", type=float, default=4.0)
    p.add_argument("--mcts-solver-max-total-walls", type=int, default=0)
    p.add_argument("--mcts-solver-node-limit", type=int, default=20_000)
    p.add_argument("--mcts-solver-time-limit-s", type=float, default=0.02)
    p.add_argument("--abandon-stragglers-below", type=int, default=0,
                   help="forwarded to selfplay_cpp.py --abandon-stragglers-below "
                        "(stop and discard the last few in-flight games instead "
                        "of waiting out the GPU-starved tail; 0 disables)")
    p.add_argument("--bf16", action="store_true",
                   help="forwarded to selfplay_cpp.py --bf16 (bfloat16 autocast inference, CUDA only)")
    p.add_argument("--compile", action="store_true",
                   help="forwarded to selfplay_cpp.py --compile (torch.compile the model)")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed for self-play; incremented once per cycle")

    # Training (train.py) options; omitted flags fall back to train.py's own defaults.
    p.add_argument("--train-positions", type=int, default=None,
                   help="forwarded to train.py --train-positions")
    p.add_argument("--batch", type=int, default=None,
                   help="forwarded to train.py --batch")

    args = p.parse_args()

    model_dir = args.model_dir if args.model_dir is not None else os.path.dirname(MODEL_PATH)
    model_path = os.path.join(model_dir, "best.pt")
    if args.data_dir is not None:
        data_dir = args.data_dir
    elif args.model_dir is not None:
        base = os.path.basename(os.path.normpath(model_dir))
        data_dir = ("data_" + base[len("models_"):]) if base.startswith("models_") else DATA_DIR
    else:
        data_dir = DATA_DIR

    if not os.path.exists(model_path):
        sys.exit(
            f"{model_path} does not exist yet. Run `python train.py` once "
            f"(without --train-only) to create an initial checkpoint, or "
            f"`python selfplay_cpp.py` without --model for a fresh random net, "
            f"before using this loop."
        )

    for cycle in range(args.cycles):
        print(f"\n{'#' * 70}\n# Cycle {cycle + 1}/{args.cycles}: self-play\n{'#' * 70}")
        selfplay_cmd = [
            sys.executable, "selfplay_cpp.py",
            "--model", model_path,
            "--out-dir", data_dir,
            "--games", str(args.games),
            "--sims", str(args.sims),
            "--threads", str(args.threads),
            "--parallel", str(args.parallel),
            "--leaf-batch", str(args.leaf_batch),
            "--max-batch", str(args.max_batch),
            "--boardsize", str(args.boardsize),
            "--walls", str(args.walls),
            "--temp-early", str(args.temp_early),
            "--temp-final", str(args.temp_final),
            "--temp-halflife", str(args.temp_halflife),
            "--temp-prune-visits", str(args.temp_prune_visits),
            "--max-moves", str(args.max_moves),
            "--tt-max-depth", str(args.tt_max_depth),
            "--tt-max-entries", str(args.tt_max_entries),
            "--solver-max-total-walls", str(args.solver_max_total_walls),
            "--solver-node-limit", str(args.solver_node_limit),
            "--solver-time-limit-s", str(args.solver_time_limit_s),
            "--mcts-solver-max-total-walls", str(args.mcts_solver_max_total_walls),
            "--mcts-solver-node-limit", str(args.mcts_solver_node_limit),
            "--mcts-solver-time-limit-s", str(args.mcts_solver_time_limit_s),
            "--abandon-stragglers-below", str(args.abandon_stragglers_below),
            "--seed", str(args.seed + cycle),
        ]
        if args.bf16:
            selfplay_cmd.append("--bf16")
        if args.compile:
            selfplay_cmd.append("--compile")
        subprocess.run(selfplay_cmd, check=True)

        print(f"\n{'#' * 70}\n# Cycle {cycle + 1}/{args.cycles}: train\n{'#' * 70}")
        train_cmd = [
            sys.executable, "train.py", "--resume", "--train-only", "--cycles", "1",
            "--model-dir", model_dir, "--data-dir", data_dir,
        ]
        if args.train_positions is not None:
            train_cmd += ["--train-positions", str(args.train_positions)]
        if args.batch is not None:
            train_cmd += ["--batch", str(args.batch)]
        if args.bf16:
            train_cmd.append("--bf16")
        subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
