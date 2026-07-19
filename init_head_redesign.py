"""One-off setup for the value/pawn head-redesign experiment (see HEAD_REDESIGN_PLAN.md).

Warm-starts a new lineage from an existing trained net: the trunk and wall
sub-heads are copied over unchanged, while the value head and pawn policy
sub-head are rebuilt with the new architecture and freshly initialized (their
old-shaped weights can't transfer). Writes both files a fresh
``models_<X>/`` lineage needs to be picked up by ``--resume``:

  <dst-dir>/best.pt                      -- inference weights (dual_network.save_model format)
  <dst-dir>/checkpoints/cycle_0000.pt    -- full training state at cycle 0
                                             (train.save_training_checkpoint format),
                                             so `train.py --resume --model-dir <dst-dir>`
                                             finds it and resumes from cycle 0 with a
                                             warm-started model but a fresh optimizer/scheduler.

Does not touch the source lineage (--src and its directory) at all.

Usage:
    python init_head_redesign.py --src models_9x9/best.pt --dst-dir models_9x9_heads

Then, e.g.:
    python cpp_train_loop.py --cycles 20 --model-dir models_9x9_heads --data-dir data_9x9_heads ...
"""
import argparse
import os

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from dual_network import DEVICE, save_model, warm_start_from_legacy
from train import LEARNING_RATE, WEIGHT_DECAY, LR_MILESTONES, LR_DECAY, save_training_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", default="models_9x9/best.pt",
                   help="existing trained checkpoint to warm-start the trunk from")
    p.add_argument("--dst-dir", default="models_9x9_heads",
                   help="new model directory for this experiment (must not already "
                        "contain best.pt/checkpoints — use a fresh lineage)")
    p.add_argument("--value-head", choices=["pooled", "legacy"], default="pooled")
    p.add_argument("--pawn-head", choices=["local", "legacy"], default="local")
    p.add_argument("--lr", type=float, default=LEARNING_RATE,
                   help=f"Adam learning rate for the fresh optimizer state (default: {LEARNING_RATE})")
    args = p.parse_args()

    dst_model_path = os.path.join(args.dst_dir, "best.pt")
    dst_ckpt_dir   = os.path.join(args.dst_dir, "checkpoints")
    dst_ckpt_path  = os.path.join(dst_ckpt_dir, "cycle_0000.pt")
    if os.path.exists(dst_model_path) or os.path.exists(dst_ckpt_path):
        raise SystemExit(
            f"{args.dst_dir!r} already has a best.pt or cycle_0000.pt — refusing to "
            f"overwrite an existing lineage. Use a fresh --dst-dir."
        )

    model = warm_start_from_legacy(
        args.src, value_head=args.value_head, pawn_head=args.pawn_head, device=DEVICE
    )
    total_params = sum(p_.numel() for p_ in model.parameters())
    print(f"\nNew net: boardsize={model.boardsize}  filters={model.filters}  "
          f"residual_blocks={model.num_residual}  gpool_every={model.gpool_every}  "
          f"value_head={model.value_head}  pawn_head={model.pawn_head}  "
          f"params={total_params:,}")

    save_model(model, dst_model_path)
    print(f"Saved inference weights -> {dst_model_path}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_DECAY)
    save_training_checkpoint(dst_ckpt_path, model, optimizer, scheduler, cycle=0)
    print(f"Saved fresh training-state checkpoint (cycle 0) -> {dst_ckpt_path}")

    base = os.path.basename(os.path.normpath(args.dst_dir))
    data_dir_hint = ("data_" + base[len("models_"):]) if base.startswith("models_") else "<data-dir>"
    print(
        f"\nNext step, e.g.:\n"
        f"  python cpp_train_loop.py --cycles 20 --model-dir {args.dst_dir} "
        f"--data-dir {data_dir_hint} ..."
    )


if __name__ == "__main__":
    main()
