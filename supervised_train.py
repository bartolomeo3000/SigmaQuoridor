"""Supervised training on recorded self-play data.

Loads the latest N cycles of npz data, creates a fresh model with the same
architecture, and trains it with cross-entropy (policy) + MSE (value) losses
for a given number of epochs.  The trained model is saved and can be compared
with the procedurally-trained model via _matchup.py.

Usage
-----
  python supervised_train.py                       # defaults below
  python supervised_train.py --cycles 80 --epochs 3 --out models_7x7/supervised.pt
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from dual_network import DEVICE, DualNetwork, ViTDualNetwork, save_model
from game import action_space_size

# ── Defaults ────────────────────────────────────────────────────────────────
BOARDSIZE    = 9
FILTERS      = 128
NUM_RESIDUAL = 10
WALLS        = 5          # used only for action_space_size
DATA_DIR     = "data_7x7"
NUM_CYCLES   = 80
EPOCHS       = 12
BATCH_SIZE   = 512
LR           = 3e-4
WEIGHT_DECAY = 1e-4
OUT_PATH     = "models_7x7/supervised_extended.pt"
# ViT defaults
EMBED_DIM  = 128
NUM_HEADS  = 4
NUM_LAYERS = 6
MLP_RATIO  = 4.0


def load_data(data_dir: str, num_cycles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(glob.glob(f"{data_dir}/cycle_*.npz"))[-num_cycles:]
    if not files:
        raise FileNotFoundError(f"No cycle files found in {data_dir}")
    print(f"Loading {len(files)} cycles  ({files[0]} … {files[-1]})")
    states_list, policies_list, values_list = [], [], []
    for f in files:
        d = np.load(f)
        states_list.append(d["states"])
        policies_list.append(d["policies"])
        values_list.append(d["values"])
    states   = np.concatenate(states_list,   axis=0)
    policies = np.concatenate(policies_list, axis=0)
    values   = np.concatenate(values_list,   axis=0)
    print(f"  Total positions: {len(states):,}")
    return states, policies, values


def train(args: argparse.Namespace) -> None:
    states_np, policies_np, values_np = load_data(args.data_dir, args.cycles)

    # Move to tensors
    states   = torch.from_numpy(states_np).float()
    policies = torch.from_numpy(policies_np).float()
    values   = torch.from_numpy(values_np.reshape(-1, 1)).float()

    dataset    = TensorDataset(states, policies, values)
    loader     = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    # Fresh model or resume
    if args.resume:
        from dual_network import load_model
        model = load_model(args.resume).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        if isinstance(model, ViTDualNetwork):
            print(f"\nResuming from {args.resume} — boardsize={model.boardsize}  "
                  f"embed_dim={model.embed_dim}  num_heads={model.num_heads}  "
                  f"num_layers={model.num_layers}  params={total_params:,}")
        else:
            print(f"\nResuming from {args.resume} — boardsize={model.boardsize}  "
                  f"filters={model.filters}  residual_blocks={model.num_residual}  params={total_params:,}")
    elif args.vit:
        model = ViTDualNetwork(
            boardsize  = args.boardsize,
            embed_dim  = args.embed_dim,
            num_heads  = args.num_heads,
            num_layers = args.num_layers,
            mlp_ratio  = args.mlp_ratio,
        ).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nFresh ViT model — boardsize={args.boardsize}  embed_dim={args.embed_dim}  "
              f"num_heads={args.num_heads}  num_layers={args.num_layers}  params={total_params:,}")
    else:
        model = DualNetwork(boardsize=args.boardsize, filters=args.filters,
                            num_residual=args.num_residual).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nFresh CNN model — boardsize={args.boardsize}  filters={args.filters}  "
              f"residual_blocks={args.num_residual}  params={total_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch, eta_min=1e-5)

    A = action_space_size(args.boardsize)

    print(f"\nTraining for {args.epochs} epoch(s)  "
          f"({steps_per_epoch} steps/epoch, batch={args.batch_size})\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = total_pol = total_val = 0.0

        for step, (s_b, p_b, v_b) in enumerate(loader, 1):
            s_b = s_b.to(DEVICE, non_blocking=True)
            p_b = p_b.to(DEVICE, non_blocking=True)
            v_b = v_b.to(DEVICE, non_blocking=True)

            logits, value_pred = model(s_b)

            # Policy loss: cross-entropy with soft targets
            # Mask illegal actions (zero probability in target)
            illegal_mask = (p_b == 0)
            masked_logits = logits.clone()
            masked_logits[illegal_mask] = -1e9
            log_probs = F.log_softmax(masked_logits, dim=-1)
            pol_loss = -(p_b * log_probs).sum(dim=-1).mean()

            # Value loss: MSE
            val_loss = F.mse_loss(value_pred, v_b)

            loss = pol_loss + val_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_pol  += pol_loss.item()
            total_val  += val_loss.item()

            if step % 500 == 0 or step == steps_per_epoch:
                avg_l = total_loss / step
                avg_p = total_pol  / step
                avg_v = total_val  / step
                elapsed = time.time() - t0
                print(f"  epoch {epoch}/{args.epochs}  step {step:>5}/{steps_per_epoch}"
                      f"  loss={avg_l:.4f}  pol={avg_p:.4f}  val={avg_v:.4f}"
                      f"  lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s")

        print(f"  ── epoch {epoch} done in {time.time()-t0:.0f}s ──\n")

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out))
    print(f"Saved → {out}")
    print(f"\nRun matchup with:")
    print(f"  python _matchup.py --model-a models_7x7/best.pt --model-b {out} "
          f"--name-a procedural --name-b supervised")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised training on self-play data")
    p.add_argument("--data-dir",     default=DATA_DIR)
    p.add_argument("--cycles",       type=int, default=NUM_CYCLES)
    p.add_argument("--epochs",       type=int, default=EPOCHS)
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--boardsize",    type=int, default=BOARDSIZE)
    p.add_argument("--filters",      type=int, default=FILTERS)
    p.add_argument("--num-residual", type=int, default=NUM_RESIDUAL)
    p.add_argument("--out",          default=OUT_PATH)
    p.add_argument("--resume",       default=None,
                   help="Path to existing model to resume training from")
    # ViT options
    p.add_argument("--vit",          action="store_true",
                   help="Train a ViTDualNetwork instead of the default CNN")
    p.add_argument("--embed-dim",    type=int,   default=EMBED_DIM)
    p.add_argument("--num-heads",    type=int,   default=NUM_HEADS)
    p.add_argument("--num-layers",   type=int,   default=NUM_LAYERS)
    p.add_argument("--mlp-ratio",    type=float, default=MLP_RATIO)
    return p.parse_args()


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    train(parse_args())
