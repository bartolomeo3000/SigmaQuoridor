import torch
import _bootstrap  # noqa: F401  (puts the repo root on sys.path)

from train import load_holdout, compute_holdout_loss, HOLDOUT_DIR, HOLDOUT_CYCLES, HOLDOUT_SIZE
from dual_network import load_model, DEVICE

holdout = load_holdout(HOLDOUT_DIR, HOLDOUT_CYCLES, HOLDOUT_SIZE)
print(f"Holdout: {len(holdout[0]):,} positions from last {HOLDOUT_CYCLES} cycles of {HOLDOUT_DIR!r}")
print(f"Device: {DEVICE}")

model = load_model("runs/models_7x7_v2/best.pt", device=DEVICE)
model.eval()

losses = compute_holdout_loss(model, *holdout, DEVICE)
print(f"loss={losses['total']:.4f}  policy={losses['policy']:.4f}  value={losses['value']:.4f}")
