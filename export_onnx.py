"""
Export the trained PyTorch model to ONNX format for use in the browser.

Usage:
    python export_onnx.py

Outputs:
    docs/models/best.onnx               (7x7/5-wall lineage)
    docs/models/supervised.onnx         (if models_7x7/supervised.pt exists)
    docs/models_9x9/best.onnx           (9x9/10-wall lineage)
    docs/models_9x9/checkpoints/*.onnx  (curated subset — see CKPT_STEP below)
"""
from pathlib import Path
import torch
from dual_network import load_model

Path("docs/models").mkdir(parents=True, exist_ok=True)
Path("docs/models_9x9").mkdir(parents=True, exist_ok=True)

EXPORTS = [
    ("models_7x7/best.pt",              "docs/models/best.onnx"),
    ("models_7x7/supervised.pt",         "docs/models/supervised.onnx"),
    ("models_7x7/supervised_extended.pt","docs/models/supervised_extended.onnx"),
    ("models_9x9/best.pt",               "docs/models_9x9/best.onnx"),
]

# Dynamically add checkpoint exports.
_ck_src = Path("models_7x7/checkpoints")
_ck_dst = Path("docs/models/checkpoints")
if _ck_src.exists():
    _ck_dst.mkdir(parents=True, exist_ok=True)
    for _pt in sorted(_ck_src.glob("cycle_*.pt")):
        EXPORTS.append((str(_pt), str(_ck_dst / (_pt.stem + ".onnx"))))

# 9x9 checkpoints are ~6x larger on disk than the 7x7 ones (128x10+gpool vs
# 64x6), so we only export a curated subset (every 20 cycles + the latest)
# rather than all of them, to keep docs/ (served via GitHub Pages) reasonably
# sized. Bump CKPT_STEP down (or add specific cycles to _ck_extra) if you
# want a denser checkpoint history in the browser picker.
CKPT_STEP = 20
_ck9_src = Path("models_9x9/checkpoints")
_ck9_dst = Path("docs/models_9x9/checkpoints")
if _ck9_src.exists():
    _ck9_dst.mkdir(parents=True, exist_ok=True)
    _all_ck9 = sorted(_ck9_src.glob("cycle_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
    _picked = {p for p in _all_ck9 if int(p.stem.split("_")[1]) % CKPT_STEP == 1}
    if _all_ck9:
        _picked.add(_all_ck9[-1])   # always include the latest checkpoint
    for _pt in sorted(_picked, key=lambda p: int(p.stem.split("_")[1])):
        EXPORTS.append((str(_pt), str(_ck9_dst / (_pt.stem + ".onnx"))))


def export(src: str, dst: str) -> None:
    src_path = Path(src)
    if not src_path.exists():
        print(f"  skip — {src} not found")
        return

    # All architecture hyper-parameters (filters, num_residual, boardsize,
    # gpool_every) are inferred automatically from the saved weight shapes.
    model = load_model(str(src_path), device=torch.device("cpu"))
    model.eval()
    boardsize = model.boardsize

    dummy = torch.zeros(1, 8, boardsize, boardsize)
    # Use legacy exporter to embed all weights in a single .onnx file
    # (new dynamo exporter writes external .data files, which ONNX RT Web can't load)
    torch.onnx.export(
        model,
        dummy,
        dst,
        input_names=["input"],
        output_names=["policy_logits", "value"],
        opset_version=17,
        dynamo=False,
    )
    size_mb = Path(dst).stat().st_size / 1e6
    print(f"  {dst}  ({size_mb:.2f} MB)")

print("Exporting models to ONNX...")
for src, dst in EXPORTS:
    export(src, dst)
print("Done.")

