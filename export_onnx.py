"""
Export the trained PyTorch model to ONNX format for use in the browser.

Usage:
    python export_onnx.py

Outputs:
    docs/models/best.onnx
    docs/models/supervised.onnx  (if models_7x7/supervised.pt exists)
"""
import sys
from pathlib import Path
import torch
from dual_network import DualNetwork

OUT_DIR = Path("docs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORTS = [
    ("models_7x7/best.pt",              "docs/models/best.onnx"),
    ("models_7x7/supervised.pt",         "docs/models/supervised.onnx"),
    ("models_7x7/supervised_extended.pt","docs/models/supervised_extended.onnx"),
]

def export(src: str, dst: str) -> None:
    src_path = Path(src)
    if not src_path.exists():
        print(f"  skip — {src} not found")
        return

    checkpoint = torch.load(src_path, map_location="cpu", weights_only=False)
    # Support both {"model": state_dict} and plain state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    boardsize = checkpoint.get("boardsize", 7) if isinstance(checkpoint, dict) else 7
    # Infer filters and num_residual from weight shapes
    filters = int(state_dict["conv.weight"].shape[0]) if "conv.weight" in state_dict else 64
    res_indices = {int(k.split(".")[1]) for k in state_dict if k.startswith("residuals.")}
    residuals = len(res_indices) if res_indices else 6

    model = DualNetwork(boardsize=boardsize, filters=filters, num_residual=residuals)
    model.load_state_dict(state_dict)
    model.eval()

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
