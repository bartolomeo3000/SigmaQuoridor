"""Load a pre-trained model, reset policy and value heads, save as best.pt.

Usage:
    python reset_heads.py --src models_9x9/supervised_big.pt --dst models_9x9/best.pt
"""
import argparse
import torch
from dual_network import load_model, save_model

HEAD_LAYERS = [
    "policy_pawn_fc",
    "policy_h_conv", "policy_h_bn",
    "policy_v_conv", "policy_v_bn",
    "value_conv",    "value_bn",
    "value_fc1",     "value_fc2",
]

def reset_heads(model: torch.nn.Module) -> None:
    for name in HEAD_LAYERS:
        layer = getattr(model, name)
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
            print(f"  reset  {name}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="models_9x9/supervised_big.pt")
    p.add_argument("--dst", default="models_9x9/best.pt")
    args = p.parse_args()

    model = load_model(args.src)
    print(f"Loaded {args.src}  (boardsize={model.boardsize}, "
          f"filters={model.filters}, residual_blocks={model.num_residual})")
    reset_heads(model)
    save_model(model, args.dst)
    print(f"Saved → {args.dst}")

if __name__ == "__main__":
    main()
