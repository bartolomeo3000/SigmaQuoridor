"""Verify the exported heads ONNX graph matches PyTorch on REALISTIC one-hot
canonical inputs — the pawn head's mask-gather only does anything on a genuine
one-hot plane, so zeros/random inputs would not exercise it. Uses onnx.reference
(pure-python spec implementation); ORT-web shares kernels with onnxruntime, but
for the standard ops here (Conv/BN/Relu/Mul/ReduceSum/ReduceMean/ReduceMax/
Gemm/Tanh/Concat/Slice) a spec-faithful match is strong evidence."""
import numpy as np
import torch
from onnx.reference import ReferenceEvaluator
import onnx
import _bootstrap  # noqa: F401  (puts the repo root on sys.path)

from game import State, PawnAction, WallAction
from dual_network import load_model

CPU = torch.device("cpu")
PT   = "runs/models_9x9_heads/best.pt"
ONNX = "docs/models_9x9/best.onnx"

def positions():
    out = []
    out.append(("start (P1)", State(boardsize=9, walls_p1=10, walls_p2=10)))
    s = State(boardsize=9, walls_p1=10, walls_p2=10).next(PawnAction(direction=(0, 1)))
    out.append(("1 ply (P2)", s))
    s2 = State(boardsize=9, walls_p1=10, walls_p2=10)
    for _ in range(3):
        s2 = s2.next(PawnAction(direction=(0, 1))).next(PawnAction(direction=(0, -1)))
    out.append(("mid (P2 flip)", s2))
    # a position with walls placed (exercises wall planes + pawn gather off-center)
    s3 = State(boardsize=9, walls_p1=10, walls_p2=10)
    s3 = s3.next(WallAction(x=3, y=3, orientation='h'))
    s3 = s3.next(WallAction(x=5, y=5, orientation='v'))
    s3 = s3.next(PawnAction(direction=(0, 1)))
    out.append(("walls+pawn", s3))
    return out

def main():
    model = load_model(PT, device=CPU); model.eval()
    onnx.checker.check_model(onnx.load(ONNX))
    sess = ReferenceEvaluator(ONNX)

    print(f"PT   : {PT}  (value_head={model.value_head}, pawn_head={model.pawn_head})")
    print(f"ONNX : {ONNX}\n")
    worst_logit = 0.0
    worst_value = 0.0
    for name, st in positions():
        x = st.to_nn_input()[None].astype(np.float32)   # (1,8,N,N) one-hot planes
        # sanity: planes 0 and 1 must each be a single 1.0 (one-hot pawn)
        oh0, oh1 = x[0, 0].sum(), x[0, 1].sum()
        with torch.no_grad():
            pl_pt, v_pt = model(torch.from_numpy(x))
        pl_pt = pl_pt.numpy()[0]; v_pt = float(v_pt.item())
        outs = sess.run(None, {"input": x})
        pl_ox = np.asarray(outs[0]).reshape(-1)
        v_ox  = float(np.asarray(outs[1]).reshape(-1)[0])
        dl = np.abs(pl_pt - pl_ox).max()
        dv = abs(v_pt - v_ox)
        worst_logit = max(worst_logit, dl); worst_value = max(worst_value, dv)
        print(f"  {name:14s} onehot(p0,p1)=({oh0:.0f},{oh1:.0f})  "
              f"value PT={v_pt:+.4f} ONNX={v_ox:+.4f}  "
              f"max|dlogit|={dl:.2e}  |dvalue|={dv:.2e}")
    print(f"\n  WORST over positions:  max|dlogit|={worst_logit:.2e}   max|dvalue|={worst_value:.2e}")
    ok = worst_logit < 1e-4 and worst_value < 1e-4
    print("  RESULT:", "OK — ONNX graph matches PyTorch on real one-hot inputs"
          if ok else "MISMATCH — investigate")

if __name__ == "__main__":
    main()
