"""Regression tests for the value/pawn head redesign (see HEAD_REDESIGN_PLAN.md).

Checks, per the plan's testing checklist:
  - old checkpoints (legacy heads) still load correctly via _infer_arch/load_model
  - a fresh new-variant net round-trips through save_model/load_model unchanged
  - forward shapes are correct for both variants
  - warm_start_from_legacy transfers trunk/wall tensors and leaves changed-head
    tensors freshly initialized, and the result is usable end-to-end (NNEvaluator)

Run:  .venv/Scripts/python test_head_redesign.py
"""
import numpy as np
import torch

import _bootstrap  # noqa: F401  (puts the repo root on sys.path)

from game import State, action_space_size
from dual_network import (DualNetwork, NNEvaluator, DEVICE, save_model, load_model,
                          warm_start_from_legacy, _infer_arch)

N = 9
OLD_CKPT = "runs/models_9x9/best.pt"


def test_old_checkpoint_still_loads():
    model = load_model(OLD_CKPT)
    assert model.value_head == "legacy" and model.pawn_head == "legacy", (
        f"expected legacy heads for a pre-redesign checkpoint, got "
        f"value_head={model.value_head} pawn_head={model.pawn_head}")
    assert model.boardsize == N and model.filters == 128 and model.num_residual == 10
    s = State(boardsize=N)
    ev = NNEvaluator(model)
    priors, value = ev(s, s.get_legal_actions())
    assert abs(sum(priors) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0
    print(f"OK  {OLD_CKPT} still loads as legacy/legacy, serves valid output")


def test_fresh_net_roundtrip(tmp_path="scratchpad_headtest.pt"):
    import os
    model = DualNetwork(boardsize=N, filters=32, num_residual=2, gpool_every=0,
                        value_head="pooled", pawn_head="local").to(DEVICE)
    model.eval()
    x = torch.from_numpy(State(boardsize=N).to_nn_input()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits1, val1 = model(x)

    save_model(model, tmp_path)
    reloaded = load_model(tmp_path)
    assert reloaded.value_head == "pooled" and reloaded.pawn_head == "local"
    assert reloaded.boardsize == N, f"boardsize_marker round-trip failed: got {reloaded.boardsize}"
    reloaded.eval()
    with torch.no_grad():
        logits2, val2 = reloaded(x)
    assert torch.allclose(logits1, logits2) and torch.allclose(val1, val2)
    os.remove(tmp_path)
    print("OK  fresh pooled/local net round-trips through save_model/load_model exactly, "
          "boardsize correctly recovered from boardsize_marker")


def test_forward_shapes():
    A = action_space_size(N)
    for value_head, pawn_head in [("pooled", "local"), ("legacy", "legacy"),
                                  ("pooled", "legacy"), ("legacy", "local")]:
        model = DualNetwork(boardsize=N, filters=16, num_residual=2,
                            value_head=value_head, pawn_head=pawn_head).to(DEVICE)
        model.eval()
        batch = torch.stack([
            torch.from_numpy(State(boardsize=N).to_nn_input()) for _ in range(4)
        ]).to(DEVICE)
        with torch.no_grad():
            logits, val = model(batch)
        assert logits.shape == (4, A), (value_head, pawn_head, logits.shape)
        assert val.shape == (4, 1), (value_head, pawn_head, val.shape)
    print(f"OK  forward shapes correct for all 4 head-variant combinations (A={A})")


def test_warm_start():
    new = warm_start_from_legacy(OLD_CKPT, value_head="pooled", pawn_head="local", device=DEVICE)
    assert new.value_head == "pooled" and new.pawn_head == "local"
    assert new.boardsize == 9 and new.filters == 128 and new.num_residual == 10 and new.gpool_every == 3

    old = load_model(OLD_CKPT)
    old_sd, new_sd = old.state_dict(), new.state_dict()

    # Trunk + wall sub-heads must be copied exactly (bit-identical).
    for k in ["conv.weight", "bn.weight", "residuals.0.conv1.weight",
             "policy_h_conv.weight", "policy_v_conv.weight"]:
        assert torch.equal(old_sd[k], new_sd[k]), f"{k} was not warm-started correctly"

    # Changed-head keys must exist in new but not old, and vice versa.
    assert "value_pool_fc.weight" in new_sd and "value_pool_fc.weight" not in old_sd
    assert "policy_pawn_fc1.weight" in new_sd and "policy_pawn_fc1.weight" not in old_sd
    assert "value_fc1.weight" in old_sd and "value_fc1.weight" not in new_sd
    assert "policy_pawn_fc.weight" in old_sd and "policy_pawn_fc.weight" not in new_sd

    # End-to-end: usable through the normal serve path.
    ev = NNEvaluator(new)
    s = State(boardsize=N)
    priors, value = ev(s, s.get_legal_actions())
    assert abs(sum(priors) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0
    print("OK  warm_start_from_legacy: trunk/wall tensors transferred, "
          "changed heads freshly initialized, result usable via NNEvaluator")


if __name__ == "__main__":
    test_old_checkpoint_still_loads()
    test_fresh_net_roundtrip()
    test_forward_shapes()
    test_warm_start()
    print("\nAll head-redesign tests passed.")
