"""Dual network (policy + value head) for AlphaZero-style MCTS.

Architecture
------------
Input: (B, 8, N, N) tensor from State.to_nn_input().

Trunk
  Conv2d(8 → F, 3x3, pad=1) → BN → ReLU
  R residual blocks (each: Conv→BN→ReLU→Conv→BN, skip-add→ReLU)

Policy head — three spatial sub-heads, concatenated into raw logits:
  ┌─ Pawn (8 logits):   global avg-pool → Linear(F → 8)
  ├─ H-walls (W² logits): Conv2d(F→1, 1x1) → BN → ReLU
  │                         → slice [:, 0, :W, :W]  (W = N-1)
  │                         → flatten row-major
  └─ V-walls (W² logits): same structure
  → cat → (B, 8 + 2*W²) raw logits   [NO softmax — applied in NNEvaluator]

Value head:
  Conv2d(F→1, 1x1) → BN → ReLU → flatten → Linear(N²→64) → ReLU
  → Linear(64→1) → tanh   → scalar in [-1, 1]

Policy head output order matches action_to_index() exactly:
  indices 0-7        : pawn directions (same order as ALL_PAWN_DIRECTIONS)
  indices 8..8+W²-1  : H-walls row-major  (index = 8 + y*W + x)
  indices 8+W²..end  : V-walls row-major  (index = 8 + W² + y*W + x)

Plug-in into MCTS
-----------------
    evaluator = NNEvaluator(load_model("models/best.pt"))
    agent = MCTSAgent(evaluator=evaluator, num_simulations=800)

For use in app.py registry:
    "available": lambda: Path("models/best.pt").exists(),
    "factory":   lambda n: MCTSAgent(
                     evaluator=make_nn_evaluator("models/best.pt"),
                     num_simulations=n,
                 ),
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import State, Action, action_to_index, action_space_size

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Standard pre-activation residual block with two 3x3 convolutions."""

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = x
        x  = F.relu(self.bn1(self.conv1(x)))
        x  = self.bn2(self.conv2(x))
        return F.relu(x + sc)


# ---------------------------------------------------------------------------
# Dual network
# ---------------------------------------------------------------------------

class DualNetwork(nn.Module):
    """
    Shared convolutional trunk feeding two heads:
    * policy head  — raw logits over all actions (no softmax)
    * value head   — scalar estimate in [-1, 1] from the current player's POV

    Parameters
    ----------
    boardsize    : int   side length of the board (7 or 9)
    filters      : int   number of convolutional channels throughout the trunk
    num_residual : int   number of residual blocks in the trunk
    """

    IN_CHANNELS = 8  # channels produced by State.to_nn_input()

    def __init__(
        self,
        boardsize:    int = 7,
        filters:      int = 64,
        num_residual: int = 6,
    ) -> None:
        super().__init__()
        self.boardsize    = boardsize
        self.filters      = filters
        self.num_residual = num_residual
        N = boardsize
        W = N - 1   # wall-anchor grid side length

        # ── Trunk ────────────────────────────────────────────────────────────
        self.conv = nn.Conv2d(self.IN_CHANNELS, filters, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(filters)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        self.residuals = nn.Sequential(
            *[ResidualBlock(filters) for _ in range(num_residual)]
        )

        # ── Policy head ──────────────────────────────────────────────────────
        # Pawn sub-head: global avg-pool preserves no spatial info (correct —
        # direction legality depends on the whole board, not one cell)
        self.policy_pawn_fc = nn.Linear(filters, 8)

        # H-wall sub-head: 1×1 conv keeps all spatial information; we slice
        # the top-left W×W corner after conv (valid anchor coordinates are
        # 0..W-1 in both axes)
        self.policy_h_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.policy_h_bn   = nn.BatchNorm2d(1)

        # V-wall sub-head: identical structure
        self.policy_v_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.policy_v_bn   = nn.BatchNorm2d(1)

        # ── Value head ───────────────────────────────────────────────────────
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(N * N, 64)
        self.value_fc2  = nn.Linear(64, 1)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 8, N, N)  float32 tensor (from State.to_nn_input())

        Returns
        -------
        policy_logits : (B, action_space_size)  raw logits — NO softmax
        value         : (B, 1)                  tanh output in [-1, 1]
        """
        B = x.size(0)
        N = self.boardsize
        W = N - 1

        # ── Trunk ────────────────────────────────────────────────────────────
        x = F.relu(self.bn(self.conv(x)))   # (B, F, N, N)
        x = self.residuals(x)               # (B, F, N, N)

        # ── Policy head ──────────────────────────────────────────────────────
        # Pawn: global average pool → (B, F) → FC → (B, 8)
        p_pawn = x.mean(dim=(2, 3))             # (B, F)
        p_pawn = self.policy_pawn_fc(p_pawn)    # (B, 8)

        # H-walls: 1×1 conv → slice W×W anchor region → flatten → (B, W²)
        h = F.relu(self.policy_h_bn(self.policy_h_conv(x)))  # (B, 1, N, N)
        h = h[:, 0, :W, :W].contiguous().view(B, -1)         # (B, W²)

        # V-walls: identical
        v = F.relu(self.policy_v_bn(self.policy_v_conv(x)))  # (B, 1, N, N)
        v = v[:, 0, :W, :W].contiguous().view(B, -1)         # (B, W²)

        # Concatenate in the same order as action_to_index()
        policy_logits = torch.cat([p_pawn, h, v], dim=1)     # (B, 8+2*W²)

        # ── Value head ──────────────────────────────────────────────────────
        val = F.relu(self.value_bn(self.value_conv(x)))  # (B, 1, N, N)
        val = val.view(B, -1)                            # (B, N²)
        val = F.relu(self.value_fc1(val))                # (B, 64)
        val = torch.tanh(self.value_fc2(val))            # (B, 1)

        return policy_logits, val


# ---------------------------------------------------------------------------
# NNEvaluator — Evaluator Protocol adapter
# ---------------------------------------------------------------------------

class NNEvaluator:
    """
    Wraps a DualNetwork and exposes the ``Evaluator`` Protocol expected by
    ``MCTSAgent``::

        evaluator(state, legal_actions) -> (priors, value)

    Priors are computed by gathering the raw policy logits at the indices of
    the legal actions, then applying softmax over those masked logits only.
    This is the correct AlphaZero masking procedure.

    The model is set to eval mode at construction and all inference runs
    inside ``torch.no_grad()``.
    """

    def __init__(
        self,
        model:  DualNetwork,
        device: torch.device | None = None,
    ) -> None:
        self.model  = model
        self.device = device or DEVICE
        self.model.eval()

    def __call__(
        self,
        state:         State,
        legal_actions: list[Action],
    ) -> tuple[list[float], float]:
        with torch.no_grad():
            # Build (1, 8, N, N) input tensor
            x = torch.from_numpy(state.to_nn_input()).unsqueeze(0).to(self.device)

            policy_logits, value = self.model(x)   # (1, A), (1, 1)
            policy_logits = policy_logits[0]       # (A,)

            # Gather logits for legal actions, then softmax over that subset
            indices = torch.tensor(
                [action_to_index(a, state.boardsize) for a in legal_actions],
                dtype=torch.long,
                device=self.device,
            )
            legal_logits = policy_logits[indices]                  # (n_legal,)
            priors       = torch.softmax(legal_logits, dim=0).cpu().tolist()

            return priors, float(value[0, 0])

    def batch_call_raw(
        self,
        arrays:             list[np.ndarray],
        legal_actions_list: list[list[Action]],
    ) -> list[tuple[list[float], float]]:
        """
        Evaluate a batch of pre-computed ``(8, N, N)`` float32 arrays in one
        forward pass.

        Accepts numpy arrays rather than ``State`` objects so that callers
        such as ``SymmetricEvaluator`` can supply pre-flipped inputs without
        reconstructing state objects.

        Returns one ``(priors, value)`` pair per input array.
        """
        inputs    = torch.from_numpy(np.stack(arrays)).to(self.device)  # (B, 8, N, N)
        boardsize = self.model.boardsize
        with torch.no_grad():
            logits_batch, values_batch = self.model(inputs)              # (B, A), (B, 1)

        results: list[tuple[list[float], float]] = []
        for la, logits, val in zip(legal_actions_list, logits_batch, values_batch):
            indices = torch.tensor(
                [action_to_index(a, boardsize) for a in la],
                dtype=torch.long,
                device=self.device,
            )
            priors = torch.softmax(logits[indices], dim=0).cpu().tolist()
            results.append((priors, float(val[0])))
        return results

    def batch_call(
        self,
        states:             list[State],
        legal_actions_list: list[list[Action]],
    ) -> list[tuple[list[float], float]]:
        """
        Evaluate a batch of game states in one forward pass.

        Converts each state to its NN input array and delegates to
        ``batch_call_raw``.
        """
        arrays = [s.to_nn_input() for s in states]
        return self.batch_call_raw(arrays, legal_actions_list)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_model(model: DualNetwork, path: str) -> None:
    """Save model weights to *path*, creating parent directories if needed.

    Writes to a temporary file first then renames atomically so that a
    concurrently running process (e.g. app.py) holding the file open does not
    cause a Windows ERROR_USER_MAPPED_FILE (error 1224) failure.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def _infer_arch(state_dict: dict) -> tuple[int, int, int]:
    """
    Infer (filters, num_residual, boardsize) from a saved state dict so that
    checkpoints can be loaded without knowing the architecture upfront.

    Derivations
    -----------
    * filters      — output channels of the first conv layer
    * num_residual — number of ResidualBlock entries in 'residuals.*'
    * boardsize    — value_fc1 maps N² → 64, so boardsize = sqrt(in_features)
    """
    filters = state_dict["conv.weight"].shape[0]
    num_residual = sum(
        1 for k in state_dict
        if k.startswith("residuals.") and k.endswith(".conv1.weight")
    )
    boardsize = int(math.sqrt(state_dict["value_fc1.weight"].shape[1]))
    return filters, num_residual, boardsize


def _infer_vit_arch(state_dict: dict) -> tuple[int, int, int, int, float]:
    """
    Infer (embed_dim, num_layers, boardsize, num_heads, mlp_ratio) from a
    saved ViTDualNetwork state dict.

    Derivations
    -----------
    * embed_dim  — output features of the patch embedding linear layer
    * num_layers — number of TransformerEncoderLayer blocks
    * boardsize  — pos_embed has shape (1, 1 + N², E), so N = sqrt(seq - 1)
    * num_heads  — stored explicitly in the ``_num_heads_buf`` buffer
    * mlp_ratio  — FFN hidden dim / embed_dim
    """
    embed_dim  = state_dict["patch_embed.weight"].shape[0]
    num_layers = sum(
        1 for k in state_dict
        if k.startswith("transformer.layers.") and k.endswith(".self_attn.in_proj_weight")
    )
    boardsize  = int(math.sqrt(state_dict["pos_embed"].shape[1] - 1))
    num_heads  = int(state_dict["_num_heads_buf"])
    ff_dim     = state_dict["transformer.layers.0.linear1.weight"].shape[0]
    mlp_ratio  = ff_dim / embed_dim
    return embed_dim, num_layers, boardsize, num_heads, mlp_ratio


def load_model(path: str, device: torch.device | None = None) -> DualNetwork | ViTDualNetwork:
    """
    Load a checkpoint from *path*.  Architecture (CNN or ViT) and all
    hyper-parameters are inferred automatically from the saved weight shapes.
    """
    dev = device or DEVICE
    sd  = torch.load(path, map_location=dev, weights_only=True)
    # Training checkpoints wrap the model state under "model_state"
    if "model_state" in sd:
        sd = sd["model_state"]
    if "patch_embed.weight" in sd:
        embed_dim, num_layers, boardsize, num_heads, mlp_ratio = _infer_vit_arch(sd)
        model: DualNetwork | ViTDualNetwork = ViTDualNetwork(
            boardsize  = boardsize,
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            num_layers = num_layers,
            mlp_ratio  = mlp_ratio,
        ).to(dev)
    else:
        filters, num_residual, boardsize = _infer_arch(sd)
        model = DualNetwork(
            boardsize    = boardsize,
            filters      = filters,
            num_residual = num_residual,
        ).to(dev)
    model.load_state_dict(sd)
    return model


def make_nn_evaluator(
    path:   str,
    device: torch.device | None = None,
) -> NNEvaluator:
    """
    Convenience function: load a checkpoint and return a ready-to-use
    ``NNEvaluator``.  Intended for use in the app.py agent registry::

        "factory": lambda n: MCTSAgent(
            evaluator=make_nn_evaluator("models/best.pt"),
            num_simulations=n,
        ),
    """
    model = load_model(path, device=device)
    return NNEvaluator(model, device=device or DEVICE)


# ---------------------------------------------------------------------------
# Initialise a fresh network if none exists
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "models/best.pt"


def init_fresh_model(path: str = DEFAULT_MODEL_PATH) -> None:
    """Create and save a randomly initialised DualNetwork if *path* is absent."""
    if Path(path).exists():
        return
    model = DualNetwork()
    save_model(model, path)
    print(f"Initialised fresh model → {path}")
    del model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from game import State, action_space_size

    N = 7
    print(f"Device: {DEVICE}")

    def _test_model(model: DualNetwork | ViTDualNetwork, label: str) -> None:
        model = model.to(DEVICE)
        total = sum(p.numel() for p in model.parameters())
        print(f"\n[{label}]  Parameters: {total:,}")

        evaluator = NNEvaluator(model)
        s = State(boardsize=N)
        legal = s.get_legal_actions()

        t0 = time.perf_counter()
        priors, value = evaluator(s, legal)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  Inference:  {dt:.1f} ms")
        print(f"  Value: {value:.4f}   priors sum: {sum(priors):.6f}   n_legal: {len(priors)}")
        assert len(priors) == len(legal)
        assert abs(sum(priors) - 1.0) < 1e-5

        B = 64
        batch = torch.stack([
            torch.from_numpy(State(boardsize=N).to_nn_input()) for _ in range(B)
        ]).to(DEVICE)
        model.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            pl, v = model(batch)
            dt = (time.perf_counter() - t0) * 1000
        print(f"  Batch B={B}: {dt:.1f} ms  shapes: {tuple(pl.shape)}, {tuple(v.shape)}")

    print(f"Action space (N={N}): {action_space_size(N)}")

    _test_model(DualNetwork(boardsize=N, filters=64, num_residual=6),    "CNN  (F=64, R=6)")

    print("\nAll checks passed.")
