"""Dual network (policy + value head) for AlphaZero-style MCTS.

Architecture
------------
Input: (B, 8, N, N) tensor from State.to_nn_input().

Trunk
  Conv2d(8 → F, 3x3, pad=1) → BN → ReLU
  R residual blocks (each: Conv→BN→ReLU→Conv→BN, skip-add→ReLU)

Policy head — three spatial sub-heads, concatenated into raw logits:
  ┌─ Pawn (8 logits): two variants, selected by ``pawn_head``:
  │    "local"  (default): gather trunk features at my/opponent pawn cells
  │                (via input one-hot planes 0/1) + global mean → MLP → 8
  │    "legacy": global avg-pool → Linear(F → 8)
  ├─ H-walls (W² logits): Conv2d(F→1, 1x1) → BN → ReLU
  │                         → slice [:, 0, :W, :W]  (W = N-1)
  │                         → flatten row-major
  └─ V-walls (W² logits): same structure
  → cat → (B, 8 + 2*W²) raw logits   [NO softmax — applied in NNEvaluator]

Value head — two variants, selected by ``value_head``:
  "pooled" (default): Conv2d(F→32,1x1) → BN → ReLU → mean+max pool → (B,64)
                       → Linear(64→128) → ReLU → Linear(128→1) → tanh
  "legacy":            Conv2d(F→1, 1x1) → BN → ReLU → flatten → Linear(N²→64)
                       → ReLU → Linear(64→1) → tanh

Both variants are self-describing: ``_infer_arch`` detects which one a saved
checkpoint uses from the presence of its variant-specific keys, so
``load_model`` reconstructs old and new checkpoints correctly without being
told which kind a given file is. See HEAD_REDESIGN_PLAN.md.

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

from game import (State, Action, action_to_index, action_space_size,
                  vert_policy_permutation)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

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


class GPoolResidualBlock(nn.Module):
    """KataGo-style residual block with a global-pooling side branch.

    conv1 produces ``c_reg + c_pool`` channels. The ``c_pool`` "pooling"
    channels are reduced to per-channel global statistics (mean and max over
    the board) and mapped by a linear layer to a per-channel bias that is
    added to the ``c_reg`` regular channels — giving every spatial position
    immediate access to global context (wall budgets, overall race state)
    without waiting for it to diffuse through many 3x3 conv layers.
    See "Accelerating Self-Play Learning in Go" (KataGo), sec. 4.1.
    """

    def __init__(self, filters: int, gpool_channels: int | None = None) -> None:
        super().__init__()
        c_pool = gpool_channels if gpool_channels is not None else max(8, filters // 4)
        c_reg  = filters - c_pool
        if c_reg <= 0:
            raise ValueError(f"gpool_channels={c_pool} must be < filters={filters}")
        self.c_reg, self.c_pool = c_reg, c_pool

        self.conv1     = nn.Conv2d(filters, c_reg + c_pool, 3, padding=1, bias=False)
        self.bn1_reg   = nn.BatchNorm2d(c_reg)
        self.bn1_pool  = nn.BatchNorm2d(c_pool)
        self.gpool_fc  = nn.Linear(2 * c_pool, c_reg)
        self.conv2     = nn.Conv2d(c_reg, filters, 3, padding=1, bias=False)
        self.bn2       = nn.BatchNorm2d(filters)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = x
        y  = self.conv1(x)
        r  = self.bn1_reg(y[:, : self.c_reg])
        p  = F.relu(self.bn1_pool(y[:, self.c_reg :]))          # (B, c_pool, N, N)
        pooled = torch.cat([p.mean(dim=(2, 3)),
                            p.amax(dim=(2, 3))], dim=1)          # (B, 2*c_pool)
        bias   = self.gpool_fc(pooled)                           # (B, c_reg)
        r  = F.relu(r + bias[:, :, None, None])
        y  = self.bn2(self.conv2(r))
        return F.relu(y + sc)


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
    gpool_every  : int   if > 0, every gpool_every-th residual block (1-based:
                         blocks gpool_every-1, 2*gpool_every-1, ...) is a
                         KataGo-style GPoolResidualBlock instead of a plain
                         ResidualBlock. 0 disables (default, matches all
                         pre-existing checkpoints).
    value_head   : str   "pooled" (default, new) or "legacy". See module
                         docstring. Only affects fresh construction —
                         load_model always reconstructs whichever variant a
                         checkpoint was actually saved with.
    pawn_head    : str   "local" (default, new) or "legacy". See module
                         docstring.
    """

    IN_CHANNELS = 8  # channels produced by State.to_nn_input()
    _VALUE_POOL_CHANNELS = 32
    _VALUE_POOL_HIDDEN   = 128
    _PAWN_LOCAL_HIDDEN   = 64

    def __init__(
        self,
        boardsize:    int = 7,
        filters:      int = 64,
        num_residual: int = 6,
        gpool_every:  int = 0,
        value_head:   str = "pooled",
        pawn_head:    str = "local",
    ) -> None:
        super().__init__()
        if value_head not in ("pooled", "legacy"):
            raise ValueError(f"value_head must be 'pooled' or 'legacy', got {value_head!r}")
        if pawn_head not in ("local", "legacy"):
            raise ValueError(f"pawn_head must be 'local' or 'legacy', got {pawn_head!r}")
        self.boardsize    = boardsize
        self.filters      = filters
        self.num_residual = num_residual
        self.gpool_every  = gpool_every
        self.value_head   = value_head
        self.pawn_head    = pawn_head
        N = boardsize
        W = N - 1   # wall-anchor grid side length

        # ── Trunk ────────────────────────────────────────────────────────────
        self.conv = nn.Conv2d(self.IN_CHANNELS, filters, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(filters)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        self.residuals = nn.Sequential(*[
            GPoolResidualBlock(filters)
            if gpool_every > 0 and (i + 1) % gpool_every == 0
            else ResidualBlock(filters)
            for i in range(num_residual)
        ])

        # ── Policy head ──────────────────────────────────────────────────────
        if pawn_head == "local":
            # Gather trunk features at my/opponent pawn cells (exact, via the
            # input one-hot planes) plus a global-mean summary, then an MLP.
            # Local features carry step/wall-adjacency context that a pure
            # avg-pool discards; the opponent-cell feature matters mainly for
            # jump/diagonal legality (relevant only when adjacent).
            H_p = self._PAWN_LOCAL_HIDDEN
            self.policy_pawn_fc1 = nn.Linear(3 * filters, H_p)
            self.policy_pawn_fc2 = nn.Linear(H_p, 8)
        else:
            # Legacy: global avg-pool preserves no spatial info.
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
        if value_head == "pooled":
            # KataGo-style: small conv + mean+max global pool feeds the FC,
            # instead of squeezing the whole trunk through one spatial channel.
            C_v = self._VALUE_POOL_CHANNELS
            H_v = self._VALUE_POOL_HIDDEN
            self.value_conv    = nn.Conv2d(filters, C_v, kernel_size=1, bias=False)
            self.value_bn      = nn.BatchNorm2d(C_v)
            self.value_pool_fc = nn.Linear(2 * C_v, H_v)
            self.value_fc2     = nn.Linear(H_v, 1)
            # value_fc1 (legacy) is board-size-specific (N*N in_features); the
            # pooled head has no such tensor, so boardsize can no longer be
            # inferred from a weight shape. Serialize it explicitly instead —
            # only pooled-head checkpoints carry this key, so it can't clash
            # with (or be required by) legacy checkpoints saved before it existed.
            self.register_buffer("boardsize_marker", torch.tensor(boardsize))
        else:
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
        x_in = x   # raw input, needed by the "local" pawn head (captured before
                   # the trunk overwrites x below); planes 0/1 are the my/opponent
                   # pawn one-hot planes (see State.to_nn_input()).

        # ── Trunk ────────────────────────────────────────────────────────────
        x = F.relu(self.bn(self.conv(x)))   # (B, F, N, N)
        x = self.residuals(x)               # (B, F, N, N)
        trunk = x

        # ── Policy head ──────────────────────────────────────────────────────
        if self.pawn_head == "local":
            # Exact differentiable gather at the one-hot pawn cell via
            # mask-multiply-and-sum (mask has a single 1.0, so the sum equals
            # the trunk feature vector at that cell) + a global-mean summary.
            my_mask   = x_in[:, 0:1]                        # (B,1,N,N)
            opp_mask  = x_in[:, 1:2]                        # (B,1,N,N)
            feat_my   = (trunk * my_mask ).sum(dim=(2, 3))  # (B,F) at my pawn
            feat_opp  = (trunk * opp_mask).sum(dim=(2, 3))  # (B,F) at opp pawn
            feat_glob = trunk.mean(dim=(2, 3))              # (B,F) global context
            p_pawn = torch.cat([feat_my, feat_opp, feat_glob], dim=1)  # (B,3F)
            p_pawn = self.policy_pawn_fc2(F.relu(self.policy_pawn_fc1(p_pawn)))  # (B,8)
        else:
            # Legacy: global average pool → (B, F) → FC → (B, 8)
            p_pawn = trunk.mean(dim=(2, 3))         # (B, F)
            p_pawn = self.policy_pawn_fc(p_pawn)    # (B, 8)

        # H-walls: 1×1 conv → slice W×W anchor region → flatten → (B, W²)
        h = F.relu(self.policy_h_bn(self.policy_h_conv(trunk)))  # (B, 1, N, N)
        h = h[:, 0, :W, :W].contiguous().view(B, -1)             # (B, W²)

        # V-walls: identical
        v = F.relu(self.policy_v_bn(self.policy_v_conv(trunk)))  # (B, 1, N, N)
        v = v[:, 0, :W, :W].contiguous().view(B, -1)             # (B, W²)

        # Concatenate in the same order as action_to_index()
        policy_logits = torch.cat([p_pawn, h, v], dim=1)     # (B, 8+2*W²)

        # ── Value head ──────────────────────────────────────────────────────
        if self.value_head == "pooled":
            val = F.relu(self.value_bn(self.value_conv(trunk)))   # (B, C_v, N, N)
            val = torch.cat([val.mean(dim=(2, 3)),
                             val.amax(dim=(2, 3))], dim=1)        # (B, 2*C_v)
            val = F.relu(self.value_pool_fc(val))                 # (B, H_v)
            val = torch.tanh(self.value_fc2(val))                 # (B, 1)
        else:
            val = F.relu(self.value_bn(self.value_conv(trunk)))  # (B, 1, N, N)
            val = val.view(B, -1)                                # (B, N²)
            val = F.relu(self.value_fc1(val))                    # (B, 64)
            val = torch.tanh(self.value_fc2(val))                # (B, 1)

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
        # Vertical action permutation (raw <-> canonical frame). The network
        # produces policy in the current player's canonical POV, so for P2 we
        # gather each real action's logit at its vertically-flipped index.
        self._vperm = torch.from_numpy(
            vert_policy_permutation(self.model.boardsize)
        ).to(self.device)

    def _canon_indices(self, legal_actions: list[Action], flip: bool) -> torch.Tensor:
        """Policy-head indices at which to read each legal action's logit.

        For P2 (``flip``), map the raw action index to its canonical (vertically
        flipped) index so the raw-frame legal action is read from the network's
        canonical-frame output.
        """
        idx = torch.tensor(
            [action_to_index(a, self.model.boardsize) for a in legal_actions],
            dtype=torch.long,
            device=self.device,
        )
        return self._vperm[idx] if flip else idx

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
            indices      = self._canon_indices(legal_actions,
                                               not state.is_player1_turn())
            legal_logits = policy_logits[indices]                  # (n_legal,)
            priors       = torch.softmax(legal_logits, dim=0).cpu().tolist()

            return priors, float(value[0, 0])

    def batch_call_raw(
        self,
        arrays:             list[np.ndarray],
        legal_actions_list: list[list[Action]],
        flips:              list[bool] | None = None,
    ) -> list[tuple[list[float], float]]:
        """
        Evaluate a batch of pre-computed ``(8, N, N)`` float32 arrays in one
        forward pass.

        Accepts numpy arrays rather than ``State`` objects so that callers
        such as ``SymmetricEvaluator`` can supply pre-flipped inputs without
        reconstructing state objects.

        ``flips[i]`` selects the canonical (vertical) frame mapping for item i
        (True for P2-to-move positions); ``None`` means no item is flipped.
        Because the arrays carry no side-to-move information, the caller must
        supply this explicitly whenever P2 positions are in the batch.

        Returns one ``(priors, value)`` pair per input array.
        """
        inputs    = torch.from_numpy(np.stack(arrays)).to(self.device)  # (B, 8, N, N)
        with torch.no_grad():
            logits_batch, values_batch = self.model(inputs)              # (B, A), (B, 1)

        if flips is None:
            flips = [False] * len(legal_actions_list)
        results: list[tuple[list[float], float]] = []
        for la, logits, val, flip in zip(legal_actions_list, logits_batch,
                                         values_batch, flips):
            indices = self._canon_indices(la, flip)
            priors  = torch.softmax(logits[indices], dim=0).cpu().tolist()
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
        ``batch_call_raw``, passing the per-state canonical-frame flip.
        """
        arrays = [s.to_nn_input() for s in states]
        flips  = [not s.is_player1_turn() for s in states]
        return self.batch_call_raw(arrays, legal_actions_list, flips)


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


def _infer_arch(state_dict: dict) -> tuple[int, int, int, int, str, str]:
    """
    Infer (filters, num_residual, boardsize, gpool_every, value_head, pawn_head)
    from a saved state dict so that checkpoints can be loaded without knowing
    the architecture upfront.

    Derivations
    -----------
    * filters      — output channels of the first conv layer
    * num_residual — number of ResidualBlock entries in 'residuals.*'
    * value_head   — "pooled" if 'value_pool_fc.weight' is present, else "legacy"
    * pawn_head    — "local" if 'policy_pawn_fc1.weight' is present, else "legacy"
    * boardsize    — legacy value head: value_fc1 maps N² → 64, so
                     boardsize = sqrt(in_features). The pooled value head has
                     no board-size-shaped tensor, so it instead carries an
                     explicit 'boardsize_marker' buffer, checked first.
    * gpool_every  — position of the first block with a 'gpool_fc' key (+1);
                     0 when no gpool blocks are present
    """
    filters = state_dict["conv.weight"].shape[0]
    num_residual = sum(
        1 for k in state_dict
        if k.startswith("residuals.") and k.endswith(".conv1.weight")
    )
    value_head = "pooled" if "value_pool_fc.weight" in state_dict else "legacy"
    pawn_head  = "local"  if "policy_pawn_fc1.weight" in state_dict else "legacy"
    if "boardsize_marker" in state_dict:
        boardsize = int(state_dict["boardsize_marker"].item())
    else:
        boardsize = int(math.sqrt(state_dict["value_fc1.weight"].shape[1]))
    gpool_idx = [
        int(k.split(".")[1]) for k in state_dict
        if k.startswith("residuals.") and k.endswith(".gpool_fc.weight")
    ]
    gpool_every = (min(gpool_idx) + 1) if gpool_idx else 0
    return filters, num_residual, boardsize, gpool_every, value_head, pawn_head


def load_model(path: str, device: torch.device | None = None) -> DualNetwork:
    """
    Load a checkpoint from *path*.  All architecture hyper-parameters are
    inferred automatically from the saved weight shapes.
    """
    dev = device or DEVICE
    sd  = torch.load(path, map_location=dev, weights_only=True)
    # Training checkpoints wrap the model state under "model_state"
    if "model_state" in sd:
        sd = sd["model_state"]
    filters, num_residual, boardsize, gpool_every, value_head, pawn_head = _infer_arch(sd)
    model = DualNetwork(
        boardsize    = boardsize,
        filters      = filters,
        num_residual = num_residual,
        gpool_every  = gpool_every,
        value_head   = value_head,
        pawn_head    = pawn_head,
    ).to(dev)
    model.load_state_dict(sd)
    return model


def warm_start_from_legacy(
    old_path:   str,
    value_head: str = "pooled",
    pawn_head:  str = "local",
    device:     torch.device | None = None,
) -> DualNetwork:
    """
    Load *old_path* and build a new ``DualNetwork`` with the same trunk
    hyperparameters but the given (presumably new) head variants, transferring
    every tensor whose key AND shape are unchanged (trunk conv/BN/residuals,
    wall sub-heads) and leaving the changed/new head tensors at their fresh
    initialization.

    Used to start a head-redesign experiment from an existing trained net
    without discarding the trunk. See HEAD_REDESIGN_PLAN.md.
    """
    dev = device or DEVICE
    old = load_model(old_path, device=dev)
    new = DualNetwork(
        boardsize    = old.boardsize,
        filters      = old.filters,
        num_residual = old.num_residual,
        gpool_every  = old.gpool_every,
        value_head   = value_head,
        pawn_head    = pawn_head,
    ).to(dev)

    old_sd = old.state_dict()
    new_sd = new.state_dict()
    copied = [k for k, v in old_sd.items() if k in new_sd and new_sd[k].shape == v.shape]
    for k in copied:
        new_sd[k] = old_sd[k]
    new.load_state_dict(new_sd)

    fresh = sorted(set(new_sd) - set(copied))
    print(f"warm-started {len(copied)}/{len(new_sd)} tensors from {old_path}")
    print(f"freshly initialized ({len(fresh)}): {fresh}")
    return new


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

    def _test_model(model: DualNetwork, label: str) -> None:
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
