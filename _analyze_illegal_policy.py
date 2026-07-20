"""Empirically inspect raw policy logits assigned to illegal moves.

The MCTS evaluator masks illegal moves before softmax. This script checks what
the network itself outputs before that mask is applied.

Canonicalization: the net emits policy in the current player's canonical
(P2-vertically-flipped) frame, so a real-board action ``a`` for P2 is read at
``vperm[action_to_index(a)]``, not ``action_to_index(a)``. This script applies
that mapping (mirroring NNEvaluator) so the legal/illegal partition is correct
for full-canonical nets. For pre-fix half-canonical nets (7x7, old 9x9 ≤339)
the net was trained in the real frame; pass --half-canonical to skip the flip.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from dual_network import load_model
from game import (
    State,
    action_space_size,
    action_to_index,
    index_to_action,
    vert_policy_permutation,
)


def _canon_legal_indices(state: State, boardsize: int, vperm: np.ndarray | None) -> np.ndarray:
    """Logit indices of ``state``'s legal actions, in the network's output frame.

    For P2 (when ``vperm`` is provided) each real action's index is mapped
    through the vertical permutation to its canonical-frame position; for P1
    (or half-canonical nets, ``vperm is None``) the raw index is used.
    """
    raw = np.array([action_to_index(a, boardsize) for a in state.get_legal_actions()],
                   dtype=np.int64)
    if vperm is not None and not state.is_player1_turn():
        return vperm[raw]
    return raw


@dataclass
class SampleStats:
    n_states: int
    n_actions: int
    mean_legal_count: float
    mean_illegal_count: float
    mean_legal_logit: float
    mean_illegal_logit: float
    median_legal_logit: float
    median_illegal_logit: float
    mean_illegal_softmax_mass: float
    median_illegal_softmax_mass: float
    p90_illegal_softmax_mass: float
    top1_illegal_rate: float
    top5_illegal_mean: float


def sample_random_states(boardsize: int, walls: int, n_states: int, seed: int) -> list[State]:
    rng = np.random.default_rng(seed)
    states: list[State] = []
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)

    while len(states) < n_states:
        if state.is_finished():
            state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
            continue

        states.append(state)
        actions = state.get_legal_actions()
        state = state.next(actions[int(rng.integers(len(actions)))])

    return states


def sample_policy_states(model, boardsize: int, walls: int, n_states: int, seed: int,
                         device: torch.device, vperm: np.ndarray | None) -> list[State]:
    rng = np.random.default_rng(seed)
    states: list[State] = []
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)

    while len(states) < n_states:
        if state.is_finished():
            state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
            continue

        states.append(state)
        actions = state.get_legal_actions()
        indices = torch.tensor(_canon_legal_indices(state, boardsize, vperm),
                               dtype=torch.long, device=device)
        x = torch.from_numpy(state.to_nn_input()).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits[0, indices], dim=0).cpu().numpy()
        chosen = int(rng.choice(len(actions), p=probs))
        state = state.next(actions[chosen])

    return states


def analyze_states(model, states: list[State], device: torch.device,
                   vperm: np.ndarray | None) -> SampleStats:
    if not states:
        raise ValueError("No states to analyze")

    boardsize = states[0].boardsize
    n_actions = action_space_size(boardsize)

    legal_logits_all: list[np.ndarray] = []
    illegal_logits_all: list[np.ndarray] = []
    legal_counts: list[int] = []
    illegal_counts: list[int] = []
    illegal_masses: list[float] = []
    top1_illegal: list[bool] = []
    top5_illegal_frac: list[float] = []

    for state in states:
        legal_idx = _canon_legal_indices(state, boardsize, vperm)
        legal_mask = np.zeros(n_actions, dtype=bool)
        legal_mask[legal_idx] = True

        x = torch.from_numpy(state.to_nn_input()).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(x)
            logits_np = logits[0].detach().cpu().numpy()

        probs = torch.softmax(torch.from_numpy(logits_np), dim=0).numpy()
        illegal_mask = ~legal_mask

        legal_logits_all.append(logits_np[legal_mask])
        illegal_logits_all.append(logits_np[illegal_mask])
        legal_counts.append(int(legal_mask.sum()))
        illegal_counts.append(int(illegal_mask.sum()))
        illegal_masses.append(float(probs[illegal_mask].sum()))

        order = np.argsort(logits_np)[::-1]
        top1_illegal.append(bool(illegal_mask[order[0]]))
        top5 = order[: min(5, len(order))]
        top5_illegal_frac.append(float(illegal_mask[top5].mean()))

    legal_logits = np.concatenate(legal_logits_all)
    illegal_logits = np.concatenate(illegal_logits_all)
    illegal_masses_arr = np.array(illegal_masses)

    return SampleStats(
        n_states=len(states),
        n_actions=n_actions,
        mean_legal_count=float(np.mean(legal_counts)),
        mean_illegal_count=float(np.mean(illegal_counts)),
        mean_legal_logit=float(np.mean(legal_logits)),
        mean_illegal_logit=float(np.mean(illegal_logits)),
        median_legal_logit=float(np.median(legal_logits)),
        median_illegal_logit=float(np.median(illegal_logits)),
        mean_illegal_softmax_mass=float(np.mean(illegal_masses_arr)),
        median_illegal_softmax_mass=float(np.median(illegal_masses_arr)),
        p90_illegal_softmax_mass=float(np.quantile(illegal_masses_arr, 0.90)),
        top1_illegal_rate=float(np.mean(top1_illegal)),
        top5_illegal_mean=float(np.mean(top5_illegal_frac)),
    )


def print_stats(label: str, stats: SampleStats) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(f"states analyzed                 : {stats.n_states}")
    print(f"action space size               : {stats.n_actions}")
    print(f"mean legal / illegal actions    : {stats.mean_legal_count:.1f} / {stats.mean_illegal_count:.1f}")
    print(f"mean legal / illegal logit      : {stats.mean_legal_logit:.4f} / {stats.mean_illegal_logit:.4f}")
    print(f"median legal / illegal logit    : {stats.median_legal_logit:.4f} / {stats.median_illegal_logit:.4f}")
    print(f"unmasked illegal softmax mass   : mean={stats.mean_illegal_softmax_mass:.3f}, median={stats.median_illegal_softmax_mass:.3f}, p90={stats.p90_illegal_softmax_mass:.3f}")
    print(f"unmasked top-1 illegal rate     : {100 * stats.top1_illegal_rate:.1f}%")
    print(f"mean illegal fraction in top-5  : {100 * stats.top5_illegal_mean:.1f}%")


def show_initial_state_top(model, boardsize: int, walls: int, device: torch.device, k: int = 12) -> None:
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
    legal_idx = {action_to_index(a, boardsize) for a in state.get_legal_actions()}
    x = torch.from_numpy(state.to_nn_input()).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x)
        logits_np = logits[0].detach().cpu().numpy()
    order = np.argsort(logits_np)[::-1][:k]
    print(f"\nTop {k} raw logits in initial state")
    print("--------------------------------")
    for rank, idx in enumerate(order, 1):
        legality = "legal" if int(idx) in legal_idx else "ILLEGAL"
        print(f"{rank:2d}. idx={int(idx):3d}  {str(index_to_action(int(idx), boardsize)):35s}  logit={logits_np[idx]:8.4f}  {legality}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw NN policy mass on illegal actions")
    parser.add_argument("--model", default="models_9x9_heads/best.pt")
    parser.add_argument("--boardsize", type=int, default=9)
    parser.add_argument("--walls", type=int, default=10)
    parser.add_argument("--states", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half-canonical", action="store_true",
                        help="net was trained in the real (pre-fix) frame — skip the P2 "
                             "policy un-flip (use for 7x7 / old 9x9 <=339 checkpoints)")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model, device=device)
    model.eval()

    vperm = None if args.half_canonical else vert_policy_permutation(args.boardsize)

    print(f"model      : {args.model}")
    print(f"device     : {device}")
    print(f"boardsize  : {args.boardsize}")
    print(f"walls      : {args.walls}")
    print(f"canonical  : {'half (no P2 un-flip)' if args.half_canonical else 'full (P2 policy un-flipped)'}")

    random_states = sample_random_states(args.boardsize, args.walls, args.states, args.seed)
    policy_states = sample_policy_states(model, args.boardsize, args.walls, args.states, args.seed + 1, device, vperm)

    print_stats("Random legal states", analyze_states(model, random_states, device, vperm))
    print_stats("States sampled from masked NN policy", analyze_states(model, policy_states, device, vperm))
    show_initial_state_top(model, args.boardsize, args.walls, device)

    print("\nNote: MCTS/NNEvaluator applies softmax only over legal actions, so illegal moves get exactly zero prior after masking.")


if __name__ == "__main__":
    main()