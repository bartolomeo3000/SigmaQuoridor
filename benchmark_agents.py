"""
Lightweight benchmark agents for Quoridor.

  RandomAgent         — uniform random over all legal actions.
  GreedyDistanceAgent — greedily picks the move that maximises
                        (opp_dist_to_goal - my_dist_to_goal) in the
                        resulting position, using State.compute_move_advantages.
                        Ties broken uniformly at random.

Both agents expose the same duck-type interface as MCTSAgent so they plug
straight into the Flask app and evaluation scripts:
  select_action(state)             -> Action
  evaluator(state, legal_actions)  -> (priors: list[float], value: float)
  num_simulations                  -> None  (not applicable)
"""

from __future__ import annotations

import random
import numpy as np

from game import State, Action


class RandomAgent:
    """Chooses a legal action uniformly at random."""

    num_simulations = None

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, state: State, training: bool = False) -> Action:
        return self.rng.choice(state.get_legal_actions())

    def evaluator(self, state: State, legal_actions: list[Action]):
        n = len(legal_actions)
        return [1.0 / n] * n, 0.0

    def get_policy(self, state: State) -> list[tuple[Action, float]]:
        legal = state.get_legal_actions()
        p = 1.0 / len(legal)
        return [(a, p) for a in legal]


class GreedyDistanceAgent:
    """Greedily maximises (opp_dist_to_goal − my_dist_to_goal) after the move.

    Uses State.compute_move_advantages, which returns
        (opp_dist_after − my_dist_after) − (opp_dist_now − my_dist_now)
    for each legal action.  Since the baseline term is the same for every
    action in a given position, argmax of the delta equals argmax of the
    absolute distance difference.
    """

    num_simulations = None

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, state: State, training: bool = False) -> Action:
        legal = state.get_legal_actions()
        advantages = state.compute_move_advantages(legal)
        best = max(advantages)
        candidates = [a for a, v in zip(legal, advantages) if v == best]
        return self.rng.choice(candidates)

    def evaluator(self, state: State, legal_actions: list[Action]):
        advantages = state.compute_move_advantages(legal_actions)
        arr = np.array(advantages, dtype=float)
        arr -= arr.max()          # numerical stability
        exp_a = np.exp(arr)
        priors = (exp_a / exp_a.sum()).tolist()
        return priors, 0.0

    def get_policy(self, state: State) -> list[tuple[Action, float]]:
        legal = state.get_legal_actions()
        advantages = state.compute_move_advantages(legal)
        best = max(advantages)
        policy = [(a, 1.0 if v == best else 0.0) for a, v in zip(legal, advantages)]
        policy.sort(key=lambda x: x[1], reverse=True)
        return policy
