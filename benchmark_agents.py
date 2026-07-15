"""
Lightweight benchmark agents for Quoridor.

    RandomAgent         — uniform random over all legal actions.
    GreedyDistanceAgent — MinimaxAgent fixed at depth 1.
    RawPolicyAgent      — argmax over evaluator policy priors only.

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


class RawPolicyAgent:
    """Chooses the argmax legal action from evaluator policy priors."""

    num_simulations = None

    def __init__(self, evaluator, seed: int | None = None):
        self.evaluator = evaluator
        self.rng = random.Random(seed)

    def select_action(self, state: State, training: bool = False) -> Action:
        legal = state.get_legal_actions()
        priors, _ = self.evaluator(state, legal)
        best = max(priors)
        candidates = [a for a, p in zip(legal, priors) if p == best]
        return self.rng.choice(candidates)

    def get_policy(self, state: State) -> list[tuple[Action, float]]:
        legal = state.get_legal_actions()
        priors, _ = self.evaluator(state, legal)
        policy = list(zip(legal, priors))
        policy.sort(key=lambda x: x[1], reverse=True)
        return policy


class MinimaxAgent:
    """Greedy-distance heuristic with alpha-beta minimax at any positive depth.

    Move ordering at every node uses ``compute_move_advantages`` (current
    player's perspective), which provides good alpha-beta cutoffs without
    any extra state copies.

    Terminal values:
      win  → +1e9   lose → -1e9   draw → 0
    Leaf heuristic: opponent_dist_to_goal − my_dist_to_goal
      (both measured from the maximising player's perspective).
    """

    num_simulations = None

    def __init__(self, depth: int = 3, seed: int | None = None):
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("MinimaxAgent depth must be a positive integer")
        self.depth = depth
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic(state: State, maximizing_player: int) -> float:
        """Distance-based leaf evaluation from the maximising player's POV."""
        p1x, p1y = state.player1pos
        p2x, p2y = state.player2pos
        p1_dist = state.p1_dist[p1y][p1x]
        p2_dist = state.p2_dist[p2y][p2x]
        if maximizing_player == 1:
            return float(p2_dist - p1_dist)
        else:
            return float(p1_dist - p2_dist)

    @staticmethod
    def _terminal_value(state: State, maximizing_player: int) -> float | None:
        """Return a finite score if the state is terminal, else None."""
        w = state.winner()
        if w == maximizing_player:
            return 1e9
        if w != 0:          # opponent won
            return -1e9
        if state.is_drawn():
            return 0.0
        return None

    @staticmethod
    def _ordered_actions(state: State, legal_actions: list) -> list:
        """Sort actions best-first for the current mover (improves α-β cutoffs).
        Immediately winning moves are placed first regardless of heuristic score."""
        advantages = state.compute_move_advantages(legal_actions)
        WIN_BONUS = 1e12
        scored = [
            (a, WIN_BONUS if state.next(a).winner() != 0 else adv)
            for a, adv in zip(legal_actions, advantages)
        ]
        return [a for a, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

    def _alphabeta(self, state: State, depth: int, alpha: float, beta: float,
                   maximizing: bool, maximizing_player: int) -> float:
        tv = self._terminal_value(state, maximizing_player)
        if tv is not None:
            return tv
        if depth == 0:
            return self._heuristic(state, maximizing_player)

        legal = state.get_legal_actions()
        if not legal:
            return self._heuristic(state, maximizing_player)

        ordered = self._ordered_actions(state, legal)

        if maximizing:
            value = -1e18
            for action in ordered:
                value = max(value, self._alphabeta(
                    state.next(action), depth - 1, alpha, beta, False, maximizing_player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = 1e18
            for action in ordered:
                value = min(value, self._alphabeta(
                    state.next(action), depth - 1, alpha, beta, True, maximizing_player))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # ------------------------------------------------------------------
    # Public interface (same duck-type as the other agents)
    # ------------------------------------------------------------------

    def select_action(self, state: State, training: bool = False) -> Action:
        maximizing_player = 1 if state.is_player1_turn() else 2
        legal = state.get_legal_actions()

        # Short-circuit: take a free win without running the full search.
        wins = [a for a in legal if state.next(a).winner() != 0]
        if wins:
            return self.rng.choice(wins)

        ordered = self._ordered_actions(state, legal)

        best_val = -1e18
        best_actions: list[Action] = []
        for action in ordered:
            val = self._alphabeta(
                state.next(action), self.depth - 1, best_val, 1e18,
                False, maximizing_player)
            if val > best_val:
                best_val = val
                best_actions = [action]
            elif val == best_val:
                best_actions.append(action)

        return self.rng.choice(best_actions)

    def evaluator(self, state: State, legal_actions: list):
        maximizing_player = 1 if state.is_player1_turn() else 2
        values = [
            self._alphabeta(state.next(a), self.depth - 1, -1e18, 1e18,
                            False, maximizing_player)
            for a in legal_actions
        ]
        arr = np.array(values, dtype=float)
        arr -= arr.max()
        exp_a = np.exp(arr)
        return (exp_a / exp_a.sum()).tolist(), 0.0

    def get_policy(self, state: State) -> list[tuple[Action, float]]:
        legal = state.get_legal_actions()
        maximizing_player = 1 if state.is_player1_turn() else 2
        values = [
            self._alphabeta(state.next(a), self.depth - 1, -1e18, 1e18,
                            False, maximizing_player)
            for a in legal
        ]
        best = max(values)
        policy = [(a, 1.0 if v == best else 0.0) for a, v in zip(legal, values)]
        policy.sort(key=lambda x: x[1], reverse=True)
        return policy


class GreedyDistanceAgent(MinimaxAgent):
    """One-ply MinimaxAgent."""

    def __init__(self, seed: int | None = None):
        super().__init__(depth=1, seed=seed)
