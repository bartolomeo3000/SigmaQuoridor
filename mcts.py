"""MCTS agent for Quoridor.

Design for NN plug-in
---------------------
The sole hook between MCTS and game evaluation is the ``Evaluator`` Protocol::

    evaluator(state, legal_actions) -> (priors, value)

* ``priors`` - list[float], one per legal action, summing to 1.  Uniform
  [1/n, …] for the pure-rollout baseline; replaced by the policy head of a
  neural network in AlphaZero mode.
* ``value``  - float in [-1, 1] from the perspective of
  ``state.get_current_player()``.  +1 means the current player is winning.

The default ``rollout_evaluator`` returns uniform priors and a random-playout
value.  To plug in a neural network, write a callable with the same signature
and pass it to ``MCTSAgent(evaluator=your_nn)``::

    agent = MCTSAgent(evaluator=my_nn_evaluator, num_simulations=800)
    action = agent.select_action(state)
"""

from __future__ import annotations

import math
import random
from typing import Protocol, runtime_checkable

import numpy as np

from game import Action, State


# ---------------------------------------------------------------------------
# Evaluator Protocol — the seam for neural-network integration
# ---------------------------------------------------------------------------

@runtime_checkable
class Evaluator(Protocol):
    """
    Callable that estimates the value of a non-terminal state and supplies
    prior probabilities over its legal actions.

    Parameters
    ----------
    state         : current non-terminal game state
    legal_actions : pre-computed legal actions for ``state``
                    (avoids redundant calls inside the agent)

    Returns
    -------
    priors : list[float]
        Prior probability for each element of ``legal_actions`` (must sum to 1).
    value  : float
        Estimated outcome in [-1, 1] from ``state.get_current_player()``'s
        perspective.  +1 = current player surely wins.
    """

    def __call__(
        self,
        state: State,
        legal_actions: list[Action],
    ) -> tuple[list[float], float]: ...


# ---------------------------------------------------------------------------
# Baseline evaluator: uniform priors + random rollout
# ---------------------------------------------------------------------------

def _random_rollout(state: State) -> float:
    """
    Play uniformly at random from *state* until the game is finished.

    Returns +1.0 if the player to move at *state* wins, -1.0 if that player
    loses, and 0.0 for a draw.
    """
    root_player = state.get_current_player()
    s = state
    while not s.is_finished():
        actions = s.get_legal_actions()
        s = s.next(random.choice(actions))
    w = s.winner()
    if w == 0:
        return 0.0
    return 1.0 if w == root_player else -1.0


def rollout_evaluator(
    state: State,
    legal_actions: list[Action],
) -> tuple[list[float], float]:
    """
    Baseline ``Evaluator``: uniform priors + single random rollout.

    This is the default for ``MCTSAgent``.  Replace it with a neural-network
    callable to switch to AlphaZero-style guided MCTS::

        agent = MCTSAgent(evaluator=my_nn_evaluator)
    """
    n = len(legal_actions)
    priors = [1.0 / n] * n
    value  = _random_rollout(state)
    return priors, value


# ---------------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------------

class MCTSNode:
    """
    One node in the MCTS search tree, representing a single game state.

    Statistics (``value_sum``, ``visit_count``) follow the **negamax
    convention**: values are always stored from the perspective of the player
    to move *at this node*.  ``q_value`` therefore represents how good the
    position is for the side about to move here.

    ``base_prior`` stores the evaluator's original prior so that Dirichlet
    noise can be re-applied consistently each time this node becomes the root,
    without compounding or drift.
    """

    __slots__ = (
        "state", "_parent_state", "parent", "action",
        "prior", "base_prior",
        "children",
        "visit_count", "value_sum",
        "is_expanded",
    )

    def __init__(
        self,
        state: State | None,
        parent: MCTSNode | None = None,
        action: Action | None = None,
        prior: float = 1.0,
        parent_state: State | None = None,
    ) -> None:
        self.state         = state
        self._parent_state = parent_state  # held until ensure_state() is called
        self.parent        = parent
        self.action        = action       # action taken at *parent* to reach this node
        self.prior         = prior        # effective prior (may include Dirichlet noise)
        self.base_prior    = prior        # evaluator's clean prior — never overwritten
        self.children:   list[MCTSNode] = []
        self.visit_count: int   = 0
        self.value_sum:   float = 0.0
        self.is_expanded: bool  = False

    def ensure_state(self) -> None:
        """
        Materialise this node's state on first visit (lazy expansion).

        Children are created with ``state=None`` and ``_parent_state`` set.
        The first call here computes ``parent_state.next(action)`` and caches
        the result; subsequent calls are no-ops.
        """
        if self.state is None:
            self.state = self._parent_state.next(self.action)
            self._parent_state = None   # release reference so parent can be GC'd

    # ------------------------------------------------------------------
    # Value / selection helpers
    # ------------------------------------------------------------------

    @property
    def q_value(self) -> float:
        """Mean value from the current player's perspective; 0.0 when unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def puct_score(self, parent_visit_count: int, c_puct: float) -> float:
        """
        PUCT score as seen from the *parent's* perspective (used to rank children).

        The minus sign on Q converts from the child-player's frame to the
        parent-player's frame (negamax: what is good for the child is bad for
        the parent)::

            score = -Q_child + c_puct x P x √N_parent / (1 + N_child)
        """
        u = (
            c_puct
            * self.prior
            * math.sqrt(parent_visit_count)
            / (1 + self.visit_count)
        )
        return -self.q_value + u

    def best_child(self, c_puct: float) -> MCTSNode:
        """Return the child with the highest PUCT score."""
        return max(
            self.children,
            key=lambda c: c.puct_score(self.visit_count, c_puct),
        )


# ---------------------------------------------------------------------------
# MCTS agent
# ---------------------------------------------------------------------------

class MCTSAgent:
    """
    Monte Carlo Tree Search agent using AlphaZero-style PUCT selection.

    Parameters
    ----------
    evaluator         : Evaluator
        Callable matching the ``Evaluator`` Protocol.  Defaults to
        ``rollout_evaluator`` (uniform priors + random rollout).  Swap with
        a neural-network callable to get AlphaZero-style guided MCTS.
    num_simulations   : int
        Number of tree-walk / leaf-evaluation calls per ``search()`` call.
        The initial root expansion is separate and does not count toward this.
    c_puct            : float
        Exploration constant in the PUCT formula.  Larger values favour
        breadth; smaller values exploit high-prior / high-value lines.
    training          : bool
        When ``True`` the agent is in **training mode**: Dirichlet noise is
        mixed into the root priors before each search, encouraging exploration
        of sub-optimal lines during self-play data generation.
        When ``False`` (the default) the agent is in **competitive mode**: no
        noise is added and ``select_action`` should be called with
        ``temperature=0`` (the default) for fully deterministic greedy play.
        Can be toggled at any time via ``agent.training = True/False``.
    temperature       : float
        Controls sharpness of the visit-count distribution in ``get_policy``
        and ``select_action``.

        * ``0.0`` (default) — deterministic argmax; always picks the
          most-visited action.  Use this for competitive play.
        * ``1.0`` — sample proportional to visit counts.  Use this during
          the opening moves of self-play games (typically first ~15 plies).
    dirichlet_alpha   : float
        Dirichlet concentration parameter — only used when ``training=True``.
        Smaller values produce sparser (more peaked) noise.  AlphaZero uses
        0.3 for chess and 0.15 for Go.
    dirichlet_epsilon : float
        Weight of the Dirichlet noise mixed into root-children priors —
        only used when ``training=True``.
    """

    def __init__(
        self,
        evaluator: Evaluator = rollout_evaluator,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        training: bool = False,
        temperature: float = 0.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        self.evaluator         = evaluator
        self.num_simulations   = num_simulations
        self.c_puct            = c_puct
        self.training          = training
        self.temperature       = temperature
        self.dirichlet_alpha   = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self._root: MCTSNode | None = None

    # ------------------------------------------------------------------
    # Core simulation primitives
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode) -> float:
        """
        Expand *node* by creating one child per legal action.

        Calls the evaluator — **this is the NN integration point**:

        * Pure MCTS  → ``rollout_evaluator`` runs a random playout.
        * AlphaZero  → a NN forward pass returns prior logits + state value.

        Returns the estimated value from the node's current player's
        perspective, to be propagated up the tree via ``_backup``.
        """
        legal_actions = node.state.get_legal_actions()
        priors, value = self.evaluator(node.state, legal_actions)
        for action, prior in zip(legal_actions, priors):
            node.children.append(
                MCTSNode(
                    state=None,
                    parent=node,
                    action=action,
                    prior=prior,
                    parent_state=node.state,
                )
            )
        node.is_expanded = True
        return value

    def _apply_dirichlet(self, node: MCTSNode) -> None:
        """
        Mix Dirichlet noise into *node*'s children priors for root exploration.

        Always computes from ``child.base_prior`` (the evaluator's clean
        output) so the result is idempotent regardless of how many times this
        method is called on the same node::

            child.prior = (1 - ε) x base_prior + ε x Dir(alpha)
        """
        if not node.children:
            return
        n     = len(node.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        eps   = self.dirichlet_epsilon
        for child, eta in zip(node.children, noise):
            child.prior = (1.0 - eps) * child.base_prior + eps * float(eta)

    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Descend from *root* via PUCT until reaching an unexpanded or terminal
        node (the leaf to evaluate next).
        """
        node = root
        while True:
            node.ensure_state()
            if not node.is_expanded or node.state.is_finished():
                break
            node = node.best_child(self.c_puct)
        return node

    @staticmethod
    def _terminal_value(state: State) -> float:
        """
        Value at a finished state from ``state.get_current_player()``'s view.

        At a terminal node the *current player* is the one whose turn would be
        next — the **previous** player just made the winning move.  Therefore:

        * winner != 0  →  current player lost  →  -1.0
        * draw         →                           0.0
        """
        if state.winner() != 0:
            return -1.0
        return 0.0

    def _backup(self, node: MCTSNode, value: float) -> None:
        """
        Walk from *node* toward the root, updating visit counts and value sums.

        *value* is from ``node.state.get_current_player()``'s perspective.
        Because players alternate turns, the sign flips at each level
        (negamax convention).
        """
        n = node
        while n is not None:
            n.visit_count += 1
            n.value_sum   += value
            value = -value
            n = n.parent

    def _run_simulation(self, root: MCTSNode) -> None:
        """One simulation: select a leaf → evaluate it → propagate the result."""
        leaf = self._select(root)
        leaf.ensure_state()   # materialise lazy child on first visit
        if leaf.state.is_finished():
            value = self._terminal_value(leaf.state)
        else:
            value = self._expand(leaf)
        self._backup(leaf, value)

    # ------------------------------------------------------------------
    # Subtree reuse
    # ------------------------------------------------------------------

    def _find_matching_node(self, target_key: tuple) -> MCTSNode | None:
        """
        Search up to two plies into the previously stored tree for a node
        whose position key matches *target_key*.

        Two plies covers the common cases: one half-move was played since the
        last ``search()`` call (the opponent replied), or two half-moves were
        played (agent moved, then opponent replied).
        """
        if self._root is None:
            return None
        for child in self._root.children:
            child.ensure_state()
            if child.state._position_key() == target_key:
                return child
            for grandchild in child.children:
                grandchild.ensure_state()
                if grandchild.state._position_key() == target_key:
                    return grandchild
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, state: State) -> MCTSNode:
        """
        Run MCTS from *state* and return the root node (with visit statistics).

        Attempts to reuse a matching subtree (up to two plies deep) from the
        previous call, falling back to a fresh tree when none is found.

        The root is expanded before the main simulation loop so that PUCT
        scores are well-defined from the first simulation onward.  This
        initial expansion is backed up but does not count toward
        ``num_simulations``.

        Side-effect: stores the new root in ``self._root`` for the next call.
        """
        target_key = state._position_key()
        reused = self._find_matching_node(target_key)

        if reused is not None:
            root = reused
            root.parent = None  # detach from old tree (allows GC of the rest)
        else:
            root = MCTSNode(state=state)

        # Expand root (if not already) and backup so visit_count > 0.
        # This ensures PUCT scores (which use sqrt(parent_visit_count)) are
        # non-degenerate from the very first simulation.
        if not root.is_expanded:
            init_value = self._expand(root)
            self._backup(root, init_value)

        # Apply Dirichlet noise to root's children prior to searching.
        # Only in training mode; _apply_dirichlet uses base_prior as the
        # source, so calling it again on a reused root is safe and produces
        # a freshly sampled noise mix each search call.
        if self.training:
            self._apply_dirichlet(root)

        for _ in range(self.num_simulations):
            self._run_simulation(root)

        self._root = root
        return root

    def get_policy(
        self,
        state: State,
    ) -> list[tuple[Action, float]]:
        """
        Run MCTS and return a probability distribution over legal actions.

        The sharpness is controlled by ``self.temperature``:

        * ``0.0`` — probability 1.0 for the most-visited action, 0.0 for all others
        * ``> 0`` — proportional to ``visit_count^(1 / temperature)``

        Returns a list of ``(Action, probability)`` pairs sorted by descending
        probability.
        """
        root     = self.search(state)
        children = root.children
        counts   = [c.visit_count for c in children]

        if self.temperature == 0.0:
            best  = counts.index(max(counts))
            probs = [1.0 if i == best else 0.0 for i in range(len(children))]
        else:
            inv_t = 1.0 / self.temperature
            raw   = [c ** inv_t for c in counts]
            total = sum(raw)
            probs = [r / total for r in raw] if total > 0 else [1.0 / len(raw)] * len(raw)

        policy = [(c.action, p) for c, p in zip(children, probs)]
        policy.sort(key=lambda x: x[1], reverse=True)
        return policy

    def select_action(self, state: State) -> Action:
        """
        Run MCTS and return a single action sampled from the visit distribution.

        Uses ``self.temperature``:
        ``0.0`` → deterministic argmax; ``1.0`` → stochastic proportional sampling.
        """
        policy  = self.get_policy(state)
        actions = [a for a, _ in policy]
        probs   = np.array([p for _, p in policy])
        idx     = int(np.random.choice(len(actions), p=probs))
        return actions[idx]
