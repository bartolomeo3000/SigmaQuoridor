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
        "heuristic_bonus",
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
        self.children:        list[MCTSNode] = []
        self.visit_count:    int   = 0
        self.value_sum:      float = 0.0
        self.is_expanded:    bool  = False
        self.heuristic_bonus: float = 0.0

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

    def puct_score(
        self,
        parent_visit_count: int,
        c_puct: float,
        parent_q: float = 0.0,
        fpu_reduction: float = 0.0,
        visited_prior_sum: float = 0.0,
    ) -> float:
        """
        PUCT score as seen from the *parent's* perspective (used to rank children).

        The minus sign on Q converts from the child-player's frame to the
        parent-player's frame (negamax: what is good for the child is bad for
        the parent)::

            score = -Q_child + c_puct x P x √N_parent / (1 + N_child)

        FPU (First Play Urgency): unvisited nodes (visit_count == 0) use
        ``parent_q - fpu_reduction * sqrt(visited_prior_sum)`` as their Q
        estimate instead of 0.0, where ``visited_prior_sum`` is the sum of
        base priors of already-visited siblings.

        This mirrors KataGo's dynamic FPU: the penalty starts at 0 when no
        siblings have been explored yet (no evidence to trust parent_q) and
        grows as more of the policy mass is visited (increasing confidence
        that remaining unvisited children are indeed worse).
        """
        u = (
            c_puct
            * self.prior
            * math.sqrt(parent_visit_count)
            / (1 + self.visit_count)
        )
        if self.visit_count == 0:
            q_from_parent = parent_q - fpu_reduction * math.sqrt(visited_prior_sum)
        else:
            q_from_parent = -self.q_value  # negamax: child's good = parent's bad
        return q_from_parent + u + self.heuristic_bonus

    def best_child(self, c_puct: float, fpu_reduction: float = 0.0) -> MCTSNode:
        """Return the child with the highest PUCT score."""
        pq = self.q_value
        visited_prior_sum = sum(
            c.base_prior for c in self.children if c.visit_count > 0
        )
        return max(
            self.children,
            key=lambda c: c.puct_score(
                self.visit_count, c_puct, pq, fpu_reduction, visited_prior_sum
            ),
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
        dist_bonus_weight: float = 0.0,
        fpu_reduction: float = 0.2,
        sim_batch_size: int = 1,
    ) -> None:
        self.evaluator          = evaluator
        self.num_simulations    = num_simulations
        self.c_puct             = c_puct
        self.training           = training
        self.temperature        = temperature
        self.dirichlet_alpha    = dirichlet_alpha
        self.dirichlet_epsilon  = dirichlet_epsilon
        self.dist_bonus_weight  = dist_bonus_weight
        self.fpu_reduction      = fpu_reduction
        self.sim_batch_size     = sim_batch_size
        self._root: MCTSNode | None = None

    # ------------------------------------------------------------------
    # Core simulation primitives
    # ------------------------------------------------------------------

    def _expand_from_eval(
        self,
        node:          MCTSNode,
        priors:        list[float],
        legal_actions: list[Action],
    ) -> None:
        """
        Expand *node* using pre-computed priors (no evaluator call).

        Used by both the standard ``_expand`` path and the batched simulation
        path, where the NN has already been called for a whole batch of leaves.
        """
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

        # Distance heuristic bonus — pre-weighted and stored on children so
        # puct_score() can add it cheaply without passing extra arguments.
        if self.dist_bonus_weight != 0.0:
            advantages = node.state.compute_move_advantages(legal_actions)
            w = self.dist_bonus_weight / node.state.boardsize
            for child, adv in zip(node.children, advantages):
                child.heuristic_bonus = w * adv

    def _expand(self, node: MCTSNode) -> float:
        """
        Expand *node* by calling the evaluator, then creating children.

        Returns the estimated value from the node's current player's
        perspective, to be propagated up the tree via ``_backup``.
        """
        legal_actions = node.state.get_legal_actions()
        priors, value = self.evaluator(node.state, legal_actions)
        self._expand_from_eval(node, priors, legal_actions)
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
            node = node.best_child(self.c_puct, self.fpu_reduction)
        return node

    def _select_with_path(
        self, root: MCTSNode
    ) -> tuple[MCTSNode, list[MCTSNode]]:
        """
        Traverse root→leaf via PUCT, applying LC0-style unscored virtual
        visits: ``visit_count`` is incremented at every node on the path so
        that subsequent traversals within the same simulation batch are
        steered toward different leaves, reducing collisions without a
        separate in-flight counter.

        Backprop after a batched gather must use ``_backup_values_only``
        (updates only ``value_sum``) because ``visit_count`` was already
        committed here.

        Returns ``(leaf, path)`` where ``path`` is [root, …, leaf].
        """
        node = root
        path: list[MCTSNode] = []
        while True:
            node.ensure_state()
            node.visit_count += 1  # unscored virtual visit (LC0 convention)
            path.append(node)
            if not node.is_expanded or node.state.is_finished():
                break
            node = node.best_child(self.c_puct, self.fpu_reduction)
        return node, path

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

    def _backup_values_only(self, path: list[MCTSNode], value: float) -> None:
        """
        Walk ``path`` (root→leaf order) updating only ``value_sum``.

        Used in batched simulation where ``visit_count`` was already
        incremented by ``_select_with_path`` (unscored virtual visit).
        Traverses in reverse so the leaf is backpropagated first.
        """
        for node in reversed(path):
            node.value_sum += value
            value = -value

    def _run_simulation(self, root: MCTSNode) -> None:
        """One simulation: select a leaf → evaluate it → propagate the result."""
        leaf = self._select(root)
        leaf.ensure_state()   # materialise lazy child on first visit
        if leaf.state.is_finished():
            value = self._terminal_value(leaf.state)
        else:
            value = self._expand(leaf)
        self._backup(leaf, value)

    def _batch_evaluate(
        self,
        nodes:              list[MCTSNode],
        legal_actions_list: list[list[Action]],
    ) -> list[tuple[list[float], float]]:
        """
        Evaluate a batch of leaf nodes in one call.

        Delegates to ``evaluator.batch_call(states, legal_actions_list)``
        when available (implemented by ``NNEvaluator`` / ``SymmetricEvaluator``),
        falling back to sequential single-item calls otherwise.
        """
        if not nodes:
            return []
        states = [n.state for n in nodes]
        if hasattr(self.evaluator, "batch_call"):
            return self.evaluator.batch_call(states, legal_actions_list)
        return [self.evaluator(s, la) for s, la in zip(states, legal_actions_list)]

    def _run_simulations_batched(self, root: MCTSNode, num_sims: int | None = None) -> None:
        """
        Run simulations with batched NN evaluation.

        Uses ``num_sims`` if provided, otherwise falls back to
        ``self.num_simulations``.  Each round collects ``self.sim_batch_size``
        leaves via ``_select_with_path`` (virtual visits discourage collisions
        across traversals within the same round), issues one batched NN call
        for all unique new leaves, expands them, and propagates values via
        ``_backup_values_only`` (visit counts were already committed).
        """
        remaining = num_sims if num_sims is not None else self.num_simulations
        while remaining > 0:
            b          = min(self.sim_batch_size, remaining)
            remaining -= b

            # ── Gather b leaves with virtual N increments ─────────────────
            all_leaves: list[MCTSNode]       = []
            all_paths:  list[list[MCTSNode]] = []
            for _ in range(b):
                leaf, path = self._select_with_path(root)
                leaf.ensure_state()
                all_leaves.append(leaf)
                all_paths.append(path)

            # ── Unique unexpanded non-terminal leaves → need NN eval ──────
            pending: dict[int, tuple[MCTSNode, list[Action]]] = {}
            for leaf in all_leaves:
                lid = id(leaf)
                if (
                    lid not in pending
                    and not leaf.is_expanded
                    and not leaf.state.is_finished()
                ):
                    pending[lid] = (leaf, leaf.state.get_legal_actions())

            # ── Batched NN call ───────────────────────────────────────────
            result_map: dict[int, tuple[list[float], float, list[Action]]] = {}
            if pending:
                eval_nodes = [v[0] for v in pending.values()]
                eval_legal = [v[1] for v in pending.values()]
                batch_out  = self._batch_evaluate(eval_nodes, eval_legal)
                for n, (priors, value), la in zip(eval_nodes, batch_out, eval_legal):
                    result_map[id(n)] = (priors, value, la)

            # ── Expand + value-only backprop ──────────────────────────────
            for leaf, path in zip(all_leaves, all_paths):
                lid = id(leaf)
                if leaf.state.is_finished():
                    value = self._terminal_value(leaf.state)
                elif lid in result_map:
                    priors, value, la = result_map[lid]
                    if not leaf.is_expanded:          # guard against same-batch collision
                        self._expand_from_eval(leaf, priors, la)
                else:
                    # Leaf was already expanded by an earlier path in this
                    # batch; use its current Q value as the backprop proxy.
                    value = leaf.q_value if leaf.visit_count > 0 else 0.0
                self._backup_values_only(path, value)

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

    def search(self, state: State, num_sims: int | None = None) -> MCTSNode:
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

        n = num_sims if num_sims is not None else self.num_simulations
        if self.sim_batch_size > 1:
            self._run_simulations_batched(root, num_sims=n)
        else:
            for _ in range(n):
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

    def select_action(self, state: State, **kwargs) -> Action:
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
