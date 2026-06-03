from abc import ABC, abstractmethod
import joblib
from pathlib import Path

import numpy as np

from game import action_to_index, action_space_size, PawnAction, WallAction

BOARDSIZE = 7

seed = 0
dtype = np.float16
dtype_int = np.uint16
#wall_bits = 3

class BaseAgent(ABC):

    def __init__(self, boardsize=BOARDSIZE, seed=seed):
        self.boardsize = boardsize
        self.num_actions = action_space_size(boardsize)
        self.rng = np.random.default_rng(seed)

    def __str__(self):
        return self.__class__.__name__
    
    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)
    
    def save(self, path, compress=False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # with open(path, "wb") as f:
        #     pickle.dump(self.__dict__, f)

        q_keys = list(self.Q.keys())
        q_indices = [np.fromiter(v.keys(), dtype=dtype_int) if len(v) > 0 else np.empty(0, dtype=dtype_int) for v in self.Q.values()]
        q_values = [np.fromiter(v.values(), dtype=dtype) if len(v) > 0 else np.empty(0, dtype=dtype) for v in self.Q.values()]

        vc_keys = list(self.visit_counts_dict.keys())
        vc_indices = [np.fromiter(v.keys(), dtype=dtype_int) if len(v) > 0 else np.empty(0, dtype=dtype_int) for v in self.visit_counts_dict.values()]
        vc_values = [np.fromiter(v.values(), dtype=dtype_int) if len(v) > 0 else np.empty(0, dtype=dtype_int) for v in self.visit_counts_dict.values()]

        data = {"boardsize": self.boardsize, "gamma": self.gamma, "alpha": self.alpha, "alpha0": self.alpha0, "alpha_beta": self.alpha_beta, "min_alpha": self.min_alpha,
                "epsilon": self.epsilon, "min_epsilon": self.min_epsilon, "epsilon_decay": self.epsilon_decay, "step_count": self.step_count, "reset_period": self.reset_period, "ucb_c": self.ucb_c,
                "eps": self.eps, "q_keys": q_keys, "q_indices": q_indices, "q_values": q_values, "vc_keys": vc_keys, "vc_indices": vc_indices, "vc_values": vc_values}

        joblib.dump(data, path, compress=('gzip', 3)) if compress else joblib.dump(data, path)


    def load(self, path):
        # with open(path, "rb") as f:
        #     data = pickle.load(f)
            
        # self.__dict__.update(data)
        # return self

        data = joblib.load(path)

        self.boardsize = data["boardsize"]
        self.gamma = data["gamma"]
        self.alpha = data["alpha"]
        self.alpha0 = data["alpha0"]
        self.alpha_beta = data["alpha_beta"]
        self.min_alpha = data["min_alpha"]
        self.epsilon = data["epsilon"]
        self.min_epsilon = data["min_epsilon"]
        self.epsilon_decay = data["epsilon_decay"]
        self.step_count = data["step_count"]
        self.reset_period = data["reset_period"]
        self.ucb_c = data["ucb_c"]
        self.eps = data["eps"]

        self.Q = {}
        for s, idxs, vals in zip(data["q_keys"], data["q_indices"], data["q_values"]):
            self.Q[int(s)] = {int(i): dtype(v) for i, v in zip(idxs, vals)}

        self.visit_counts_dict = {}
        for s, idxs, vals in zip(data["vc_keys"], data["vc_indices"], data["vc_values"]):
            self.visit_counts_dict[int(s)] = {int(i): dtype_int(v) for i, v in zip(idxs, vals)}

        return self

    
    def _state_to_key(self, state):
        N = self.boardsize
        p1x, p1y = state.player1pos
        p2x, p2y = state.player2pos

        hbits = 0
        for i, v in enumerate(state.hwalls):
            if v:
                hbits |= (1 << i)

        vbits = 0
        for i, v in enumerate(state.vwalls):
            if v:
                vbits |= (1 << i)

        n_h = len(state.hwalls)
        n_v = len(state.vwalls)
        cb = (N - 1).bit_length()  # bits needed per coordinate: 3 for 7x7, 4 for 9x9
        base = (
            p1x |
            (p1y << cb) |
            (p2x << (2 * cb)) |
            (p2y << (3 * cb)) |
            (hbits << (4 * cb)) |
            (vbits << (4 * cb + n_h)) |
            (state.walls_p1 << (4 * cb + n_h + n_v)) |
            (state.walls_p2 << (4 * cb + n_h + n_v + cb)) |
            (state.get_current_player() << (4 * cb + n_h + n_v + 2 * cb))
        )

        mp1x = N - 1 - p1x
        mp2x = N - 1 - p2x

        mhbits = 0
        for y in range(N):
            row = y * N
            for x in range(N):
                if state.hwalls[row + x]:
                    mhbits |= (1 << (row + (N - 1 - x)))

        mvbits = 0
        for y in range(N):
            row = y * N
            for x in range(N):
                if state.vwalls[row + x]:
                    mvbits |= (1 << (row + (N - 1 - x)))

        mirrored = (
            mp1x |
            (p1y << cb) |
            (mp2x << (2 * cb)) |
            (p2y << (3 * cb)) |
            (mhbits << (4 * cb)) |
            (mvbits << (4 * cb + n_h)) |
            (state.walls_p1 << (4 * cb + n_h + n_v)) |
            (state.walls_p2 << (4 * cb + n_h + n_v + cb)) |
            (state.get_current_player() << (4 * cb + n_h + n_v + 2 * cb))
        )

        # key = min(base, mirrored)
        # cached = self.state_key_cache.get(key)
        # if cached is not None:
        #     return cached

        # self.state_key_cache[key] = key
        return min(base, mirrored)
    
    def _action_to_index_cached(self, action):
        return action_to_index(action, self.boardsize)
    
    def _reflect_position(self, pos):
        x, y = pos
        return (self.boardsize - 1 - x, y)

    def _reflect_state(self, state):
        N = self.boardsize
        reflected = state.copy()

        reflected.player1pos = self._reflect_position(state.player1pos)
        reflected.player2pos = self._reflect_position(state.player2pos)

        reflected.hwalls = bytearray(N * N)
        reflected.vwalls = bytearray(N * N)

        reflected.hwall_anchors = {(N - 2 - x, y) for x, y in state.hwall_anchors}
        reflected.vwall_anchors = {(N - 2 - x, y) for x, y in state.vwall_anchors}

        for ax, ay in reflected.hwall_anchors:
            reflected.hwalls[ay * N + ax] = 1
            reflected.hwalls[ay * N + ax + 1] = 1
        for ax, ay in reflected.vwall_anchors:
            reflected.vwalls[ay * N + ax] = 1
            reflected.vwalls[(ay + 1) * N + ax] = 1

        reflected._recompute_dists()
        reflected._legal_actions_cache = None
        return reflected

    def _reflect_action(self, action):
        if isinstance(action, PawnAction):
            dx, dy = action.direction
            return PawnAction(direction=(-dx, dy))
        return WallAction(x=self.boardsize - 2 - action.x, y=action.y, orientation=action.orientation)

    def _reflect_transition(self, state, action, next_state):
        return self._reflect_state(state), self._reflect_action(action), self._reflect_state(next_state)

    @abstractmethod
    def select_action(self, state, training=True):
        pass

    @abstractmethod
    def get_policy(self, state) -> list:
        """Return [(action, probability), ...] sorted by descending probability."""
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass


class TDZeroLearningBaseAgent(BaseAgent):

    def __init__(self, boardsize=BOARDSIZE, gamma=0.995, alpha=0.8, alpha_beta_coef=0.6, min_alpha=0.01, epsilon=0.25, min_epsilon=0.01, epsilon_decay=0.99999, ucb_coef=None,
                 cyclical_step_count=0, cyclical_reset_period=20000, seed=seed, eps=1e-8):

        super().__init__(boardsize, seed)

        self.gamma = gamma

        self.alpha = alpha
        self.alpha0 = alpha
        self.alpha_beta = alpha_beta_coef
        self.min_alpha = min_alpha

        self.epsilon = epsilon
        self.epsilon0 = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.step_count = cyclical_step_count
        self.reset_period = cyclical_reset_period
        self.ucb_c = self.ucb_c = 1.4 * (1.0 / (1.0 + 0.000002 * self.step_count)) if ucb_coef is None else ucb_coef

        self.Q = {}
        self.visit_counts_dict = {}

        self.rng = np.random.default_rng(seed)
        self.eps = eps

    def _q_values(self, state_key, Q):
        q = Q.get(state_key)
        if q is None:
            q = {} #np.zeros(self.num_actions, dtype=dtype)
            Q[state_key] = q

        return q
    
    def _visit_counts(self, state_key):
        n = self.visit_counts_dict.get(state_key)
        if n is None:
            n = {} #np.zeros(self.num_actions, dtype=dtype_int)
            self.visit_counts_dict[state_key] = n
        return n
    
    def _legal_action_indices(self, state):
        actions = state.get_legal_actions()
        indices = np.empty(len(actions), dtype=np.int16)

        for i, a in enumerate(actions):
            indices[i] = self._action_to_index_cached(a)
        return (actions, indices)
    
    def _policy_probs(self, state):
        legal_actions, indices = self._legal_action_indices(state)
        n = len(indices)
        probs = np.full(n, self.epsilon / n, dtype=dtype)

        q = self._q_values(self._state_to_key(state), self.Q)
        probs[np.argmax(np.array([q.get(int(i), 0.0) for i in indices], dtype=dtype))] += 1.0 - self.epsilon

        return legal_actions, indices, probs
    
    def select_action(self, state, training=True):
        legal_actions, indices = self._legal_action_indices(state)
        s = self._state_to_key(state)

        if not training:
            return self.get_policy(state)[0][0]

        visit_counts = self._visit_counts(s)
        q = self._q_values(s, self.Q)
        counts = np.array([visit_counts.get(int(i), 0) for i in indices], dtype=dtype_int)
        scores = np.nan_to_num(np.array([q.get(int(i), 0.0) for i in indices], dtype=dtype) + self.ucb_c * np.sqrt(np.log((counts.sum() + 1) + 1) / (counts + 1)), self.eps)

        max_idxs = np.flatnonzero(scores == scores.max())
        return legal_actions[max_idxs[self.rng.integers(len(max_idxs))]]

    def get_policy(self, state) -> list:
        legal_actions, indices = self._legal_action_indices(state)
        q = self._q_values(self._state_to_key(state), self.Q)
        qvals = np.array([q.get(int(i), 0.0) for i in indices], dtype=dtype)
        best = int(np.argmax(qvals))
        probs = [1.0 if i == best else 0.0 for i in range(len(legal_actions))]
        policy = list(zip(legal_actions, probs))
        policy.sort(key=lambda x: x[1], reverse=True)
        return policy

    def _update_single(self, state, action, reward, next_state, is_terminal_state):
        s = self._state_to_key(state)
        a = self._action_to_index_cached(action)

        visit_counts = self._visit_counts(s)
        visit_counts[a] = visit_counts.get(a, 0) + 1

        target = reward
        if not is_terminal_state:
            target -= self.gamma * self._update_policy(next_state, self._state_to_key(next_state))

        self.alpha = max(self.min_alpha, self.alpha0 / (visit_counts[a] ** self.alpha_beta))

        q_s = self._q_values(s, self.Q)
        old_q = q_s.get(a, 0.0)
        q_s[a] = dtype(old_q + self.alpha * (target - old_q))

    def update(self, state, action, reward, next_state, is_terminal_state):
        self._update_single(state, action, reward, next_state, is_terminal_state)
        reflected_state, reflected_action, reflected_next_state = self._reflect_transition(state, action, next_state)
        self._update_single(reflected_state, reflected_action, reward, reflected_next_state, is_terminal_state)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.step_count += 1
        if self.step_count % self.reset_period == 0:
            self.epsilon = min(self.epsilon0, self.epsilon * 2.0)

    @abstractmethod
    def _update_policy(self, next_state, ns):
        pass


class QLearningAgent(TDZeroLearningBaseAgent):
    def _update_policy(self, next_state, ns):
        indices = self._legal_action_indices(next_state)[1]
        q = self._q_values(ns, self.Q)
        return max((q.get(int(i), 0.0) for i in indices), default=0.0)
    
class SarsaAgent(TDZeroLearningBaseAgent):
    def _update_policy(self, next_state, ns):
        actions, _, probs = self._policy_probs(next_state)
        return self._q_values(ns, self.Q).get(self._action_to_index_cached(actions[self.rng.choice(len(actions), p=probs)]), 0.0)

class ExpectedSarsaAgent(TDZeroLearningBaseAgent):
    def _update_policy(self, next_state, ns):
        _, indices, probs = self._policy_probs(next_state)
        q = self._q_values(ns, self.Q)
        return np.dot(probs, np.array([q.get(int(i), 0.0) for i in indices], dtype=dtype))
    

class DoubleTDZeroLearningBaseAgent(TDZeroLearningBaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.QA = {}
        self.QB = {}

    def select_action(self, state, training=True):
        legal_actions, indices = self._legal_action_indices(state)
        s = self._state_to_key(state)

        qa = self._q_values(s, self.QA)
        qb = self._q_values(s, self.QB)

        if not training:
            return self.get_policy(state)[0][0]

        visit_counts = self._visit_counts(s)
        counts = np.array([visit_counts.get(int(i), 0) for i in indices], dtype=dtype_int)
        scores = np.nan_to_num(np.array([qa.get(int(i), 0.0) + qb.get(int(i), 0.0) for i in indices], dtype=dtype) +
                               self.ucb_c * np.sqrt(np.log((counts.sum() + 1) + 1) / (counts + 1)), self.eps)
        max_idxs = np.flatnonzero(scores == scores.max())
        return legal_actions[max_idxs[self.rng.integers(len(max_idxs))]]

    def get_policy(self, state) -> list:
        legal_actions, indices = self._legal_action_indices(state)
        s = self._state_to_key(state)
        qa = self._q_values(s, self.QA)
        qb = self._q_values(s, self.QB)
        qvals = np.array([qa.get(int(i), 0.0) + qb.get(int(i), 0.0) for i in indices], dtype=dtype)
        best = int(np.argmax(qvals))
        probs = [1.0 if i == best else 0.0 for i in range(len(legal_actions))]
        policy = list(zip(legal_actions, probs))
        policy.sort(key=lambda x: x[1], reverse=True)
        return policy

    def _policy_probs(self, state):
        legal_actions, indices = self._legal_action_indices(state)
        n = len(indices)
        probs = np.full(n, self.epsilon / n, dtype=dtype)

        s = self._state_to_key(state)
        qa = self._q_values(s, self.QA)
        qb = self._q_values(s, self.QB)

        probs[np.argmax(np.array([qa.get(int(i), 0.0) + qb.get(int(i), 0.0) for i in indices], dtype=dtype))] += 1.0 - self.epsilon

        return legal_actions, indices, probs

    def _update_single(self, state, action, reward, next_state, is_terminal_state):
        s = self._state_to_key(state)
        a = self._action_to_index_cached(action)

        visit_counts = self._visit_counts(s)
        visit_counts[a]  = visit_counts.get(a, 0) + 1

        update_a = self.rng.random() < 0.5
        q_update = self._q_values(s, self.QA) if update_a else self._q_values(s, self.QB)

        target = reward
        if not is_terminal_state:
            target -= self.gamma * self._double_update_policy(next_state, self._state_to_key(next_state), update_a)

        self.alpha = max(self.min_alpha, self.alpha0 / (visit_counts[a] ** self.alpha_beta))

        old_q = q_update.get(a, 0.0)
        q_update[a] = dtype(old_q + self.alpha * (target - old_q))

    def _update_policy(self, next_state, ns):
        pass
    
    @abstractmethod
    def _double_update_policy(self, next_state, ns, update_a):
        pass

class DoubleQLearningAgent(DoubleTDZeroLearningBaseAgent):
    def _double_update_policy(self, next_state, ns, update_a):
        indices = self._legal_action_indices(next_state)[1]
        qa = self._q_values(ns, self.QA)
        qb = self._q_values(ns, self.QB)
        qa_vals = np.array([qa.get(int(i), 0.0) for i in indices], dtype=dtype)
        qb_vals = np.array([qb.get(int(i), 0.0) for i in indices], dtype=dtype)

        return qb_vals[np.argmax(qa_vals)] if update_a else qa_vals[np.argmax(qb_vals)]


class DoubleSarsaAgent(DoubleTDZeroLearningBaseAgent):
    def _double_update_policy(self, next_state, ns, update_a):
        actions, _, probs = self._policy_probs(next_state)
        return self._q_values(ns, self.QB if update_a else self.QA).get(self._action_to_index_cached(actions[self.rng.choice(len(actions), p=probs)]), 0.0)


class DoubleExpectedSarsaAgent(DoubleTDZeroLearningBaseAgent):
    def _double_update_policy(self, next_state, ns, update_a):
        _, indices, probs = self._policy_probs(next_state)
        q = self._q_values(ns, self.QB if update_a else self.QA)
        return np.dot(probs, np.array([q.get(int(i), 0.0) for i in indices], dtype=dtype))
    