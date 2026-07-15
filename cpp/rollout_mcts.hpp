// Simple MCTS agent using random-playout leaf evaluation (no neural network).
//
// This mirrors mcts.py's `MCTSAgent(evaluator=rollout_evaluator)` baseline:
// PUCT selection with uniform priors over legal actions, and leaf values
// estimated by playing one uniform-random game to completion instead of a
// neural-network forward pass. Ported to C++ purely for speed -- the tree
// walk / PUCT formula is unchanged, only the (cheap, NN-free) leaf evaluator
// differs from selfplay.hpp's batched NN-guided search.
//
// Deliberately simple/single-threaded (unlike SelfPlayManager): one game,
// one fresh arena per move, full GameState materialized per node (states are
// small structs-of-bitsets/arrays, so this costs a few hundred bytes/node --
// trivial at the simulation counts this baseline targets). Repetition rule
// is enforced faithfully while descending/expanding the real search tree
// (base history + path overlay, same convention as selfplay.hpp), but is
// deliberately NOT enforced during the throwaway random rollout itself (a
// harmless approximation for a Monte-Carlo value estimate).
//
// Optional distance-heuristic nudge (dist_bonus_weight, default 0.0 = off):
// classic MCTS "progressive bias" (Chaslot et al.) -- each child gets an
// extra additive term h(c) / (1 + N(c)) folded into the selection score,
// where h(c) is a heuristic value proportional to how much that move changes
// (my_dist_to_goal - opp_dist_to_goal) relative to the parent (GameState::
// move_advantages), scaled by dist_bonus_weight / boardsize. Priors P(c)
// stay uniform; unlike a constant additive bonus, this term fades as the
// child accumulates its own visits, so (like the usual U(c) exploration
// term) its influence vanishes with search rather than permanently biasing
// selection.

#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "engine.hpp"

namespace quoridor {

struct RolloutNode {
    GameState state;
    int32_t   parent      = -1;
    uint16_t  action      = 0;      // action taken at parent to reach this node
    float     prior       = 0.0f;   // uniform 1/n over legal siblings
    float     bias        = 0.0f;   // progressive-bias h(c) (0 when disabled)
    int32_t   visits      = 0;
    float     value_sum   = 0.0f;
    int32_t   first_child = -1;
    uint16_t  nchild      = 0;
    bool      expanded    = false;
    bool      terminal    = false;
    float     tvalue      = 0.0f;   // terminal value (POV of player to move here)
};

class RolloutMCTS {
public:
    // dist_bonus_weight == 0.0 (default) reproduces the "pure" rollout MCTS
    // exactly (no heuristic). Non-zero adds a progressive-bias term
    // h(c)/(1+N(c)) to each child's score, h(c) = dist_bonus_weight /
    // boardsize * advantage -- see file header for rationale.
    RolloutMCTS(double c_puct, int max_moves, uint64_t seed,
                double dist_bonus_weight = 0.0)
        : c_puct_(c_puct), max_moves_(max_moves), rng_(seed),
          dist_bonus_weight_(dist_bonus_weight) {}

    // Runs num_simulations PUCT/rollout simulations from root_state (root
    // expansion itself is not counted, matching mcts.py's convention) and
    // returns the most-visited root action. Optionally reports the full
    // (action, visit_count) distribution over root children.
    uint16_t search(const GameState& root_state,
                     const std::unordered_map<uint64_t, int>& history,
                     int num_simulations,
                     std::vector<uint16_t>* out_actions = nullptr,
                     std::vector<int32_t>* out_visits = nullptr) {
        history_ = &history;
        arena_.clear();
        arena_.reserve(size_t(num_simulations) * 4 + 16);

        RolloutNode root{};
        root.state = root_state;
        arena_.push_back(std::move(root));

        if (!expand_node(0)) {
            return 0;  // terminal at root -- shouldn't happen mid-game
        }

        for (int i = 0; i < num_simulations; ++i) simulate_once();

        const RolloutNode& root_ref = arena_[0];
        int32_t best = -1, best_visits = -1;
        for (int k = 0; k < root_ref.nchild; ++k) {
            const int32_t ci = root_ref.first_child + k;
            if (out_actions) out_actions->push_back(arena_[ci].action);
            if (out_visits)  out_visits->push_back(arena_[ci].visits);
            if (arena_[ci].visits > best_visits) {
                best_visits = arena_[ci].visits;
                best = ci;
            }
        }
        return best >= 0 ? arena_[best].action : 0;
    }

private:
    double c_puct_;
    int max_moves_;
    std::mt19937_64 rng_;
    double dist_bonus_weight_;
    std::vector<RolloutNode> arena_;
    const std::unordered_map<uint64_t, int>* history_ = nullptr;

    // Zobrist hashes from (but excluding) the root down to node_idx, for the
    // RepCounter path-overlay (mirrors selfplay.hpp::gather's `overlay`).
    void build_overlay(int32_t node_idx, std::vector<uint64_t>& overlay) const {
        overlay.clear();
        int32_t cur = node_idx;
        while (cur != 0) {
            overlay.push_back(arena_[cur].state.zhash);
            cur = arena_[cur].parent;
        }
    }

    // Expand a fresh node: terminal check, then uniform-prior children.
    // Returns false if the node turned out to be terminal (no children).
    bool expand_node(int32_t node_idx) {
        const GameState st = arena_[node_idx].state;  // copy before any push_back
        const int w = st.winner();
        if (w != 0) {
            arena_[node_idx].terminal = true;
            arena_[node_idx].expanded = true;
            arena_[node_idx].tvalue = -1.0f;  // mover at this node already lost
            return false;
        }
        if (st.depth >= max_moves_) {
            arena_[node_idx].terminal = true;
            arena_[node_idx].expanded = true;
            arena_[node_idx].tvalue = 0.0f;   // draw
            return false;
        }

        std::vector<uint64_t> overlay;
        build_overlay(node_idx, overlay);
        RepCounter rc{history_, &overlay};
        std::vector<uint16_t> legal;
        GameState mut = st;  // gen_legal is non-const (toggles wall bits internally)
        mut.gen_legal(rc, legal);
        if (legal.empty()) {
            arena_[node_idx].terminal = true;
            arena_[node_idx].expanded = true;
            arena_[node_idx].tvalue = 0.0f;   // draw (no legal moves)
            return false;
        }

        const int32_t first = int32_t(arena_.size());
        const float p = 1.0f / float(legal.size());
        std::vector<float> adv;
        if (dist_bonus_weight_ != 0.0) {
            adv.resize(legal.size());
            st.move_advantages(legal, adv.data());
        }
        const float wb = dist_bonus_weight_ != 0.0
                       ? float(dist_bonus_weight_) / float(st.N) : 0.0f;
        for (size_t k = 0; k < legal.size(); ++k) {
            const uint16_t a = legal[k];
            RolloutNode child{};
            child.state = st;
            child.state.apply(a);
            child.parent = node_idx;
            child.action = a;
            child.prior = p;
            if (wb != 0.0f) child.bias = wb * adv[k];
            arena_.push_back(std::move(child));
        }
        arena_[node_idx].first_child = first;
        arena_[node_idx].nchild = uint16_t(legal.size());
        arena_[node_idx].expanded = true;
        return true;
    }

    // PUCT child selection (mirror of MCTSNode.best_child / selfplay.hpp::select_child,
    // fpu_reduction = 0 -- unvisited children default to the parent's own Q).
    int32_t select_child(int32_t node_idx) const {
        const RolloutNode& n = arena_[node_idx];
        const float parent_q = n.visits > 0 ? n.value_sum / n.visits : 0.0f;
        const float sq = std::sqrt(float(n.visits));

        int32_t best = n.first_child;
        float best_score = -1e30f;
        for (int k = 0; k < n.nchild; ++k) {
            const int32_t ci = n.first_child + k;
            const RolloutNode& c = arena_[ci];
            const float u = float(c_puct_) * c.prior * sq / (1.0f + c.visits);
            const float q = c.visits == 0 ? parent_q : -(c.value_sum / c.visits);
            const float s = q + u + c.bias / (1.0f + c.visits);
            if (s > best_score) { best_score = s; best = ci; }
        }
        return best;
    }

    void backup(const std::vector<int32_t>& path, float v) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            arena_[*it].value_sum += v;
            v = -v;
        }
    }

    // Uniform-random playout to terminal from `st` (by value). Repetition
    // rule is not enforced here (see file header comment). Returns +1 if the
    // player to move at `st` eventually wins, -1 if they lose, 0 for a draw.
    float random_rollout(GameState st) {
        const int root_player = st.current_player();
        std::vector<uint16_t> legal;
        RepCounter rc{nullptr, nullptr};
        while (st.winner() == 0 && st.depth < max_moves_) {
            st.gen_legal(rc, legal);
            if (legal.empty()) break;
            st.apply(legal[rng_() % legal.size()]);
        }
        const int w = st.winner();
        if (w == 0) return 0.0f;
        return w == root_player ? 1.0f : -1.0f;
    }

    void simulate_once() {
        int32_t cur = 0;
        std::vector<int32_t> path;
        path.push_back(cur);
        ++arena_[cur].visits;
        while (arena_[cur].expanded && !arena_[cur].terminal) {
            cur = select_child(cur);
            ++arena_[cur].visits;
            path.push_back(cur);
        }
        if (arena_[cur].terminal) {
            backup(path, arena_[cur].tvalue);
            return;
        }
        const bool ok = expand_node(cur);
        const float v = ok ? random_rollout(arena_[cur].state) : arena_[cur].tvalue;
        backup(path, v);
    }
};

}  // namespace quoridor
