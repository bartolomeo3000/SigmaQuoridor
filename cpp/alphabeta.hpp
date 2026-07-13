// Exact alpha-beta solver for Quoridor endgame positions (few/no walls
// remaining). Searches to full terminal depth -- no heuristic leaf
// evaluation -- so a returned value is the EXACT game-theoretic result
// (+1 win / -1 loss / 0 draw) from the perspective of the player to move
// at the root, not an estimate. Move ordering uses the existing distance-
// advantage heuristic (GameState::move_advantages) purely to maximise
// alpha-beta pruning effectiveness; it never affects correctness since
// every branch is still fully explored (or proven irrelevant by the
// alpha-beta bound) before returning.
//
// Among moves with the same game-theoretic result, the solver additionally
// prefers the FASTEST win / SLOWEST loss (classic "mate distance" style
// tie-break) via a (result, dist) Score, where `dist` is plies remaining
// to the terminal state along the chosen line. This makes the transposition
// table double as a ready-made principal-variation store: extract_pv()
// walks best_action pointers from the (already-solved) root to the
// terminal state.
//
// Correctness notes
// ------------------
// * Terminal detection mirrors the real engine: winner() != 0 ends the
//   game immediately (checked BEFORE generating further moves), and
//   gen_legal() returning empty is a draw (matches engine.hpp's own
//   "draw: no legal actions" rule -- this happens when every pawn move
//   would create an illegal 3rd repetition and no walls are left).
// * Repetition is tracked exactly like the real engine: `hist_base` is
//   the real game's position-count map up to the root (so mid-game
//   candidate positions solve correctly), and `path_` is the search
//   path's zhash sequence, pushed/popped as a stack during DFS -- mirrors
//   the RepCounter overlay pattern already used by MCTS's gather().
// * Transposition table entries store a bound TYPE (EXACT / LOWERBOUND /
//   UPPERBOUND), not just a raw value -- storing only "best value found"
//   without the bound type is a classic alpha-beta+TT bug: a value found
//   under a narrowed alpha-beta window is not safe to reuse verbatim
//   under a different (e.g. wider) window later. This implementation
//   follows the standard fail-soft negamax+TT recipe. Pruning decisions
//   use only the `result` component of Score (dist never affects whether
//   a position is won/lost/drawn, only which equally-good line is chosen).
#pragma once

#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quoridor {

struct AlphaBetaResult {
    int value = 0;                  // +1 win / -1 loss / 0 draw (perspective: root's player to move)
    int dist = 0;                   // plies to terminal along the chosen (fastest-win/slowest-loss) line
    uint16_t best_action = 0xFFFF;  // 0xFFFF = none (draw with no legal actions, or timed out immediately)
    long long nodes = 0;
    bool timed_out = false;         // if true, value/dist/best_action are best-effort, NOT proven exact
};

class AlphaBetaSolver {
public:
    // node_limit < 0 disables the node cap; time_limit_s < 0 disables the
    // wall-clock cap. At least one should normally be set for safety.
    AlphaBetaSolver(long long node_limit = -1, double time_limit_s = -1.0)
        : node_limit_(node_limit), time_limit_s_(time_limit_s) {}

    AlphaBetaResult solve(GameState root, const std::unordered_map<uint64_t, int>* hist_base) {
        nodes_ = 0;
        timed_out_ = false;
        tt_.clear();
        path_.clear();
        t0_ = std::chrono::steady_clock::now();

        uint16_t best_action;
        const Score s = negamax(root, -2, 2, hist_base, best_action);

        AlphaBetaResult r;
        r.value = s.result;
        r.dist = s.dist;
        r.best_action = best_action;
        r.nodes = nodes_;
        r.timed_out = timed_out_;
        return r;
    }

    // Walk best_action pointers from `state` (must be the same position
    // solve() was called on, or any state visited during that search) to
    // the terminal state, using this solver instance's transposition
    // table. Only meaningful immediately after a successful (non-timed-
    // out) solve() call on the same instance. Returns (state-before-move,
    // action) pairs in play order; stops early (without error) after
    // `max_len` steps as a safety cap (e.g. to respect an external
    // max_moves ceiling).
    std::vector<std::pair<GameState, uint16_t>> extract_pv(GameState state, int max_len) const {
        std::vector<std::pair<GameState, uint16_t>> pv;
        for (int i = 0; i < max_len; ++i) {
            if (state.winner() != 0) break;
            auto it = tt_.find(state.zhash);
            if (it == tt_.end() || it->second.best_action == 0xFFFF) break;
            const uint16_t a = it->second.best_action;
            pv.emplace_back(state, a);
            state.apply(a);
        }
        return pv;
    }

private:
    // (result, dist): result is +1/0/-1 from the perspective of the player
    // to move at the state this Score describes; dist is plies remaining
    // to the terminal state along the best-found line from that state.
    struct Score {
        int8_t result;
        int16_t dist;
    };

    enum class Bound : uint8_t { EXACT, LOWER, UPPER };

    struct TTEntry {
        Score score;
        Bound bound;
        uint16_t best_action;
    };

    long long node_limit_;
    double time_limit_s_;
    std::chrono::steady_clock::time_point t0_;
    long long nodes_ = 0;
    bool timed_out_ = false;
    std::unordered_map<uint64_t, TTEntry> tt_;
    std::vector<uint64_t> path_;

    // True iff `a` is strictly preferable to `b` for the maximizing side
    // at a node (both scores are from that same side's perspective):
    // higher result wins outright; among equal results, faster wins
    // (smaller dist) or slower losses (larger dist) are preferred; draws
    // have no dist preference.
    static bool better(Score a, Score b) {
        if (a.result != b.result) return a.result > b.result;
        if (a.result > 0) return a.dist < b.dist;   // win: prefer faster
        if (a.result < 0) return a.dist > b.dist;   // loss: prefer slower
        return false;                                // draw: no preference
    }

    bool budget_exceeded() {
        if (timed_out_) return true;
        if (node_limit_ >= 0 && nodes_ >= node_limit_) { timed_out_ = true; return true; }
        if (time_limit_s_ >= 0) {
            const double el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0_).count();
            if (el >= time_limit_s_) { timed_out_ = true; return true; }
        }
        return false;
    }

    // Fail-soft negamax with alpha-beta pruning + bounded transposition
    // table. Returns the exact Score from the perspective of the player
    // to move at `state`, UNLESS timed_out_ becomes true mid-search, in
    // which case the return value is only a lower bound on the true
    // result (whatever was proven from the children actually explored).
    Score negamax(GameState& state, int alpha, int beta,
                 const std::unordered_map<uint64_t, int>* hist_base,
                 uint16_t& out_best_action) {
        ++nodes_;
        out_best_action = 0xFFFF;
        const int orig_alpha = alpha;

        const int w = state.winner();
        if (w != 0) return {int8_t((w == state.current_player()) ? 1 : -1), 0};

        auto tt_it = tt_.find(state.zhash);
        if (tt_it != tt_.end()) {
            const TTEntry& e = tt_it->second;
            out_best_action = e.best_action;
            if (e.bound == Bound::EXACT) return e.score;
            if (e.bound == Bound::LOWER) alpha = std::max(alpha, int(e.score.result));
            else                          beta  = std::min(beta,  int(e.score.result));
            if (alpha >= beta) return e.score;
        }

        if (budget_exceeded()) return {int8_t(alpha), 0};   // unknown; caller must check timed_out

        RepCounter rc{hist_base, &path_};
        std::vector<uint16_t> legal;
        state.gen_legal(rc, legal);

        if (legal.empty()) {
            tt_[state.zhash] = {{0, 0}, Bound::EXACT, 0xFFFF};
            return {0, 0};   // draw: no legal actions (matches engine's own rule)
        }

        // Move ordering by distance-advantage heuristic (descending) --
        // pruning aid only, does not affect the exactness of the result.
        std::vector<float> adv(legal.size());
        state.move_advantages(legal, adv.data());
        std::vector<int> order(legal.size());
        for (size_t i = 0; i < legal.size(); ++i) order[i] = int(i);
        std::sort(order.begin(), order.end(),
                 [&](int a, int b) { return adv[a] > adv[b]; });

        bool have_best = false;
        Score best{0, 0};
        uint16_t best_action = 0xFFFF;

        for (int idx : order) {
            const uint16_t a = legal[idx];
            GameState child = state;
            child.apply(a);
            path_.push_back(child.zhash);
            uint16_t child_best;
            const Score child_score = negamax(child, -beta, -alpha, hist_base, child_best);
            path_.pop_back();

            // A child that itself ran out of budget returns an unreliable
            // (not fully-proven) score -- never let it feed into best;
            // stop here and keep whatever was already proven from earlier
            // siblings (still a legitimate lower bound).
            if (timed_out_) break;

            const Score v{int8_t(-child_score.result), int16_t(child_score.dist + 1)};
            if (!have_best || better(v, best)) { best = v; best_action = a; have_best = true; }
            alpha = std::max(alpha, int(best.result));
            if (alpha >= beta) break;   // beta cutoff (pruning uses result only)
        }

        if (!have_best) {
            // Nothing resolved at all (timed out before finishing even the
            // first child) -- no safe bound to cache; caller must check
            // timed_out_ and not trust the returned value.
            out_best_action = legal[order[0]];
            return {int8_t(alpha), 0};
        }
        out_best_action = best_action;

        // Only cache a bound we can vouch for. On timeout mid-loop, best
        // is still a legitimate LOWER bound (every contributing child was
        // fully solved before we stopped), just not provably EXACT/UPPER.
        Bound bound;
        if (timed_out_)                    bound = Bound::LOWER;
        else if (best.result <= orig_alpha) bound = Bound::UPPER;
        else if (best.result >= beta)       bound = Bound::LOWER;
        else                                 bound = Bound::EXACT;
        tt_[state.zhash] = {best, bound, best_action};

        return best;
    }
};

}  // namespace quoridor
