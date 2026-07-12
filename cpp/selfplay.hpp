// Self-play manager: many concurrent games multiplexed over worker threads,
// all feeding one shared inference queue consumed from Python.
//
// Architecture (SEED-RL style, in-process):
//
//   worker threads (pure C++, GIL-free)          Python inference thread
//   ┌───────────────────────────────┐            ┌──────────────────────┐
//   │ pick ready game               │            │ get_batch()  ← blocks│
//   │  ├ run MCTS traversals        │  eval      │   (GIL released)     │
//   │  ├ collect leaf batch         │  queue     │ model(batch) on GPU  │
//   │  └ submit leaves ─────────────┼──────────▶ │ put_results()        │
//   │ game parks until results      │ ◀──────────┼── routes results,    │
//   └───────────────────────────────┘  requeue   │   re-readies games   │
//                                                └──────────────────────┘
//
// MCTS semantics mirror mcts.py's batched mode:
//   * PUCT with KataGo-style dynamic FPU
//   * LC0-style unscored virtual visits during leaf collection
//     (visit_count incremented on the way down, value backed up later)
//   * Dirichlet noise on root children (training mode)
//   * policy target = visit distribution with temperature schedule
//     (tau=1 for depth < temp_threshold, tau=0 after)
//   * root expansion is a separate eval, not counted toward num_simulations
//
// Deliberate deviation from mcts.py: no tree reuse between moves (fresh tree
// per move). At the 32-64 sim budgets this pipeline targets, reuse buys
// little and per-move arenas keep memory flat across 100+ parallel games.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "engine.hpp"

namespace quoridor {

struct Config {
    int boardsize          = 7;
    int walls              = 5;
    int num_simulations    = 64;
    int leaf_batch         = 8;
    double c_puct          = 1.0;
    double dirichlet_alpha = 0.3;
    double dirichlet_eps   = 0.25;
    double fpu_reduction   = 0.1;
    int temp_threshold     = 20;
    int max_moves          = 200;
    double dist_bonus_max  = 0.0;   // per-side weight ~ U[0, max] per game
    bool training          = true;  // Dirichlet noise + temperature sampling
    uint64_t seed          = 0;

    // Shared transposition table: cache raw NN (logits, value) outputs for
    // states with real+simulated depth <= tt_max_depth, shared across all
    // parallel games. 0 disables the TT entirely; negative means unlimited
    // depth (no ceiling -- every eligible node may be cached/looked up).
    int tt_max_depth       = 8;

    // Hard cap on total cached entries across all shards (bounds worst-case
    // memory regardless of tt_max_depth/game volume). Split evenly across
    // kTTShards; each shard evicts an arbitrary entry once full. 0 disables
    // the TT entirely (nothing is ever cached, regardless of tt_max_depth).
    // Pass a very large value (e.g. SIZE_MAX) for effectively unlimited
    // growth -- not recommended for long/large runs without knowing the
    // available RAM.
    size_t tt_max_entries  = 2'000'000;
};

// ---------------------------------------------------------------------------
// Tree node (per-move arena; states are NOT stored, they are replayed
// incrementally during descent)
// ---------------------------------------------------------------------------

struct Node {
    float    prior       = 0.0f;   // effective prior (Dirichlet-mixed at root)
    float    base_prior  = 0.0f;   // clean NN prior (for FPU visited_prior_sum)
    float    bonus       = 0.0f;   // distance-heuristic bonus (0 when disabled)
    float    value_sum   = 0.0f;
    int32_t  visits      = 0;
    int32_t  first_child = -1;
    uint16_t nchild      = 0;
    uint16_t action      = 0;
    uint8_t  flags       = 0;      // EXPANDED | TERMINAL | PENDING
    float    tvalue      = 0.0f;   // terminal value (current player's POV)
};

constexpr uint8_t F_EXPANDED = 1;
constexpr uint8_t F_TERMINAL = 2;
constexpr uint8_t F_PENDING  = 4;

// ---------------------------------------------------------------------------
// Shared transposition table (raw NN outputs, keyed on full board state).
//
// Keyed on (zhash, walls_p1, walls_p2) rather than zhash alone: zhash only
// covers pawn positions + wall cells + side-to-move, NOT which player placed
// how many walls (the board doesn't record wall ownership), so two distinct
// game lines can reach identical pawn/wall layouts with different remaining
// wall budgets -- and walls_p1/walls_p2 are separate NN input planes. Real-
// game repetition history is deliberately NOT part of the key: it only
// affects which actions are *legal* (3-fold-repetition avoidance), never the
// NN's raw per-action logits/value, and `legal` is always recomputed fresh
// per lookup anyway -- masking cached raw logits against the caller's own
// (possibly repetition-restricted) legal set is always correct.
// ---------------------------------------------------------------------------

struct TTKey {
    uint64_t zhash;
    uint32_t walls;   // (walls_p1 << 16) | walls_p2
    bool operator==(const TTKey& o) const noexcept {
        return zhash == o.zhash && walls == o.walls;
    }
};

struct TTKeyHash {
    size_t operator()(const TTKey& k) const noexcept {
        return std::hash<uint64_t>()(k.zhash ^ (uint64_t(k.walls) * 0x9E3779B97F4A7C15ULL));
    }
};

struct TTEntry {
    std::vector<float> logits;   // raw (unmasked) policy logits, size A
    float value = 0.0f;
};

inline TTKey tt_key(const GameState& st) {
    return TTKey{st.zhash, (uint32_t(st.walls_p1) << 16) | uint32_t(st.walls_p2)};
}

// Masked softmax over `legal` actions from a raw (A,) logits row -- shared by
// the real NN-result path (put_results) and the transposition-cache-hit path.
inline void masked_softmax(const float* row, const std::vector<uint16_t>& legal,
                           std::vector<float>& priors) {
    priors.resize(legal.size());
    float mx = -1e30f;
    for (uint16_t a : legal) mx = std::max(mx, row[a]);
    double sum = 0.0;
    for (size_t j = 0; j < legal.size(); ++j) {
        const double e = std::exp(double(row[legal[j]] - mx));
        priors[j] = float(e);
        sum += e;
    }
    const float inv = float(1.0 / (sum > 0 ? sum : 1.0));
    for (auto& pr : priors) pr *= inv;
}

// One leaf awaiting NN evaluation.
struct PendingLeaf {
    int32_t node = -1;
    bool is_root = false;
    std::vector<uint16_t> legal;                 // legal action indices at leaf
    std::vector<float> adv;                      // dist-heuristic advantages
    std::vector<std::vector<int32_t>> paths;     // all root→leaf paths hitting it
    std::vector<float> planes;                   // (8*N*N) NN input
    std::vector<float> priors;                   // filled by put_results
    float value = 0.0f;                          // filled by put_results

    // Transposition-table bookkeeping (see TTKey above).
    bool resolved = false;    // true if priors/value already filled from cache
    bool cacheable = false;   // true if a miss here should be inserted on arrival
    TTKey cache_key{};
};

// ---------------------------------------------------------------------------
// One self-play game (state machine driven by worker threads)
// ---------------------------------------------------------------------------

struct Game {
    GameState state;
    std::unordered_map<uint64_t, int> history;   // real-game position counts
    std::vector<uint16_t> root_legal;            // legal actions at real position

    std::vector<Node> arena;                     // per-move search tree
    int32_t root = -1;
    int sims_done = 0;

    std::vector<PendingLeaf> pending;
    std::unordered_map<int32_t, int> pending_by_node;
    int results_missing = 0;

    std::mt19937_64 rng;
    float w_p1 = 0.0f, w_p2 = 0.0f;              // dist-bonus weights per side
    float w_cur = 0.0f;                          // weight for the player at root

    // Recorded trajectory (values assigned at game end).
    std::vector<float> rec_planes;               // plies * 8*N*N
    std::vector<float> rec_policy;               // plies * A
    std::vector<int8_t> rec_player;
    int plies = 0;

    // Diagnostic only: post-move Zobrist hash for the first few real plies,
    // used to measure cross-game transposition overlap (see get_openings()).
    static constexpr int kOpeningTrack = 16;
    std::array<uint64_t, kOpeningTrack> opening_hashes{};
    int n_opening = 0;

    void reset(const Config& cfg, uint64_t seed) {
        state.init(cfg.boardsize, cfg.walls);
        history.clear();
        history[state.zhash] = 1;
        arena.clear();
        root = -1;
        sims_done = 0;
        pending.clear();
        pending_by_node.clear();
        results_missing = 0;
        rng.seed(seed);
        if (cfg.dist_bonus_max > 0.0) {
            std::uniform_real_distribution<float> u(0.0f, float(cfg.dist_bonus_max));
            w_p1 = u(rng);
            w_p2 = u(rng);
        } else {
            w_p1 = w_p2 = 0.0f;
        }
        rec_planes.clear();
        rec_policy.clear();
        rec_player.clear();
        plies = 0;
        n_opening = 0;
        RepCounter rc{&history, nullptr};
        state.gen_legal(rc, root_legal);
    }
};

// ---------------------------------------------------------------------------
// SelfPlayManager
// ---------------------------------------------------------------------------

class SelfPlayManager {
public:
    SelfPlayManager(Config cfg, int num_threads, int parallel_games)
        : cfg_(cfg), num_threads_(num_threads), parallel_games_(parallel_games) {}

    ~SelfPlayManager() { stop(); }

    void start(int total_games) {
        stop();  // in case of restart
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_flag_ = false;
            total_games_ = total_games;
            games_finished_ = 0;
            evalq_.clear();
            outstanding_.clear();
            ready_.clear();
            const int n = std::min(parallel_games_, total_games);
            games_.clear();
            games_.reserve(n);
            seed_counter_ = cfg_.seed * 0x9E3779B97F4A7C15ULL + 1;
            for (int i = 0; i < n; ++i) {
                games_.push_back(std::make_unique<Game>());
                games_[i]->reset(cfg_, next_seed());
                ready_.push_back(i);
            }
            games_started_ = n;
        }
        // Transposition-table entries are only valid for a fixed set of NN
        // weights; clear on (re)start since a caller reusing one manager
        // across cycles will have loaded a different model in between.
        for (auto& shard : tt_) {
            std::lock_guard<std::mutex> lk(shard.mu);
            shard.map.clear();
        }
        for (int t = 0; t < num_threads_; ++t)
            workers_.emplace_back([this] { worker_loop(); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_flag_ = true;
        }
        cv_ready_.notify_all();
        cv_eval_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
        workers_.clear();
    }

    bool is_done() const {
        std::lock_guard<std::mutex> lk(mu_);
        return games_finished_ >= total_games_;
    }

    int games_finished() const {
        std::lock_guard<std::mutex> lk(mu_);
        return games_finished_;
    }

    // Blocks (caller should release the GIL) until leaves are available or all
    // games are done. Returns batch_id, fills `out` with (count * 8*N*N)
    // floats. count == 0 signals completion.
    int64_t get_batch(int max_batch, int flush_us,
                      std::vector<float>& out, int& count) {
        const int NN = cfg_.boardsize * cfg_.boardsize;
        std::unique_lock<std::mutex> lk(mu_);
        cv_eval_.wait(lk, [&] {
            return !evalq_.empty() || games_finished_ >= total_games_ || stop_flag_;
        });
        if (evalq_.empty()) { count = 0; return -1; }

        if (int(evalq_.size()) < max_batch && flush_us > 0) {
            const auto deadline = std::chrono::steady_clock::now()
                                + std::chrono::microseconds(flush_us);
            cv_eval_.wait_until(lk, deadline, [&] {
                return int(evalq_.size()) >= max_batch
                    || games_finished_ >= total_games_ || stop_flag_;
            });
        }

        const int b = std::min<int>(max_batch, int(evalq_.size()));
        std::vector<EvalRef> refs(evalq_.begin(), evalq_.begin() + b);
        evalq_.erase(evalq_.begin(), evalq_.begin() + b);

        out.resize(size_t(b) * 8 * NN);
        for (int i = 0; i < b; ++i) {
            const PendingLeaf& p = games_[refs[i].game]->pending[refs[i].pidx];
            std::memcpy(out.data() + size_t(i) * 8 * NN,
                        p.planes.data(), sizeof(float) * 8 * NN);
        }
        const int64_t id = next_batch_id_++;
        outstanding_[id] = std::move(refs);
        count = b;
        return id;
    }

    // logits: (B, A) row-major float32, values: (B,) float32.
    void put_results(int64_t batch_id, const float* logits,
                     const float* values, int B, int A) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = outstanding_.find(batch_id);
        if (it == outstanding_.end())
            throw std::runtime_error("put_results: unknown batch_id");
        std::vector<EvalRef> refs = std::move(it->second);
        outstanding_.erase(it);
        if (int(refs.size()) != B)
            throw std::runtime_error("put_results: batch size mismatch");
        if (A != action_space(cfg_.boardsize))
            throw std::runtime_error("put_results: action-space mismatch");

        for (int i = 0; i < B; ++i) {
            Game& g = *games_[refs[i].game];
            PendingLeaf& p = g.pending[refs[i].pidx];
            const float* row = logits + size_t(i) * A;

            masked_softmax(row, p.legal, p.priors);
            p.value = values[i];
            if (p.cacheable) tt_insert(p.cache_key, row, A, p.value);

            if (--g.results_missing == 0) {
                ready_.push_back(refs[i].game);
                cv_ready_.notify_one();
            }
        }
    }

    // Drain completed-game data. Each output row i of `states` is 8*N*N
    // floats; `policies` rows are A floats.
    void get_data(std::vector<float>& states, std::vector<float>& policies,
                  std::vector<float>& values, long long& n_positions) {
        std::lock_guard<std::mutex> lk(data_mu_);
        states.swap(out_states_);
        policies.swap(out_policies_);
        values.swap(out_values_);
        n_positions = static_cast<long long>(values.size());
        out_states_.clear();
        out_policies_.clear();
        out_values_.clear();
    }

    struct Stats {
        long long p1_wins = 0, p2_wins = 0, draws = 0;
        long long total_plies = 0, total_walls = 0;
        int min_plies = 1 << 30, max_plies = 0;
        long long games = 0;
    };

    Stats stats() const {
        std::lock_guard<std::mutex> lk(data_mu_);
        return stats_;
    }

    const Config& config() const { return cfg_; }

private:
    struct EvalRef { int game; int pidx; };

    // ------------------------------------------------------------------
    // Worker side
    // ------------------------------------------------------------------

    void worker_loop() {
        for (;;) {
            int gi;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_ready_.wait(lk, [&] {
                    return stop_flag_ || !ready_.empty()
                        || games_finished_ >= total_games_;
                });
                if (stop_flag_) return;
                if (ready_.empty()) {
                    if (games_finished_ >= total_games_) return;
                    continue;
                }
                gi = ready_.front();
                ready_.pop_front();
            }
            process_game(gi);
        }
    }

    // Runs one game until it blocks on evaluation or the game quota is hit.
    void process_game(int gi) {
        Game& g = *games_[gi];
        if (!g.pending.empty()) integrate_results(g);

        for (;;) {
            if (g.root < 0) {
                if (submit_root(gi, g)) return;   // queued a real eval; wait
                continue;                          // resolved from cache; keep going
            }

            if (g.sims_done < cfg_.num_simulations) {
                gather(g);
                if (!g.pending.empty()) {
                    submit_pending(gi, g);
                    if (g.results_missing > 0) return;  // some real evals queued; wait
                    integrate_results(g);   // batch fully resolved from cache
                }
                continue;  // all traversals hit terminals; keep searching
            }

            play_move(g);

            bool finished = false;
            int w = g.state.winner();
            if (w != 0 || g.state.depth >= cfg_.max_moves) {
                finished = true;
            } else {
                RepCounter rc{&g.history, nullptr};
                g.state.gen_legal(rc, g.root_legal);
                if (g.root_legal.empty()) finished = true;
            }

            if (finished) {
                finalize_game(g, w);
                bool more, all_done;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    ++games_finished_;
                    more = games_started_ < total_games_;
                    if (more) ++games_started_;
                    all_done = games_finished_ >= total_games_;
                }
                if (all_done) {
                    cv_eval_.notify_all();
                    cv_ready_.notify_all();
                }
                if (!more) return;
                g.reset(cfg_, next_seed());
            }
        }
    }

    uint64_t next_seed() {
        return seed_counter_.fetch_add(0x9E3779B97F4A7C15ULL) ^ 0xD1B54A32D192ED03ULL;
    }

    // PUCT child selection (mirror of MCTSNode.best_child / puct_score).
    int32_t select_child(Game& g, int32_t ni) const {
        const Node& n = g.arena[ni];
        const float parent_q = n.visits > 0 ? n.value_sum / n.visits : 0.0f;
        const float sq = std::sqrt(float(n.visits));

        float vps = 0.0f;
        for (int k = 0; k < n.nchild; ++k) {
            const Node& c = g.arena[n.first_child + k];
            if (c.visits > 0) vps += c.base_prior;
        }
        const float fpu_q = parent_q - float(cfg_.fpu_reduction) * std::sqrt(vps);

        int32_t best = n.first_child;
        float best_score = -1e30f;
        for (int k = 0; k < n.nchild; ++k) {
            const int32_t ci = n.first_child + k;
            const Node& c = g.arena[ci];
            const float u = float(cfg_.c_puct) * c.prior * sq / (1.0f + c.visits);
            const float q = c.visits == 0 ? fpu_q : -(c.value_sum / c.visits);
            const float s = q + u + c.bonus;
            if (s > best_score) { best_score = s; best = ci; }
        }
        return best;
    }

    // Value-only backprop; virtual visits were committed during descent.
    void backup(Game& g, const std::vector<int32_t>& path, float v) const {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            g.arena[*it].value_sum += v;
            v = -v;
        }
    }

    // Collect up to leaf_batch traversals; unresolved leaves go to g.pending.
    void gather(Game& g) {
        const int b = std::min(cfg_.leaf_batch,
                               cfg_.num_simulations - g.sims_done);
        std::vector<int32_t> path;
        std::vector<uint64_t> overlay;
        std::vector<uint16_t> legal;

        for (int t = 0; t < b; ++t) {
            GameState st = g.state;
            path.clear();
            overlay.clear();

            int32_t cur = g.root;
            ++g.arena[cur].visits;   // unscored virtual visit (LC0 convention)
            path.push_back(cur);
            while (g.arena[cur].flags & F_EXPANDED) {
                const int32_t nxt = select_child(g, cur);
                st.apply(g.arena[nxt].action);
                overlay.push_back(st.zhash);
                cur = nxt;
                ++g.arena[cur].visits;
                path.push_back(cur);
            }

            Node& leaf = g.arena[cur];
            if (leaf.flags & F_TERMINAL) {
                backup(g, path, leaf.tvalue);
                continue;
            }
            if (leaf.flags & F_PENDING) {
                g.pending[g.pending_by_node[cur]].paths.push_back(path);
                continue;
            }

            // Fresh leaf: terminal check, else queue for NN eval.
            float tv = 0.0f;
            bool terminal = false;
            if (st.winner() != 0) {
                tv = -1.0f;               // player to move at leaf just lost
                terminal = true;
            } else if (st.depth >= cfg_.max_moves) {
                terminal = true;          // draw
            } else {
                RepCounter rc{&g.history, &overlay};
                st.gen_legal(rc, legal);
                if (legal.empty()) terminal = true;  // draw (no moves)
            }
            if (terminal) {
                leaf.flags |= F_TERMINAL;
                leaf.tvalue = tv;
                backup(g, path, tv);
                continue;
            }

            PendingLeaf p;
            p.node = cur;
            p.legal = legal;
            p.paths.push_back(path);
            if (g.w_cur != 0.0f) {
                p.adv.resize(legal.size());
                st.move_advantages(legal, p.adv.data());
            }

            bool cache_hit = false;
            if (tt_eligible(st.depth)) {
                const TTKey key = tt_key(st);
                TTEntry entry;
                if (tt_lookup(key, entry)) {
                    masked_softmax(entry.logits.data(), legal, p.priors);
                    p.value = entry.value;
                    p.resolved = true;
                    cache_hit = true;
                } else {
                    p.cacheable = true;
                    p.cache_key = key;
                }
            }
            if (!cache_hit) {
                const int NN = cfg_.boardsize * cfg_.boardsize;
                p.planes.resize(8 * NN);
                st.nn_input(p.planes.data());
            }

            leaf.flags |= F_PENDING;
            g.pending_by_node[p.node] = int(g.pending.size());
            g.pending.push_back(std::move(p));
        }
        g.sims_done += b;
    }

    // Create root node + queue its evaluation. Returns true if a real NN
    // round trip was queued (caller should block/return); false if the root
    // was resolved immediately from the transposition cache (caller should
    // keep processing this game without yielding to the eval queue).
    bool submit_root(int gi, Game& g) {
        g.arena.clear();
        g.arena.push_back(Node{});
        g.root = 0;
        g.w_cur = g.state.is_p1_turn() ? g.w_p1 : g.w_p2;

        PendingLeaf p;
        p.node = 0;
        p.is_root = true;
        p.legal = g.root_legal;
        if (g.w_cur != 0.0f) {
            p.adv.resize(p.legal.size());
            g.state.move_advantages(p.legal, p.adv.data());
        }

        bool cache_hit = false;
        if (tt_eligible(g.state.depth)) {
            const TTKey key = tt_key(g.state);
            TTEntry entry;
            if (tt_lookup(key, entry)) {
                masked_softmax(entry.logits.data(), p.legal, p.priors);
                p.value = entry.value;
                p.resolved = true;
                cache_hit = true;
            } else {
                p.cacheable = true;
                p.cache_key = key;
            }
        }
        if (!cache_hit) {
            const int NN = cfg_.boardsize * cfg_.boardsize;
            p.planes.resize(8 * NN);
            g.state.nn_input(p.planes.data());
        }

        g.pending_by_node[0] = 0;
        g.pending.push_back(std::move(p));

        if (cache_hit) {
            g.results_missing = 0;
            integrate_results(g);
            return false;
        }

        g.results_missing = 1;
        std::lock_guard<std::mutex> lk(mu_);
        evalq_.push_back({gi, 0});
        cv_eval_.notify_all();
        return true;
    }

    // Queues only the leaves not already resolved from cache. Sets
    // g.results_missing to the number still awaiting a real NN result (may
    // be 0 if every leaf in this batch was a cache hit).
    void submit_pending(int gi, Game& g) {
        int missing = 0;
        for (const PendingLeaf& p : g.pending) if (!p.resolved) ++missing;
        g.results_missing = missing;
        if (missing == 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < int(g.pending.size()); ++i)
            if (!g.pending[i].resolved) evalq_.push_back({gi, i});
        cv_eval_.notify_all();
    }

    // Expand nodes + back up NN values for every pending leaf.
    void integrate_results(Game& g) {
        for (PendingLeaf& p : g.pending) {
            const int32_t ni = p.node;
            if (!(g.arena[ni].flags & F_EXPANDED)) {
                const int32_t fc = int32_t(g.arena.size());
                const int nc = int(p.legal.size());
                const float wb = g.w_cur != 0.0f
                               ? g.w_cur / float(cfg_.boardsize) : 0.0f;
                for (int k = 0; k < nc; ++k) {
                    Node c;
                    c.action = p.legal[k];
                    c.prior = c.base_prior = p.priors[k];
                    if (wb != 0.0f) c.bonus = wb * p.adv[k];
                    g.arena.push_back(c);
                }
                Node& n = g.arena[ni];   // re-fetch: push_back may reallocate
                n.first_child = fc;
                n.nchild = uint16_t(nc);
                n.flags = uint8_t((n.flags | F_EXPANDED) & ~F_PENDING);
            }

            if (p.is_root) {
                if (cfg_.training && cfg_.dirichlet_eps > 0.0) {
                    Node& r = g.arena[ni];
                    std::gamma_distribution<double> gd(cfg_.dirichlet_alpha, 1.0);
                    std::vector<double> noise(r.nchild);
                    double s = 0.0;
                    for (auto& x : noise) { x = gd(g.rng); s += x; }
                    if (s > 0.0) {
                        const double eps = cfg_.dirichlet_eps;
                        for (int k = 0; k < r.nchild; ++k) {
                            Node& c = g.arena[r.first_child + k];
                            c.prior = float((1.0 - eps) * c.base_prior
                                            + eps * noise[k] / s);
                        }
                    }
                }
                // Initial root backup (mirror of search(): expand + _backup).
                g.arena[ni].visits += 1;
                g.arena[ni].value_sum += p.value;
            } else {
                for (const auto& path : p.paths) backup(g, path, p.value);
            }
        }
        g.pending.clear();
        g.pending_by_node.clear();
    }

    // ------------------------------------------------------------------
    // Shared transposition table: sharded by zhash to keep contention low
    // across the (small, core-bound) worker-thread pool. Lookups happen
    // lock-free w.r.t. `mu_` (called from gather()/submit_root() on worker
    // threads); inserts happen from put_results() while `mu_` is held, but
    // shard mutexes are never held while acquiring `mu_`, so lock order is
    // always mu_ -> shard, never the reverse -- no deadlock risk.
    // ------------------------------------------------------------------

    static constexpr int kTTShards = 64;
    struct TTShard {
        mutable std::mutex mu;
        std::unordered_map<TTKey, TTEntry, TTKeyHash> map;
    };
    std::array<TTShard, kTTShards> tt_;

    // 0 => TT off entirely; negative tt_max_depth => unlimited (no ceiling);
    // otherwise only depths <= tt_max_depth are eligible.
    bool tt_eligible(int depth) const {
        if (cfg_.tt_max_depth == 0 || cfg_.tt_max_entries == 0) return false;
        return cfg_.tt_max_depth < 0 || depth <= cfg_.tt_max_depth;
    }

    bool tt_lookup(const TTKey& key, TTEntry& out) const {
        const auto& shard = tt_[key.zhash % kTTShards];
        std::lock_guard<std::mutex> lk(shard.mu);
        auto it = shard.map.find(key);
        if (it == shard.map.end()) return false;
        out = it->second;   // copy out while holding the shard lock
        return true;
    }

    void tt_insert(const TTKey& key, const float* row, int A, float value) {
        if (cfg_.tt_max_entries == 0) return;  // TT storage fully disabled
        auto& shard = tt_[key.zhash % kTTShards];
        std::lock_guard<std::mutex> lk(shard.mu);
        auto it = shard.map.find(key);
        if (it == shard.map.end()) {
            // Only a brand-new key can grow the shard -- overwrites of an
            // existing key never need eviction. Cap is per-shard so the
            // total table size across all shards stays bounded. Guaranteed
            // >= 1 so any positive tt_max_entries still caches something,
            // even if it's smaller than kTTShards.
            const size_t cap = std::max<size_t>(1, cfg_.tt_max_entries / kTTShards);
            if (shard.map.size() >= cap) {
                // Arbitrary (not LRU) eviction: O(1), no extra bookkeeping
                // on the hot lookup path. Evicting just means a future hit
                // becomes a normal NN eval again -- always correct, never
                // stale, since eviction can't return wrong data.
                shard.map.erase(shard.map.begin());
            }
            it = shard.map.emplace(key, TTEntry{}).first;
        }
        it->second.logits.assign(row, row + A);
        it->second.value = value;
    }

    // Record policy target, sample a move, advance the real game state.
    void play_move(Game& g) {
        const Node& r = g.arena[g.root];
        const int nc = r.nchild;
        const int A = action_space(cfg_.boardsize);
        const int NN = cfg_.boardsize * cfg_.boardsize;

        std::vector<double> probs(nc, 0.0);
        long long total = 0;
        int argmax = 0;
        for (int k = 0; k < nc; ++k) {
            const int v = g.arena[r.first_child + k].visits;
            total += v;
            if (v > g.arena[r.first_child + argmax].visits) argmax = k;
        }
        const bool tau1 = cfg_.training && g.state.depth < cfg_.temp_threshold;
        if (!tau1) {
            probs[argmax] = 1.0;
        } else if (total > 0) {
            for (int k = 0; k < nc; ++k)
                probs[k] = double(g.arena[r.first_child + k].visits) / double(total);
        } else {
            for (int k = 0; k < nc; ++k) probs[k] = 1.0 / nc;
        }

        // Record training example (values assigned at game end).
        const size_t sp = g.rec_planes.size();
        g.rec_planes.resize(sp + 8 * NN);
        g.state.nn_input(g.rec_planes.data() + sp);
        const size_t pp = g.rec_policy.size();
        g.rec_policy.resize(pp + A, 0.0f);
        for (int k = 0; k < nc; ++k)
            g.rec_policy[pp + g.arena[r.first_child + k].action] = float(probs[k]);
        g.rec_player.push_back(int8_t(g.state.current_player()));
        ++g.plies;

        // Sample the move from the same distribution.
        int chosen = argmax;
        if (tau1) {
            std::uniform_real_distribution<double> u(0.0, 1.0);
            double x = u(g.rng), acc = 0.0;
            for (int k = 0; k < nc; ++k) {
                acc += probs[k];
                if (x <= acc) { chosen = k; break; }
            }
        }
        g.state.apply(g.arena[r.first_child + chosen].action);
        ++g.history[g.state.zhash];
        if (g.n_opening < Game::kOpeningTrack)
            g.opening_hashes[g.n_opening++] = g.state.zhash;

        // Fresh tree for the next move.
        g.arena.clear();
        g.root = -1;
        g.sims_done = 0;
    }

    void finalize_game(Game& g, int winner) {
        const int A = action_space(cfg_.boardsize);
        const int NN = cfg_.boardsize * cfg_.boardsize;
        const int walls_placed = 2 * g.state.walls_initial
                               - g.state.walls_p1 - g.state.walls_p2;

        std::lock_guard<std::mutex> lk(data_mu_);
        out_states_.insert(out_states_.end(),
                           g.rec_planes.begin(), g.rec_planes.end());
        out_policies_.insert(out_policies_.end(),
                             g.rec_policy.begin(), g.rec_policy.end());
        for (int i = 0; i < g.plies; ++i) {
            float v = 0.0f;
            if (winner != 0) v = (g.rec_player[i] == winner) ? 1.0f : -1.0f;
            out_values_.push_back(v);
        }
        (void)A; (void)NN;

        ++stats_.games;
        if (winner == 1) ++stats_.p1_wins;
        else if (winner == 2) ++stats_.p2_wins;
        else ++stats_.draws;
        stats_.total_plies += g.plies;
        stats_.total_walls += walls_placed;
        stats_.min_plies = std::min(stats_.min_plies, g.plies);
        stats_.max_plies = std::max(stats_.max_plies, g.plies);

        out_openings_.push_back(g.opening_hashes);
        out_opening_lens_.push_back(g.n_opening);
    }

    // ------------------------------------------------------------------

    Config cfg_;
    int num_threads_;
    int parallel_games_;

    mutable std::mutex mu_;
    std::condition_variable cv_ready_, cv_eval_;
    std::vector<std::unique_ptr<Game>> games_;
    std::deque<int> ready_;
    std::deque<EvalRef> evalq_;
    std::unordered_map<int64_t, std::vector<EvalRef>> outstanding_;
    int64_t next_batch_id_ = 0;
    int total_games_ = 0, games_started_ = 0, games_finished_ = 0;
    bool stop_flag_ = false;
    std::vector<std::thread> workers_;
    std::atomic<uint64_t> seed_counter_{1};

    mutable std::mutex data_mu_;
    std::vector<float> out_states_, out_policies_, out_values_;
    Stats stats_;

    // Diagnostic only (see get_openings()).
    std::vector<std::array<uint64_t, Game::kOpeningTrack>> out_openings_;
    std::vector<int> out_opening_lens_;

public:
    void get_openings(std::vector<std::array<uint64_t, Game::kOpeningTrack>>& openings,
                      std::vector<int>& lens) {
        std::lock_guard<std::mutex> lk(data_mu_);
        openings.swap(out_openings_);
        lens.swap(out_opening_lens_);
        out_openings_.clear();
        out_opening_lens_.clear();
    }
};

}  // namespace quoridor
