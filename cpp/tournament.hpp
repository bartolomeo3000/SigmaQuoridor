// Tournament manager: many concurrent cross-play games between arbitrary
// (model, sims) "agents", multiplexed over worker threads, heavily adapted
// from selfplay.hpp's SelfPlayManager architecture.
//
// Differences from SelfPlayManager (this is evaluation/tournament play, not
// training data generation -- mirrors mcts.py's MCTSAgent(training=False)):
//   * No Dirichlet root noise, no dist-heuristic bonus, no tau1/tau0
//     temp_threshold ply-schedule -- each agent has one constant
//     `temperature` for the whole game (0 = deterministic argmax, matching
//     tools/_matchup.py; >0 = visits^(1/temperature) sampling, matching
//     tournament.py/tournament_simcounts.py).
//   * No recorded training trajectory -- only the final (winner, plies) is
//     reported per game.
//   * No shared transposition table (skipped for simplicity -- tournament
//     runs are far smaller than self-play data-generation runs, and TT
//     entries would need to be keyed per-model since different weights
//     produce different logits for the same state).
//   * A game's two players can each use a DIFFERENT model, so leaf/root NN
//     evaluations are routed to one of several per-model inference queues
//     (`evalq_[model_id]`) instead of a single shared queue. Each player's
//     agent (c_puct, fpu_reduction, num_simulations, leaf_batch, temperature)
//     is looked up per-move via `TGame::cur_agent()`.
//   * Games are drawn from an explicit, heterogeneous match list (each entry
//     names its own p1_agent/p2_agent) rather than a homogeneous game count.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#include "engine.hpp"

namespace quoridor {

// One player's engine configuration for a tournament game: which model to
// query (routes evals to that model's own inference queue) and how to
// search with it.
struct AgentSpec {
    int    model_id        = 0;
    int    num_simulations = 800;
    int    leaf_batch      = 1;
    double c_puct          = 1.0;
    double fpu_reduction   = 0.1;
    double temperature     = 0.0;   // 0 = deterministic argmax
};

// One scheduled game: which two registered agents play, who is P1.
struct MatchSpec {
    int p1_agent = 0;
    int p2_agent = 1;
};

struct TConfig {
    int boardsize = 7;
    int walls     = 5;
    int max_moves = 200;
    uint64_t seed = 0;
    std::vector<AgentSpec> agents;   // indexed by MatchSpec::p1_agent/p2_agent
};

struct TNode {
    float    prior       = 0.0f;
    float    value_sum   = 0.0f;
    int32_t  visits      = 0;
    int32_t  first_child = -1;
    uint16_t nchild      = 0;
    uint16_t action      = 0;
    uint8_t  flags       = 0;
    float    tvalue      = 0.0f;
};

constexpr uint8_t TF_EXPANDED = 1;
constexpr uint8_t TF_TERMINAL = 2;
constexpr uint8_t TF_PENDING  = 4;

// Masked softmax over `legal` actions from a raw (A,) logits row. The network
// outputs policy in the current player's canonical POV, so P2 leaves (`flip`)
// read each real action's logit at its vertically-flipped index.
inline void t_masked_softmax(const float* row, const std::vector<uint16_t>& legal,
                             std::vector<float>& priors, int N, bool flip) {
    priors.resize(legal.size());
    float mx = -1e30f;
    for (uint16_t a : legal) mx = std::max(mx, row[flip ? vflip_action(a, N) : a]);
    double sum = 0.0;
    for (size_t j = 0; j < legal.size(); ++j) {
        const int idx = flip ? vflip_action(legal[j], N) : int(legal[j]);
        const double e = std::exp(double(row[idx] - mx));
        priors[j] = float(e);
        sum += e;
    }
    const float inv = float(1.0 / (sum > 0 ? sum : 1.0));
    for (auto& pr : priors) pr *= inv;
}

struct TPendingLeaf {
    int32_t node = -1;
    bool is_root = false;
    std::vector<uint16_t> legal;
    std::vector<std::vector<int32_t>> paths;
    std::vector<float> planes;
    std::vector<float> priors;
    float value = 0.0f;
    bool flip = false;                    // P2 leaf: gather priors in canonical frame
};

struct TGame {
    GameState state;
    std::unordered_map<uint64_t, int> history;
    std::vector<uint16_t> root_legal;

    std::vector<TNode> arena;
    int32_t root = -1;
    int sims_done = 0;

    std::vector<TPendingLeaf> pending;
    std::unordered_map<int32_t, int> pending_by_node;
    int results_missing = 0;

    std::mt19937_64 rng;

    int match_index = -1;   // index into the caller's original match list
    int p1_agent = 0, p2_agent = 1;
    int plies = 0;

    const AgentSpec* agents = nullptr;   // points into TournamentManager::cfg_.agents

    const AgentSpec& cur_agent() const {
        return agents[state.is_p1_turn() ? p1_agent : p2_agent];
    }

    void reset(const TConfig& cfg, int match_idx, int p1a, int p2a, uint64_t seed) {
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
        match_index = match_idx;
        p1_agent = p1a;
        p2_agent = p2a;
        agents = cfg.agents.data();
        plies = 0;
        RepCounter rc{&history, nullptr};
        state.gen_legal(rc, root_legal);
    }
};

struct MatchResult {
    int match_index;
    int winner;   // 0 = draw, 1 = p1, 2 = p2
    int plies;
};

class TournamentManager {
public:
    TournamentManager(TConfig cfg, int num_threads, int parallel_games)
        : cfg_(std::move(cfg)), num_threads_(num_threads), parallel_games_(parallel_games) {
        int max_model = -1;
        for (const auto& a : cfg_.agents) max_model = std::max(max_model, a.model_id);
        num_models_ = max_model + 1;
        evalq_.resize(std::max(num_models_, 1));
        outstanding_.resize(std::max(num_models_, 1));
    }

    ~TournamentManager() { stop(); }

    void start(std::vector<MatchSpec> matches) {
        stop();  // in case of restart
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_flag_ = false;
            matches_ = std::move(matches);
            total_games_ = int(matches_.size());
            games_finished_ = 0;
            for (auto& q : evalq_) q.clear();
            for (auto& m : outstanding_) m.clear();
            ready_.clear();
            const int n = std::min(parallel_games_, total_games_);
            games_.clear();
            games_.reserve(n);
            seed_counter_ = cfg_.seed * 0x9E3779B97F4A7C15ULL + 1;
            for (int i = 0; i < n; ++i) {
                games_.push_back(std::make_unique<TGame>());
                const MatchSpec& m = matches_[i];
                games_[i]->reset(cfg_, i, m.p1_agent, m.p2_agent, next_seed());
                ready_.push_back(i);
            }
            games_started_ = n;
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

    int num_models() const { return num_models_; }

    // Blocks (caller should release the GIL) until leaves for `model_id` are
    // available or all games are done. Returns batch_id, fills `out` with
    // (count * 8*N*N) floats. count == 0 signals no more work for this model.
    int64_t get_batch(int model_id, int max_batch, int flush_us,
                      std::vector<float>& out, int& count) {
        const int NN = cfg_.boardsize * cfg_.boardsize;
        std::unique_lock<std::mutex> lk(mu_);
        auto& q = evalq_.at(model_id);
        cv_eval_.wait(lk, [&] {
            return !q.empty() || games_finished_ >= total_games_ || stop_flag_;
        });
        if (q.empty()) { count = 0; return -1; }

        if (int(q.size()) < max_batch && flush_us > 0) {
            const auto deadline = std::chrono::steady_clock::now()
                                + std::chrono::microseconds(flush_us);
            cv_eval_.wait_until(lk, deadline, [&] {
                return int(q.size()) >= max_batch
                    || games_finished_ >= total_games_ || stop_flag_;
            });
        }

        const int b = std::min<int>(max_batch, int(q.size()));
        std::vector<EvalRef> refs(q.begin(), q.begin() + b);
        q.erase(q.begin(), q.begin() + b);

        out.resize(size_t(b) * 8 * NN);
        for (int i = 0; i < b; ++i) {
            const TPendingLeaf& p = games_[refs[i].game]->pending[refs[i].pidx];
            std::memcpy(out.data() + size_t(i) * 8 * NN,
                        p.planes.data(), sizeof(float) * 8 * NN);
        }
        const int64_t id = next_batch_id_++;
        outstanding_.at(model_id)[id] = std::move(refs);
        count = b;
        return id;
    }

    void put_results(int model_id, int64_t batch_id, const float* logits,
                     const float* values, int B, int A) {
        std::lock_guard<std::mutex> lk(mu_);
        auto& omap = outstanding_.at(model_id);
        auto it = omap.find(batch_id);
        if (it == omap.end())
            throw std::runtime_error("put_results: unknown batch_id");
        std::vector<EvalRef> refs = std::move(it->second);
        omap.erase(it);
        if (int(refs.size()) != B)
            throw std::runtime_error("put_results: batch size mismatch");
        if (A != action_space(cfg_.boardsize))
            throw std::runtime_error("put_results: action-space mismatch");

        for (int i = 0; i < B; ++i) {
            TGame& g = *games_[refs[i].game];
            TPendingLeaf& p = g.pending[refs[i].pidx];
            const float* row = logits + size_t(i) * A;

            t_masked_softmax(row, p.legal, p.priors, cfg_.boardsize, p.flip);
            p.value = values[i];

            if (--g.results_missing == 0) {
                ready_.push_back(refs[i].game);
                cv_ready_.notify_one();
            }
        }
    }

    // Drain finished-game results accumulated so far.
    std::vector<MatchResult> get_results() {
        std::lock_guard<std::mutex> lk(data_mu_);
        std::vector<MatchResult> r;
        r.swap(out_results_);
        return r;
    }

    const TConfig& config() const { return cfg_; }

private:
    struct EvalRef { int game; int pidx; };

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

    void process_game(int gi) {
        TGame& g = *games_[gi];
        if (!g.pending.empty()) integrate_results(g);

        for (;;) {
            if (g.root < 0) {
                submit_root(gi, g);
                return;   // always a real eval (no TT); caller waits
            }

            const AgentSpec& ag = g.cur_agent();
            if (g.sims_done < ag.num_simulations) {
                gather(g);
                if (!g.pending.empty()) {
                    // NOTE: once submit_pending has queued refs, ownership of
                    // this game transfers to the inference thread -- do not
                    // touch `g` again after a true return (see submit_pending).
                    if (submit_pending(gi, g)) return;  // real evals queued; wait
                    integrate_results(g);   // (unreachable: no TT means every
                                             // pending leaf always needs a
                                             // real eval; kept for symmetry)
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
                int next_match_idx = -1;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    ++games_finished_;
                    more = games_started_ < total_games_;
                    if (more) { next_match_idx = games_started_; ++games_started_; }
                    all_done = games_finished_ >= total_games_;
                }
                if (all_done) {
                    cv_eval_.notify_all();
                    cv_ready_.notify_all();
                }
                if (!more) return;
                const MatchSpec& m = matches_[next_match_idx];
                g.reset(cfg_, next_match_idx, m.p1_agent, m.p2_agent, next_seed());
            }
        }
    }

    uint64_t next_seed() {
        return seed_counter_.fetch_add(0x9E3779B97F4A7C15ULL) ^ 0xD1B54A32D192ED03ULL;
    }

    int32_t select_child(TGame& g, int32_t ni) const {
        const AgentSpec& ag = g.cur_agent();
        const TNode& n = g.arena[ni];
        const float parent_q = n.visits > 0 ? n.value_sum / n.visits : 0.0f;
        const float sq = std::sqrt(float(n.visits));

        float vps = 0.0f;
        for (int k = 0; k < n.nchild; ++k) {
            const TNode& c = g.arena[n.first_child + k];
            if (c.visits > 0) vps += c.prior;
        }
        const float fpu_q = parent_q - float(ag.fpu_reduction) * std::sqrt(vps);

        int32_t best = n.first_child;
        float best_score = -1e30f;
        for (int k = 0; k < n.nchild; ++k) {
            const int32_t ci = n.first_child + k;
            const TNode& c = g.arena[ci];
            const float u = float(ag.c_puct) * c.prior * sq / (1.0f + c.visits);
            const float q = c.visits == 0 ? fpu_q : -(c.value_sum / c.visits);
            const float s = q + u;
            if (s > best_score) { best_score = s; best = ci; }
        }
        return best;
    }

    void backup(TGame& g, const std::vector<int32_t>& path, float v) const {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            g.arena[*it].value_sum += v;
            v = -v;
        }
    }

    void gather(TGame& g) {
        const AgentSpec& ag = g.cur_agent();
        const int b = std::min(ag.leaf_batch, ag.num_simulations - g.sims_done);
        std::vector<int32_t> path;
        std::vector<uint64_t> overlay;
        std::vector<uint16_t> legal;

        for (int t = 0; t < b; ++t) {
            GameState st = g.state;
            path.clear();
            overlay.clear();

            int32_t cur = g.root;
            ++g.arena[cur].visits;
            path.push_back(cur);
            while (g.arena[cur].flags & TF_EXPANDED) {
                const int32_t nxt = select_child(g, cur);
                st.apply(g.arena[nxt].action);
                overlay.push_back(st.zhash);
                cur = nxt;
                ++g.arena[cur].visits;
                path.push_back(cur);
            }

            TNode& leaf = g.arena[cur];
            if (leaf.flags & TF_TERMINAL) {
                backup(g, path, leaf.tvalue);
                continue;
            }
            if (leaf.flags & TF_PENDING) {
                g.pending[g.pending_by_node[cur]].paths.push_back(path);
                continue;
            }

            float tv = 0.0f;
            bool terminal = false;
            if (st.winner() != 0) {
                tv = -1.0f;
                terminal = true;
            } else if (st.depth >= cfg_.max_moves) {
                terminal = true;
            } else {
                RepCounter rc{&g.history, &overlay};
                st.gen_legal(rc, legal);
                if (legal.empty()) terminal = true;
            }
            if (terminal) {
                leaf.flags |= TF_TERMINAL;
                leaf.tvalue = tv;
                backup(g, path, tv);
                continue;
            }

            TPendingLeaf p;
            p.node = cur;
            p.legal = legal;
            p.flip = !st.is_p1_turn();
            p.paths.push_back(path);
            const int NN = cfg_.boardsize * cfg_.boardsize;
            p.planes.resize(8 * NN);
            st.nn_input(p.planes.data());

            leaf.flags |= TF_PENDING;
            g.pending_by_node[p.node] = int(g.pending.size());
            g.pending.push_back(std::move(p));
        }
        g.sims_done += b;
    }

    void submit_root(int gi, TGame& g) {
        g.arena.clear();
        g.arena.push_back(TNode{});
        g.root = 0;

        TPendingLeaf p;
        p.node = 0;
        p.is_root = true;
        p.legal = g.root_legal;
        p.flip = !g.state.is_p1_turn();
        const int NN = cfg_.boardsize * cfg_.boardsize;
        p.planes.resize(8 * NN);
        g.state.nn_input(p.planes.data());

        g.pending_by_node[0] = 0;
        g.pending.push_back(std::move(p));
        g.results_missing = 1;

        const int model_id = g.cur_agent().model_id;
        std::lock_guard<std::mutex> lk(mu_);
        evalq_.at(model_id).push_back({gi, 0});
        cv_eval_.notify_all();
    }

    // Returns true if real NN evals were queued (caller must stop touching
    // the game: the moment the refs are published under mu_, the inference
    // thread may finish them and hand the game to another worker -- reading
    // g.results_missing after that races with put_results decrementing it
    // on a different thread. Mirrors the fix applied to selfplay.hpp for
    // the identical Windows crash there.) Returns false if there was
    // nothing to submit (empty pending list).
    bool submit_pending(int gi, TGame& g) {
        const int n = int(g.pending.size());
        if (n == 0) {
            g.results_missing = 0;
            return false;
        }
        const int model_id = g.cur_agent().model_id;
        std::lock_guard<std::mutex> lk(mu_);
        g.results_missing = n;
        for (int i = 0; i < n; ++i)
            evalq_.at(model_id).push_back({gi, i});
        cv_eval_.notify_all();
        return true;
    }

    void integrate_results(TGame& g) {
        for (TPendingLeaf& p : g.pending) {
            const int32_t ni = p.node;
            if (!(g.arena[ni].flags & TF_EXPANDED)) {
                const int32_t fc = int32_t(g.arena.size());
                const int nc = int(p.legal.size());
                for (int k = 0; k < nc; ++k) {
                    TNode c;
                    c.action = p.legal[k];
                    c.prior = p.priors[k];
                    g.arena.push_back(c);
                }
                TNode& n = g.arena[ni];   // re-fetch: push_back may reallocate
                n.first_child = fc;
                n.nchild = uint16_t(nc);
                n.flags = uint8_t((n.flags | TF_EXPANDED) & ~TF_PENDING);
            }

            if (p.is_root) {
                g.arena[ni].visits += 1;
                g.arena[ni].value_sum += p.value;
            } else {
                for (const auto& path : p.paths) backup(g, path, p.value);
            }
        }
        g.pending.clear();
        g.pending_by_node.clear();
    }

    // Record policy target visit distribution -> sample/argmax a move,
    // advance the real game state. No training-data recording.
    void play_move(TGame& g) {
        const AgentSpec& ag = g.cur_agent();
        const TNode& r = g.arena[g.root];
        const int nc = r.nchild;

        int argmax = 0;
        for (int k = 1; k < nc; ++k)
            if (g.arena[r.first_child + k].visits > g.arena[r.first_child + argmax].visits)
                argmax = k;

        int chosen = argmax;
        if (ag.temperature > 0.0) {
            std::vector<double> probs(nc, 0.0);
            const double inv_t = 1.0 / ag.temperature;
            double total = 0.0;
            for (int k = 0; k < nc; ++k) {
                probs[k] = std::pow(double(g.arena[r.first_child + k].visits), inv_t);
                total += probs[k];
            }
            if (total > 0.0) {
                for (auto& x : probs) x /= total;
            } else {
                for (auto& x : probs) x = 1.0 / nc;
            }
            std::uniform_real_distribution<double> u(0.0, 1.0);
            double x = u(g.rng), acc = 0.0;
            for (int k = 0; k < nc; ++k) {
                acc += probs[k];
                if (x <= acc) { chosen = k; break; }
            }
        }

        ++g.plies;
        g.state.apply(g.arena[r.first_child + chosen].action);
        ++g.history[g.state.zhash];

        g.arena.clear();
        g.root = -1;
        g.sims_done = 0;
    }

    void finalize_game(TGame& g, int winner) {
        std::lock_guard<std::mutex> lk(data_mu_);
        out_results_.push_back(MatchResult{g.match_index, winner, g.plies});
    }

    // ------------------------------------------------------------------

    TConfig cfg_;
    int num_threads_;
    int parallel_games_;
    int num_models_ = 1;

    mutable std::mutex mu_;
    std::condition_variable cv_ready_, cv_eval_;
    std::vector<std::unique_ptr<TGame>> games_;
    std::deque<int> ready_;
    std::vector<std::deque<EvalRef>> evalq_;                          // per model_id
    std::vector<std::unordered_map<int64_t, std::vector<EvalRef>>> outstanding_;  // per model_id
    int64_t next_batch_id_ = 0;
    std::vector<MatchSpec> matches_;
    int total_games_ = 0, games_started_ = 0, games_finished_ = 0;
    bool stop_flag_ = false;
    std::vector<std::thread> workers_;
    std::atomic<uint64_t> seed_counter_{1};

    mutable std::mutex data_mu_;
    std::vector<MatchResult> out_results_;
};

}  // namespace quoridor
