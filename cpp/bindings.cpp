// pybind11 bindings for the C++ Quoridor engine + self-play manager.
//
// Exposes:
//   State               — single-game state (parity testing / scripting)
//   SelfPlayManager     — threaded self-play with a shared inference queue
//   random_playouts     — engine-only microbenchmark (no NN)
//   action_space_size   — 8 + 2*(N-1)^2

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <limits>

#include "alphabeta.hpp"
#include "rollout_mcts.hpp"
#include "selfplay.hpp"
#include "tournament.hpp"

namespace py = pybind11;
using namespace quoridor;

// ---------------------------------------------------------------------------
// State wrapper (real game line, with position history for repetition rule)
// ---------------------------------------------------------------------------

class PyState {
public:
    PyState(int boardsize, int walls, int max_moves = 200)
        : max_moves_(max_moves) {
        st_.init(boardsize, walls);
        hist_[st_.zhash] = 1;
    }

    std::vector<int> legal_actions() {
        RepCounter rc{&hist_, nullptr};
        std::vector<uint16_t> v;
        st_.gen_legal(rc, v);
        return std::vector<int>(v.begin(), v.end());
    }

    void apply(int action) {
        if (action < 0 || action >= action_space(st_.N))
            throw std::out_of_range("action index out of range");
        st_.apply(uint16_t(action));
        ++hist_[st_.zhash];
    }

    int  winner() const { return st_.winner(); }
    int  depth() const { return st_.depth; }
    int  current_player() const { return st_.current_player(); }
    int  walls_p1() const { return st_.walls_p1; }
    int  walls_p2() const { return st_.walls_p2; }

    bool is_finished() {
        if (st_.winner() != 0) return true;
        if (st_.depth >= max_moves_) return true;
        return legal_actions().empty();
    }

    py::array_t<float> nn_input() const {
        const int N = st_.N;
        py::array_t<float> arr({8, N, N});
        st_.nn_input(arr.mutable_data());
        return arr;
    }

    // Simple no-NN baseline: PUCT MCTS with uniform priors + random-rollout
    // leaf evaluation, optionally nudged by a progressive-bias distance
    // heuristic (see rollout_mcts.hpp). Fresh arena every call (no tree
    // reuse across moves, same deliberate simplification as SelfPlayManager).
    int rollout_action(int num_simulations, double c_puct, uint64_t seed,
                        double dist_bonus_weight) {
        int action;
        {
            py::gil_scoped_release rel;
            RolloutMCTS mcts(c_puct, max_moves_, seed, dist_bonus_weight);
            action = mcts.search(st_, hist_, num_simulations);
        }
        return action;
    }

    // Exact alpha-beta solve of the current position (searches to full
    // terminal depth -- no heuristic leaf evaluation -- so the result is
    // a proven game-theoretic value, not an estimate, UNLESS the node/time
    // budget was exhausted first (check "timed_out" in the returned dict).
    py::dict solve_alphabeta(long long node_limit = -1, double time_limit_s = -1.0) {
        AlphaBetaResult r;
        {
            py::gil_scoped_release rel;
            AlphaBetaSolver solver(node_limit, time_limit_s);
            r = solver.solve(st_, &hist_);
        }
        py::dict d;
        d["value"] = r.value;
        d["dist"] = r.dist;
        d["best_action"] = r.best_action == 0xFFFF ? py::object(py::none()) : py::cast(int(r.best_action));
        d["nodes"] = r.nodes;
        d["timed_out"] = r.timed_out;
        return d;
    }

private:
    GameState st_;
    std::unordered_map<uint64_t, int> hist_;
    int max_moves_;
};

// ---------------------------------------------------------------------------
// Engine-only benchmark: random-legal-move playouts (no NN, GIL released)
// ---------------------------------------------------------------------------

static py::dict random_playouts(int num_games, int boardsize, int walls,
                                uint64_t seed, int max_moves) {
    long long total_moves = 0, total_legal_calls = 0;
    double seconds = 0.0;
    {
        py::gil_scoped_release rel;
        std::mt19937_64 rng(seed);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<uint16_t> legal;
        for (int gme = 0; gme < num_games; ++gme) {
            GameState st;
            st.init(boardsize, walls);
            std::unordered_map<uint64_t, int> hist;
            hist[st.zhash] = 1;
            while (st.winner() == 0 && st.depth < max_moves) {
                RepCounter rc{&hist, nullptr};
                st.gen_legal(rc, legal);
                ++total_legal_calls;
                if (legal.empty()) break;
                st.apply(legal[rng() % legal.size()]);
                ++hist[st.zhash];
                ++total_moves;
            }
        }
        seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    }
    py::dict d;
    d["games"] = num_games;
    d["moves"] = total_moves;
    d["legal_gen_calls"] = total_legal_calls;
    d["seconds"] = seconds;
    d["moves_per_sec"] = seconds > 0 ? double(total_moves) / seconds : 0.0;
    return d;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(quoridor_cpp, m) {
    m.doc() = "C++ Quoridor engine, MCTS and batched self-play for SigmaQuoridor";

    m.def("action_space_size", &action_space, py::arg("boardsize"));

    py::class_<PyState>(m, "State")
        .def(py::init<int, int, int>(),
             py::arg("boardsize") = 7, py::arg("walls") = 5,
             py::arg("max_moves") = 200)
        .def("legal_actions", &PyState::legal_actions)
        .def("apply", &PyState::apply, py::arg("action"))
        .def("winner", &PyState::winner)
        .def("depth", &PyState::depth)
        .def("current_player", &PyState::current_player)
        .def("walls_p1", &PyState::walls_p1)
        .def("walls_p2", &PyState::walls_p2)
        .def("is_finished", &PyState::is_finished)
        .def("nn_input", &PyState::nn_input)
        .def("rollout_action", &PyState::rollout_action,
             py::arg("num_simulations") = 400, py::arg("c_puct") = 1.4,
             py::arg("seed") = 0, py::arg("dist_bonus_weight") = 0.0,
             "No-NN baseline: PUCT MCTS with uniform priors + random-rollout "
             "leaf evaluation, optionally nudged by a classic MCTS "
             "progressive-bias term h(c)/(1+N(c)) based on "
             "(my_dist_to_goal - opp_dist_to_goal) (dist_bonus_weight, "
             "0.0 = disabled). Returns the most-visited root action.")
        .def("solve_alphabeta", &PyState::solve_alphabeta,
             py::arg("node_limit") = -1, py::arg("time_limit_s") = -1.0,
             "Exact alpha-beta solve of the current position (searches to "
             "full terminal depth, distance-heuristic move ordering only -- "
             "does not affect exactness). Returns dict(value, best_action, "
             "nodes, timed_out); value/best_action are only proven exact "
             "when timed_out is False.");

    m.def("random_playouts", &random_playouts,
          py::arg("num_games") = 100, py::arg("boardsize") = 7,
          py::arg("walls") = 5, py::arg("seed") = 0,
          py::arg("max_moves") = 200,
          "Engine-only microbenchmark: uniform random legal playouts.");

    py::class_<SelfPlayManager>(m, "SelfPlayManager")
        .def(py::init([](int boardsize, int walls, int num_simulations,
                         int leaf_batch, int num_threads, int parallel_games,
                         double c_puct, double dirichlet_alpha,
                         double dirichlet_epsilon, double fpu_reduction,
                         double temp_early, double temp_final,
                         double temp_halflife, int temp_prune_visits,
                         int max_moves,
                         double dist_bonus_max, bool training, uint64_t seed,
                         int tt_max_depth, long long tt_max_entries,
                         int solver_max_total_walls, long long solver_node_limit,
                         double solver_time_limit_s,
                         int mcts_solver_max_total_walls, long long mcts_solver_node_limit,
                         double mcts_solver_time_limit_s,
                         double pcr_full_prob, int pcr_cheap_sims,
                         bool pcr_cheap_noise) {
                 Config cfg;
                 cfg.boardsize       = boardsize;
                 cfg.walls           = walls;
                 cfg.num_simulations = num_simulations;
                 cfg.leaf_batch      = leaf_batch;
                 cfg.c_puct          = c_puct;
                 cfg.dirichlet_alpha = dirichlet_alpha;
                 cfg.dirichlet_eps   = dirichlet_epsilon;
                 cfg.fpu_reduction   = fpu_reduction;
                 cfg.temp_early      = temp_early;
                 cfg.temp_final      = temp_final;
                 cfg.temp_halflife   = temp_halflife;
                 cfg.temp_prune_visits = temp_prune_visits;
                 cfg.max_moves       = max_moves;
                 cfg.dist_bonus_max  = dist_bonus_max;
                 cfg.training        = training;
                 cfg.seed            = seed;
                 cfg.tt_max_depth    = tt_max_depth;
                 // -1 (or any negative) means "unlimited" (sugar for a huge
                 // cap); 0 disables the TT entirely; positive N is a real cap.
                 cfg.tt_max_entries  = tt_max_entries < 0
                     ? std::numeric_limits<size_t>::max()
                     : static_cast<size_t>(tt_max_entries);
                 cfg.solver_max_total_walls = solver_max_total_walls;
                 cfg.solver_node_limit      = solver_node_limit;
                 cfg.solver_time_limit_s    = solver_time_limit_s;
                 cfg.mcts_solver_max_total_walls = mcts_solver_max_total_walls;
                 cfg.mcts_solver_node_limit      = mcts_solver_node_limit;
                 cfg.mcts_solver_time_limit_s    = mcts_solver_time_limit_s;
                 cfg.pcr_full_prob   = pcr_full_prob;
                 cfg.pcr_cheap_sims  = pcr_cheap_sims;
                 cfg.pcr_cheap_noise = pcr_cheap_noise;
                 return new SelfPlayManager(cfg, num_threads, parallel_games);
             }),
             py::arg("boardsize") = 7, py::arg("walls") = 5,
             py::arg("num_simulations") = 64, py::arg("leaf_batch") = 8,
             py::arg("num_threads") = 8, py::arg("parallel_games") = 128,
             py::arg("c_puct") = 1.0, py::arg("dirichlet_alpha") = 0.3,
             py::arg("dirichlet_epsilon") = 0.25, py::arg("fpu_reduction") = 0.1,
             py::arg("temp_early") = 1.0, py::arg("temp_final") = 0.2,
             py::arg("temp_halflife") = 10.0, py::arg("temp_prune_visits") = 4,
             py::arg("max_moves") = 200,
             py::arg("dist_bonus_max") = 0.0, py::arg("training") = true,
             py::arg("seed") = 0, py::arg("tt_max_depth") = 8,
             py::arg("tt_max_entries") = 2'000'000,
             py::arg("solver_max_total_walls") = 2,
             py::arg("solver_node_limit") = 5'000'000,
             py::arg("solver_time_limit_s") = 4.0,
             py::arg("mcts_solver_max_total_walls") = 0,
             py::arg("mcts_solver_node_limit") = 20'000,
             py::arg("mcts_solver_time_limit_s") = 0.02,
             py::arg("pcr_full_prob") = 0.25,
             py::arg("pcr_cheap_sims") = 160,
             py::arg("pcr_cheap_noise") = false)
        .def("start", &SelfPlayManager::start, py::arg("total_games"),
             "Spawn worker threads and begin playing total_games games.")
        .def("stop", &SelfPlayManager::stop)
        .def("is_done", &SelfPlayManager::is_done)
        .def("games_finished", &SelfPlayManager::games_finished)
        .def("get_batch",
             [](SelfPlayManager& self, int max_batch, int flush_us) {
                 std::vector<float> buf;
                 int count = 0;
                 int64_t id;
                 {
                     py::gil_scoped_release rel;
                     id = self.get_batch(max_batch, flush_us, buf, count);
                 }
                 const int N = self.config().boardsize;
                 py::array_t<float> arr({count, 8, N, N});
                 if (count > 0)
                     std::memcpy(arr.mutable_data(), buf.data(),
                                 sizeof(float) * buf.size());
                 return py::make_tuple(id, arr);
             },
             py::arg("max_batch") = 256, py::arg("flush_us") = 500,
             "Block (GIL released) until leaves are available; returns "
             "(batch_id, states[B,8,N,N]). B == 0 means all games finished.")
        .def("put_results",
             [](SelfPlayManager& self, int64_t batch_id,
                py::array_t<float, py::array::c_style | py::array::forcecast> logits,
                py::array_t<float, py::array::c_style | py::array::forcecast> values) {
                 if (logits.ndim() != 2)
                     throw std::invalid_argument("logits must be (B, A)");
                 const int B = int(logits.shape(0));
                 const int A = int(logits.shape(1));
                 if (values.size() != B)
                     throw std::invalid_argument("values must have B elements");
                 self.put_results(batch_id, logits.data(), values.data(), B, A);
             },
             py::arg("batch_id"), py::arg("logits"), py::arg("values"),
             "Deliver NN outputs for a batch returned by get_batch. "
             "logits: raw policy logits (B, A); values: (B,) in [-1, 1].")
        .def("get_data",
             [](SelfPlayManager& self) {
                 std::vector<float> s, p, v, pte;
                 long long n = 0;
                 self.get_data(s, p, v, pte, n);
                 const int N = self.config().boardsize;
                 const int A = action_space(N);
                 py::array_t<float> states({int(n), 8, N, N});
                 py::array_t<float> policies({int(n), A});
                 py::array_t<float> values(py::array::ShapeContainer{int(n)});
                 py::array_t<float> plies_to_end(py::array::ShapeContainer{int(n)});
                 if (n > 0) {
                     std::memcpy(states.mutable_data(), s.data(),
                                 sizeof(float) * s.size());
                     std::memcpy(policies.mutable_data(), p.data(),
                                 sizeof(float) * p.size());
                     std::memcpy(values.mutable_data(), v.data(),
                                 sizeof(float) * v.size());
                     std::memcpy(plies_to_end.mutable_data(), pte.data(),
                                 sizeof(float) * pte.size());
                 }
                 py::dict d;
                 d["states"] = states;
                 d["policies"] = policies;
                 d["values"] = values;
                 d["plies_to_end"] = plies_to_end;
                 return d;
             },
             "Drain positions from completed games: dict(states, policies, values, "
             "plies_to_end).")
        .def("get_openings",
             [](SelfPlayManager& self) {
                 std::vector<std::array<uint64_t, Game::kOpeningTrack>> openings;
                 std::vector<int> lens;
                 self.get_openings(openings, lens);
                 const int n = int(openings.size());
                 py::array_t<uint64_t> hashes({n, Game::kOpeningTrack});
                 py::array_t<int> out_lens(py::array::ShapeContainer{n});
                 for (int i = 0; i < n; ++i) {
                     std::memcpy(hashes.mutable_data(i, 0), openings[i].data(),
                                 sizeof(uint64_t) * Game::kOpeningTrack);
                 }
                 if (n > 0)
                     std::memcpy(out_lens.mutable_data(), lens.data(),
                                 sizeof(int) * n);
                 py::dict d;
                 d["hashes"] = hashes;   // (n, 4) post-move Zobrist hash per ply
                 d["lens"] = out_lens;   // number of tracked plies actually played
                 return d;
             },
             "Diagnostic: drain per-game post-move Zobrist hashes for the "
             "first few real plies (for measuring transposition overlap).")
        .def("stats",
             [](SelfPlayManager& self) {
                 const auto s = self.stats();
                 py::dict d;
                 d["games"] = s.games;
                 d["p1_wins"] = s.p1_wins;
                 d["p2_wins"] = s.p2_wins;
                 d["draws"] = s.draws;
                 d["mean_plies"] = s.games ? double(s.total_plies) / s.games : 0.0;
                 d["mean_walls"] = s.games ? double(s.total_walls) / s.games : 0.0;
                 d["min_plies"] = s.games ? s.min_plies : 0;
                 d["max_plies"] = s.max_plies;
                 d["solver_calls"] = s.solver_calls;
                 d["solver_timeouts"] = s.solver_timeouts;
                 d["solver_positions"] = s.solver_positions;
                 d["mcts_solver_calls"] = s.mcts_solver_calls;
                 d["mcts_solver_timeouts"] = s.mcts_solver_timeouts;
                 d["mcts_solver_hits"] = s.mcts_solver_hits;
                 d["pcr_full_turns"] = s.pcr_full_turns;
                 d["pcr_cheap_turns"] = s.pcr_cheap_turns;
                 return d;
             });

    py::class_<TournamentManager>(m, "TournamentManager")
        .def(py::init([](int boardsize, int walls, int max_moves, uint64_t seed,
                         int num_threads, int parallel_games,
                         std::vector<int> agent_model_ids,
                         std::vector<int> agent_num_simulations,
                         std::vector<int> agent_leaf_batch,
                         std::vector<double> agent_c_puct,
                         std::vector<double> agent_fpu_reduction,
                         std::vector<double> agent_temperature) {
                 const size_t k = agent_model_ids.size();
                 if (agent_num_simulations.size() != k || agent_leaf_batch.size() != k ||
                     agent_c_puct.size() != k || agent_fpu_reduction.size() != k ||
                     agent_temperature.size() != k)
                     throw std::invalid_argument("agent_* lists must all have the same length");
                 TConfig cfg;
                 cfg.boardsize = boardsize;
                 cfg.walls     = walls;
                 cfg.max_moves = max_moves;
                 cfg.seed      = seed;
                 cfg.agents.resize(k);
                 for (size_t i = 0; i < k; ++i) {
                     AgentSpec& a = cfg.agents[i];
                     a.model_id        = agent_model_ids[i];
                     a.num_simulations = agent_num_simulations[i];
                     a.leaf_batch      = agent_leaf_batch[i];
                     a.c_puct          = agent_c_puct[i];
                     a.fpu_reduction   = agent_fpu_reduction[i];
                     a.temperature     = agent_temperature[i];
                 }
                 return new TournamentManager(cfg, num_threads, parallel_games);
             }),
             py::arg("boardsize") = 7, py::arg("walls") = 5,
             py::arg("max_moves") = 200, py::arg("seed") = 0,
             py::arg("num_threads") = 8, py::arg("parallel_games") = 128,
             py::arg("agent_model_ids"), py::arg("agent_num_simulations"),
             py::arg("agent_leaf_batch"), py::arg("agent_c_puct"),
             py::arg("agent_fpu_reduction"), py::arg("agent_temperature"),
             "agent_* are parallel lists, one entry per registered agent "
             "(indexed by p1_agent/p2_agent in start()).")
        .def("num_models", &TournamentManager::num_models)
        .def("start",
             [](TournamentManager& self, std::vector<int> p1_agents,
                std::vector<int> p2_agents) {
                 if (p1_agents.size() != p2_agents.size())
                     throw std::invalid_argument("p1_agents/p2_agents must have the same length");
                 std::vector<MatchSpec> matches(p1_agents.size());
                 for (size_t i = 0; i < matches.size(); ++i) {
                     matches[i].p1_agent = p1_agents[i];
                     matches[i].p2_agent = p2_agents[i];
                 }
                 self.start(std::move(matches));
             },
             py::arg("p1_agents"), py::arg("p2_agents"),
             "Spawn worker threads and play one game per (p1_agents[i], p2_agents[i]) entry.")
        .def("stop", &TournamentManager::stop)
        .def("is_done", &TournamentManager::is_done)
        .def("games_finished", &TournamentManager::games_finished)
        .def("get_batch",
             [](TournamentManager& self, int model_id, int max_batch, int flush_us) {
                 std::vector<float> buf;
                 int count = 0;
                 int64_t id;
                 {
                     py::gil_scoped_release rel;
                     id = self.get_batch(model_id, max_batch, flush_us, buf, count);
                 }
                 const int N = self.config().boardsize;
                 py::array_t<float> arr({count, 8, N, N});
                 if (count > 0)
                     std::memcpy(arr.mutable_data(), buf.data(),
                                 sizeof(float) * buf.size());
                 return py::make_tuple(id, arr);
             },
             py::arg("model_id"), py::arg("max_batch") = 256, py::arg("flush_us") = 500,
             "Block (GIL released) until leaves for model_id are available; "
             "returns (batch_id, states[B,8,N,N]). B == 0 means no more work "
             "for this model (either the queue is momentarily empty at the "
             "end of the run, or the whole tournament is done).")
        .def("put_results",
             [](TournamentManager& self, int model_id, int64_t batch_id,
                py::array_t<float, py::array::c_style | py::array::forcecast> logits,
                py::array_t<float, py::array::c_style | py::array::forcecast> values) {
                 if (logits.ndim() != 2)
                     throw std::invalid_argument("logits must be (B, A)");
                 const int B = int(logits.shape(0));
                 const int A = int(logits.shape(1));
                 if (values.size() != B)
                     throw std::invalid_argument("values must have B elements");
                 self.put_results(model_id, batch_id, logits.data(), values.data(), B, A);
             },
             py::arg("model_id"), py::arg("batch_id"), py::arg("logits"), py::arg("values"),
             "Deliver NN outputs for a batch returned by get_batch(model_id, ...).")
        .def("get_results",
             [](TournamentManager& self) {
                 std::vector<MatchResult> results = self.get_results();
                 const int n = int(results.size());
                 py::array_t<int> match_index(py::array::ShapeContainer{n});
                 py::array_t<int> winner(py::array::ShapeContainer{n});
                 py::array_t<int> plies(py::array::ShapeContainer{n});
                 for (int i = 0; i < n; ++i) {
                     match_index.mutable_at(i) = results[i].match_index;
                     winner.mutable_at(i)      = results[i].winner;
                     plies.mutable_at(i)       = results[i].plies;
                 }
                 py::dict d;
                 d["match_index"] = match_index;
                 d["winner"] = winner;
                 d["plies"] = plies;
                 return d;
             },
             "Drain results of games finished since the last call: "
             "dict(match_index, winner, plies), aligned with the arrays "
             "passed to start().");
}
