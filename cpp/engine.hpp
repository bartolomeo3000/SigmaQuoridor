// Quoridor game engine — C++ port of game.py with identical semantics.
//
// Parity contract with game.py (verified by test_cpp_parity.py):
//   * action indexing:  0-7 pawn directions (ALL_PAWN_DIRECTIONS order),
//                       8 + y*W + x        horizontal walls (W = N-1),
//                       8 + W^2 + y*W + x  vertical walls
//   * legal move generation order: pawn moves first, then per (y, x) cell
//     H wall then V wall (matters only for tie-breaking)
//   * pawn jumps: straight jump encoded as the orthogonal direction,
//     go-around jumps as diagonals, legal only when the straight
//     continuation is blocked
//   * repetition rule: a pawn move that would create a third occurrence of
//     a position key (p1pos, p2pos, side-to-move, wall config) is illegal
//   * draw: depth >= max_moves (200) or no legal actions
//   * NN input: 8 planes identical to State.to_nn_input(), including the
//     vertical perspective flip when it is P2's turn
//
// Positions are hashed with Zobrist keys (pawn cells per player, wall
// anchors, side to move); the key plays the role of game.py's
// _position_key() tuple.

#pragma once

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace quoridor {

constexpr int MAXN = 9;
constexpr int MAXC = MAXN * MAXN;

// Order must match game.py ALL_PAWN_DIRECTIONS exactly.
constexpr int8_t DIRS[8][2] = {
    {0, 1},  {0, -1}, {-1, 0}, {1, 0},     // up, down, left, right
    {-1, 1}, {1, 1},  {-1, -1}, {1, -1},   // diagonals
};

inline int action_space(int N) { return 8 + 2 * (N - 1) * (N - 1); }

// ---------------------------------------------------------------------------
// Zobrist hashing
// ---------------------------------------------------------------------------

inline const uint64_t* zobrist_tables() {
    // Layout: [p1 MAXC][p2 MAXC][hw MAXC][vw MAXC][side]
    static uint64_t tbl[4 * MAXC + 1];
    static bool init = false;
    if (!init) {
        std::mt19937_64 r(0x51600D0BULL);
        for (int i = 0; i < 4 * MAXC + 1; ++i) tbl[i] = r();
        init = true;
    }
    return tbl;
}
inline const uint64_t* Z_P1()  { return zobrist_tables(); }
inline const uint64_t* Z_P2()  { return zobrist_tables() + MAXC; }
inline const uint64_t* Z_HW()  { return zobrist_tables() + 2 * MAXC; }
inline const uint64_t* Z_VW()  { return zobrist_tables() + 3 * MAXC; }
inline uint64_t        Z_SIDE(){ return zobrist_tables()[4 * MAXC]; }

// ---------------------------------------------------------------------------
// Repetition counter: base history (real game line) + path overlay (tree line)
// ---------------------------------------------------------------------------

struct RepCounter {
    const std::unordered_map<uint64_t, int>* base = nullptr;
    const std::vector<uint64_t>* overlay = nullptr;

    int count(uint64_t key) const {
        int c = 0;
        if (base) {
            auto it = base->find(key);
            if (it != base->end()) c = it->second;
        }
        if (overlay)
            for (uint64_t h : *overlay) c += (h == key);
        return c;
    }
};

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------

struct GameState {
    int N = 7;
    int depth = 0;
    int p1x = 0, p1y = 0, p2x = 0, p2y = 0;
    int walls_p1 = 0, walls_p2 = 0, walls_initial = 0;
    std::bitset<MAXC> hwalls, vwalls;    // wall segment bits, index y*N+x
    std::bitset<MAXC> hanchor, vanchor;  // wall anchor bits
    int16_t p1_dist[MAXC];               // BFS dist to row N-1
    int16_t p2_dist[MAXC];               // BFS dist to row 0
    std::bitset<MAXC> hpath[2], vpath[2];  // greedy shortest-path edges, [0]=P1
    bool path_valid[2] = {false, false};
    uint64_t zhash = 0;

    void init(int boardsize, int walls) {
        if (boardsize % 2 == 0 || boardsize > MAXN)
            throw std::invalid_argument("boardsize must be odd and <= 9");
        N = boardsize;
        depth = 0;
        p1x = N / 2; p1y = 0;
        p2x = N / 2; p2y = N - 1;
        walls_p1 = walls_p2 = walls_initial = walls;
        hwalls.reset(); vwalls.reset();
        hanchor.reset(); vanchor.reset();
        zhash = Z_P1()[p1y * N + p1x] ^ Z_P2()[p2y * N + p2x];
        recompute_dists();
    }

    bool is_p1_turn() const { return depth % 2 == 0; }
    int  current_player() const { return is_p1_turn() ? 1 : 2; }
    bool in_bounds(int x, int y) const { return 0 <= x && x < N && 0 <= y && y < N; }

    int winner() const {
        if (p1y == N - 1) return 1;
        if (p2y == 0)     return 2;
        return 0;
    }

    // O(1): is the boundary from (x,y) toward (dx,dy) blocked by a wall?
    bool edge_blocked(int x, int y, int dx, int dy) const {
        if (dy ==  1) return hwalls[y * N + x];
        if (dy == -1) return hwalls[(y - 1) * N + x];
        if (dx ==  1) return vwalls[y * N + x];
        if (dx == -1) return vwalls[y * N + x - 1];
        return false;
    }

    // ------------------------------------------------------------------
    // BFS helpers
    // ------------------------------------------------------------------

    // Multi-source BFS from every cell of goal_row; INF = N*N (matches game.py).
    void bfs_dist(int goal_row, int16_t* dist) const {
        const int16_t INF = int16_t(N * N);
        const int NN = N * N;
        for (int i = 0; i < NN; ++i) dist[i] = INF;
        int q[MAXC]; int qh = 0, qt = 0;
        for (int x = 0; x < N; ++x) {
            dist[goal_row * N + x] = 0;
            q[qt++] = goal_row * N + x;
        }
        while (qh < qt) {
            const int c = q[qh++];
            const int x = c % N, y = c / N;
            const int16_t d = int16_t(dist[c] + 1);
            if (y + 1 < N && dist[c + N] == INF && !hwalls[c])        { dist[c + N] = d; q[qt++] = c + N; }
            if (y > 0     && dist[c - N] == INF && !hwalls[c - N])    { dist[c - N] = d; q[qt++] = c - N; }
            if (x + 1 < N && dist[c + 1] == INF && !vwalls[c])        { dist[c + 1] = d; q[qt++] = c + 1; }
            if (x > 0     && dist[c - 1] == INF && !vwalls[c - 1])    { dist[c - 1] = d; q[qt++] = c - 1; }
        }
    }

    // Plain reachability BFS (used while probing candidate walls).
    bool reach(int sx, int sy, int goal_row) const {
        if (sy == goal_row) return true;
        std::bitset<MAXC> vis;
        int q[MAXC]; int qh = 0, qt = 0;
        vis.set(sy * N + sx);
        q[qt++] = sy * N + sx;
        while (qh < qt) {
            const int c = q[qh++];
            const int x = c % N, y = c / N;
            if (y + 1 < N && !vis[c + N] && !hwalls[c]) {
                if (y + 1 == goal_row) return true;
                vis.set(c + N); q[qt++] = c + N;
            }
            if (y > 0 && !vis[c - N] && !hwalls[c - N]) {
                if (y - 1 == goal_row) return true;
                vis.set(c - N); q[qt++] = c - N;
            }
            if (x + 1 < N && !vis[c + 1] && !vwalls[c]) {
                if (y == goal_row) return true;
                vis.set(c + 1); q[qt++] = c + 1;
            }
            if (x > 0 && !vis[c - 1] && !vwalls[c - 1]) {
                if (y == goal_row) return true;
                vis.set(c - 1); q[qt++] = c - 1;
            }
        }
        return false;
    }

    // Greedy shortest-path edge set (mirror of _shortest_path_edge_set).
    void compute_path_edges(int px, int py, const int16_t* dist,
                            std::bitset<MAXC>& hset, std::bitset<MAXC>& vset,
                            bool& valid) const {
        hset.reset(); vset.reset();
        int d = dist[py * N + px];
        if (d >= N * N) { valid = false; return; }
        valid = true;
        static constexpr int8_t D4[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int x = px, y = py;
        int guard = 4 * N * N;
        while (d > 0 && guard-- > 0) {
            for (const auto& dd : D4) {
                const int nx = x + dd[0], ny = y + dd[1];
                if (!in_bounds(nx, ny)) continue;
                if (dist[ny * N + nx] != d - 1) continue;
                if (edge_blocked(x, y, dd[0], dd[1])) continue;
                if      (dd[1] ==  1) hset.set(y * N + x);
                else if (dd[1] == -1) hset.set((y - 1) * N + x);
                else if (dd[0] ==  1) vset.set(y * N + x);
                else                  vset.set(y * N + x - 1);
                x = nx; y = ny; --d;
                break;
            }
        }
    }

    void recompute_path_edges() {
        compute_path_edges(p1x, p1y, p1_dist, hpath[0], vpath[0], path_valid[0]);
        compute_path_edges(p2x, p2y, p2_dist, hpath[1], vpath[1], path_valid[1]);
    }

    void recompute_dists() {
        bfs_dist(N - 1, p1_dist);
        bfs_dist(0,     p2_dist);
        recompute_path_edges();
    }

    // ------------------------------------------------------------------
    // Pawn moves
    // ------------------------------------------------------------------

    bool pawn_legal(int di) const {
        const int dx = DIRS[di][0], dy = DIRS[di][1];
        const bool p1t = is_p1_turn();
        const int cx = p1t ? p1x : p2x, cy = p1t ? p1y : p2y;
        const int ox = p1t ? p2x : p1x, oy = p1t ? p2y : p1y;

        if (dx == 0 || dy == 0) {
            const int tx = cx + dx, ty = cy + dy;
            if (!in_bounds(tx, ty)) return false;
            if (edge_blocked(cx, cy, dx, dy)) return false;
            if (tx != ox || ty != oy) return true;
            const int jx = tx + dx, jy = ty + dy;
            return in_bounds(jx, jy) && !edge_blocked(tx, ty, dx, dy);
        }
        // Diagonal go-around jump: try both intermediate axes.
        const int steps[2][2][2] = {{{dx, 0}, {0, dy}}, {{0, dy}, {dx, 0}}};
        for (int k = 0; k < 2; ++k) {
            const int sdx = steps[k][0][0], sdy = steps[k][0][1];
            const int ldx = steps[k][1][0], ldy = steps[k][1][1];
            const int mx = cx + sdx, my = cy + sdy;
            if (mx != ox || my != oy) continue;
            if (edge_blocked(cx, cy, sdx, sdy)) continue;
            const int px_ = mx + sdx, py_ = my + sdy;
            const bool straight_blocked =
                !in_bounds(px_, py_) || edge_blocked(mx, my, sdx, sdy);
            if (!straight_blocked) continue;
            const int lx = mx + ldx, ly = my + ldy;
            if (!in_bounds(lx, ly)) continue;
            if (edge_blocked(mx, my, ldx, ldy)) continue;
            return true;
        }
        return false;
    }

    void pawn_dest(int di, int& outx, int& outy) const {
        const int dx = DIRS[di][0], dy = DIRS[di][1];
        const bool p1t = is_p1_turn();
        const int cx = p1t ? p1x : p2x, cy = p1t ? p1y : p2y;
        const int ox = p1t ? p2x : p1x, oy = p1t ? p2y : p1y;
        if (dx == 0 || dy == 0) {
            const int tx = cx + dx, ty = cy + dy;
            if (tx == ox && ty == oy) { outx = cx + 2 * dx; outy = cy + 2 * dy; }
            else                       { outx = tx;          outy = ty; }
        } else {
            outx = cx + dx; outy = cy + dy;
        }
    }

    // Hash of the position that would result from pawn action di
    // (matches _prospective_pawn_key: same walls, moved pawn, flipped side).
    uint64_t prospective_pawn_hash(int di) const {
        int nx, ny;
        pawn_dest(di, nx, ny);
        const bool p1t = is_p1_turn();
        const int cx = p1t ? p1x : p2x, cy = p1t ? p1y : p2y;
        const uint64_t* zp = p1t ? Z_P1() : Z_P2();
        return zhash ^ zp[cy * N + cx] ^ zp[ny * N + nx] ^ Z_SIDE();
    }

    // ------------------------------------------------------------------
    // Legal action generation
    // ------------------------------------------------------------------

    // Number of existing-wall/board-edge contact points (0..3) touched by a
    // candidate horizontal wall anchored at (cx, cy) [W-space]. A wall
    // touching <=1 point can never complete a seal around either player:
    // sealing requires a barrier bridging two existing features (or itself),
    // and a single 2-unit segment with <=1 anchor point cannot bridge that.
    // This lets gen_legal() skip the reach() BFS even when the candidate
    // crosses the current shortest-path edge set.
    int h_wall_contacts(int cx, int cy) const {
        const int W = N - 1;
        auto vtouch = [&](int vx, int vy) {
            return vx >= 0 && vx < W && vy >= 0 && vy < W && vanchor[vy * N + vx];
        };
        auto htouch = [&](int hx, int hy) {
            return hx >= 0 && hx < W && hy >= 0 && hy < W && hanchor[hy * N + hx];
        };
        const bool left = (cx == 0)
                        || htouch(cx - 2, cy)
                        || vtouch(cx - 1, cy - 1) || vtouch(cx - 1, cy) || vtouch(cx - 1, cy + 1);
        const bool mid  = vtouch(cx, cy - 1) || vtouch(cx, cy + 1);
        const bool right = (cx + 2 == N)
                        || htouch(cx + 2, cy)
                        || vtouch(cx + 1, cy - 1) || vtouch(cx + 1, cy) || vtouch(cx + 1, cy + 1);
        return int(left) + int(mid) + int(right);
    }

    // Mirror of h_wall_contacts() for a candidate vertical wall.
    int v_wall_contacts(int cx, int cy) const {
        const int W = N - 1;
        auto vtouch = [&](int vx, int vy) {
            return vx >= 0 && vx < W && vy >= 0 && vy < W && vanchor[vy * N + vx];
        };
        auto htouch = [&](int hx, int hy) {
            return hx >= 0 && hx < W && hy >= 0 && hy < W && hanchor[hy * N + hx];
        };
        const bool top = (cy == 0)
                       || vtouch(cx, cy - 2)
                       || htouch(cx - 1, cy - 1) || htouch(cx, cy - 1) || htouch(cx + 1, cy - 1);
        const bool mid = htouch(cx - 1, cy) || htouch(cx + 1, cy);
        const bool bot = (cy + 2 == N)
                       || vtouch(cx, cy + 2)
                       || htouch(cx - 1, cy + 1) || htouch(cx, cy + 1) || htouch(cx + 1, cy + 1);
        return int(top) + int(mid) + int(bot);
    }

    // Non-const: temporarily toggles wall bits while probing candidates.
    void gen_legal(const RepCounter& rc, std::vector<uint16_t>& out) {
        out.clear();

        // Pawn actions, repetition-filtered (mirror of _get_legal_pawn_actions).
        for (int di = 0; di < 8; ++di) {
            if (!pawn_legal(di)) continue;
            if (rc.count(prospective_pawn_hash(di)) >= 2) continue;
            out.push_back(uint16_t(di));
        }

        const int mywalls = is_p1_turn() ? walls_p1 : walls_p2;
        if (mywalls <= 0) return;

        const int W = N - 1;
        for (int y = 0; y < W; ++y) {
            for (int x = 0; x < W; ++x) {
                const int a = y * N + x;   // anchor / segment base index

                // --- H wall at (x, y) ---
                bool h_over = hanchor[a] || vanchor[a]
                           || (x > 0 && hanchor[a - 1])
                           || (x + 1 < W && hanchor[a + 1]);
                if (!h_over) {
                    const bool p1_needs = !path_valid[0] || hpath[0][a] || hpath[0][a + 1];
                    const bool p2_needs = !path_valid[1] || hpath[1][a] || hpath[1][a + 1];
                    bool ok = true;
                    if ((p1_needs || p2_needs) && h_wall_contacts(x, y) >= 2) {
                        hwalls.set(a); hwalls.set(a + 1);
                        ok = (!p1_needs || reach(p1x, p1y, N - 1))
                          && (!p2_needs || reach(p2x, p2y, 0));
                        hwalls.reset(a); hwalls.reset(a + 1);
                    }
                    if (ok) out.push_back(uint16_t(8 + y * W + x));
                }

                // --- V wall at (x, y) ---
                bool v_over = vanchor[a] || hanchor[a]
                           || (y > 0 && vanchor[a - N])
                           || (y + 1 < W && vanchor[a + N]);
                if (!v_over) {
                    const bool p1_needs = !path_valid[0] || vpath[0][a] || vpath[0][a + N];
                    const bool p2_needs = !path_valid[1] || vpath[1][a] || vpath[1][a + N];
                    bool ok = true;
                    if ((p1_needs || p2_needs) && v_wall_contacts(x, y) >= 2) {
                        vwalls.set(a); vwalls.set(a + N);
                        ok = (!p1_needs || reach(p1x, p1y, N - 1))
                          && (!p2_needs || reach(p2x, p2y, 0));
                        vwalls.reset(a); vwalls.reset(a + N);
                    }
                    if (ok) out.push_back(uint16_t(8 + W * W + y * W + x));
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Apply action (no legality check — mirror of State.next fast path)
    // ------------------------------------------------------------------

    void apply(uint16_t a) {
        const bool p1t = is_p1_turn();
        if (a < 8) {
            int nx, ny;
            pawn_dest(int(a), nx, ny);
            const int cx = p1t ? p1x : p2x, cy = p1t ? p1y : p2y;
            const uint64_t* zp = p1t ? Z_P1() : Z_P2();
            zhash ^= zp[cy * N + cx] ^ zp[ny * N + nx];
            if (p1t) { p1x = nx; p1y = ny; } else { p2x = nx; p2y = ny; }
            recompute_path_edges();          // dist grids unchanged by pawn moves
        } else {
            const int W = N - 1;
            int wi = a - 8;
            const bool horiz = wi < W * W;
            if (!horiz) wi -= W * W;
            const int x = wi % W, y = wi / W;
            const int base = y * N + x;
            if (horiz) {
                hwalls.set(base); hwalls.set(base + 1);
                hanchor.set(base);
                zhash ^= Z_HW()[base];
            } else {
                vwalls.set(base); vwalls.set(base + N);
                vanchor.set(base);
                zhash ^= Z_VW()[base];
            }
            if (p1t) --walls_p1; else --walls_p2;
            recompute_dists();               // walls changed
        }
        zhash ^= Z_SIDE();
        ++depth;
    }

    // ------------------------------------------------------------------
    // NN input planes — exact mirror of State.to_nn_input()
    // ------------------------------------------------------------------

    void nn_input(float* out) const {
        const int NN = N * N;
        std::memset(out, 0, sizeof(float) * 8 * NN);
        const float norm = walls_initial > 0 ? float(walls_initial) : 1.0f;
        const float maxd = float(NN - 1);
        const int16_t INF = int16_t(NN);

        if (is_p1_turn()) {
            out[0 * NN + p1y * N + p1x] = 1.0f;
            out[1 * NN + p2y * N + p2x] = 1.0f;
            for (int i = 0; i < NN; ++i) {
                out[2 * NN + i] = hwalls[i] ? 1.0f : 0.0f;
                out[3 * NN + i] = vwalls[i] ? 1.0f : 0.0f;
                out[4 * NN + i] = walls_p1 / norm;
                out[5 * NN + i] = walls_p2 / norm;
                out[6 * NN + i] = p1_dist[i] < INF ? p1_dist[i] / maxd : 1.0f;
                out[7 * NN + i] = p2_dist[i] < INF ? p2_dist[i] / maxd : 1.0f;
            }
        } else {
            // P2 perspective: vertical flip (new_y = N-1-old_y).
            out[0 * NN + (N - 1 - p2y) * N + p2x] = 1.0f;
            out[1 * NN + (N - 1 - p1y) * N + p1x] = 1.0f;
            for (int r = 0; r < N; ++r) {
                for (int x = 0; x < N; ++x) {
                    const int o = r * N + x;
                    // flipped_hw[:N-1] = hw[:N-1][::-1]; row N-1 stays zero
                    out[2 * NN + o] = (r < N - 1 && hwalls[(N - 2 - r) * N + x]) ? 1.0f : 0.0f;
                    out[3 * NN + o] = vwalls[(N - 1 - r) * N + x] ? 1.0f : 0.0f;
                    out[4 * NN + o] = walls_p2 / norm;
                    out[5 * NN + o] = walls_p1 / norm;
                    const int fi = (N - 1 - r) * N + x;
                    out[6 * NN + o] = p2_dist[fi] < INF ? p2_dist[fi] / maxd : 1.0f;
                    out[7 * NN + o] = p1_dist[fi] < INF ? p1_dist[fi] / maxd : 1.0f;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Distance heuristic (mirror of compute_move_advantages)
    // ------------------------------------------------------------------

    void move_advantages(const std::vector<uint16_t>& legal, float* out) const {
        const bool p1t = is_p1_turn();
        const int mex = p1t ? p1x : p2x, mey = p1t ? p1y : p2y;
        const int opx = p1t ? p2x : p1x, opy = p1t ? p2y : p1y;
        const int16_t* myd = p1t ? p1_dist : p2_dist;
        const int16_t* opd = p1t ? p2_dist : p1_dist;
        const int me = p1t ? 0 : 1, op = 1 - me;
        const int my_now = myd[mey * N + mex];
        const int opp_now = opd[opy * N + opx];
        const int W = N - 1;

        for (size_t i = 0; i < legal.size(); ++i) {
            const uint16_t a = legal[i];
            int my_after, opp_after;
            if (a < 8) {
                int dx_, dy_;
                pawn_dest(int(a), dx_, dy_);
                my_after  = myd[dy_ * N + dx_];
                opp_after = opp_now;
            } else {
                int wi = a - 8;
                const bool horiz = wi < W * W;
                if (!horiz) wi -= W * W;
                const int x = wi % W, y = wi / W;
                const int s1 = y * N + x;
                const int s2 = horiz ? s1 + 1 : s1 + N;
                int my_delta = 0, opp_delta = 0;
                if (horiz) {
                    if (path_valid[op] && (hpath[op][s1] || hpath[op][s2])) opp_delta = 1;
                    if (path_valid[me] && (hpath[me][s1] || hpath[me][s2])) my_delta  = 1;
                } else {
                    if (path_valid[op] && (vpath[op][s1] || vpath[op][s2])) opp_delta = 1;
                    if (path_valid[me] && (vpath[me][s1] || vpath[me][s2])) my_delta  = 1;
                }
                my_after  = my_now + my_delta;
                opp_after = opp_now + opp_delta;
            }
            out[i] = float((opp_after - my_after) - (opp_now - my_now));
        }
    }
};

}  // namespace quoridor
