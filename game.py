""" Quoridor game logic implementation. """
from collections import deque
from dataclasses import dataclass, field
from typing import Union, List
import numpy as np


@dataclass
class PawnDirections:
    """
    Class to define all possible pawn movement directions in Quoridor.
    Orthogonal moves (up, down, left, right) are defined explicitly.
    Diagonal moves are calculated dynamically from orthogonal moves.

    IMPORTANT -- ORIENTATION:
    Coordinates are (x, y), where x is horizontal and y is vertical.
    The board origin (0, 0) is bottom-left.
    - Moving up increases y.
    - Moving down decreases y.
    - Moving left decreases x.
    - Moving right increases x.
    """
    up: tuple = (0, 1)
    down: tuple = (0, -1)
    left: tuple = (-1, 0)
    right: tuple = (1, 0)
    up_and_left: tuple = field(init=False)
    up_and_right: tuple = field(init=False)
    down_and_left: tuple = field(init=False)
    down_and_right: tuple = field(init=False)
    
    def __post_init__(self):
        self.up_and_left = (self.up[0] + self.left[0], self.up[1] + self.left[1])
        self.up_and_right = (self.up[0] + self.right[0], self.up[1] + self.right[1])
        self.down_and_left = (self.down[0] + self.left[0], self.down[1] + self.left[1])
        self.down_and_right = (self.down[0] + self.right[0], self.down[1] + self.right[1])


@dataclass(slots=True)
class PawnAction:
    """Action to move the pawn in a given direction."""
    direction: tuple  # e.g., (0, 1) for up


@dataclass(slots=True)
class WallAction:
    """
    Action to place a wall of length 2 on the board.

    Horizontal wall is placed on (x, y) and (x+1, y). So on (x,y) and one cell to the right.
    It means you can't move up from these two cells, e.g. from (x, y) to (x, y+1). Vice versa, you can't move down from (x, y+1) to (x, y).
    Same goes for moving between (x+1, y) and (x+1, y+1).

    Vertical wall is placed on (x, y) and (x, y+1). So on (x,y) and one cell up.
    It means you can't move right from these two cells, e.g. from (x, y) to (x+1, y). Vice versa, you can't move left from (x+1, y) to (x, y).
    Same goes for moving between (x, y+1) and (x+1, y+1).
    """
    x: int
    y: int
    orientation: str  # 'h' for horizontal, 'v' for vertical

    def wall_cells(self):
        """Return the list of cells occupied by this wall."""
        if self.orientation == 'h':
            return [(self.x, self.y), (self.x + 1, self.y)]
        elif self.orientation == 'v':
            return [(self.x, self.y), (self.x, self.y + 1)]
        else:
            raise ValueError("Invalid wall orientation: {}".format(self.orientation))

# Unified action type
Action = Union[PawnAction, WallAction]

# All possible pawn movement directions (orthogonal then diagonal).
ALL_PAWN_DIRECTIONS = (
    (0, 1), (0, -1), (-1, 0), (1, 0),    # up, down, left, right
    (-1, 1), (1, 1), (-1, -1), (1, -1),  # diagonals
)

# Global cache: (frozenset hwall_anchors, frozenset vwall_anchors, p1pos, p2pos)
# -> list[WallAction].  Legal wall actions are fully determined by these four
# values, so this cache is valid across all states with the same configuration
# and survives copy()/next(), making it useful for MCTS transpositions.
_WALL_ACTIONS_CACHE: dict = {}


class State:
    """
    Class to represent the state of the game, including the board size, player positions, wall placements, and the current player's turn.
    Implements the game logic and move legality checks for both pawn moves and wall placements.
    The main functionality is the next() method, which takes an action and returns the resulting state after applying that action.
    Other important method is get_legal_actions(), which generates all legal actions from the current state, essential for MCTS and optimized for speed.
    Also includes is_finished() method to check if the game has reached a terminal state.
    """
    def __init__(self, boardsize=7, depth=0, player1pos=None, player2pos=None, hwalls=None, vwalls=None, walls_p1=5, walls_p2=5, hwall_anchors=None, vwall_anchors=None, walls_initial=None, position_history=None, p1_dist=None, p2_dist=None, p1_path_edges=None, p2_path_edges=None):
        if boardsize % 2 == 0:
            raise ValueError("Board size must be odd")
        self.boardsize = boardsize

        # depth is used to determine how long the game has progressed and which player's turn it is
        self.depth = depth

        # Initialize player positions and walls if not provided
        self.player1pos = player1pos if player1pos is not None else (boardsize // 2, 0)
        self.player2pos = player2pos if player2pos is not None else (boardsize // 2, boardsize - 1)

        # Initialize walls as flat bytearrays (N*N bytes, indexed as y*N+x).
        # bytearray uses 1 byte per cell with no Python object overhead per element,
        # and copies in a single C memcpy via bytearray(other).
        self.hwalls: bytearray = hwalls if hwalls is not None else bytearray(boardsize * boardsize)
        self.vwalls: bytearray = vwalls if vwalls is not None else bytearray(boardsize * boardsize)

        # Track remaining walls for each player
        self.walls_p1 = walls_p1
        self.walls_p2 = walls_p2

        # Sets of (x, y) anchor positions for placed walls.
        # Needed for O(1) overlap and crossing checks in is_wall_legal().
        # (Using only the matrices is ambiguous: e.g. vwalls[y][x]==1 could be
        # the top segment of a wall anchored at (x, y-1) OR the bottom of (x, y).)
        self.hwall_anchors: set = hwall_anchors if hwall_anchors is not None else set()
        self.vwall_anchors: set = vwall_anchors if vwall_anchors is not None else set()

        # Initial wall count per player — stored for normalisation in to_nn_input().
        # Defaults to walls_p1 at construction time (correct when building a fresh game).
        self.walls_initial: int = walls_initial if walls_initial is not None else walls_p1

        # Cached BFS distance grids and shortest-path edge sets (see is_wall_legal).
        # p1_dist[y][x] = moves from (x,y) to P1's goal row (N-1); p2_dist toward row 0.
        # Distance grids are recomputed after every WallAction; edge sets after any action.
        # When provided (from copy()), skip recomputation entirely — the parent's grids
        # are still valid and the objects are safe to share (they're only replaced wholesale).
        if p1_dist is not None:
            self.p1_dist = p1_dist
            self.p2_dist = p2_dist
            self.p1_path_edges = p1_path_edges
            self.p2_path_edges = p2_path_edges
        else:
            self._recompute_dists()

        # Position history for threefold repetition detection.
        # Keys are position tuples (see _position_key); values are occurrence counts.
        if position_history is not None:
            self.position_history = position_history
        else:
            self.position_history = {}
            self._record_position()

        # Per-state cache for get_legal_actions().  Set to None at construction and
        # populated on the first call.  Safe because State objects are never mutated
        # after next() produces them — only _apply_wall_bits/_remove_wall_bits touch
        # the bytearrays transiently during legality probing, not between calls.
        self._legal_actions_cache: list | None = None

    def _position_key(self) -> tuple:
        """Return a hashable key uniquely identifying the current board position."""
        return (
            self.player1pos,
            self.player2pos,
            self.depth % 2,
            frozenset(self.hwall_anchors),
            frozenset(self.vwall_anchors),
        )

    def _record_position(self) -> None:
        """Increment the occurrence count of the current position in history."""
        key = self._position_key()
        self.position_history[key] = self.position_history.get(key, 0) + 1

    def _recompute_dists(self):
        """Recompute dist grids and path-edge sets from current walls and positions."""
        self.p1_dist = self._bfs_dist_grid(self.hwalls, self.vwalls, self.boardsize - 1, self.boardsize)
        self.p2_dist = self._bfs_dist_grid(self.hwalls, self.vwalls, 0, self.boardsize)
        self._recompute_path_edges()

    def _recompute_path_edges(self):
        """Recompute path-edge sets from current player positions and cached dist grids."""
        self.p1_path_edges = self._shortest_path_edge_set(self.player1pos, self.p1_dist)
        self.p2_path_edges = self._shortest_path_edge_set(self.player2pos, self.p2_dist)

    def _shortest_path_edge_set(self, start, dist_grid):
        """
        Return the set of wall-segment coordinates that lie on ONE shortest path
        from `start` to the goal represented by dist_grid (goal cells have dist 0).

        Greedily descends from start, at each cell picking the first neighbour
        with dist == d - 1.  If a wall disconnects the player it must cut ALL
        shortest paths, so it must cut this one too — making this set a valid
        filter for skipping BFS.

        Each element is a tuple ('h', row, col) or ('v', row, col) identifying
        the wall-matrix cell that would block that traversal step.

        Returns None if start is unreachable (dist == N*N).
        """
        N = self.boardsize
        sx, sy = start
        d = dist_grid[sy][sx]
        if d == N * N:  # player already disconnected
            return None
        edge_set = set()
        x, y = sx, sy
        while d > 0:
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < N and 0 <= ny < N):
                    continue
                if dist_grid[ny][nx] != d - 1:
                    continue
                if self._is_edge_blocked(x, y, dx, dy):
                    continue  # edge is wall-blocked; dist[ny][nx] was reached via another route
                if   dy ==  1: edge_set.add(('h', y,     x    ))
                elif dy == -1: edge_set.add(('h', y - 1, x    ))
                elif dx ==  1: edge_set.add(('v', y,     x    ))
                else:          edge_set.add(('v', y,     x - 1))
                x, y, d = nx, ny, d - 1
                break
        return edge_set

    def playerspos_matrix(self):
        matrix = [[0 for _ in range(self.boardsize)] for _ in range(self.boardsize)]
        matrix[self.player1pos[1]][self.player1pos[0]] = 1
        matrix[self.player2pos[1]][self.player2pos[0]] = 2
        return matrix

    def print_playerspos(self):
        """Print the player positions on the board, with 0 for empty cells, 1 for player 1, and 2 for player 2."""
        for row in reversed(self.playerspos_matrix()):
            print(' '.join(str(cell) for cell in row))

    def print_hwalls(self):
        N = self.boardsize
        for y in reversed(range(N)):
            print(' '.join(str(self.hwalls[y * N + x]) for x in range(N)))

    def print_vwalls(self):
        N = self.boardsize
        for y in reversed(range(N)):
            print(' '.join(str(self.vwalls[y * N + x]) for x in range(N)))

    def _is_edge_blocked(self, x: int, y: int, dx: int, dy: int) -> bool:
        """
        O(1) check: is the cell-boundary from (x, y) in direction (dx, dy) blocked by a wall?

        Horizontal walls (hwalls[r][c] == 1) block vertical movement across row boundary r↔r+1 at column c.
        Vertical walls (vwalls[r][c] == 1) block horizontal movement across column boundary c↔c+1 at row r.

        Only pure orthogonal directions (one of dx, dy is zero) are valid inputs.
        """
        N = self.boardsize
        if dy == 1:    # moving up: blocked if hwalls[y][x] is set
            return bool(self.hwalls[y * N + x])
        if dy == -1:   # moving down: blocked if hwalls[y-1][x] is set
            return bool(self.hwalls[(y - 1) * N + x])
        if dx == 1:    # moving right: blocked if vwalls[y][x] is set
            return bool(self.vwalls[y * N + x])
        if dx == -1:   # moving left: blocked if vwalls[y][x-1] is set
            return bool(self.vwalls[y * N + x - 1])
        return False

    def get_current_player(self):
        """Returns 1 if it's player 1's turn, 2 if it's player 2's turn."""
        return 1 if self.depth % 2 == 0 else 2

    def is_player1_turn(self):
        return self.depth % 2 == 0

    def is_drawn(self):
        """Game is drawn at depth 100 or on threefold repetition of any position."""
        if self.depth >= 100:
            return True
        if max(self.position_history.values(), default=0) >= 3:
            return True
        return False
    
    def winner(self):
        """Check if either player has won by reaching the opposite side of the board."""
        if self.player1pos[1] == self.boardsize - 1:
            return 1  # Player 1 wins
        elif self.player2pos[1] == 0:
            return 2  # Player 2 wins
        else:
            return 0  # No winner yet

    def is_finished(self):
        if self.is_drawn():
            return True
        if self.winner() != 0:
            return True
        return False

    def is_action_legal(self, action: Action) -> bool:
        """Check if the given action is legal based on the current state of the game."""
        if isinstance(action, PawnAction):
            return self.is_pawn_move_legal(action)
        elif isinstance(action, WallAction):
            return self.is_wall_legal(action)
        return False

    def _get_legal_pawn_actions(self) -> List[PawnAction]:
        """Return all legal pawn moves from the current position."""
        actions = []
        for d in ALL_PAWN_DIRECTIONS:
            action = PawnAction(direction=d)
            if self.is_pawn_move_legal(action):
                actions.append(action)
        return actions

    def _build_overlap_sets(self) -> tuple[set, set]:
        """
        Return (h_illegal, v_illegal): sets of anchor positions that are
        forbidden due to overlap or crossing with already-placed walls.

        Built in O(k) from the anchor sets (k = walls placed) rather than
        checked per-position, replacing the per-call _wall_overlaps() pass.
        """
        h_illegal: set = set()
        v_illegal: set = set()
        for ax, ay in self.hwall_anchors:
            h_illegal.update(((ax - 1, ay), (ax, ay), (ax + 1, ay)))
            v_illegal.add((ax, ay))
        for ax, ay in self.vwall_anchors:
            v_illegal.update(((ax, ay - 1), (ax, ay), (ax, ay + 1)))
            h_illegal.add((ax, ay))
        return h_illegal, v_illegal

    def _get_legal_wall_actions(self) -> List[WallAction]:
        """Return all legal wall placements for the current player."""
        current_player = self.get_current_player()
        if (self.walls_p1 if current_player == 1 else self.walls_p2) == 0:
            return []

        # Legal wall actions depend only on wall config and player positions,
        # not on depth or whose turn it is — safe to cache globally.
        cache_key = (
            frozenset(self.hwall_anchors), frozenset(self.vwall_anchors),
            self.player1pos, self.player2pos,
        )
        cached = _WALL_ACTIONS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        N = self.boardsize
        h_illegal, v_illegal = self._build_overlap_sets()

        # Cache hot attributes as locals — avoids repeated self-attribute lookups
        # inside the (N-1)^2 * 2 inner iterations.
        pe1    = self.p1_path_edges
        pe2    = self.p2_path_edges
        hwalls = self.hwalls
        vwalls = self.vwalls
        p1pos  = self.player1pos
        p2pos  = self.player2pos
        goal1  = N - 1

        actions = []
        for y in range(N - 1):
            for x in range(N - 1):

                # --- H-wall at (x, y) ---
                if (x, y) not in h_illegal:
                    seg_a = ('h', y, x);  seg_b = ('h', y, x + 1)
                    p1_needs = pe1 is None or seg_a in pe1 or seg_b in pe1
                    p2_needs = pe2 is None or seg_a in pe2 or seg_b in pe2
                    if not p1_needs and not p2_needs:
                        actions.append(WallAction(x, y, 'h'))
                    else:
                        idx = y * N + x
                        hwalls[idx] = 1;  hwalls[idx + 1] = 1
                        p1_ok = (not p1_needs) or self.BFS(p1pos, goal1)
                        p2_ok = (not p2_needs) or self.BFS(p2pos, 0)
                        hwalls[idx] = 0;  hwalls[idx + 1] = 0
                        if p1_ok and p2_ok:
                            actions.append(WallAction(x, y, 'h'))

                # --- V-wall at (x, y) ---
                if (x, y) not in v_illegal:
                    seg_a = ('v', y, x);  seg_b = ('v', y + 1, x)
                    p1_needs = pe1 is None or seg_a in pe1 or seg_b in pe1
                    p2_needs = pe2 is None or seg_a in pe2 or seg_b in pe2
                    if not p1_needs and not p2_needs:
                        actions.append(WallAction(x, y, 'v'))
                    else:
                        idx = y * N + x
                        vwalls[idx] = 1;  vwalls[idx + N] = 1
                        p1_ok = (not p1_needs) or self.BFS(p1pos, goal1)
                        p2_ok = (not p2_needs) or self.BFS(p2pos, 0)
                        vwalls[idx] = 0;  vwalls[idx + N] = 0
                        if p1_ok and p2_ok:
                            actions.append(WallAction(x, y, 'v'))

        _WALL_ACTIONS_CACHE[cache_key] = actions
        return actions

    def get_legal_actions(self) -> List[Action]:
        """Get all legal actions from the current state. Essential for MCTS."""
        if self._legal_actions_cache is None:
            self._legal_actions_cache = (
                self._get_legal_pawn_actions() + self._get_legal_wall_actions()
            )
        return self._legal_actions_cache

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.boardsize and 0 <= y < self.boardsize

    def is_pawn_move_legal(self, action: PawnAction) -> bool:
        """
        Check legality of a pawn move.

        Orthogonal move (dx, dy) with one component zero:
          - Target cell must be in bounds and not wall-blocked.
          - If the target cell is occupied by the opponent, the move becomes a jump:
              * Straight jump: land two steps ahead if in bounds and not wall-blocked.
              * Diagonal jump: only when the straight continuation is blocked (wall or edge);
                the diagonal direction's component must also not be wall-blocked.

        Diagonal move (both dx and dy non-zero) — "go around" jump:
          - The intermediate orthogonal step toward the opponent (in the primary axis)
            must be occupied AND the straight continuation past the opponent must be
            blocked (wall or board edge). The diagonal step from the opponent's cell
            must itself not be wall-blocked.
          - Only one axis can be the "jump axis": we try both (dx,0) and (0,dy) as
            intermediate steps.
        """
        dx, dy = action.direction
        cur_pos = self.player1pos if self.is_player1_turn() else self.player2pos
        opp_pos = self.player2pos if self.is_player1_turn() else self.player1pos
        cx, cy = cur_pos

        # --- Orthogonal move ---
        if dx == 0 or dy == 0:
            tx, ty = cx + dx, cy + dy
            if not self._in_bounds(tx, ty):
                return False
            if self._is_edge_blocked(cx, cy, dx, dy):
                return False
            if (tx, ty) != opp_pos:
                # Normal step onto empty cell
                return True
            # Opponent is on the target cell: attempt a jump
            jx, jy = tx + dx, ty + dy
            if self._in_bounds(jx, jy) and not self._is_edge_blocked(tx, ty, dx, dy):
                # Straight jump is valid — this orthogonal action represents it
                return True
            # Straight jump blocked: diagonal jumps are encoded as separate diagonal actions
            return False

        # --- Diagonal move ---
        # A diagonal (dx, dy) is legal only as a "go-around" jump.
        # The player must be adjacent to the opponent via one orthogonal axis,
        # and the straight continuation past the opponent is blocked.
        for step_dir, side_dir in [((dx, 0), (0, dy)), ((0, dy), (dx, 0))]:
            sdx, sdy = step_dir
            ldx, ldy = side_dir
            mid_x, mid_y = cx + sdx, cy + sdy
            if (mid_x, mid_y) != opp_pos:
                continue
            if self._is_edge_blocked(cx, cy, sdx, sdy):
                continue  # wall between current and opponent — can't even reach opponent
            # Check the straight continuation past opponent is blocked
            past_x, past_y = mid_x + sdx, mid_y + sdy
            straight_blocked = (
                not self._in_bounds(past_x, past_y)
                or self._is_edge_blocked(mid_x, mid_y, sdx, sdy)
            )
            if not straight_blocked:
                continue  # straight jump available → diagonal not legal from this axis
            # Check we can actually step sideways from the opponent's cell
            land_x, land_y = mid_x + ldx, mid_y + ldy
            if not self._in_bounds(land_x, land_y):
                continue
            if self._is_edge_blocked(mid_x, mid_y, ldx, ldy):
                continue
            return True
        return False

    def _wall_path_check(self, action: WallAction) -> bool:
        """
        Return True if placing `action` leaves both players with a path to their goal.

        Uses cached path-edge sets to skip BFS for players whose greedy shortest
        path is not touched by the new wall.  Temporarily applies then removes the
        wall so BFS sees the updated board.
        """
        x, y = action.x, action.y
        if action.orientation == 'h':
            seg_a, seg_b = ('h', y, x), ('h', y, x + 1)
        else:
            seg_a, seg_b = ('v', y, x), ('v', y + 1, x)

        pe1, pe2 = self.p1_path_edges, self.p2_path_edges
        p1_needs_bfs = pe1 is None or seg_a in pe1 or seg_b in pe1
        p2_needs_bfs = pe2 is None or seg_a in pe2 or seg_b in pe2

        if not p1_needs_bfs and not p2_needs_bfs:
            return True  # neither player's path can be cut by this wall

        # Use bits-only helpers to avoid anchor-set bookkeeping during the
        # temporary apply/remove; BFS only reads the bytearray matrices.
        self._apply_wall_bits(action)
        p1_ok = (not p1_needs_bfs) or self.BFS(self.player1pos, self.boardsize - 1)
        p2_ok = (not p2_needs_bfs) or self.BFS(self.player2pos, 0)
        self._remove_wall_bits(action)

        return p1_ok and p2_ok

    def _wall_overlaps(self, action: WallAction) -> bool:
        """Return True if `action` overlaps or crosses an already-placed wall."""
        x, y = action.x, action.y
        if action.orientation == 'h':
            return (
                (x, y)     in self.hwall_anchors or
                (x - 1, y) in self.hwall_anchors or
                (x + 1, y) in self.hwall_anchors or
                (x, y)     in self.vwall_anchors
            )
        else:
            return (
                (x, y)     in self.vwall_anchors or
                (x, y - 1) in self.vwall_anchors or
                (x, y + 1) in self.vwall_anchors or
                (x, y)     in self.hwall_anchors
            )

    def is_wall_legal(self, action: WallAction) -> bool:
        """
        Check legality of placing a wall.

        A wall placement is illegal if any of the following hold:
        1. The anchor is out of the valid range (x or y > boardsize-2), because
           the second cell of the wall would fall outside the board.
        2. Same-orientation overlap: the new wall shares a cell with an existing
           wall of the same orientation.  For 'h' at (x,y) the two cells are
           (x,y) and (x+1,y), so we check anchors (x,y), (x-1,y), (x+1,y).
           For 'v' at (x,y): anchors (x,y), (x,y-1), (x,y+1).
        3. Crossing overlap: an 'h' wall at (x,y) and a 'v' wall at (x,y) share
           the same central pivot and cross each other.
        4. Path blocked: after the hypothetical placement, at least one player
           has no path to their goal row.
        """
        if action.orientation not in ('h', 'v'):
            raise ValueError("Invalid wall orientation: {}".format(action.orientation))

        x, y = action.x, action.y
        limit = self.boardsize - 2  # max valid anchor coordinate

        # 1. Bounds check
        if not (0 <= x <= limit and 0 <= y <= limit):
            return False

        # 2+3. Overlap and crossing check
        if self._wall_overlaps(action):
            return False

        # 4. Path check: does either player get disconnected from their goal?
        return self._wall_path_check(action)

    def _apply_wall_bits(self, action: WallAction) -> None:
        """Set wall bits only — no anchor-set update. Used during legality probing."""
        x, y, N = action.x, action.y, self.boardsize
        if action.orientation == 'h':
            self.hwalls[y * N + x] = 1;      self.hwalls[y * N + x + 1] = 1
        else:
            self.vwalls[y * N + x] = 1;      self.vwalls[(y + 1) * N + x] = 1

    def _remove_wall_bits(self, action: WallAction) -> None:
        """Clear wall bits only — no anchor-set update. Used during legality probing."""
        x, y, N = action.x, action.y, self.boardsize
        if action.orientation == 'h':
            self.hwalls[y * N + x] = 0;      self.hwalls[y * N + x + 1] = 0
        else:
            self.vwalls[y * N + x] = 0;      self.vwalls[(y + 1) * N + x] = 0

    def _apply_wall(self, action: WallAction) -> None:
        """Write wall bits and register the anchor (does not check legality)."""
        x, y = action.x, action.y
        N = self.boardsize
        if action.orientation == 'h':
            self.hwalls[y * N + x] = 1
            self.hwalls[y * N + x + 1] = 1
            self.hwall_anchors.add((x, y))
        else:
            self.vwalls[y * N + x] = 1
            self.vwalls[(y + 1) * N + x] = 1
            self.vwall_anchors.add((x, y))

    def _remove_wall(self, action: WallAction) -> None:
        """Clear wall bits and deregister the anchor."""
        x, y = action.x, action.y
        N = self.boardsize
        if action.orientation == 'h':
            self.hwalls[y * N + x] = 0
            self.hwalls[y * N + x + 1] = 0
            self.hwall_anchors.discard((x, y))
        else:
            self.vwalls[y * N + x] = 0
            self.vwalls[(y + 1) * N + x] = 0
            self.vwall_anchors.discard((x, y))

    def place_wall(self, action: WallAction, check_legal=True):
        """Place a wall based on action.orientation ('h' or 'v')."""
        if check_legal and not self.is_wall_legal(action):
            raise ValueError(
                "Illegal {} wall placement at ({}, {})".format(
                    action.orientation,
                    action.x,
                    action.y,
                )
            )
        self._apply_wall(action)
        
    def move_pawn(self, action: PawnAction):
        """
        Move the current player's pawn according to the action direction.

        For an orthogonal direction where the adjacent cell is occupied by the
        opponent, the pawn jumps two steps (straight jump). Otherwise it moves
        one step. For a diagonal direction the pawn lands one step in each
        component (the go-around jump destination).
        """
        dx, dy = action.direction
        is_p1 = self.is_player1_turn()
        cur_pos = self.player1pos if is_p1 else self.player2pos
        opp_pos = self.player2pos if is_p1 else self.player1pos
        cx, cy = cur_pos

        if dx == 0 or dy == 0:
            # Orthogonal: check for straight jump over opponent
            tx, ty = cx + dx, cy + dy
            if (tx, ty) == opp_pos:
                # Straight jump: land two cells ahead
                new_pos = (cx + 2 * dx, cy + 2 * dy)
            else:
                new_pos = (tx, ty)
        else:
            # Diagonal go-around: always lands one step in each component
            new_pos = (cx + dx, cy + dy)

        if is_p1:
            self.player1pos = new_pos
        else:
            self.player2pos = new_pos

    def BFS(self, start, goal_row):
        """
        Return True if there is a path from `start` (x, y) to any cell in
        `goal_row` (a y-coordinate), respecting the current wall configuration.

        Uses plain BFS over the grid; each edge is checked with _is_edge_blocked.
        Complexity: O(boardsize^2).
        """
        sx, sy = start
        if sy == goal_row:
            return True
        N = self.boardsize
        visited = bytearray(N * N)
        visited[sy * N + sx] = 1
        queue = deque([(sx, sy)])
        hwalls = self.hwalls
        vwalls = self.vwalls
        while queue:
            x, y = queue.popleft()
            yi = y * N
            # up
            if y + 1 < N and not visited[yi + N + x] and not hwalls[yi + x]:
                if y + 1 == goal_row: return True
                visited[yi + N + x] = 1;  queue.append((x, y + 1))
            # down
            if y > 0 and not visited[yi - N + x] and not hwalls[yi - N + x]:
                if y - 1 == goal_row: return True
                visited[yi - N + x] = 1;  queue.append((x, y - 1))
            # right
            if x + 1 < N and not visited[yi + x + 1] and not vwalls[yi + x]:
                if y == goal_row: return True
                visited[yi + x + 1] = 1;  queue.append((x + 1, y))
            # left
            if x > 0 and not visited[yi + x - 1] and not vwalls[yi + x - 1]:
                if y == goal_row: return True
                visited[yi + x - 1] = 1;  queue.append((x - 1, y))
        return False

    def copy(self):
        return State(
            boardsize=self.boardsize,
            depth=self.depth,
            player1pos=self.player1pos,
            player2pos=self.player2pos,
            hwalls=bytearray(self.hwalls),
            vwalls=bytearray(self.vwalls),
            walls_p1=self.walls_p1,
            walls_p2=self.walls_p2,
            hwall_anchors=set(self.hwall_anchors),
            vwall_anchors=set(self.vwall_anchors),
            walls_initial=self.walls_initial,
            position_history=dict(self.position_history),
            p1_dist=self.p1_dist,
            p2_dist=self.p2_dist,
            p1_path_edges=self.p1_path_edges,
            p2_path_edges=self.p2_path_edges,
        )

    def _deduct_wall(self) -> None:
        """Decrement the current player's remaining wall count by one."""
        if self.get_current_player() == 1:
            self.walls_p1 -= 1
        else:
            self.walls_p2 -= 1

    def next(self, action: Action, check_legal = False) -> 'State':
        """Apply action to the current state and return the resulting state after the action is applied.
        Ensures all game rules are followed.
        """
        if check_legal and not self.is_action_legal(action):
            raise ValueError(f"Illegal action: {action}")
        
        state = self.copy()
        
        if isinstance(action, PawnAction):
            state.move_pawn(action)
            state._recompute_path_edges()  # positions changed; dist grids still valid
        elif isinstance(action, WallAction):
            state.place_wall(action, check_legal=False)
            state._deduct_wall()
            state._recompute_dists()  # walls changed; update cached dist grids
        
        state.depth += 1
        state._record_position()
        return state

    @staticmethod
    def _bfs_dist_grid(hwalls, vwalls, goal_row, N):
        """
        Multi-source BFS from all cells in goal_row outward.
        Returns an N x N list of ints: the shortest wall-aware number of orthogonal
        moves needed to reach goal_row from each cell.
        Unreachable cells get value N*N (guaranteed > any real distance).

        hwalls and vwalls must already be in the perspective-flipped coordinate
        system (i.e. as if the current player is always moving toward row N-1).
        """
        INF = N * N
        dist = [[INF] * N for _ in range(N)]
        queue = deque()
        for x in range(N):
            dist[goal_row][x] = 0
            queue.append((x, goal_row))
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < N and 0 <= ny < N):
                    continue
                if dist[ny][nx] != INF:
                    continue
                # Edge-blocked checks mirror _is_edge_blocked() semantics
                if dy == 1  and hwalls[y * N + x]:           continue
                if dy == -1 and hwalls[(y - 1) * N + x]:     continue
                if dx == 1  and vwalls[y * N + x]:           continue
                if dx == -1 and vwalls[y * N + x - 1]:       continue
                dist[ny][nx] = dist[y][x] + 1
                queue.append((nx, ny))
        return dist

    def _get_nn_perspective(self, hw: np.ndarray, vw: np.ndarray):
        """
        Return board tensors and scalar counts from the perspective of the current
        player, flipping the board vertically when it is P2's turn so the network
        always sees the current player moving toward row N-1.

        Returns (my_pos, opp_pos, my_walls, opp_walls, hw, vw, my_dist_raw, opp_dist_raw).
        """
        N = self.boardsize
        if self.is_player1_turn():
            return (
                self.player1pos, self.player2pos,
                self.walls_p1, self.walls_p2,
                hw, vw,
                self.p1_dist, self.p2_dist,
            )
        # P2's turn: flip the board vertically so P2 moves toward row N-1
        flipped_hw = np.zeros((N, N), dtype=np.float32)
        flipped_hw[:N - 1] = hw[:N - 1][::-1]
        return (
            (self.player2pos[0], N - 1 - self.player2pos[1]),
            (self.player1pos[0], N - 1 - self.player1pos[1]),
            self.walls_p2, self.walls_p1,
            flipped_hw, vw[::-1].copy(),
            self.p2_dist[::-1], self.p1_dist[::-1],
        )

    def to_nn_input(self):
        """
        Encode the state as a (8, N, N) float32 numpy array from the perspective
        of the current player, who is always treated as moving toward row N-1.

        When it is P2's turn the board is flipped vertically (new_y = N-1-old_y)
        so that the network sees a canonical orientation regardless of which
        player is to move.

        Channels
        --------
        0  — my pawn position (1-hot)
        1  — opponent pawn position (1-hot)
        2  — horizontal walls  (1 where a horizontal wall segment exists)
        3  — vertical walls    (1 where a vertical wall segment exists)
        4  — my remaining walls, constant plane normalised by walls_initial
        5  — opponent remaining walls, constant plane normalised by walls_initial
        6  — BFS distance from every cell to MY goal row (row N-1), normalised
             to [0, 1]; unreachable cells get 1.0
        7  — BFS distance from every cell to OPPONENT's goal row (row 0),
             normalised to [0, 1]; unreachable cells get 1.0

        Returns
        -------
        np.ndarray of shape (8, N, N), dtype float32
        """
        N = self.boardsize
        hw = np.frombuffer(self.hwalls, dtype=np.uint8).reshape(N, N).astype(np.float32)
        vw = np.frombuffer(self.vwalls, dtype=np.uint8).reshape(N, N).astype(np.float32)
        my_pos, opp_pos, my_walls, opp_walls, hw, vw, my_dist_raw, opp_dist_raw = \
            self._get_nn_perspective(hw, vw)

        max_dist = N * N - 1
        INF = N * N
        norm = self.walls_initial if self.walls_initial > 0 else 1

        planes = np.zeros((8, N, N), dtype=np.float32)

        # Pawn positions
        planes[0, my_pos[1],  my_pos[0]]  = 1.0
        planes[1, opp_pos[1], opp_pos[0]] = 1.0

        # Wall matrices
        planes[2] = hw
        planes[3] = vw

        # Remaining walls (scalar broadcast to full plane)
        planes[4] = my_walls  / norm
        planes[5] = opp_walls / norm

        # BFS distance planes (from cached dist grids, no extra BFS calls)
        d = np.array(my_dist_raw,  dtype=np.float32)
        planes[6] = np.where(d < INF, d / max_dist, 1.0)

        d = np.array(opp_dist_raw, dtype=np.float32)
        planes[7] = np.where(d < INF, d / max_dist, 1.0)

        return planes


# ---------------------------------------------------------------------------
# Action ↔ policy-index mapping (for neural-network policy head)
# ---------------------------------------------------------------------------
#
# Index layout for a board of size N:
#
#   0 – 3          PawnAction  orthogonal:  up, down, left, right
#   4 – 7          PawnAction  diagonal:    up-left, up-right, down-left, down-right
#   8 … 8+(N-1)²-1            WallAction horizontal, row-major: idx = 8 + y*(N-1) + x
#   8+(N-1)² … 8+2*(N-1)²-1  WallAction vertical,   row-major: idx = 8 + (N-1)² + y*(N-1) + x
#
#   Total actions:  8 + 2*(N-1)²
#   7×7 board  →  8 + 2×36  =  80
#   9×9 board  →  8 + 2×64  = 136
#
# The mapping is purely positional: indices 0–7 mirror ALL_PAWN_DIRECTIONS order
# and wall indices increase left-to-right, then bottom-to-top — matching the
# board coordinate system (origin bottom-left).

def action_space_size(boardsize: int) -> int:
    """Total number of distinct actions on a board of the given size."""
    return 8 + 2 * (boardsize - 1) ** 2


def action_to_index(action: Action, boardsize: int) -> int:
    """
    Convert an ``Action`` to its policy-head index.

    Raises ``ValueError`` for unknown action types or out-of-range coordinates.
    """
    if isinstance(action, PawnAction):
        return ALL_PAWN_DIRECTIONS.index(action.direction)
    if isinstance(action, WallAction):
        W = boardsize - 1
        offset = 0 if action.orientation == 'h' else W * W
        return 8 + offset + action.y * W + action.x
    raise ValueError(f"Unknown action type: {type(action)}")


def index_to_action(index: int, boardsize: int) -> Action:
    """
    Convert a policy-head index back to an ``Action``.

    Raises ``ValueError`` for indices outside ``[0, action_space_size(boardsize))``.
    """
    total = action_space_size(boardsize)
    if not (0 <= index < total):
        raise ValueError(f"Index {index} out of range [0, {total}) for boardsize {boardsize}")
    if index < 8:
        return PawnAction(direction=ALL_PAWN_DIRECTIONS[index])
    W = boardsize - 1
    wall_idx = index - 8
    if wall_idx < W * W:
        return WallAction(x=wall_idx % W, y=wall_idx // W, orientation='h')
    wall_idx -= W * W
    return WallAction(x=wall_idx % W, y=wall_idx // W, orientation='v')


# ---------------------------------------------------------------------------
# Left-right symmetry helpers
# ---------------------------------------------------------------------------
#
# Quoridor has left-right (mirror) symmetry: if you flip the board
# horizontally, the resulting position is equally valid and strategically
# equivalent.  These helpers let callers exploit that symmetry for free data
# augmentation and MCTS evaluator symmetrisation.

# Pawn direction indices after a left-right flip.
# Under flip: dx → -dx, dy stays the same.
# ALL_PAWN_DIRECTIONS: 0=(0,1) 1=(0,-1) 2=(-1,0) 3=(1,0)
#                      4=(-1,1) 5=(1,1) 6=(-1,-1) 7=(1,-1)
_LR_PAWN_FLIP = (0, 1, 3, 2, 5, 4, 7, 6)


def flip_nn_input_lr(planes: np.ndarray) -> np.ndarray:
    """
    Flip a (8, N, N) nn_input array left-right (mirror along the vertical axis).

    Column x maps to column N-1-x.  This is a pure numpy operation — no game
    state reconstruction needed.

    The wall channels (2=hwalls, 3=vwalls) encode wall *segments* on the NxN
    grid; under a column flip those segments flip with the board, so
    ``np.flip(..., axis=2)`` is both necessary and sufficient for all 8 channels.
    """
    return np.flip(planes, axis=2).copy()


def flip_policy_lr(policy: np.ndarray, boardsize: int) -> np.ndarray:
    """
    Remap a full-size policy vector (length = action_space_size(boardsize))
    to the mirrored board.

    Pawn moves: swap left↔right and the two pairs of diagonals (see _LR_PAWN_FLIP).
    Wall moves: anchor x → W-1-x for both H and V walls (y unchanged).
    """
    W = boardsize - 1
    flipped = np.empty_like(policy)

    # Pawn actions (indices 0-7)
    for orig, mapped in enumerate(_LR_PAWN_FLIP):
        flipped[mapped] = policy[orig]

    # H-wall actions (indices 8 .. 8+W²-1), anchor grid is W×W, row-major
    h_start = 8
    h_block = policy[h_start: h_start + W * W].reshape(W, W)      # (W, W) [y, x]
    flipped[h_start: h_start + W * W] = np.flip(h_block, axis=1).ravel()

    # V-wall actions (indices 8+W² .. end), same layout
    v_start = 8 + W * W
    v_block = policy[v_start: v_start + W * W].reshape(W, W)
    flipped[v_start: v_start + W * W] = np.flip(v_block, axis=1).ravel()

    return flipped


if __name__ == "__main__":
    # Example usage
    state = State()
    state.print_playerspos()
    print()
    state.print_hwalls()
    print()
    state.print_vwalls()
    print()
    directions = PawnDirections()
    print(f"Up: {directions.up}")
    print(f"Up-left: {directions.up_and_left}")
    print()
    print("Placing a horizontal wall at (2, 3)")
    state.place_wall(WallAction(x=2, y=3, orientation='h'), check_legal=False)
    state.print_hwalls()