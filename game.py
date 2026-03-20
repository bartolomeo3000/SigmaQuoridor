# tu bedzie logika gry
from dataclasses import dataclass, field
from typing import Union, List


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


@dataclass
class PawnAction:
    """Action to move the pawn in a given direction."""
    direction: tuple  # e.g., (0, 1) for up


@dataclass
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


# Unified action type
Action = Union[PawnAction, WallAction]


class State:
    """
    Class to represent the state of the game, including the board size, player positions, wall placements, and the current player's turn.
    Implements the game logic and move legality checks for both pawn moves and wall placements.
    The main functionality is the next() method, which takes an action and returns the resulting state after applying that action.
    Other important method is get_legal_actions(), which generates all legal actions from the current state, essential for MCTS.
    Also includes is_finished() method to check if the game has reached a terminal state.
    """
    def __init__(self, boardsize=7, depth=0, player1pos=None, player2pos=None, hwalls=None, vwalls=None, walls_p1=5, walls_p2=5):
        if boardsize % 2 == 0:
            raise ValueError("Board size must be odd")
        self.boardsize = boardsize

        # depth is used to determine how long the game has progressed and which player's turn it is
        self.depth = depth

        # Initialize player positions and walls if not provided
        self.player1pos = player1pos if player1pos is not None else (boardsize // 2, 0)
        self.player2pos = player2pos if player2pos is not None else (boardsize // 2, boardsize - 1)

        # Initialize walls as 2D lists if not provided
        if hwalls is not None:
            self.hwalls = hwalls
        else:
            self.hwalls = [[0 for _ in range(boardsize)] for _ in range(boardsize)]

        if vwalls is not None:
            self.vwalls = vwalls
        else:
            self.vwalls = [[0 for _ in range(boardsize)] for _ in range(boardsize)]
        
        # Track remaining walls for each player
        self.walls_p1 = walls_p1
        self.walls_p2 = walls_p2

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
        for row in reversed(self.hwalls):
            print(' '.join(str(cell) for cell in row))
    
    def print_vwalls(self):
        for row in reversed(self.vwalls):
            print(' '.join(str(cell) for cell in row))

    def get_current_player(self):
        """Returns 1 if it's player 1's turn, 2 if it's player 2's turn."""
        return 1 if self.depth % 2 == 0 else 2

    def is_player1_turn(self):
        return self.depth % 2 == 0

    def is_drawn(self):
        """Game is drawn if it reaches a certain depth without a winner, to prevent infinite games."""
        # Later can add a check for 3 repetitions of the same position, but for now we just use the depth limit.
        if self.depth >= 100:
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

    def get_legal_actions(self) -> List[Action]:
        """
        Get all legal actions from the current state. Essential for MCTS.
        
        Note: I wrote the basic structure here, but you can change it as you like.
        """
        actions = []
        
        # Add legal pawn moves
        pawn_directions = PawnDirections()
        directions = [pawn_directions.up, pawn_directions.down, pawn_directions.left, pawn_directions.right,
                      pawn_directions.up_and_left, pawn_directions.up_and_right, 
                      pawn_directions.down_and_left, pawn_directions.down_and_right]
        
        # Basic idea is to loop through all possible directions and check legality.
        # Can also be smarter and check diagonal directions only if the corresponding orthogonal move is blocked by an opponent's pawn with a wall behind.
        for direction in directions:    
            action = PawnAction(direction=direction)
            if self.is_pawn_move_legal(action):
                actions.append(action)
        
        # Add legal wall placements (respecting wall count)
        current_player = self.get_current_player()
        has_remaining_walls = (self.walls_p1 > 0 if current_player == 1 else self.walls_p2 > 0)
        if not has_remaining_walls:
            return actions  # No wall placements possible if no walls left
        
        # TODO: Loop through all possible wall placements and check legality
        # for x in range(self.boardsize - 1):
        #     for y in range(self.boardsize - 1):
        #         h_action = WallAction(x=x, y=y, orientation='h')
        #         if self.is_wall_legal(h_action):
        #             actions.append(h_action)
        #         v_action = WallAction(x=x, y=y, orientation='v')
        #         if self.is_wall_legal(v_action):
        #             actions.append(v_action)
        return actions

    def is_pawn_move_legal(self, action: PawnAction) -> bool:
        """Check legality of pawn move, including:
        - Whether the move is within the bounds of the board
        - Whether the move is blocked by a wall
        """
        # TODO: Implement pawn move legality checks
        ...
        pass

    def is_wall_legal(self, action: WallAction) -> bool:
        # TODO: Check legality of placing a wall based on action.orientation.
        # Horizontal ('h') wall occupies (x, y) and (x+1, y).
        # Vertical ('v') wall occupies (x, y) and (x, y+1).
        # This should include checks for:
        # - Whether the wall overlaps with existing horizontal or vertical walls
        # - Whether the wall blocks all paths for either player to reach their goal (BFS)
        if action.orientation not in ('h', 'v'):
            return False
        ...
        return False

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

        if action.orientation == 'h':
            self.hwalls[action.y][action.x] = 1
            self.hwalls[action.y][action.x + 1] = 1
        elif action.orientation == 'v':
            self.vwalls[action.y][action.x] = 1
            self.vwalls[action.y + 1][action.x] = 1
        else:
            raise ValueError("Invalid wall orientation: {}".format(action.orientation))
        
    def move_pawn(self, action: PawnAction):
        """Move the current player's pawn in the specified direction."""
        # TODO: Implement pawn move logic
        pass

    def BFS(self, start, goal_row):
        """Breadth-First Search to check if there's a path from start to any cell in goal_row, considering the walls."""
        # TODO: Implement BFS to check for path existence, taking into account the walls on the board.
        # Lets implement it at the very end.
        return True  # placeholder for BFS result

    def copy(self):
        return State(
            boardsize=self.boardsize,
            depth=self.depth,
            player1pos=self.player1pos,
            player2pos=self.player2pos,
            hwalls=[row[:] for row in self.hwalls],
            vwalls=[row[:] for row in self.vwalls],
            walls_p1=self.walls_p1,
            walls_p2=self.walls_p2
        )

    def next(self, action: Action, check_legal = False) -> 'State':
        """Apply action to the current state and return the resulting state after the action is applied.
        Ensures all game rules are followed.
        """
        if check_legal and not self.is_action_legal(action):
            raise ValueError(f"Illegal action: {action}")
        
        state = self.copy()
        
        if isinstance(action, PawnAction):
            state.move_pawn(action)
        elif isinstance(action, WallAction):
            state.place_wall(action, check_legal=False)
            current_player = state.get_current_player()
            if current_player == 1:
                state.walls_p1 -= 1
            else:
                state.walls_p2 -= 1
        
        state.depth += 1
        return state


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