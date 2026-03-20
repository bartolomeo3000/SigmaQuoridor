# tu bedzie logika gry
class PostionAction:
    def __init__(self):
        self.up = (-1, 0)
        self.down = (1, 0)
        self.left = (0, -1)
        self.right = (0, 1)

class State:
    def __init__(self, boardsize = 7):
        if boardsize % 2 == 0:
            raise ValueError("Board size must be odd")
        self.boardsize = boardsize
        self.player1pos = (0, boardsize // 2)
        self.player2pos = (boardsize - 1, boardsize // 2)

        self.hwalls = [[0 for _ in range(boardsize)] for _ in range(boardsize)]
        self.vwalls = [[0 for _ in range(boardsize)] for _ in range(boardsize)]

    def playerspos_matrix(self):
        matrix = [[0 for _ in range(self.boardsize)] for _ in range(self.boardsize)]
        matrix[self.player1pos[0]][self.player1pos[1]] = 1
        matrix[self.player2pos[0]][self.player2pos[1]] = 2
        return matrix

    def print_playerspos(self):
        for row in self.playerspos_matrix():
            print(' '.join(str(cell) for cell in row))

    def print_hwalls(self):
        for row in self.hwalls:
            print(' '.join(str(cell) for cell in row))
    
    def print_vwalls(self):
        for row in self.vwalls:
            print(' '.join(str(cell) for cell in row))

    def place_hwall(self, x, y):
        if self.hwalls[x][y] == 1:
            raise ValueError("There is already a horizontal wall at this position")
        self.hwalls[x][y] = 1

    def place_vwall(self, x, y):
        if self.vwalls[x][y] == 1:
            raise ValueError("There is already a vertical wall at this position")
        self.vwalls[x][y] = 1
    
    def next(self, action):
        state = State(self.boardsize)
        # logic to apply the action and update the state
        return state  # placeholder for the next state after applying the action

if __name__ == "__main__":
    state = State()
    state.print_playerspos()
    print()
    state.print_hwalls()
    print()
    state.print_vwalls()