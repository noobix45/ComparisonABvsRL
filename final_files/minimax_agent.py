import numpy as np

PATTERNS = {
    "11111" : 100000,
    "011110": 10000,
    "011112": 1000,
    "211110": 1000,
    "10111": 1000,
    "11011": 1000,
    "11101": 1000,
    "001110": 500,
    "011100": 500,        
    "010110": 500,
    "011010": 500,
    "001100": 50,
    "000110": 50,
    "011000": 50
}

#precalcularea scorurilor
_SCORE_TABLE = np.zeros(2187, dtype=np.int32)
for i in range(2187):
    val = i
    chars = []
    for _ in range(7):
        chars.append(str(val % 3))
        val //= 3
    s = "".join(chars)
    
    score = 0
    for p, s_val in PATTERNS.items():
        score += s.count(p) * s_val
    _SCORE_TABLE[i] = score

class MinimaxAgent2:
    def __init__(self, player_id, max_depth=2, shared_memo=None):
        self.player_id = player_id
        self.max_depth = max_depth
        self.nodes_visited = 0
        self.memo = shared_memo if shared_memo is not None else {}

    def get_action(self, env):
        self.nodes_visited = 0
        best_score = -float('inf')
        best_move = None
        possible_moves = list(env.valid_moves)

        # sortare dupa zonele care sunt mai aglomerate
        possible_moves.sort(
            key=lambda m: np.count_nonzero(
                env.board[max(0, m[0]-1):m[0]+2, max(0, m[1]-1):m[1]+2]
            ), 
            reverse=True
        )

        alpha = -float("inf")
        beta = float("inf")

        for move in possible_moves:
            r, c = move
            env.make_move(r, c)
            score = self.minimax(env, self.max_depth-1, alpha, beta, is_maximizing=False)
            env.undo_move()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

        return best_move
    
    def minimax(self, env, depth, alpha, beta, is_maximizing):
        self.nodes_visited += 1

        if env.winner == self.player_id: 
            return 1000000 
        if env.winner == -self.player_id:
            return -1000000 
        if depth == 0 or not env.valid_moves:
            return self.evaluate(env)
        
        possible_moves = list(env.valid_moves)

        if is_maximizing:
            max_eval = -float('inf')
            for r, c in possible_moves:
                env.make_move(r, c)
                eval = self.minimax(env, depth-1, alpha, beta, False) # ia evaluarea la nivelul asta
                env.undo_move() # da mutarea inapoi
                max_eval = max(max_eval, eval) # acvtualizeaza maxim
                alpha = max(alpha, eval)
                if beta <= alpha: # pruning
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for r, c in possible_moves:
                env.make_move(r, c)
                eval = self.minimax(env, depth-1, alpha, beta, True)
                env.undo_move()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: # pruning
                    break
            return min_eval

    def evaluate(self, env):
        board_bytes = env.board.tobytes('C')
        if board_bytes in self.memo:
            return self.memo[board_bytes]
        if env.move_count == 0:
            return 0

        my_board = np.zeros_like(env.board, dtype=np.int32)
        my_board[env.board == self.player_id] = 1
        my_board[env.board == -self.player_id] = 2

        op_board = np.zeros_like(env.board, dtype=np.int32)
        op_board[env.board == -self.player_id] = 1
        op_board[env.board == self.player_id] = 2

        my_padded = np.pad(my_board, pad_width=1, constant_values=2)
        op_padded = np.pad(op_board, pad_width=1, constant_values=2)

        def get_board_score(B):
            W_horiz = (B[:, :-6] + B[:, 1:-5]*3 + B[:, 2:-4]*9 + B[:, 3:-3]*27 + B[:, 4:-2]*81 + B[:, 5:-1]*243 + B[:, 6:]*729)
            W_vert = (B[:-6, :] + B[1:-5, :]*3 + B[2:-4, :]*9 + B[3:-3, :]*27 + B[4:-2, :]*81 + B[5:-1, :]*243 + B[6:, :]*729)
            W_diag1 = (B[:-6, :-6] + B[1:-5, 1:-5]*3 + B[2:-4, 2:-4]*9 + B[3:-3, 3:-3]*27 + B[4:-2, 4:-2]*81 + B[5:-1, 5:-1]*243 + B[6:, 6:]*729)
            W_diag2 = (B[:-6, 6:] + B[1:-5, 5:-1]*3 + B[2:-4, 4:-2]*9 + B[3:-3, 3:-3]*27 + B[4:-2, 2:-4]*81 + B[5:-1, 1:-5]*243 + B[6:, :-6]*729)
            return int(np.sum(_SCORE_TABLE[W_horiz]) + np.sum(_SCORE_TABLE[W_vert]) + np.sum(_SCORE_TABLE[W_diag1]) + np.sum(_SCORE_TABLE[W_diag2]))

        final_score = get_board_score(my_padded) - get_board_score(op_padded)
        self.memo[board_bytes] = final_score
        self.memo[(-env.board).tobytes('C')] = -final_score
        
        return final_score