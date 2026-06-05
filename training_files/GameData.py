import os
import json
import numpy as np
import pygame
import random

class GameEngine:
    def __init__(self, size=10, players=["X","O"]):
        self.size = size
        # 0 liber; 1 jucator 1 = x; -1 jucator 2 = o
        self.board = np.zeros((size,size), dtype=np.int8)
        self.players = players
        self.current_player = 1
        self.move_count = 0
        self.history = []

        self.winner = 0
        self.winning_line = None

        # initializarea frontierei de mutari valide - pentru agent
        self.valid_moves = set()
        if self.size % 2 == 0:
            c1, c2 = self.size //2-1, self.size//2
            self.valid_moves.update([(c1,c1), (c1,c2), (c2,c1), (c2,c2)])
        else:
            self.valid_moves.add((self.size//2, self.size//2))

        self._eval_agent_p1 = None # agent plus1
        self._eval_agent_m1 = None #agent minus1
    
    """ apelat doar la prima mutare, verifica daca mutarea a fost facuta in centru"""
    def is_center(self, r, c):
        # 0 1 2 3 |4 5| 6 7 8 9
        if self.size % 2 == 0:
            center_indices = [self.size // 2 - 1, self.size//2]
            return r in center_indices and c in center_indices # daca pozitia row si column este in centru sau nu
        else:
            return r == self.size // 2 and c == self.size // 2

    """ verifica daca mutarea e adiacenta """
    def is_adjacent(self, r, c):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc # new row, new column
            if 0 <= nr < self.size and 0 <= nc < self.size: # incearca toate 4 directiile
                if self.board[nr, nc] != 0: # daca gaseste o pozitie ocupata e o pozitie valida; 
                    #am garantia ca nu imi gaseste o pozitie invalida pentru ca prima e valida
                    return True
        return False
        
    """ verifica daca o mutare e valida, daca e o face si da toggle la player"""
    def make_move(self, r, c):
        if not(0 <= r < self.size and 0 <= c < self.size): # safety check ca pozitia sa fie in board
            return False, "Outside board"
        if self.board[r, c] != 0:
            return False, "Occupied"
        
        if self.move_count == 0:
            if not self.is_center(r,c):
                return False, "First move must be in a center position"
        else:
            if not self.is_adjacent(r,c):
                return False, "Move must be adjacent"
            
        old_valid_moves = self.valid_moves.copy() # salvez copia frontierei inainte sa fac o mutare
        
        self.board[r, c] = self.current_player # pun in env simbolul

        if self.move_count == 0:
            self.valid_moves.clear()
        else:
            # scot pozitia simbolului plasat din frontiere pentru ca nu mai e o mutare valida
            if(r,c) in self.valid_moves:
                self.valid_moves.remove((r,c))

        # actualizarea frontierei
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc # new row, new column
            if 0 <= nr < self.size and 0 <= nc < self.size: # incearca toate 4 directiile
                if self.board[nr, nc] == 0: # daca gaseste o pozitie goala e o pozitie valida;
                    self.valid_moves.add((nr,nc))

        self.move_count += 1 # o mutare valida a fost facuta 
        # current_player e jucatorul care a facut mutarea (r,c) din variantele (old_valid_moves) - from now on frontiera reala e valid_moves
        self.history.append((r,c, self.current_player, old_valid_moves))

        is_win, line_coords = self.check_win(r,c)
        if is_win:
            self.winner = self.current_player
            self.winning_line = line_coords
            return True, "Game over"

        self.current_player *= -1 # toggle la player
        return True, "OK"
    
    def undo_move(self):
        if not self.history: # nu am istoric inca nu am ce undo sa fac
            return False
        r,c,player,old_moves = self.history.pop() # scot ultima mutare care a fost facuta din frontiera

        # reset la pozitia respectiva din tabla, playerul curent si daca a castigat sau nu
        self.board[r,c] = 0
        self.valid_moves = old_moves
        self.move_count-=1
        self.current_player = player
        self.winner = 0
        self.winning_line = None
        return True
    
    """ verifica daca ultimul simbol plasat fomreaza o linie de 5"""
    def check_win(self, r, c):
        symbol = self.board[r, c]
        if symbol == 0:
            return False, None
        
        axes = [
            (0, 1),   # Orizontala (stanga <-> dreapta)
            (1, 0),   # Verticala (sus <-> jos)
            (1, 1),   # Diagonala principala (stanga-sus <-> dreapta-jos)
            (1, -1)   # Diagonala secundara (stanga-jos <-> dreapta-sus)
        ]

        for dr, dc in axes: # pentru toate directiile
            count = 1

            start_r, start_c = r,c
            end_r, end_c = r,c
            
            for i in range(1,5): # verifica forward
                nr = r + dr * i
                nc = c + dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == symbol:
                    count += 1
                    end_r, end_c = nr, nc
                else:
                    break

            for i in range(1,5): # verifica backwards
                nr = r - dr * i
                nc = c - dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == symbol:
                    count += 1
                    start_r, start_c = nr, nc
                else:
                    break
                    
            if count >= 5:
                return True, ((start_r, start_c), (end_r, end_c))
        return False, None
    
    def evaluate_for_player(self, player_id):
        if player_id == 1: # daca id ul e 1
            if self._eval_agent_p1 is None: # creearea agentului se face o singura data cand e None
                self._eval_agent_p1 = MinimaxAgent2(1) # evaluez cu agent cu player id = 1
            return self._eval_agent_p1.evaluate(self)
        else:
            if self._eval_agent_m1 is None: 
                self._eval_agent_m1 = MinimaxAgent2(-1) # daca id e -1 se face cu id = -1
            return self._eval_agent_m1.evaluate(self)

def load_data(path = None):

    if path == None: # daca nu am un path predefinit iau ultimul fisier de date - convenabil
        cwd = os.getcwd()
        files = os.listdir(cwd)
        json_files = sorted([file for file in files if file.endswith(".json")])

        last_data_file = json_files[-1]
        path = cwd + "/" + last_data_file

    with open(path, "r") as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    # print(metadata)

    title = f"{metadata['agent_type']} played {metadata['iterations']} times, with depth {metadata['depth']} in {metadata['total_benchmark_time_seconds']:.2f} seconds"

    games = data["games"]

    times, move_coordss, frontier_sizes, memo_sizes, playerss = [], [], [], [], []

    for game in games:
        stats = [game[i] for i in range (len(game))]
        move_coords, time, frontier_size, memo_size, players = [], [], [], [], []

        for stat in stats:
            # print(stat)
            move_coords.append(stat.get("move_coords"))
            time.append(stat["time"])
            frontier_size.append(stat["frontier_size"])
            memo_size.append(stat["memo_size"])
            players.append(stat["player"])

        times.append(time)
        move_coordss.append(move_coords)
        frontier_sizes.append(frontier_size)
        memo_sizes.append(memo_size)
        playerss.append(players)
    
    return title, times, move_coordss, frontier_sizes, memo_sizes, playerss

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

_SCORE_TABLE = np.zeros(2187, dtype=np.int32)
for i in range(2187):
    val = i
    chars = []
    # Acum generăm 7 caractere
    for _ in range(7):
        chars.append(str(val % 3))
        val //= 3
    s = "".join(chars)
    
    score = 0
    for p, s_val in PATTERNS.items():
        # de cate ori apare sirul p in sirul s
        score += s.count(p) * s_val # aduna toate scorurile pe toate patternurile din window
    _SCORE_TABLE[i] = score

class MinimaxAgent2:
    def __init__(self, player_id, max_depth = 2, shared_memo=None):
        self.player_id = player_id # 1 pentru x; -1 pentru O
        self.max_depth = max_depth # adancimea maxima de cautare
        self.nodes_visited = 0

        if shared_memo is not None:
            self.memo = shared_memo
        else:
            self.memo = {}

    """ returneaza un tuplu (row, col) pentru mutarea pe care o alege conform minimax"""
    def get_action(self, env):

        self.nodes_visited = 0
        best_score = -float('inf') # init la scor cu -inf pentru agentul max
        best_move = None # inca nu am un best move

        possible_moves = list(env.valid_moves) # toate mutarile valide din frontiera

        # sorteaza mutarile dupa cat de aglomerata e zona din jur intr un window de 3x3
        possible_moves.sort(
            key=lambda m: np.count_nonzero(
                env.board[max(0, m[0]-1):m[0]+2, max(0, m[1]-1):m[1]+2]
            ), 
            reverse=True
        )

        alpha = -float("inf")
        beta = float("inf")

        for move in possible_moves: # ma uit in toate mutarile pe care le am posibile = branching factor = b = nr de mutari din frontiera
            r,c = move
            env.make_move(r,c) # fac cate o mutare din possible pe rand
             # urmeaza jucatorul adevers, Maximizing = False (adica Min); am facut deja o mutare deci depth-1
            score = self.minimax(env, self.max_depth-1, alpha, beta, is_maximizing=False) # iau scorul recursiv
            env.undo_move() # sterg mutarea pe care tocmai am facut o

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

        # print(f"AI ({'X' if self.player_id==1 else 'O'}) a explorat {self.nodes_visited} noduri")
        return best_move
    
    def minimax(self, env, depth, alpha, beta, is_maximizing):
        self.nodes_visited += 1

        if env.winner == self.player_id: 
            return 1000000 # castig garantat
        if env.winner == -self.player_id:
            return -1000000 # castiga adversarul
        if depth == 0 or not env.valid_moves: # nu mai am mutari valide sau am ajuns la adancimea maxima stabilita
            return self.evaluate(env)
        
        possible_moves = list(env.valid_moves)

        if is_maximizing: # agentul max
            max_eval = -float('inf')
            for r,c in possible_moves:
                env.make_move(r,c)
                eval = self.minimax(env, depth-1, alpha, beta, False) # il cheama pe cel MIN
                env.undo_move()
                max_eval = max(max_eval, eval)

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval
        else:
            min_eval = float('inf') # agentul min
            for r,c in possible_moves:
                env.make_move(r,c)
                eval = self.minimax(env, depth-1, alpha, beta, True) # il cheama pe cel MAX
                env.undo_move()
                min_eval = min(min_eval, eval)

                beta = min(beta, eval)
                if beta<=alpha:
                    break
                
            return min_eval

    def evaluate(self, env):
        board_bytes = env.board.tobytes('C') # sir de biti
        if board_bytes in self.memo: # daca e in cache return instant
            return self.memo[board_bytes]

        if env.move_count == 0: # prima mutare nu se evalueaza
            return 0

        # mapping
        my_board = np.zeros_like(env.board, dtype=np.int32)
        my_board[env.board == self.player_id] = 1
        my_board[env.board == -self.player_id] = 2

        op_board = np.zeros_like(env.board, dtype=np.int32)
        op_board[env.board == -self.player_id] = 1
        op_board[env.board == self.player_id] = 2

        #padding pentru pattern corect
        my_padded = np.pad(my_board, pad_width=1, constant_values=2)
        op_padded = np.pad(op_board, pad_width=1, constant_values=2)

        # separa linii coloane si diagonale ina ferestre de cate 7
        # pe un rand 14 ferestre
        def get_board_score(B):
            # Orizontal
            # tot randul mai putin ultimele 6
            # elimina 1 de la inceput 5 de la final etc
            W_horiz = (
                B[:, :-6] + B[:, 1:-5]*3 + B[:, 2:-4]*9 + 
                B[:, 3:-3]*27 + B[:, 4:-2]*81 + B[:, 5:-1]*243 + B[:, 6:]*729
            )
            # Vertical
            W_vert = (
                B[:-6, :] + B[1:-5, :]*3 + B[2:-4, :]*9 + 
                B[3:-3, :]*27 + B[4:-2, :]*81 + B[5:-1, :]*243 + B[6:, :]*729
            )
            # Diagonala principală
            W_diag1 = (
                B[:-6, :-6] + B[1:-5, 1:-5]*3 + B[2:-4, 2:-4]*9 + 
                B[3:-3, 3:-3]*27 + B[4:-2, 4:-2]*81 + B[5:-1, 5:-1]*243 + B[6:, 6:]*729
            )
            # Diagonala secundară
            W_diag2 = (
                B[:-6, 6:] + B[1:-5, 5:-1]*3 + B[2:-4, 4:-2]*9 + 
                B[3:-3, 3:-3]*27 + B[4:-2, 2:-4]*81 + B[5:-1, 1:-5]*243 + B[6:, :-6]*729
            )
            
            # scorul pe toata tabla e suma scorurilor pe linii coloane si diagonale care este suma scorurilor tuturor patternurilor din fiecare linie coloana si diagonala
            return int(
                np.sum(_SCORE_TABLE[W_horiz]) + 
                np.sum(_SCORE_TABLE[W_vert]) + 
                np.sum(_SCORE_TABLE[W_diag1]) + 
                np.sum(_SCORE_TABLE[W_diag2])
            )

        my_score = get_board_score(my_padded)
        op_score = get_board_score(op_padded)
        final_score = my_score - op_score

        # salveaza configuratia curenta si cea opusa
        self.memo[board_bytes] = final_score
        self.memo[(-env.board).tobytes('C')] = -final_score
        
        return final_score
    

class FastHeuristicAgent:
    def __init__(self, player_id):
        self.player_id = player_id

    def get_action(self, engine):

        possible_moves = list(engine.valid_moves) # iau mutarile posibile

        opponent_id = -self.player_id
        axes = [(0,1), (1,0), (1,1), (1,-1)]
        
        blocking_move = None

        # ray casting
        for (r, c) in possible_moves:
            for dr, dc in axes:
                my_count = 0
                op_count = 0
                
                # verific piese proprii
                for sign in [1, -1]:
                    for i in range(1, 5):
                        nr, nc = r + sign*dr*i, c + sign*dc*i
                        if 0 <= nr < engine.size and 0 <= nc < engine.size and engine.board[nr, nc] == self.player_id:
                            my_count += 1
                        else:
                            break
                
                # daca am gasit 4 inseamna ca daca as pune aici fac 5 si castig instant
                if my_count >= 4:
                    return (r, c)
                
                # verific piesele adversarului
                for sign in [1, -1]:
                    for i in range(1, 5):
                        nr, nc = r + sign*dr*i, c + sign*dc*i
                        if 0 <= nr < engine.size and 0 <= nc < engine.size and engine.board[nr, nc] == opponent_id:
                            op_count += 1
                        else:
                            break
                
                # daca adversasrul are 4 inseamna ca urmeaza sa castige, salvez mutarea in caz ca trebuie sa il blochez
                # dar poate gasesc o mutare cu care castig eu
                if op_count >= 4:
                    blocking_move = (r, c)

        # daca s a terminat nu am gasit mutare castigatoare pentru mine deci il blochez daca el are mutare castigatoare
        if blocking_move:
            return blocking_move

        # daca mutarea nu e urgenta muta random
        if possible_moves:
            return random.choice(possible_moves)
        else:
            return None