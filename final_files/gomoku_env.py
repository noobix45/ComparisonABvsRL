import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from CnnPolicy import CustomCNN

class GameEngine:
    def __init__(self, size=20, players=["X","O"]):
        self.size = size
        self.board = np.zeros((size,size), dtype=np.int8)
        self.players = players
        self.current_player = 1
        self.move_count = 0
        self.history = []
        self.winner = 0
        self.winning_line = None
        self.valid_moves = set()
        
        if self.size % 2 == 0:
            c1, c2 = self.size // 2 - 1, self.size // 2
            self.valid_moves.update([(c1,c1), (c1,c2), (c2,c1), (c2,c2)])
        else:
            self.valid_moves.add((self.size//2, self.size//2))

    def is_center(self, r, c):
        if self.size % 2 == 0:
            center_indices = [self.size // 2 - 1, self.size//2]
            return r in center_indices and c in center_indices
        return r == self.size // 2 and c == self.size // 2

    def is_adjacent(self, r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] != 0:
                return True
        return False
        
    def make_move(self, r, c):
        if not(0 <= r < self.size and 0 <= c < self.size): return False, "Outside board"
        if self.board[r, c] != 0: return False, "Occupied"
        
        if self.move_count == 0:
            if not self.is_center(r,c): return False, "First move must be center"
        elif not self.is_adjacent(r,c): return False, "Move must be adjacent"
            
        old_valid_moves = self.valid_moves.copy()
        self.board[r, c] = self.current_player

        if self.move_count == 0:
            self.valid_moves.clear()
        elif (r,c) in self.valid_moves:
            self.valid_moves.remove((r,c))

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == 0:
                self.valid_moves.add((nr,nc))

        self.move_count += 1
        self.history.append((r, c, self.current_player, old_valid_moves))

        is_win, line_coords = self.check_win(r,c)
        if is_win:
            self.winner = self.current_player
            self.winning_line = line_coords
            return True, "Game over"

        self.current_player *= -1
        return True, "OK"
    
    def undo_move(self):
        if not self.history: return False
        r, c, player, old_moves = self.history.pop()
        self.board[r,c] = 0
        self.valid_moves = old_moves
        self.move_count -= 1
        self.current_player = player
        self.winner = 0
        self.winning_line = None
        return True
    
    def check_win(self, r, c):
        symbol = self.board[r, c]
        if symbol == 0: return False, None
        
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            start_r, start_c, end_r, end_c = r, c, r, c
            
            for i in range(1,5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == symbol:
                    count += 1; end_r, end_c = nr, nc
                else: break

            for i in range(1,5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == symbol:
                    count += 1; start_r, start_c = nr, nc
                else: break
                    
            if count >= 5: return True, ((start_r, start_c), (end_r, end_c))
        return False, None

# warapper pentru ppo ca sa converteasca mutarea in tuplu pentru engine
class PPOOpponentAgent:
    def __init__(self, model_path, deterministic=True):
        custom_objects = {"features_extractor_class": CustomCNN}
        self.model = MaskablePPO.load(model_path, device='cpu', custom_objects=custom_objects)
        self.player_id = 1
        self.deterministic = deterministic

    def get_action(self, engine):
        # construirea spatiului de observare pe care functioneaza agnetul - cele 4 canale
        my = (engine.board == self.player_id).astype(np.float32)
        enemy = (engine.board == -self.player_id).astype(np.float32)
        frontier = np.zeros((engine.size, engine.size), dtype=np.float32)
        threat_map = np.zeros((engine.size, engine.size), dtype=np.float32)

        enemy_id = -self.player_id
        for (r, c) in engine.valid_moves:
            frontier[r, c] = 1.0
            for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                count = 0
                for sign in [1, -1]:
                    for i in range(1, 5):
                        nr, nc = r + sign*dr*i, c + sign*dc*i
                        if 0 <= nr < engine.size and 0 <= nc < engine.size and engine.board[nr, nc] == enemy_id:
                            count += 1
                        else: break
                if count > 0:
                    threat_map[r,c] = max(threat_map[r, c], min(count / 4.0, 1.0))

        obs = np.stack([my, enemy, frontier, threat_map], axis=0)[np.newaxis]

        mask = np.zeros(engine.size * engine.size, dtype=bool)
        for(r, c) in engine.valid_moves:
            mask[r * engine.size + c] = True
            
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        action = int(np.array(action).flatten()[0])
        return (action // engine.size, action % engine.size) # //size %size imi da tuplul cu linia si coloana mutarii