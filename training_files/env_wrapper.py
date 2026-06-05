import gymnasium as gym
from gymnasium import spaces
from GameData import GameEngine
import numpy as np
import random
import os
import time

# folosesc pattern recognition din minimax - acolo scorurile sunt foarte mari
# aici le trec print NN asa ca trebuie scalate

MAX_DELTA = 5000.0
SCALE = 0.05 / MAX_DELTA

class GomokuGymEnv(gym.Env):
    def __init__(self, opponent_agent = None, use_shaping=True): # daca sa faca reward shaping sau nu
        super().__init__() 
        
        # construiesc env ul care e identic cu cel pentru minimax
        self.size = 20
        self.engine = GameEngine(size = self.size)
        
        self.opponent_agent = opponent_agent
        self.use_shaping = use_shaping

        # spatiul de actiuni este discret de la 0 la 399 - toate spatiile in care pot pune
        self.action_space = spaces.Discrete(self.size*self.size)

        #spatiul de observations este "continuu" 0 daca e spatiu gol sau 1 daca e spatiu oucpat/valid
        # shape este 3 - 3 canale
            # piesele agentului; piesele advsersarului; frontiera cu mutari valide
        self.observation_space = spaces.Box(
            low = 0, high=1, shape = (4, self.size, self.size), dtype = np.float32
        )

        #agentul joaca cu id 1
        self.agent_player = 1

        #heartbeat logging
        self.pid = os.getpid()

        #data agumentation - definirea indicilor de simetrie (0,3 rotire) (4-7 oglindire si rotire)
        self.action_to_canonical = {}
        base_indices = np.arange(self.size * self.size).reshape((self.size, self.size))

        for sym in range(8):
            transformed = base_indices.copy()
            if sym >= 4:
                transformed = np.fliplr(transformed) # Oglindire
            transformed = np.rot90(transformed, k=sym % 4) # Rotatie 0, 90, 180, 270

        # Maparea inversa: index_transformat -> index_canonic
            self.action_to_canonical[sym] = transformed.flatten()

        self.current_symmetry = 0

    """ resetarea env-ului cu prob 50/50 sa inceapa agentul sau adversarul """
    def reset(self, seed = None, options = None):
        super().reset(seed = seed, options = options)
        self.current_symmetry = random.randint(0, 7) # se alege o transformare random - joaca un meci intr o anumita transformare din grupul de simetrie
        self.engine = GameEngine(size = self.size) # un nou env

        if random.random() < 0.5:
            self.agent_player = 1
        else:
            self.agent_player = -1 # adversarul incepe - deci trebuie ca envul sa fie pornit cu o mutare deja facuta
            opp_move = self._get_opponent_action()
            self.engine.make_move(opp_move[0], opp_move[1])

        return self.get_obs(), {}

    """ ce se vede in env = canalele care intra in NN
        agentul vede mereu piesele lui """
    def get_obs(self):
        # mereu din perspectiva agentului
        my_pieces = (self.engine.board == self.agent_player).astype(np.float32) # canal 1
        enemy_pieces = (self.engine.board == -self.agent_player).astype(np.float32) # canal 2

        # spatiile ocupate cu 1 sunt spatii in care se poate pune o piesa
        frontier = np.zeros((self.size, self.size), dtype = np.float32) # canal 3
        threat_map = np.zeros((self.size, self.size), dtype = np.float32)

        enemy_id = -self.agent_player
        for (r, c) in self.engine.valid_moves:
            frontier[r, c] = 1.0
            
            axes = [(0,1),(1,0),(1,1),(1,-1)]
            for dr, dc in axes:
                count = 0
                for sign in [1, -1]:
                    for i in range(1, 5):
                        nr, nc = r + sign*dr*i, c + sign*dc*i
                        if 0 <= nr < self.size and 0 <= nc < self.size and self.engine.board[nr, nc] == enemy_id:
                            count += 1
                        else:
                            break
                if count > 0: # heat map pentru cat de urgenta este mutarea - 0.25 (pt 1 piesa), 0.5 (pt 2), 0.75 (pt 3), 1.0 (pt 4+)
                    threat_map[r,c] = max(threat_map[r, c], min(count/4.0, 1.0))
        
        # toate canalele adunate
        obs = np.stack([my_pieces, enemy_pieces, frontier, threat_map], axis=0)

        # iau spatiul de observare si ii aplica aceeasi transformare
        if self.current_symmetry > 0:
            new_obs = np.zeros_like(obs)
            for i in range(obs.shape[0]):
                plane = obs[i]
                if self.current_symmetry >= 4:
                    plane = np.fliplr(plane)
                new_obs[i] = np.rot90(plane, k=self.current_symmetry % 4)
            return new_obs

        return obs
    
    """masca de actiuni valide - folosita doar de MaskablePPO Algo"""
    def action_masks(self):
        # o lista de 400 de zero uri
        canonical_mask = np.zeros(self.size * self.size, dtype=bool) # masca reala
        for (r, c) in self.engine.valid_moves:
            canonical_mask[r * self.size + c] = True

        # parcurg lista de mutari valide ca rand coloana - matrice
        # si pun in lista mea mutarile valide
        transformed_mask = canonical_mask[self.action_to_canonical[self.current_symmetry]] # aplic simetria pe masca
        return transformed_mask
    
    """ mediul are 2 agenti - cumva trebuie ca mediul sa faca mutarea
        agentul nu stie sa isi faca decat mutarea lui - mediul / oponentul trebuie sa raspunda """
    def _get_opponent_action(self):
        # faza 1 - agentul nu are adversar - se bate cu random - deci mediul raspunde random
        if self.opponent_agent is None:
            return random.choice(list(self.engine.valid_moves))

        # daca adversarul are un id
        if hasattr(self.opponent_agent, 'player_id'):
            self.opponent_agent.player_id = -self.agent_player # id ul lui devine inversul agentului
        return self.opponent_agent.get_action(self.engine)
    
    """"""
    def step(self, action):
        # actiunea e un numar intre 0-399
        canonical_action = self.action_to_canonical[self.current_symmetry][action]
        row = canonical_action // self.size
        col = canonical_action % self.size

        # scorul inainte de mutare
        score_before = self.engine.evaluate_for_player(self.agent_player)
        
        # se face mutarea
        success, msg = self.engine.make_move(row, col)

        if not success: # teoretic imposibil pentru ca am masca
            return self.get_obs(), -10.0, True, False, {"msg": "Invalid move", "pid": self.pid}
        
        # daca castiga agentul sau e remiza
        if self.engine.winner == self.agent_player:
            return self.get_obs(), 1.0, True, False, {"msg": "Agent won", "pid": self.pid}
        if len(self.engine.valid_moves) == 0:
            return self.get_obs(), 0.0, True, False, {"msg": "Draw", "pid": self.pid}
        
        shaped_reward = 0.0
        if self.use_shaping:
            score_after = self.engine.evaluate_for_player(self.agent_player)
            delta = score_after - score_before
            shaped_reward = np.tanh(delta / 500.0) *0.1 # tanh are valori in [-1, 1]; /500 normalizat; *0.1 pentru a micsora scorul din (0,1) in si mai mic
        
        opp_move = self._get_opponent_action()
        self.engine.make_move(opp_move[0], opp_move[1])

        # daca castiga adversarul sau e remiza
        if self.engine.winner == -self.agent_player:
            return self.get_obs(), -1.0, True, False, {"msg": "Opponent won", "pid": self.pid}
        if len(self.engine.valid_moves) == 0:
            return self.get_obs(), 0.0, True, False, {"msg": "Draw", "pid": self.pid}

        return self.get_obs(), shaped_reward, False, False, {"pid": self.pid}

    def render(self):
        print(self.engine.board)
        print("-" * 30)
