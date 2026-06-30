import time
import tracemalloc
import os
import numpy as np
import matplotlib.pyplot as plt
from gomoku_env import GameEngine, PPOOpponentAgent
from minimax_agent import MinimaxAgent2
import random

def get_disk_size(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

def benchmark_games(n_games=5, minimax_depth=2, ppo_model_path="remove.zip", deterministic = True, n_random_moves = 1):
    print(f"Începem benchmark: PPO vs Minimax(Depth {minimax_depth}) | Jocuri: {n_games}")
    
    agent_ppo = PPOOpponentAgent(ppo_model_path, deterministic=deterministic)
    agent_mm = MinimaxAgent2(player_id=-1, max_depth=minimax_depth)
    
    wins = {"PPO": 0, "Minimax": 0, "Draw": 0}

    for i in range(n_games):
        engine = GameEngine(size=20)
        
        ppo_id = 1 if i % 2 == 0 else -1
        mm_id = -ppo_id
        agent_ppo.player_id = ppo_id
        agent_mm.player_id = mm_id
        
        
        while engine.winner == 0 and len(engine.valid_moves) > 0:
            current_id = engine.current_player

            #### prima mutare se face random ca sa sparga din determnism chiar si in cazul in care agentul ppo joaca determinist
            if deterministic:
                if engine.move_count < n_random_moves:
                    move = random.choice(list(engine.valid_moves))
                    engine.make_move(move[0], move[1])
                    continue
            
            if current_id == ppo_id:
                move = agent_ppo.get_action(engine)
            else:
                move = agent_mm.get_action(engine)
            
            
                
            engine.make_move(move[0], move[1])
            
        if engine.winner == ppo_id: wins["PPO"] += 1
        elif engine.winner == mm_id: wins["Minimax"] += 1
        else: wins["Draw"] += 1

        print(f"game{i} done")

    print(f"for {ppo_model_path} deterministic = {deterministic} n_random_moves={n_random_moves}")
    print(f"\n[FINAL] Rezultate: PPO {wins['PPO']} - {wins['Minimax']} Minimax ({wins['Draw']} Remize)")

    

if __name__ == "__main__":
    # deterministic true - face n random moves
    # deterministic false - n_random_moves e ignorat
    benchmark_games(n_games=20, minimax_depth=3, ppo_model_path="../move/models/phase23/snapshot_98800000", deterministic=True, n_random_moves=1)
