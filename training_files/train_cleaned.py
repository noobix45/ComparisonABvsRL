# cell 5 train_cloud
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
torch.set_num_threads(1)

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import random
import json
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from env_wrapper import GomokuGymEnv
from GameData import MinimaxAgent2, GameEngine
from GameData import FastHeuristicAgent


def make_env(use_shaping = False):
    def _init():
        env = GomokuGymEnv(opponent_agent=None, use_shaping = use_shaping) # o instanta de environment cu adversar random (agent random)
        return Monitor(env)
    return _init

""" definirea unui callback pentru opponent pool"""
class OpponentPoolCallback(BaseCallback):
    def __init__(
            self,
            pool_file, # fisierul json in care o sa fie path urile catre adversari
            snapshot_dir, # directorul de checkpointuri
            snapshot_freq = 200_000, # cat de des se face un checkpoint
            win_threshold = 0.55, # scorul necesar ca sa treaca mai departe
            max_pool_size = 15, # limita de modele incarcarcate in ram la un moment dat
            decay_factor = 0.75, # factor de scadere exponentiala
            verbose=0
        ):
        super().__init__(verbose)
        self.pool_file = pool_file
        self.snapshot_dir = snapshot_dir
        self.snapshot_freq = snapshot_freq
        self.win_threshold = win_threshold
        self.max_pool_size = max_pool_size
        self.decay_factor = decay_factor

    """ la fiecare freq steps, face un checkpoint si evalueaza agentul, 
        daca e suficient de bun, il pun in lista de adversari, daca nu ramane doar checkpoint"""
    def _on_step(self):
        if self.num_timesteps % self.snapshot_freq == 0: # la fiecare <snapshot_freq> timesteps se face un checkpoint
            path = f"{self.snapshot_dir}/snapshot_{self.num_timesteps}"
            self.model.save(path)
            
            import time
            zip_path = f"{path}.zip"
            # Așteptăm maximum 5 secunde dacă fișierul nu a apărut încă pe disc (I/O lag în cloud)
            for _ in range(5):
                if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
                    break
                time.sleep(1)

            # priuma evaluare trebuie sa treaca minimal de fast agent - dac nu nu il bag oricum in pool
            win_rate_fast = self._eval_vs_fast_agent(n_games = 10)

            if win_rate_fast < 0.50:
                if self.verbose:
                    print(f"Snapshot respins (Stage 1). Win rate vs Fast: {win_rate_fast:.2f}")
                return  True
            
            with open(self.pool_file, "r") as f:
                pool_data = json.load(f)

            win_rate_pool = self._eval_vs_pool(pool_data["paths"], pool_data["weights"], n_games = 20)

            if win_rate_pool > self.win_threshold:
                # daca e suficient de bun ajunge si in lista de adversari
                pool_data["paths"].append(path)

                #elimin modelele prea vechi - coada fifo - primul snapshot pus e primul care e sters
                if len(pool_data["paths"]) > self.max_pool_size:
                    pool_data["paths"].pop(0)

                # recalculare de ponderi cu exponential decay - modele mai nnoi sunt mai bune - au sansa mai mare sa fie alese
                n_models = len(pool_data["paths"]) # total
                # raw_weights = [self.decay_factor ** (n_models - 1 - i) for i in range(n_models)] #ultimul model (cel mai nou) primeste decay_factor^0 = 1; cel anterior decay_factor^1

                # normalizare dse probabilitati
                # total_weight = sum(raw_weights)
                pool_data["weights"] = [1.0 / n_models] * n_models

                #rescriu modificarea in json
                temp_file = self.pool_file + ".tmp"
                with open(temp_file, "w") as f:
                    json.dump(pool_data, f)
                os.replace(temp_file, self.pool_file)
                
                if self.verbose:
                    print(f"Snapshot aprobat! FastWR: {win_rate_fast:.2f} | PoolWR: {win_rate_pool:.2f}. Pool size: {n_models}")
            else:
                if self.verbose:
                    print(f"Snapshot respins (Stage 2). FastWR: {win_rate_fast:.2f} | PoolWR: {win_rate_pool:.2f}")
        return True
    
    def _eval_vs_fast_agent(self, n_games = 30):
        wins = 0
        heuristic_agent = FastHeuristicAgent(player_id=-1)

        for i in range(n_games):

            env = GomokuGymEnv(opponent_agent=heuristic_agent, use_shaping=False)
            obs, _ = env.reset() # fac un nou env

            done = False
            while not done: # flow ul jocului masca agentul muta
                mask = env.action_masks()
                action, _ = self.model.predict(obs, action_masks = mask, deterministic=True)
                obs, reward, done, _, _ = env.step(int(action)) # envul raspunde prin fast agent

            if reward > 0:
                wins += 1

        return wins / n_games
    
    def _eval_vs_pool(self, paths, weights, n_games=50):
        wins = 0

        for _ in range(n_games):
            opponent_path = random.choices(paths, weights=weights)[0] # unul singurl din lista
            opponent = None if opponent_path is None else _load_ppo_agent(opponent_path)
            
            # creeaza env ul 
            env = GomokuGymEnv(opponent_agent=opponent, use_shaping=False)
            obs, _ = env.reset() # responsabil pentru init env
            done = False

            while not done:
                mask = env.action_masks()
                action, _ = self.model.predict(obs, action_masks = mask, deterministic=True) # mutarea agentului 
                obs, reward, done, _, _ = env.step(int(action)) # se efectueaza mutarea si envul raspunde
            
            if reward > 0:
                wins+=1

        return wins/n_games # rata de castig
    
""" definita adversarului (wrapper pentru ca env sa raspunda agentului)"""
class PPOOpponentAgent:
    def __init__(self, model, deterministic = False):
        self.model = model
        self.player_id = -1 # default dar se seteaza in env in functie de perspectiva
        self.deterministic = deterministic

    def get_action(self, engine):
        # cele 4 canale cu perspectiva
        my = (engine.board == self.player_id).astype(np.float32)
        enemy = (engine.board == -self.player_id).astype(np.float32)
        frontier = np.zeros((engine.size, engine.size), dtype=np.float32)
        threat_map = np.zeros((engine.size, engine.size), dtype=np.float32)

        enemy_id = -self.player_id
        for (r, c) in engine.valid_moves:
            frontier[r, c] = 1.0
            
            axes = [(0,1),(1,0),(1,1),(1,-1)] #orizontala verticala si diagonalele
            for dr, dc in axes:
                count = 0
                for sign in [1, -1]: # verific axele in ambele sensuri
                    for i in range(1, 5):
                        nr, nc = r + sign*dr*i, c + sign*dc*i
                        if 0 <= nr < engine.size and 0 <= nc < engine.size and engine.board[nr, nc] == enemy_id:
                            count += 1
                        else:
                            break
                if count > 0: # heat map pentru cat de urgenta este mutarea - 0.25 (pt 1 piesa), 0.5 (pt 2), 0.75 (pt 3), 1.0 (pt 4+)
                    threat_map[r,c] = max(threat_map[r, c], min(count / 4.0, 1.0))

        obs = np.stack([my, enemy, frontier, threat_map], axis=0)[np.newaxis] # creez spatiul de observatii

        mask = np.zeros(engine.size * engine.size, dtype=bool) # masca necesara pentru model
        for(r, c) in engine.valid_moves:
            mask[r*engine.size+c]=True
        action, _ = self.model.predict(obs, action_masks = mask, deterministic = self.deterministic) # actiunea returnata

        action = int(np.array(action).flatten()[0])
        return (action // engine.size, action % engine.size) # linia si coloana la care muta - env va muta acolo

class DynamicPoolEnv(GomokuGymEnv):
    def __init__(self,pool_file, use_shaping = False):
        super().__init__(opponent_agent=None, use_shaping=use_shaping)
        self.pool_file = pool_file
        self.model_cache = {}
        self.default_use_shaping = use_shaping
        self.shared_memo = {}
        self.minimax_d1 = MinimaxAgent2(player_id=-1, max_depth=1, shared_memo=self.shared_memo)
        self.minimax_d2 = MinimaxAgent2(player_id=-1, max_depth=2, shared_memo=self.shared_memo)

    def reset(self, seed = None, options =None):

        if len(self.shared_memo) > 500_000:
            self.shared_memo.clear()
        
        rand_val = random.random()
        self.use_shaping = self.default_use_shaping

        if rand_val < 0.025:
            self.opponent_agent = FastHeuristicAgent(player_id=-1)
        elif rand_val < 0.05:
            self.opponent_agent = None
        elif rand_val < 0.075: #2.5% minimax d1
            self.minimax_d1.player_id=-1
            self.opponent_agent = self.minimax_d1
        elif rand_val < 0.125: #5% minimax d2
            self.minimax_d2.player_id = -1
            self.opponent_agent = self.minimax_d2
        else: # self-play
            try:
                with open(self.pool_file, "r") as f: # la fiecare reset recitesc fisierul json ca sa aleg un adeverar
                    pool = json.load(f)

                #sterg din cache modelele eliminate din pool
                valid_paths = set(pool["paths"]) # toate pathurile din pool file
                keys_to_delete = [k for k in self.model_cache.keys() if k not in valid_paths] # daca sunt in cache dar nu sunt in path le sterg din cache
                for k in keys_to_delete:
                    del self.model_cache[k]

                path = random.choices(pool["paths"], weights=pool["weights"])[0]
                if path is None:
                    self.opponent_agent = None
                else:
                    # daca nu am mai intalnit modelul asta il incarc de pe disc si iltin in ram
                    if path not in self.model_cache:
                        self.model_cache[path] = _load_ppo_agent(path)
                    # il incarc din cache
                    self.opponent_agent = self.model_cache[path]
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.opponent_agent = FastHeuristicAgent(player_id=-1)
            except Exception as e:
                print(f"Eroare neașteptată la încărcarea adversarului: {e}")
                self.opponent_agent = FastHeuristicAgent(player_id=-1)

        ####aded initial state randomization
        while True:
            #apelez resetul tablei care face si prima mutare a adversarului daca le incepe
            obs, info = super().reset(seed=seed, options=options)

            n_random_moves = random.randint(0, 3)

            # e la reset deci am garantia ca sunt primele n mutari facute
            for _ in range(n_random_moves): 
                if self.engine.winner == 0 and len(self.engine.valid_moves) > 0:
                    move = random.choice(list(self.engine.valid_moves))
                    self.engine.make_move(move[0], move[1])

            if self.engine.winner == 0 and self.engine.current_player != self.agent_player: # daca dupa mutari nu e randul agentului
                opp_move = self._get_opponent_action() # pun env ul sa faca o mutare
                if opp_move:
                    self.engine.make_move(opp_move[0], opp_move[1])

            if self.engine.winner == 0:
                obs = self.get_obs()
                break

        # intai resetez agentul adversar si apoi resetz efectiv tabla din parent env
        # return super().reset(seed=seed, options=options)

        return obs, info


def _load_ppo_agent(path, deterministic = False):
    model = MaskablePPO.load(path, device = 'cpu')
    return PPOOpponentAgent(model, deterministic=deterministic)

""" antrenare in self-play cu opponent pool"""
def train_phase2(phase1_checkpoint, save_path, total_timesteps=10_000_000, n_envs=8, extra_pool_models = None):

    snapshot_dir = f"{save_path}/snapshots"
    tb_log_dir = f"{save_path}/logs"

    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # cum arata pool initial (la inceputul antrenarii)
    pool_file = f"{save_path}/pool.json"
    paths = [phase1_checkpoint] # primul op din pool este el insusi
    if extra_pool_models: # daca mai exista si alte modele specificate in main sunt adaugate in pool
        paths.extend(extra_pool_models)
    weights = [1.0/len(paths)] * len(paths) # distributie uniforma - 1/n de n ori
    initial_pool = {"paths": paths, "weights": weights}

    # fiser json impartit intre cele n_env procese
    with open(pool_file, "w") as f:
        json.dump(initial_pool, f)

    # definirea unui environment cu opponent pool
    def make_pool_env(use_shaping = False):
        def _init():
            env = DynamicPoolEnv(pool_file=pool_file, use_shaping = use_shaping)
            return Monitor(env)
        return _init
    
    env = SubprocVecEnv([make_pool_env(use_shaping=False) for _ in range(n_envs)]) # crearea env-urilor vectoriale
    custom_objects = {
        "learning_rate": 3.0e-5,
        "batch_size": 512,
        "ent_coef": 0.02,
        "target_kl": 0.05,
        "clip_range": 0.1,
        "vf_coef": 0.75 # critic mai puternic
    }
    model = MaskablePPO.load(phase1_checkpoint, env = env, custom_objects=custom_objects) # incarcarea modelului de start - cel care bate random - cel care se antreneaza

    logger = configure(tb_log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    pool_callback = OpponentPoolCallback(
        pool_file = pool_file,
        snapshot_dir=snapshot_dir,
        snapshot_freq=200_000,
        win_threshold=0.55,
        verbose=1
    )


    callbacks = CallbackList([pool_callback])

    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)
    model.save(f"{save_path}/phase31_final")
    
    return model

""" benchmark extern pipe line ului pentur validare impotriva minimax"""
def run_benchmark(model_path, depth=1, n_games=20):
    print(f"running for {model_path}")
    model = MaskablePPO.load(model_path) # modelul asta e wrapper site sa lucreze direct pe engine cu tuplu (r, c)
    ppo_agent = PPOOpponentAgent(model)
    
    wins, losses, draws = 0, 0, 0
    
    for i in range(n_games):
        engine = GameEngine(size=20)
        
        ppo_id = 1 if i % 2 == 0 else -1 # id alternant un meci incepe cineva un meci celalalt
        mm_id = -ppo_id
        
        ppo_agent.player_id = ppo_id
        mm_agent = MinimaxAgent2(player_id=mm_id, max_depth=depth)
        
        while engine.winner == 0 and len(engine.valid_moves) > 0: # cat nu castiga nimeni si mai sunt mutari valide
            if engine.current_player == ppo_id:
                move = ppo_agent.get_action(engine)
            else:
                move = mm_agent.get_action(engine)
            engine.make_move(*move)
        
        if engine.winner == ppo_id:
            wins += 1
        elif engine.winner == 0:
            draws += 1
        else: losses += 1
    
    print(f"PPO vs Minimax d{depth}: {wins}W/{losses}L/{draws}D") 
    return wins / n_games

if __name__ == "__main__":
    import shutil

    KAGGLE_WORK_DIR = "/kaggle/working/"


#     #### rainbowkl
#     #### catalinionita
    DATASET_DIR = "/kaggle/input/datasets/rainbowkl/gomoku-phase30-models" # folderul de input dataset din kaggle
### resume: phase30_final
### pool: phase27_final, phase28_final, phase29_final, 
    # kaggle dezarhiveaza automat
    UNZIPPED_RESUME = os.path.join(DATASET_DIR, "phase30_final")
    UNZIPPED_EXTRA_1 = os.path.join(DATASET_DIR, "phase27_final")
    UNZIPPED_EXTRA_2 = os.path.join(DATASET_DIR, "phase28_final")
    UNZIPPED_EXTRA_3 = os.path.join(DATASET_DIR, "phase29_final")
    # UNZIPPED_EXTRA_4 = os.path.join(DATASET_DIR, "")
    # UNZIPPED_EXTRA_5 = os.path.join(DATASET_DIR, "")
    # UNZIPPED_EXTRA_6 = os.path.join(DATASET_DIR, "")

    REPACK_DIR = os.path.join(KAGGLE_WORK_DIR, "repacked_models") # directorul dezarhivat
    os.makedirs(REPACK_DIR, exist_ok=True)

    RESUME_MODEL_ZIP = os.path.join(REPACK_DIR, "phase30_final")
    EXTRA_1_ZIP = os.path.join(REPACK_DIR, "phase27_final")
    EXTRA_2_ZIP = os.path.join(REPACK_DIR, "phase28_final")
    EXTRA_3_ZIP = os.path.join(REPACK_DIR, "phase29_final")
    # EXTRA_4_ZIP = os.path.join(REPACK_DIR, "")
    # EXTRA_5_ZIP = os.path.join(REPACK_DIR, "")
    # EXTRA_6_ZIP = os.path.join(REPACK_DIR, "")

    print("Rearhivăm modelele pentru SB3...")
    for source, dest in zip(
        [UNZIPPED_RESUME, UNZIPPED_EXTRA_1, UNZIPPED_EXTRA_2, UNZIPPED_EXTRA_3], 
        [RESUME_MODEL_ZIP, EXTRA_1_ZIP, EXTRA_2_ZIP, EXTRA_3_ZIP]
    ):
        if os.path.exists(source):
            base_name_without_zip = dest.replace('.zip', '')
            shutil.make_archive(base_name_without_zip, 'zip', source)
            print(f"Rearhivat corect: {dest}")
        else:
            print(f"ATENȚIE: Nu am găsit folderul {source}")

    PHASE31_SAVE_DIR = os.path.join(KAGGLE_WORK_DIR, "models/phase31")
    os.makedirs(PHASE31_SAVE_DIR, exist_ok=True)
#####

    # run_benchmark("./models/phase25/snapshot_1080", depth=3)
    
    #### Faza 2 — pornit manual cand faza 1 e decenta
    print("Start train phase31")
    train_phase2(
        phase1_checkpoint=RESUME_MODEL_ZIP,
        save_path=PHASE31_SAVE_DIR,
        total_timesteps=5_000_000,
        n_envs = 8,
        extra_pool_models=[EXTRA_1_ZIP, EXTRA_2_ZIP, EXTRA_3_ZIP]   
    )
