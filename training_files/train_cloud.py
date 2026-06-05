# cell 5 train_cloud
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import random
import json
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from CnnPolicy import CustomCNN
from env_wrapper import GomokuGymEnv
from GameData import MinimaxAgent2, GameEngine
from GameData import FastHeuristicAgent

import wandb
from wandb.integration.sb3 import WandbCallback
# from kaggle_secrets import UserSecretsClient

# # Setup pentru autentificare W&B
# user_secrets = UserSecretsClient()
# wandb_key = user_secrets.get_secret("WANDB_API_KEY")
# wandb.login(key=wandb_key)

def make_env(use_shaping = False):
    def _init():
        env = GomokuGymEnv(opponent_agent=None, use_shaping = use_shaping) # o instanta de environment cu adversar random (agent random)
        return Monitor(env)
    return _init

""" antrenare paralela in 8 env-uri simultan"""
def train_phase1(save_path, total_timesteps = 5_000_000, n_envs = 8):
    env = SubprocVecEnv([make_env(use_shaping=False) for _ in range(n_envs)])

    eval_env = SubprocVecEnv([make_env(use_shaping=False)]) # un singur env de evaluare

    # aplicarea CNN ului custom pe 4 canale cu iesire 512 de features (increased from older versions)
    policy_kwargs = dict(
        features_extractor_class = CustomCNN,
        features_extractor_kwargs = dict(features_dim=512)
    )

    model = MaskablePPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4, # cat de brusc se ajusteaza NN
        n_steps=2048, # dupa cati pasi se ajusteaza NN - per env => 2048*8 total
        n_epochs = 5,
        batch_size=256, # cate mutari sunt analizate simultan la ajustarea NN ului din cele 2048*8 de mai sus
        ent_coef=0.01, # coeficient de entropie - explorare
        clip_range = 0.1, # limiteaza update uri mai mari de 0.1*100%=10%
        vf_coef = 0.5, # coeficientul de invatare al criticului
        # max_grad_norm = 0.5,
        verbose = 1, 
        tensorboard_log="./logs/"
    )
    # fun fact: da shuffle la 2048*8 date si le trce de <n_epochs> ori prin NN in batchuri de <batch_size>

    run = wandb.init(
        project = "gomoku_phase1",
        sync_tensorboard = True,
        save_code = True
    )
    model.tensorboard_log = f"/kaggle/working/logs/phase1_{run.id}"


    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=save_path,
        eval_freq=100_000 // n_envs, # cat de des vreau sa fac evaluare (dupa cate mutari) // numarul de env uri - pentru ca pasii se fac in paralel
        n_eval_episodes=50, # evaluare timp de 50 de meciuri
        deterministic=True, # vreau ca agentul sa joace la potential maxim in aceste 50 de meciuri
        verbose=1
    )

    wandb_callback = WandbCallback(
        model_save_path=f"{save_path}/wandb_models",
        verbose=2,
    )
    callbacks = CallbackList([eval_callback, wandb_callback])

    model.learn(total_timesteps = total_timesteps, callback=callbacks) # antrenarea
    model.save(f"{save_path}/phase1_final") # si salvarea
    run.finish()
    return model

""" continuarea faza 1 de la un checkpoint"""
def resume_phase1(checkpoint_path, save_path, additional_timesteps = 2_000_000, n_envs=8):
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    eval_env = SubprocVecEnv([make_env()])

    model = MaskablePPO.load(checkpoint_path, env = env)

    # aceeasi parametri (cei importanti)
    model.learning_rate = 1e-4
    model.clip_range = lambda _: 0.1
    model.ent_coef = 0.005
    model.batch_size = 256

    # acelasi callback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=save_path,
        eval_freq=100_000 // n_envs,
        n_eval_episodes=50,
        deterministic=True,
        verbose=1
    )

    model.learn(
        total_timesteps=additional_timesteps,
        callback=eval_callback,
        reset_num_timesteps=False # continua de unde a ramas
    )
    model.save(f"{save_path}/phase1_resumed")
    return model

""" for debugging to checks if processes are still alive"""
class WandbHeartbeatCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.pid_steps = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            pid = str(info.get("pid", "unknown"))
            if pid not in self.pid_steps:
                self.pid_steps[pid] = 0
            self.pid_steps[pid] += 1

        if self.n_calls % 256 == 0: 
            log_dict = {}
            for pid, steps in self.pid_steps.items():
                log_dict[f"heartbeat/pid_{pid}_steps"] = steps
            wandb.log(log_dict)
            
        return True

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

            if wandb.run is not None and os.path.exists(zip_path):
                try:
                    artifact = wandb.Artifact(name=f"snapshot-{self.num_timesteps}", type="model")
                    artifact.add_file(zip_path) 
                    wandb.log_artifact(artifact)
                    if self.verbose:
                        print(f"Artifact snapshot-{self.num_timesteps} trimis cu succes către W&B!")
                except Exception as e:
                    print(f"Eroare la trimiterea artefactului către W&B: {e}")

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
    def __init__(self, model):
        self.model = model
        self.player_id = -1 # default dar se seteaza in env in functie de perspectiva

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
        action, _ = self.model.predict(obs, action_masks = mask, deterministic = False) # actiunea returnata

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
        rand_val = random.random()
        self.use_shaping = self.default_use_shaping

        if rand_val < 0.025:
            self.opponent_agent = FastHeuristicAgent(player_id=-1)
        elif rand_val < 0.05:
            self.opponent_agent = None
        elif rand_val < 0.075: #2.5% minimax d1
            self.minimax_d1.player_id=-1
            self.opponent_agent = self.minimax_d1
        elif rand_val < 0.15: #7.5% minimax d2
            self.minimax_d2.player_id = -1
            self.opponent_agent = self.minimax_d2
        else: #80% self-play
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
        # intai resetez agentul adversar si apoi resetz efectiv tabla din parent env
        return super().reset(seed=seed, options=options)


def _load_ppo_agent(path):
    model = MaskablePPO.load(path, device = 'cpu')
    return PPOOpponentAgent(model)

""" antrenare in self-play cu opponent pool"""
def train_phase2(phase1_checkpoint, save_path, total_timesteps=10_000_000, n_envs=8, extra_pool_models = None):

    snapshot_dir = f"{save_path}/snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

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
        "learning_rate": 1.0e-5,
        "batch_size": 512,
        "ent_coef": 0.025,
        "target_kl": 0.03,
        "clip_range": 0.05,
        "vf_coef": 0.75 # critic mai puternic
    }
    model = MaskablePPO.load(phase1_checkpoint, env = env, custom_objects=custom_objects) # incarcarea modelului de start - cel care bate random - cel care se antreneaza

    heartbeat_callback = WandbHeartbeatCallback()

    pool_callback = OpponentPoolCallback(
        pool_file = pool_file,
        snapshot_dir=snapshot_dir,
        snapshot_freq=200_000,
        win_threshold=0.55,
        verbose=1
    )

    #logare cu wandb - antrenarea in cloud
    run = wandb.init(
        project = "gomoku_self_play",
        sync_tensorboard = True,
        save_code = True
    )
    model.tensorboard_log = f"/kaggle/working/logs/{run.id}"

    wandb_callback = WandbCallback(
        model_save_path=f"{save_path}/wandb_models",
        verbose=2,
    )

    wandb_callback.log_model = True

    callbacks = CallbackList([pool_callback, wandb_callback, heartbeat_callback])

    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)
    model.save(f"{save_path}/phase24_final")
    run.finish() # inchide sesiune de wandb
    
    return model

""" benchmark extern pipe line ului pentur validare impotriva minimax"""
def run_benchmark(model_path, depth=1, n_games=20):
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

    # KAGGLE_WORK_DIR = "/kaggle/working/"

# ## com/ decom this for phase 1
#     # PHASE1_SAVE_DIR = os.path.join(KAGGLE_WORK_DIR, "models/phase1")
#     # os.makedirs(PHASE1_SAVE_DIR, exist_ok=True)
# #############3

# ### com/decom this for phase 2
#     #### rainbowkl
#     DATASET_DIR = "/kaggle/input/datasets/catalinionita/gomoku-phase23-models" # folderul de input dataset din kaggle
# ### resume: snapshot_98800000
# ### pool: proud_dragon_final, snapshot_97000000, 
#     # kaggle dezarhiveaza automat
#     UNZIPPED_RESUME = os.path.join(DATASET_DIR, "snapshot_98800000")
#     UNZIPPED_EXTRA_1 = os.path.join(DATASET_DIR, "proud_dragon_final")
#     UNZIPPED_EXTRA_2 = os.path.join(DATASET_DIR, "snapshot_97000000")
#     # UNZIPPED_EXTRA_3 = os.path.join(DATASET_DIR, "")
#     # UNZIPPED_EXTRA_4 = os.path.join(DATASET_DIR, "")
#     # UNZIPPED_EXTRA_5 = os.path.join(DATASET_DIR, "")
#     # UNZIPPED_EXTRA_6 = os.path.join(DATASET_DIR, "")

#     REPACK_DIR = os.path.join(KAGGLE_WORK_DIR, "repacked_models") # directorul dezarhivat
#     os.makedirs(REPACK_DIR, exist_ok=True)

#     RESUME_MODEL_ZIP = os.path.join(REPACK_DIR, "snapshot_98800000")
#     EXTRA_1_ZIP = os.path.join(REPACK_DIR, "proud_dragon_final")
#     EXTRA_2_ZIP = os.path.join(REPACK_DIR, "snapshot_97000000")
#     # EXTRA_3_ZIP = os.path.join(REPACK_DIR, "")
#     # EXTRA_4_ZIP = os.path.join(REPACK_DIR, "")
#     # EXTRA_5_ZIP = os.path.join(REPACK_DIR, "")
#     # EXTRA_6_ZIP = os.path.join(REPACK_DIR, "")

#     print("Rearhivăm modelele pentru SB3...")
#     for source, dest in zip(
#         [UNZIPPED_RESUME, UNZIPPED_EXTRA_1, UNZIPPED_EXTRA_2], 
#         [RESUME_MODEL_ZIP, EXTRA_1_ZIP, EXTRA_2_ZIP]
#     ):
#         if os.path.exists(source):
#             base_name_without_zip = dest.replace('.zip', '')
#             shutil.make_archive(base_name_without_zip, 'zip', source)
#             print(f"Rearhivat corect: {dest}")
#         else:
#             print(f"ATENȚIE: Nu am găsit folderul {source}")

#     PHASE24_SAVE_DIR = os.path.join(KAGGLE_WORK_DIR, "models/phase24")
#     os.makedirs(PHASE24_SAVE_DIR, exist_ok=True)
#####

    # Faza 1
    # print(f"Start train Faza 1. Modelul se va salva in: {PHASE1_SAVE_DIR}")
    # train_phase1(save_path=PHASE1_SAVE_DIR, total_timesteps=3_000_000)
    
    # Benchmark dupa faza 1
    run_benchmark("./models/phase24/remove1038", depth=3) 
    
    #### Faza 2 — pornit manual cand faza 1 e decenta
    # print("Start train phase24")
    # train_phase2(
    #     phase1_checkpoint=RESUME_MODEL_ZIP,
    #     save_path=PHASE24_SAVE_DIR,
    #     total_timesteps=5_000_000,
    #     n_envs = 8,
    #     extra_pool_models=[EXTRA_1_ZIP, EXTRA_2_ZIP]   
    # )