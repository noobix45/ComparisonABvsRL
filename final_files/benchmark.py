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

def benchmark_games(n_games=5, minimax_depth=2, ppo_model_path="remove.zip"):
    print(f"Începem benchmark: PPO vs Minimax(Depth {minimax_depth}) | Jocuri: {n_games}")
    
    ppo_size = get_disk_size(ppo_model_path)
    minimax_size = get_disk_size("minimax_agent.py")
    print(f"[DISK] PPO: {ppo_size:.2f} MB | Minimax: {minimax_size:.4f} MB (doar script)")
    
    agent_ppo = PPOOpponentAgent(ppo_model_path, deterministic=False)
    agent_mm = MinimaxAgent2(player_id=-1, max_depth=minimax_depth)
    
    wins = {"PPO": 0, "Minimax": 0, "Draw": 0}
    
    global_move_timeline = []
    
    # Overhead per mutare (se resetează per mutare)
    global_ppo_times, global_mm_times = [], []
    global_ppo_mems, global_mm_mems = [], []
    
    # Memorie cumulativa
    total_cumul_history = []
    ppo_cumul_history = []
    mm_cumul_history = []
    
    # Cache Minimax
    minimax_cache_sizes = []
    
    #ram adaugat de fiecare pe parcurs
    cumul_ppo_ram = 0.0
    cumul_mm_ram = 0.0

    print("\n--- REZULTATE PER JOC ---")
    print(f"{'Joc':<5} | {'Agent':<8} | {'Min Timp':<9} | {'Max Timp':<9} | {'Avg Timp':<9} | {'Mediana T.':<10} | {'Min Mem':<9} | {'Max Mem':<9} | {'Avg Mem':<9} | {'Mediana M.':<10}")
    print("-" * 115)

    tracemalloc.start() # incep sa tin cont de memoria alocata
    total_moves_counter = 0

    for i in range(n_games):
        engine = GameEngine(size=20)
        
        ppo_id = 1 if i % 2 == 0 else -1
        mm_id = -ppo_id
        agent_ppo.player_id = ppo_id
        agent_mm.player_id = mm_id
        
        times_ppo, mems_ppo = [], []
        times_mm, mems_mm = [], []
        
        while engine.winner == 0 and len(engine.valid_moves) > 0:
            current_id = engine.current_player

            #### prima mutare se face random ca sa sparga din determnism chiar si in cazul in care agentul ppo joaca determinist
            # if engine.move_count == 0:
            #     move = random.choice(list(engine.valid_moves))
            #     engine.make_move(move[0], move[1])
            #     continue
            
            mem_baseline, _ = tracemalloc.get_traced_memory() # memoria intainte de mutare
            
            # timer pentru cand dureaza o mutare
            start_t = time.perf_counter()
            if current_id == ppo_id:
                move = agent_ppo.get_action(engine)
            else:
                move = agent_mm.get_action(engine)
            end_t = time.perf_counter()
                
            time_taken = end_t - start_t
            mem_after_move, peak_during_move = tracemalloc.get_traced_memory() # starea memoriei dupa mutare
            
            # memoria adugata pentru a gandi mutarea
            move_overhead_mb = (peak_during_move - mem_baseline) / (1024 * 1024)
            if move_overhead_mb < 0: move_overhead_mb = 0.0
            
            # ram ul care ramane in memorie (cumulativ)
            net_growth_mb = (mem_after_move - mem_baseline) / (1024 * 1024)
            
            # pun valorile iun ppo sau in minimax in functie de id
            if current_id == ppo_id:
                times_ppo.append(time_taken)
                mems_ppo.append(move_overhead_mb)
                global_ppo_times.append(time_taken)
                global_ppo_mems.append(move_overhead_mb)
                
                if net_growth_mb > 0: cumul_ppo_ram += net_growth_mb
            else:
                times_mm.append(time_taken)
                mems_mm.append(move_overhead_mb)
                global_mm_times.append(time_taken)
                global_mm_mems.append(move_overhead_mb)
                
                if net_growth_mb > 0: cumul_mm_ram += net_growth_mb
                
            engine.make_move(move[0], move[1])
            
            # axa x
            total_moves_counter += 1
            global_move_timeline.append(total_moves_counter)
            
            total_cumul_history.append(mem_after_move / (1024 * 1024))
            ppo_cumul_history.append(cumul_ppo_ram)
            mm_cumul_history.append(cumul_mm_ram)
            minimax_cache_sizes.append(len(agent_mm.memo))
            
        if engine.winner == ppo_id: wins["PPO"] += 1
        elif engine.winner == mm_id: wins["Minimax"] += 1
        else: wins["Draw"] += 1

        def print_row(agent_name, t_arr, m_arr):
            if not t_arr: return
            print(f"{i+1:<5} | {agent_name:<8} | {np.min(t_arr):.4f}s  | {np.max(t_arr):.4f}s  | {np.mean(t_arr):.4f}s  | {np.median(t_arr):.4f}s   | {np.min(m_arr):.4f}MB | {np.max(m_arr):.4f}MB | {np.mean(m_arr):.4f}MB | {np.median(m_arr):.4f}MB")

        print_row("PPO", times_ppo, mems_ppo)
        print_row("Minimax", times_mm, mems_mm)
        print("-" * 115)

    tracemalloc.stop()
    print(f"\n[FINAL] Rezultate: PPO {wins['PPO']} - {wins['Minimax']} Minimax ({wins['Draw']} Remize)")

    plot_dashboard(global_ppo_times, global_mm_times, global_ppo_mems, global_mm_mems, 
                   global_move_timeline, total_cumul_history, ppo_cumul_history, mm_cumul_history, minimax_cache_sizes)

def plot_dashboard(ppo_times, mm_times, ppo_mems, mm_mems, timeline, total_cumul, ppo_cumul, mm_cumul, cache_history):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Analiză Comparativă: Reinforcement Learning (PPO) vs Minimax', fontsize=16, fontweight='bold')
    
    # timpii de raspuns
    axs[0, 0].plot(ppo_times, label='PPO', color='blue', alpha=0.8, linewidth=1.5)
    axs[0, 0].plot(mm_times, label='Minimax', color='red', alpha=0.8, linewidth=1.5)
    axs[0, 0].set_title('Timpul de Răspuns per Mutare')
    axs[0, 0].set_xlabel('Index Mutare (Proprie)')
    axs[0, 0].set_ylabel('Secunde')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # overhead per mutare
    axs[0, 1].plot(ppo_mems, label='PPO (Peak)', color='cyan', alpha=0.8, linewidth=1.5)
    axs[0, 1].plot(mm_mems, label='Minimax (Peak)', color='orange', alpha=0.8, linewidth=1.5)
    axs[0, 1].set_title('Overhead Memorie Alocată per Mutare')
    axs[0, 1].set_xlabel('Index Mutare (Proprie)')
    axs[0, 1].set_ylabel('Memorie RAM (MB)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # ram cumulativ consumat
    axs[1, 0].plot(timeline, total_cumul, color='darkviolet', linewidth=2, label='Total Aplicație')
    axs[1, 0].plot(timeline, mm_cumul, color='red', linewidth=2, linestyle='-.', label='Minimax (Creștere netă)')
    axs[1, 0].plot(timeline, ppo_cumul, color='blue', linewidth=2, linestyle=':', label='PPO (Creștere netă)')
    axs[1, 0].set_title('Memorie RAM Cumulativă Reținută')
    axs[1, 0].set_xlabel('Index Mutare Globală (Toate Meciurile)')
    axs[1, 0].set_ylabel('Memorie RAM (MB)')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # numar de transpozitii memorate
    axs[1, 1].plot(timeline, cache_history, color='teal', linewidth=2)
    axs[1, 1].set_title('Stări Învățate de Minimax (Transposition Table)')
    axs[1, 1].set_xlabel('Index Mutare Globală (Toate Meciurile)')
    axs[1, 1].set_ylabel('Număr Configurații Salvate')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92) 
    plt.show()

if __name__ == "__main__":
    benchmark_games(n_games=5, minimax_depth=3, ppo_model_path="../move/models/phase24/remove1038.zip")