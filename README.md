# Comparison between the Alpha-Beta algorithm and Reinforcement Learning: From Brute Force to Emergent Behavior in Gomoku

**Ioniță Cătălin-Alexandru**
Year III; CTI; Group: 364

## 1. Introduction: Why Gomoku and why Reinforcement Learning?
For this project, I chose Gomoku, a much larger-scale generalization of classic Tic-Tac-Toe. The reasoning was twofold: on the one hand, it is a game with simple and elegant mechanics that I was familiar with, and on the other hand, from an algorithmic perspective, it is sufficiently complex. 

The main goal of this project was not merely to build an agent capable of playing correctly, but to explore the boundary between deterministic logic and artificial intuition. Therefore, the paper analyzes a direct confrontation between the brute force of the classic Minimax algorithm (with Alpha-Beta pruning) and a neural agent trained via Reinforcement Learning (RL).

## 2. Controlling the Combinatorial Explosion: Environment Constraints
In its original variant, Gomoku is played on a 19x19 board, and pieces can be placed anywhere on the board (in a free space). The first player to place 5 of their pieces in a row, column, or diagonal wins (unlike 3 in Tic-Tac-Toe).

To make the RL agent's training viable and to minimize the number of moves analyzed by Minimax, I imposed the following constraints:
- i) The first move must be made in the center of the board (in my 20x20 dimension variant)
- ii) Any subsequent move must be adjacent to an already existing piece on the board (up, down, left, or right)

This decision limits the game to a single dense cluster.

## 3. Deterministic Architecture: Building and Optimizing the Minimax Agent
I started by building the base Environment, the game board, the rules, and a Rendering mode to make the games visible, then I tested it with a random agent.

### 3.1 Heuristics
Since a depth-first search to the end of the game is impossible in Gomoku due to the branching factor, the algorithm must stop at a certain depth and evaluate the position at that moment. To do this efficiently, I limited the position analysis strictly to the area where pieces exist on the board.

I used a dictionary of scores (key = sequence (in String format), value = associated_score) for Pattern Matching (sequences extracted from the board were converted to String to retrieve the value from the dictionary). Initially, the values associated with the sequences (scores per piece sequence) varied linearly, which led to reward hacking within the Minimax algorithm. It refused to win because it received a higher overall score from multiple sequences of 3 or 4 pieces in a row. The solution was an exponential scaling of the sequence scores.

Also, due to the board size (20x20), sending states to be evaluated in the form of nodes in a tree was not viable. It would have meant that such a node would hold an entire board (mostly empty), an entire state. Therefore, I used an "undo" function which, after the Minimax agent makes a move and evaluates it (and keeps the information if it is the best move), resets the state to the moment before the move. In the case of a depth $d=2$, within the search, 2 moves are made, followed by 2 undos.

### 3.2. Shared Memory (Transposition Tables) and the Dihedral Group D4
Even with Alpha-Beta pruning, evaluating the board was costly. I noticed that, during recursion, Minimax frequently reached identical game states, but obtained through a different order of moves. Moreover, the Gomoku board is symmetrical.

To resolve this bottleneck, I implemented a global cache (`shared_memo`), passed as a reference to all instantiated agents. The real optimization came from exploiting geometric symmetries. I applied transformations from the dihedral group $D_4$ for each board state, generating the 8 possible canonical configurations (4 rotations at 90° and their 4 corresponding reflections).

Before evaluating a state, the algorithm checks if any of these 8 symmetrical forms has already been calculated in the past. If so, it extracts the score directly from memory in $O(1)$ time.

### 3.3. The Need for Speed: Optimization via Window Convolution
The biggest challenge appeared later in the project. When I started training the Reinforcement Learning agent, I needed Minimax to act as a "teacher", generating millions of moves per hour. Classic iteration through matrices and string concatenation to find patterns (e.g., `_XXXX_`) was too slow.

To achieve production speeds, I completely rewrote the heuristic evaluation engine, taking inspiration from the mechanics of convolutional neural networks. The process works as follows:

- **The Sliding Windows Principle**: Instead of analyzing entire rows of variable dimensions, I applied a fixed filter that "slides" over the board step by step. The optimal size of this window was set to 7. The reason is purely mathematical: to win, 5 consecutive pieces are needed, and to properly evaluate the offensive or defensive potential of a sequence, the algorithm needs the immediate context at both ends (1 space + 5 pieces + 1 space = 7).
- **Mathematical Encoding in Base 3**: Manipulating String structures in memory (allocation, concatenation, comparison) is a slow operation that generates too much overhead. To eliminate this drawback, I looked at the cell states at a purely numerical level. Each cell on the Gomoku board can have exactly 3 states: empty space (0), current player's piece (1), or opponent's piece (2). Looking at things this way, any sequence of 7 pieces simply becomes a number written in a base-3 numeral system.
- **Generating the Precalculated Score Table (Lookup Array)**: The unique value of any 7-cell window is calculated by multiplying each cell value by the corresponding power of 3 ($3^0, 3^1, ..., 3^6$). The maximum possible value for such a window is $3^7 - 1$, which is exactly 2186. Having such a small and strictly delimited state space (2187 possible combinations), I was able to completely drop the Hash Map (Dictionary) structure in favor of a simple one-dimensional array allocated contiguously in memory.
- At game initialization (boot time), the system iterates from 0 to 2186. Each number is decoded back into a sequence of 7 states and evaluated slowly based on exponential heuristic rules (checking if it's an open line of 4, a line of 3 blocked at one end, etc.). The obtained score is saved in the array at the exact index corresponding to that number. This costly operation is executed only once during the entire run.
- **Real-Time Extraction and Evaluation (*O(1)*)**: The algorithm traverses the rows, columns, and diagonals strictly within the limits of the active area (bounding box) and sequentially extracts the 7-element windows. For each window, the polynomial value in base 3 is calculated instantaneously via a dot product between the cell states vector and a precalculated vector of powers of 3. The entire heuristic evaluation process of that pattern is now reduced to accessing a memory address: `score = lookup_array[base_3_value]`.

## 4. Reinforcement Learning Architecture
Once I had a Minimax engine capable of playing and evaluating states at high speeds, I was able to start training the Reinforcement Learning model. I used the PPO (Proximal Policy Optimization) algorithm, an Actor-Critic type architecture, specifically Maskable PPO. This is a suitable optimization that fits with the valid moves frontier from the Environment, thus eliminating all invalid moves.

### 4.1 Architecture: CNN and Parallelization
Initially, the convolutional network used for feature extraction was a simple one that scanned the board on a single channel, like a simple image. Later, I separated the board into 2 channels: one for the agent's pieces and one for the opponent's pieces.

By the end, the network had 4 channels with two more added: one channel for the valid moves frontier and a "threat map" channel encoding the positions where the opponent can win. 

For extra performance, I used `SubprocVecEnv` and OMP to parallelize the agent's training across 8 environments simultaneously.

### 4.2 Curriculum Learning and Opponent Pool
The agent was trained progressively with increasingly competent opponents, including simple Minimax variants to accelerate the process of learning defense. Simultaneously with the periodic introduction of better static opponents, the agent consistently played against versions of itself from a dynamic pool.

At the start of each match, the environment randomly chose the opponent based on strictly controlled weights:
1) The agent played exclusively against a completely Random agent, rapidly learning to recognize immediate opportunities to make a line of 5.
2) I introduced a Fast Heuristic agent: if the agent can win, it will do so; if not, it looks to block the opponent; and if neither option is possible, it moves randomly.
3) Once the agent demonstrated consistency, I allocated the majority of the weight (80-90%) to a Self-Play Pool. The system periodically saved the best versions (snapshots) of the network and forced them to play against each other. Here, the "Red Queen" phenomenon became visible: the agent had periods where it won a lot, leading to stronger opponents being added to the pool. Then, the agent had to win against these better variants of itself, which forced it to explore more, adapt, and implicitly evolve.
4) To prevent "Catastrophic Forgetting", I permanently maintained a small weight of static agents (2.5% Minimax D1, 5% Minimax D2). Whenever the RL agent forgot to block a simple attack, Minimax instantly penalized it.

### 4.3 Data Augmentation
After tens of millions of training steps, the agent reached excellent scores but suffered from a generalization problem. The neural network had learned to defend very well in the center, but when the game became more complex, the agent no longer knew how to block further away from the central area and would have needed a lot of matches to accumulate sufficient context. 

For this reason, I applied the same transformations as for the Minimax agent (rotations at 90°, 180°, 270°, and reflections).

## 5. Comparison
For the final comparison, I had the RL agent (PPO) play 5 matches against the Minimax agent with a maximum search depth of 3. In this comparison, I included the following metrics:
- Final score
- Disk space occupied (to store the PPO model / the source code of the Minimax algorithm)
- Minimum and maximum move time
- Average move time and its median
- Minimum and maximum memory allocated during the matches
- Average and median allocated memory
- Graphs for Response Time and Allocated Memory

**[DISK]** PPO: 77.78 MB | Minimax: 0.0043 MB (script only)  
**[FINAL] Results:** PPO 4 - 1 Minimax (0 Draws)

### Aggregated Performance Metrics

| Agent | Min Time (s) | Max Time (s) | Avg Time (s) | Median Time (s) | Min Mem (MB) | Max Mem (MB) | Avg Mem (MB) | Median Mem (MB) |
|---|---|---|---|---|---|---|---|---|
| **PPO** | 0.0040 | 0.0209 | 0.0072 | 0.0056 | 0.0223 | 0.3049 | 0.0761 | 0.0293 |
| **Minimax** | 0.0423 | 2.5941 | 0.8370 | 0.6257 | 0.0764 | 1.7205 | 0.5361 | 0.3889 |

As can be seen from the aggregated data over the 5 matches, the Minimax agent is significantly slower and occupies more memory (especially due to the time optimization using transposition tables), whereas the PPO Reinforcement Learning agent has a minimum response time and almost 0 allocated memory. The price paid for the PPO agent, however, is disk space and significantly more training time (I estimate roughly 120 hours of total training time).

## 6. Conclusions
This project explored the boundary between deterministic computation, based on brute force, and artificial intuition developed through Reinforcement Learning. Following the implementation and evaluation of the two agents, Minimax (Alpha-Beta Pruning) and PPO (Proximal Policy Optimization), I have drawn the following fundamental conclusions:

- **The Trade-off between Training and Inference**: While Minimax requires no training, it becomes very slow as the search depth increases, consuming significant computational resources for each move. In contrast, the PPO agent required a huge computational cost in the training phase, but offers an almost instantaneous and constant response time in the game phase (inference), making it ideal for real-time applications.
- **Generalization vs. Memorization Capacity**: PPO demonstrated a superior capacity for generalization, managing to defeat tactical opponents (such as Minimax D3) that it did not encounter directly during training. However, its performance is directly dependent on the quality of the critic and the balance between exploration and exploitation.

## 7. Bibliography
- Raffin, A. et al. (2021). Stable Baselines3: Reliable Reinforcement Learning Implementations. GitHub repository. Available at: https://stable-baselines3.readthedocs.io/
- Farama Foundation. Gymnasium: A standard API for reinforcement learning. Available at: https://gymnasium.farama.org/
- Nicholas Renotte (2022). Reinforcement Learning in 3 hours | Full Course using Python. https://www.youtube.com/watch?v=Mut_u40Sqz4

---

## Appendix A: Complete Logs and Training Graphs

**Figure 1: Response Time per Move** ![Response Time per Move](resources/response_time.png)

**Figure 2: Memory Allocated per Move (Overhead)** ![Overhead Memory Allocated](resources/overhead_mem_alloc.png)

**Figure 3: Cumulative RAM Consumed** ![Cumulative RAM](resources/cumulative_ram.png)

**Figure 4: Number of Symmetries Stored by Minimax (Transposition Table)** ![Transposition Table Sizes](resources/transpositions_learned.png)

### Complete Data Table (Per Game)

| Game | Agent | Min Time (s) | Max Time (s) | Avg Time (s) | Median Time (s) | Min Mem (MB) | Max Mem (MB) | Avg Mem (MB) | Median Mem (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PPO | 0.0040 | 0.0088 | 0.0052 | 0.0047 | 0.0220 | 0.0288 | 0.0229 | 0.0224 |
| 1 | Minimax | 0.1303 | 3.3702 | 1.0412 | 0.7669 | 0.0983 | 2.1564 | 0.7068 | 0.4639 |
| 2 | PPO | 0.0042 | 0.0670 | 0.0126 | 0.0067 | 0.0222 | 0.5402 | 0.1413 | 0.0223 |
| 2 | Minimax | 0.0659 | 1.8260 | 0.8396 | 0.7367 | 0.1989 | 0.9515 | 0.5203 | 0.5180 |
| 3 | PPO | 0.0040 | 0.0090 | 0.0060 | 0.0059 | 0.0223 | 0.0378 | 0.0274 | 0.0229 |
| 3 | Minimax | 0.0066 | 2.1335 | 0.9249 | 0.8997 | 0.0294 | 1.1960 | 0.5047 | 0.4547 |
| 4 | PPO | 0.0041 | 0.0110 | 0.0058 | 0.0045 | 0.0224 | 0.8769 | 0.1559 | 0.0436 |
| 4 | Minimax | 0.0022 | 2.2298 | 0.7984 | 0.6490 | 0.0220 | 2.7880 | 0.6743 | 0.4690 |
| 5 | PPO | 0.0037 | 0.0089 | 0.0066 | 0.0064 | 0.0224 | 0.0410 | 0.0332 | 0.0352 |
| 5 | Minimax | 0.0064 | 3.4112 | 0.5810 | 0.0762 | 0.0333 | 1.5104 | 0.2745 | 0.0387 |

---

## Installation Guide

Follow these steps to set up your local development environment and install all the necessary dependencies to run and test the Gomoku AI agents.

### Prerequisites
- Python 3.8 or higher
- `pip` (Python package manager)
- `git`

### 1. Clone the Repository
If you haven't already, clone the repository to your local machine and navigate into the project root directory:
```bash
git clone git@github.com:your-username/your-repo-name.git
cd your-repo-name
```
### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to keep the project dependencies isolated from your global system packages:
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
Activate the environment based on your operating system:
- On Windows (Command Prompt):
```bash
venv\Scripts\activate.bat
```
- On Windows (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```

### 4. Upgrade pip
Ensure your package manager is up to date before installing heavy neural network libraries:
```bash
pip install --upgrade pip
```

### 5. Install Dependencies
Install all required frameworks, including Stable-Baselines3, SB3-Contrib (required for the Maskable PPO architecture), Gymnasium (the reinforcement learning environment standard), and Pygame for the visual rendering interface:
```bash
pip install stable-baselines3[extra] sb3-contrib gymnasium pygame numpy
```

### Alternative: Conda Environment

Alternatively, you can use **Conda** (via Anaconda or Miniconda) to manage your virtual environment. Conda is highly popular in the Machine Learning ecosystem because it excels at managing complex, non-Python system dependencies alongside standard Python packages. 

If you prefer this workflow, you can simply create and activate a new Conda environment:
```bash
conda create -n gomoku_env python=3.10
conda activate gomoku_env
```
