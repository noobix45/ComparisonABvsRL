import pygame
import time
import numpy as np
from GameData import GameEngine, MinimaxAgent2
from train_cloud import PPOOpponentAgent, _load_ppo_agent # Asigura-te ca poti importa asta, sau copiaza clasa PPOOpponentAgent direct aici

# Constante
CELL_SIZE = 30
GRID_SIZE = 20
BOARD_SIZE = CELL_SIZE * GRID_SIZE
BG_COLOR = (240, 240, 240)
LINE_COLOR = (200, 200, 200)
X_COLOR = (200, 50, 50)
O_COLOR = (50, 50, 200)

def draw_board(screen, engine):
    screen.fill(BG_COLOR)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, LINE_COLOR, rect, 1)

            val = engine.board[r, c]
            center_x = c * CELL_SIZE + CELL_SIZE // 2
            center_y = r * CELL_SIZE + CELL_SIZE // 2

            if val == 1:
                offset = CELL_SIZE // 4
                pygame.draw.line(screen, X_COLOR, (center_x - offset, center_y - offset), (center_x + offset, center_y + offset), 3)
                pygame.draw.line(screen, X_COLOR, (center_x - offset, center_y + offset), (center_x + offset, center_y - offset), 3)
            elif val == -1:
                radius = CELL_SIZE // 3
                pygame.draw.circle(screen, O_COLOR, (center_x, center_y), radius, 3)

    if engine.winning_line:
        (r1, c1), (r2, c2) = engine.winning_line
        x1, y1 = c1 * CELL_SIZE + CELL_SIZE // 2, r1 * CELL_SIZE + CELL_SIZE // 2
        x2, y2 = c2 * CELL_SIZE + CELL_SIZE // 2, r2 * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.line(screen, (40, 200, 40), (x1, y1), (x2, y2), 6)

    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("AI vs AI Benchmark")

    engine = GameEngine(size=GRID_SIZE)

    # =============== CONFIGUREAZA MECIUL AICI ===============
    
    # EXEMPLE:
    # 1. PPO Phase 2 vs Minimax Depth 2
    agent1 = MinimaxAgent2(player_id=1, max_depth=2)
    agent2 = _load_ppo_agent("./models/phase24/remove1030.zip")
    agent2.player_id = -1
    
    
    # 2. Daca vrei PPO vs PPO, decomenteaza linia de mai jos:
    # agent2 = _load_ppo_agent("./models/phase24/remove1010.zip")
    # agent2.player_id = -1   
    
    # ========================================================

    clock = pygame.time.Clock()
    running = True
    game_over = False

    draw_board(screen, engine)
    pygame.time.wait(1000) # pauza scurta inainte sa inceapa

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            # Alege a cui e randul
            if engine.current_player == 1:
                move = agent1.get_action(engine)
            else:
                move = agent2.get_action(engine)

            if move:
                r, c = move
                engine.make_move(r, c)
                draw_board(screen, engine)
                pygame.time.wait(300) # Delay vizual (200ms) ca sa poti urmari mutarile!

                if engine.winner != 0 or len(engine.valid_moves) == 0:
                    game_over = True
                    winner_str = "Agent 1 (X)" if engine.winner == 1 else "Agent 2 (O)" if engine.winner == -1 else "Remiza"
                    print(f"Joc terminat! Castigator: {winner_str}")

        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()