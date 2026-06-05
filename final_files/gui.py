import pygame
import json
import time
import random
import os
from gomoku_env import GameEngine, PPOOpponentAgent
from minimax_agent import MinimaxAgent2

# Constante UI
CELL_SIZE = 30
GRID_SIZE = 20
BOARD_SIZE = CELL_SIZE * GRID_SIZE
PANEL_HEIGHT = 60
WINDOW_SIZE_Y = BOARD_SIZE + PANEL_HEIGHT

BG_COLOR = (240, 240, 240)
LINE_COLOR = (200, 200, 200)
X_COLOR = (200, 50, 50)
O_COLOR = (50, 50, 200)
BTN_COLOR = (180, 180, 190)
BTN_HOVER = (200, 200, 210)
TEXT_COLOR = (30, 30, 30)

def draw_button(screen, font, rect, text):
    mouse_pos = pygame.mouse.get_pos()
    color = BTN_HOVER if rect.collidepoint(mouse_pos) else BTN_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, (100, 100, 100), rect, 2, border_radius=8)
    text_surf = font.render(text, True, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

def menu_choice(screen, font, title, options):
    #defineste butoanele cu optiunile date
    buttons = []
    start_y = 100
    for i, (label, value) in enumerate(options):
        rect = pygame.Rect(BOARD_SIZE // 2 - 125, start_y + i * 60, 250, 40)
        buttons.append((rect, value, label))
    
    # tine windowul pana primeste input
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                for rect, value, _ in buttons:
                    if rect.collidepoint(event.pos):
                        return value
                        
        screen.fill(BG_COLOR)
        
        #deseneaza titlu
        title_surf = font.render(title, True, TEXT_COLOR)
        title_rect = title_surf.get_rect(center=(BOARD_SIZE // 2, 50))
        screen.blit(title_surf, title_rect)
        
        #deseneaza butoane
        for rect, _, label in buttons:
            draw_button(screen, font, rect, label)
            
        pygame.display.flip()

#instantiaza agentul
def load_agent(agent_type, player_id, depth=2):
    if agent_type == "minimax":
        return MinimaxAgent2(player_id=player_id, max_depth=depth)
    elif agent_type == "ppo":
        agent = PPOOpponentAgent("../move/models/phase24/remove1038.zip") 
        agent.player_id = player_id
        return agent
    return None

def play_game(screen, font, p1_type, p2_type, depth):
    engine = GameEngine(size=GRID_SIZE)
    p1 = load_agent(p1_type, 1, depth)
    p2 = load_agent(p2_type, -1, depth)
    
    clock = pygame.time.Clock()
    game_over = False
    match_history = []

    def draw_board():
        screen.fill(BG_COLOR)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, LINE_COLOR, rect, 1)
                
                val = engine.board[r, c]
                center_x, center_y = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
                if val == 1:
                    offset = CELL_SIZE // 4
                    pygame.draw.line(screen, X_COLOR, (center_x - offset, center_y - offset), (center_x + offset, center_y + offset), 3)
                    pygame.draw.line(screen, X_COLOR, (center_x - offset, center_y + offset), (center_x + offset, center_y - offset), 3)
                elif val == -1:
                    pygame.draw.circle(screen, O_COLOR, (center_x, center_y), CELL_SIZE // 3, 3)

        if engine.winning_line:
            (r1, c1), (r2, c2) = engine.winning_line
            pygame.draw.line(screen, (40, 200, 40), (c1*CELL_SIZE + 15, r1*CELL_SIZE + 15), (c2*CELL_SIZE + 15, r2*CELL_SIZE + 15), 6)

        pygame.draw.rect(screen, (210, 210, 220), (0, BOARD_SIZE, BOARD_SIZE, PANEL_HEIGHT))
        
        if game_over:
            status = f"Câștigător: {'X' if engine.winner == 1 else 'O'}! Click pt meniu."
        else:
            status = f"Rândul: {'X' if engine.current_player == 1 else 'O'} ({p1_type if engine.current_player == 1 else p2_type})"
            
        screen.blit(font.render(status, True, TEXT_COLOR), (20, BOARD_SIZE + 20))
        pygame.display.flip()

    draw_board()
    
    while True:
        current_agent = p1 if engine.current_player == 1 else p2
        
        #input ai
        if current_agent is not None and not game_over:
            draw_board()
            
            #prima mutare e random la agenti
            if engine.move_count == 0:
                move = random.choice(list(engine.valid_moves))
            else:
                move = current_agent.get_action(engine)
                
            if move:
                engine.make_move(move[0], move[1])
                match_history.append({"player": engine.current_player * -1, "move": [move[0], move[1]]})
                if engine.winner != 0 or not engine.valid_moves:
                    game_over = True
            draw_board()

        # input user
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return match_history
            if event.type == pygame.MOUSEBUTTONDOWN and current_agent is None and not game_over:
                mouse_x, mouse_y = event.pos
                if mouse_y < BOARD_SIZE:
                    row, col = mouse_y // CELL_SIZE, mouse_x // CELL_SIZE
                    success, _ = engine.make_move(row, col)
                    if success:
                        match_history.append({"player": engine.current_player * -1, "move": [row, col]})
                        if engine.winner != 0 or not engine.valid_moves:
                            game_over = True
                        draw_board()
            
            # un click ca sa ma intorc in meniu
            if event.type == pygame.MOUSEBUTTONDOWN and game_over:
                return match_history
                
        clock.tick(30)

def main():
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, WINDOW_SIZE_Y))
    pygame.display.set_caption("Gomoku Arena")
    font = pygame.font.SysFont("Arial", 20, bold=True)

    while True:
        modes = [
            ("Om vs Minimax", ("human", "minimax")), 
            ("Om vs PPO", ("human", "ppo")), 
            ("Minimax vs PPO", ("minimax", "ppo")), 
            ("Minimax vs Minimax", ("minimax", "minimax")),
            ("PPO vs PPO", ("ppo", "ppo"))
        ]
        mode_choice = menu_choice(screen, font, "Alege Modul de Joc", modes)
        if not mode_choice: break
        p1_type, p2_type = mode_choice # cei 2 agenti intorsi din menu options

        # alegerae adancimii daca unul din agenti este minimax
        depth = 2
        if "minimax" in [p1_type, p2_type]:
            depths = [("Adâncime 1 (Rapid)", 1), ("Adâncime 2 (Mediu)", 2), ("Adâncime 3 (Greu)", 3)]
            depth_choice = menu_choice(screen, font, "Alege Dificultatea Minimax", depths)
            if not depth_choice: break
            depth = depth_choice

        # aleg ordinea in care incepe jocul
        starters = [
            (f"Începe {p1_type.upper()} (X)", 1),
            (f"Începe {p2_type.upper()} (X)", 2)
        ]
        starter_choice = menu_choice(screen, font, "Cine face prima mutare?", starters)
        if not starter_choice: break
        
        # jucatorul p1 incepe byu default, daca vreau sa inceapa p1 al doilea inversez tipurile si id urile
        if starter_choice == 2:
            p1_type, p2_type = p2_type, p1_type

        history = play_game(screen, font, p1_type, p2_type, depth)
        
        if history:
            filename = f"replay_{int(time.time())}.json"
            with open(filename, "w") as f:
                json.dump({"p1": p1_type, "p2": p2_type, "moves": history}, f, indent=4)
            print(f"Meci salvat în {filename}")

if __name__ == "__main__":
    main()