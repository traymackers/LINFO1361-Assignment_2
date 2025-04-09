import multiprocessing
from agent import Agent
import past_agents
from visual_game_manager import VisualGameManager
import history_manager

def run_game(game_id):
    red_agent = Agent(player=1, depth=3)
    black_agent = past_agents.all_agents[4](player=-1, depth=3)

    game = VisualGameManager(red_agent, black_agent, total_time=300, min_agent_play_time=0.2)
    results = game.play()

    history_manager.update_history(
        red_agent=str(red_agent),
        black_agent=str(black_agent),
        winner=results[0],
        total_moves_red=results[1],
        total_moves_black=results[2],
        used_time_red=results[3],
        used_time_black=results[4]
    )

if __name__ == "__main__":
    NUM_GAMES = 30  # ← change ici pour faire plus de parties

    # Lance les parties en parallèle
    with multiprocessing.Pool(processes=4) as pool:  # ← tu peux ajuster à ton nombre de cœurs
        pool.map(run_game, range(NUM_GAMES))
