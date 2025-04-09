import json
from pathlib import Path

def update_history(red_agent: str, black_agent: str, winner: int, total_moves_red: int, total_moves_black : int, used_time_red: float, used_time_black: float):

    history_file = Path("Assignment2/code/history.txt")
    history = {}

    # Read the existing history file if it exists
    if history_file.exists():
        with open(history_file, "r") as file:
            try:
                # Attempt to load the JSON content
                history = json.load(file)
            except json.JSONDecodeError:
                # If the file is empty or invalid, initialize as an empty dictionary
                history = {}
            

    # Create a unique key for the battle log
    battle_key = f"{red_agent}_vs_{black_agent}"

    # Initialize or update the battle log
    if battle_key not in history:
        history[battle_key] = {
            "red_agent": red_agent,
            "black_agent": black_agent,
            "wins_red": 0,
            "wins_black": 0,
            "average_moves_red": 0,
            "average_moves_black": 0,
            "total_games": 0,
            "average_time_red": 0.0,
            "average_time_black": 0.0
        }

    # Update the battle log
    battle_log = history[battle_key]
    battle_log["total_games"] += 1
    battle_log["average_moves_red"] = ((battle_log["average_moves_red"] * (battle_log["total_games"] - 1)) + total_moves_red) / battle_log["total_games"]
    battle_log["average_moves_black"] = ((battle_log["average_moves_black"] * (battle_log["total_games"] - 1)) + total_moves_black) / battle_log["total_games"]

    if winner == 1:
        battle_log["wins_red"] += 1
    elif winner == -1:
        battle_log["wins_black"] += 1
    battle_log["average_time_red"] = ((battle_log["average_time_red"] * (battle_log["total_games"] - 1)) + used_time_red) / battle_log["total_games"]
    battle_log["average_time_black"] = ((battle_log["average_time_black"] * (battle_log["total_games"] - 1)) + used_time_black) / battle_log["total_games"]

    # Write the updated history back to the file
    with open(history_file, "w") as file:
        json.dump(history, file, indent=4)