import numpy as np
import pandas as pd
from scipy.optimize import minimize, maximize

# Sample data: 10 players with their skill scores
players = pd.read_json('players.json', lines=True)

# Constraint: ensure the difference is below a certain value, e.g., 3
desired_difference = 500
max_retries = 1000  # Maximum number of retries
players_skill = players['skill']
players_positions = players['positions']


# Objective function: minimize the difference between the two teams' skill scores
def skill_difference(x):
    x = np.round(x)  # Round to ensure integer team assignments
    team1_indices = np.argsort(x)[:5]  # First 5 indices for team 1
    team2_indices = np.argsort(x)[5:]  # Last 5 indices for team 2

    team1_skill = np.sum(players_skill[team1_indices])
    team2_skill = np.sum(players_skill[team2_indices])

    return abs(team1_skill - team2_skill)


def get_positions(team_indices):
    positions_filled = dict()
    team_positions = players_positions[team_indices].sort_values(key=lambda s: s.str.len())
    for i, positions in team_positions.items():
        positions.reverse()
        for position in positions:
            if position not in positions_filled:
                positions_filled[position] = i
                break
    return positions_filled


def skill_difference_constraint(x):
    return desired_difference - skill_difference(x)


# Add a team size constraint to ensure each team has exactly 5 players
def team_size_constraint(x):
    return np.sum(np.round(x)) - 5


def position_constraint(x):
    x = np.round(x)  # Round to ensure integer team assignments
    team1_indices = np.argsort(x)[:5]  # First 5 indices for team 1
    team2_indices = np.argsort(x)[5:]  # Last 5 indices for team 2

    return 0 if len(get_positions(team1_indices)) == 5 and len(get_positions(team2_indices)) == 5 else 1


# Constraints list
constraints = [{'type': 'ineq', 'fun': skill_difference_constraint},
               {'type': 'eq', 'fun': team_size_constraint},
               {'type': 'eq', 'fun': position_constraint}]

# Retry mechanism
for attempt in range(max_retries):
    # Initial guess: random assignment
    initial_guess = np.random.rand(10)

    # Perform the optimization using SLSQP with the constraints
    result = minimize(skill_difference, initial_guess, method='SLSQP', constraints=constraints)

    # Check if the constraint was met
    if skill_difference(result.x) <= desired_difference and position_constraint(result.x) == 0:
        final_assignment = np.round(result.x)
        team1_indices = np.argsort(final_assignment)[:5]
        team2_indices = np.argsort(final_assignment)[5:]

        team1 = players.iloc[team1_indices][['name', 'skill']]
        team2 = players.iloc[team2_indices][['name', 'skill']]

        team1_positions = get_positions(team1_indices)
        team2_positions = get_positions(team2_indices)

        team1.loc[list(team1_positions.values()), ['positions']] = list(team1_positions.keys())
        team2.loc[list(team2_positions.values()), ['positions']] = list(team2_positions.keys())

        col = team1.pop("positions")
        team1.insert(0, col.name, col)

        col = team2.pop("positions")
        team2.insert(0, col.name, col)

        # Output the results
        print(f"Success on attempt {attempt + 1}:")
        print("Team 1:\n", team1.sort_values('positions').to_string(index=False))
        print("Team 2:\n", team2.sort_values('positions').to_string(index=False))
        print("Skill Difference:", skill_difference(result.x))
        break
else:
    print(f"Failed to meet the constraints after {max_retries} attempts.")
