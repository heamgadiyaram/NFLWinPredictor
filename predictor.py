import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

play_data_2024 = pd.read_csv("pbp-2024.csv")
nfl_scores_2024 = pd.read_csv("box_scores.csv", sep='\t')

play_data_2024['GameDate'] = pd.to_datetime(play_data_2024['GameDate'], format='%m/%d/%y').dt.strftime('%-m/%-d/%y')
nfl_scores_2024['Date'] = pd.to_datetime(nfl_scores_2024['Date'], format='%m/%d/%y').dt.strftime('%-m/%-d/%y')

team_mapping = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS'
}

nfl_scores_2024['Visitor'] = nfl_scores_2024['Visitor'].map(team_mapping)
nfl_scores_2024['Home'] = nfl_scores_2024['Home'].map(team_mapping)

merge_visitor = play_data_2024.merge(
    nfl_scores_2024,
    left_on=['GameDate', 'OffenseTeam', 'DefenseTeam'],
    right_on=['Date', 'Visitor', 'Home'],
    how='left'
)

merge_home = play_data_2024.merge(
    nfl_scores_2024,
    left_on=['GameDate', 'OffenseTeam', 'DefenseTeam'],
    right_on=['Date', 'Home', 'Visitor'],
    how='left'
)

merged_scores = play_data_2024.copy()
for col in ['Visitor', 'VisitorScore', 'Home', 'HomeScore']:
    merged_scores[col] = merge_visitor[col].combine_first(merge_home[col])

merged_scores['HomeWon'] = merged_scores['HomeScore'] > merged_scores['VisitorScore']

nfl_schedule = pd.read_csv('nfl_schedule.csv')
nfl_schedule['Home'] = nfl_schedule['Home'].map(team_mapping)
nfl_schedule['Away'] = nfl_schedule['Away'].map(team_mapping)

avg_points_scored_home = merged_scores.groupby('Home')['HomeScore'].mean()
avg_points_scored_visitor = merged_scores.groupby('Visitor')['VisitorScore'].mean()

avg_points_allowed_home = merged_scores.groupby('Home')['VisitorScore'].mean()
avg_points_allowed_visitor = merged_scores.groupby('Visitor')['HomeScore'].mean()

#calculate avg points
overall_avg_points_scored = (avg_points_scored_home + avg_points_scored_visitor) / 2
overall_avg_points_allowed = (avg_points_allowed_home + avg_points_allowed_visitor) / 2

#total wins
home_wins = merged_scores.groupby('Home')['HomeWon'].sum()
visitor_wins = (1 - merged_scores['HomeWon']).groupby(merged_scores['Visitor']).sum()

total_games_home = merged_scores['Home'].value_counts()
total_games_visitor = merged_scores['Visitor'].value_counts()

#calc win percentage
overall_wins = home_wins + visitor_wins
total_games = total_games_home + total_games_visitor
total_win_rate = overall_wins / total_games

#calculate yards per play
avg_yards_per_play = merged_scores.groupby('OffenseTeam')['Yards'].mean()

#calculate yards per game
total_yards_per_game = merged_scores.groupby(['GameId', 'OffenseTeam'])['Yards'].sum().reset_index()
avg_yards_per_game = total_yards_per_game.groupby('OffenseTeam')['Yards'].mean()

#calculate average pass completion rate
avg_pass_completion_rate_home = 1 - merged_scores.groupby('Home')['IsIncomplete'].mean()
avg_pass_completion_rate_visitor = 1 - merged_scores.groupby('Visitor')['IsIncomplete'].mean()
overall_avg_pass_completion = (avg_pass_completion_rate_home + avg_pass_completion_rate_visitor) / 2

#calculate touchdowns per game
avg_touchdowns_per_game_home = merged_scores.groupby('Home')['IsTouchdown'].mean()
avg_touchdowns_per_game_visitor = merged_scores.groupby('Visitor')['IsTouchdown'].mean()
overall_avg_touchdowns_per_game = (avg_touchdowns_per_game_home + avg_touchdowns_per_game_visitor) / 2

#calculate rush yards
avg_rush_yards_home = merged_scores[merged_scores['IsRush'] == 1].groupby('Home')['Yards'].mean()
avg_rush_yards_visitor = merged_scores[merged_scores['IsRush'] == 1].groupby('Visitor')['Yards'].mean()

overall_avg_rush_yards = (avg_rush_yards_home + avg_rush_yards_visitor) / 2


#defensive features

#create 'SuccessfulPlay' column in merged df
merged_scores['SuccessfulPlay'] = merged_scores['IsTouchdown'].astype(bool) | (~merged_scores['IsInterception'].astype(bool) & ~merged_scores['IsFumble'].astype(bool))

#calculate successful play rates
avg_success_rate_home = merged_scores.groupby('Home')['SuccessfulPlay'].mean()
avg_success_rate_visitor = merged_scores.groupby('Visitor')['SuccessfulPlay'].mean()
overall_avg_successful_plays = (avg_success_rate_home + avg_success_rate_visitor) / 2

#create column for turnovers
merged_scores['Turnover'] = merged_scores['IsInterception'] | merged_scores['IsFumble']

#calculate turnover rates
avg_turnover_rate_home = merged_scores.groupby('Home')['Turnover'].mean()
avg_turnover_rate_visitor = merged_scores.groupby('Visitor')['Turnover'].mean()
overall_avg_tunover_rate = (avg_turnover_rate_home + avg_turnover_rate_visitor)/ 2


#create df for newly calculated features
team_features = pd.DataFrame({
    'AveragePointsScored': overall_avg_points_scored,
    'AveragePointsAllowed': overall_avg_points_allowed,
    'WinRate': total_win_rate,
    'AverageYardsPerPlay': avg_yards_per_play,
    'AverageYardsPerGame': avg_yards_per_game,
    'AveragePassCompletionRate': overall_avg_pass_completion,
    'AverageTouchdownsPerGame': overall_avg_touchdowns_per_game,
    'AverageRushYards': overall_avg_rush_yards,
    'AverageSuccessFulPlayRate': overall_avg_successful_plays,
    'AverageTurnoverRate': overall_avg_tunover_rate
})

team_features.reset_index(inplace=True)
team_features.rename(columns={'index': 'Team'}, inplace=True)

print(team_features.head())




