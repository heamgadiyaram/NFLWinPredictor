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
visitor_wins = merged_scores.groupby('Visitor').apply(lambda x: len(x) - x['HomeWon'].sum())
total_games_home = merged_scores['Home'].value_counts()
total_games_visitor = merged_scores['Visitor'].value_counts()

#calc win percentage
overall_wins = home_wins + visitor_wins
total_games = total_games_home + total_games_visitor
total_wins = overall_wins / total_games

print(merged_scores.columns)
#calculate yards per play
avg_yards_per_play_home = merged_scores.groupby('Home')['Yards'].mean()
avg_yards_per_play_visitor = merged_scores.groupby('Visitor'['Yards']).mean()

#calculate yards per game
avg_yards_per_game_home = merged_scores.groupby(['SeasonYear', 'Home'])['Yards'].mean()
avg_yards_per_game_visitor = merged_scores.groupby(['SeasonYear', 'Visitor'])['Yards'].mean()
overall_avg_yards_per_game = (avg_yards_per_game_home + avg_yards_per_game_visitor) / 2

#calculate average pass completion rate
avg_pass_completion_rate_home = merged_scores.groupby('Home').apply(lambda x: 1 - x['IsIncomplete'].mean())
avg_pass_completion_rate_visitor = merged_scores.groupby('Visitor').apply(lambda x: 1 - x['IsIncomplete'].mean())
overall_avg_pass_completion = (avg_pass_completion_rate_home + avg_yards_per_game_visitor) / 2

#calculate touchdowns per game
avg_touchdowns_per_game_home = merged_scores.groupby(['SeasonYear', 'Home'])['IsTouchdown'].mean()
avg_touchdowns_per_game_visitor = merged_scores.groupby(['SeasonYear', 'Home'])['IsTouchdown'].mean()
overall_avg_touchdowns_per_game = (avg_touchdowns_per_game_home + avg_touchdowns_per_game_visitor) / 2

#calculate rush success rate
avg_rush_yards_home = merged_scores.groupby('Home').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean())
avg_rush_yards_visitor = merged_scores.groupby('Visitor').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean())
overall_avg_rush_yards = (avg_rush_yards_home + avg_rush_yards_visitor) / 2





