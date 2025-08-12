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

merged_scores['Homewon'] = merged_scores['HomeScore'] > merged_scores['VisitorScore']

nfl_schedule = pd.read_csv('nfl_schedule.csv')
nfl_schedule['Home'] = nfl_schedule['Home'].map(team_mapping)
nfl_schedule['Away'] = nfl_schedule['Away'].map(team_mapping)



