# This script creates various columns on last X fights and sees which are the best predictors


import csv
import gc
import itertools
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

# Read in global objects and packages
exec(open('global.py').read())


# Read in the data
events = fp.ParquetFile('./data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
upcoming_events = fp.ParquetFile('./data/' + config['upcoming_event_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fighters = fp.ParquetFile('./data/' + config['fighter_tott_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fight_results = fp.ParquetFile('./data/' + config['fight_results_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
upcoming_fight_details = fp.ParquetFile('./data/' + config['upcoming_fight_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fight_stats = fp.ParquetFile('./data/' + config['fight_stats_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fight_odds = fp.ParquetFile('./data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
event_odds = fp.ParquetFile('./data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
betfair_odds = fp.ParquetFile('./data/' + config['betfair_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
betfair_upcoming_odds = fp.ParquetFile('./data/' + config['betfair_upcoming_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()

# Read in the average odds model
average_odds_model = joblib.load('./results/average_odds_model.joblib')

###########################################################################################
# UPCOMING

# Add upcoming fights to fight stats
fight_stats = add_upcoming_to_fight_stats(fight_stats, upcoming_fight_details, events, upcoming_events)

# Add upcoming events
events = pd.concat([upcoming_events, events], axis = 0)

# Add the date to the upcoming fights
upcoming_fight_details = upcoming_fight_details.merge(upcoming_events[['event', 'date']], on = 'event', how = 'left')

# Add upcoming fights to the results with empty results
fight_results = pd.concat([upcoming_fight_details.drop('url', axis = 1), fight_results])


###########################################################################################
# FIGHT STATS PROCESSING

# Calculate the historical stats of each fighter at the fight date
fight_stats = fight_stats.sort_values(by=['fighter', 'date'], axis = 0)

# Copy the fight stats but for against a fighter, then merge onto stats
fight_stats_against = fight_stats.copy()
fight_stats_against.rename(columns = lambda x: x + "_against" if "attempt" in x else x, inplace = True)
fight_stats_against.rename(columns = lambda x: x + "_against" if "landed" in x else x, inplace = True)
fight_stats_against.rename(columns = lambda x: x + "_against" if "ctrl_minutes" in x else x, inplace = True)
fight_stats_against.rename(columns = lambda x: x + "_against" if "kd" in x else x, inplace = True)
fight_stats_against.rename(columns = lambda x: x + "_against" if "sub.att" in x else x, inplace = True)

# Split the bout into fighters
# Set the opponent to the fighter for merging
fight_stats_against[['fighter1', 'fighter2']] = fight_stats_against['bout'].str.split('vs\\.', expand = True)
fight_stats_against['fighter'] = np.where(fight_stats_against['fighter'] == fight_stats_against['fighter1'], fight_stats_against['fighter2'], fight_stats_against['fighter1'])
# Get only the columns we want
against_columns = [col for col in fight_stats_against.columns if "against" in col]
fight_stats_against = fight_stats_against[['event', 'bout', 'fighter'] + against_columns]

# Merge onto the fight stats
rows_before = len(fight_stats)
fight_stats = fight_stats.merge(fight_stats_against, on = ['event', 'bout', 'fighter'], how = 'left')

# Check that all the rows are correct
if rows_before == len(fight_stats):
    print('Row check PASSED')
else:
    print('Row check FAILED')
    raise SystemExit
del fight_stats_against


# Now aggrgate
fight_stats_hist_1 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_",
    X = 1,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_2 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_",
    X = 2,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_3 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_",
    X = 3,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_4 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_",
    X = 4,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_5 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_",
    X = 5,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_6 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_",
    X = 6,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_7 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_",
    X = 7,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])



merged_data = merge_df_both_fighters(
    left_df =  fight_results[['fighter1', 'fighter2', 'outcome']],
    df_list = [
        fight_stats_hist_1,
        fight_stats_hist_2,
        fight_stats_hist_3,
        fight_stats_hist_4,
        fight_stats_hist_5,
        fight_stats_hist_6,
        fight_stats_hist_7
    ]
).reset_index(drop = True)