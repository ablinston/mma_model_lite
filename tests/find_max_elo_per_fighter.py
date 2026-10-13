# imports
from ablinston_ml import stop_if_not
import fastparquet as fp
import joblib
import numpy as np
import pandas as pd

from mma_functions.cleaning import log_final_data_stats
from mma_functions.dataset import add_opponent_column, add_upcoming_to_fight_stats, flag_train_test_oot, merge_df_both_fighters
from mma_functions.elo import derive_start_elos, prep_stats_for_elo_calc
from mma_functions.features import add_fighter_differentials, aggregate_last_X_fight_data, bayesian_priors, calculate_per_minute_bayesian_stats, calculate_percentages, calculate_win_loss_stats, define_fighter_type
from mma_functions.odds import add_event_and_bout_to_odds_data, calculate_all_true_odds_columns, process_odds_data
from mma_functions.utils import check_identical_excl_string, safe_divide_np
from mma_functions.config import config, data_processing_config, elo_config, model_hyper_params, odds_scrape_config

# Read in the data
events = fp.ParquetFile("./data/" + config["event_details_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
upcoming_events = fp.ParquetFile("./data/" + config["upcoming_event_details_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
fighters = fp.ParquetFile("./data/" + config["fighter_tott_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
fight_results = fp.ParquetFile("./data/" + config["fight_results_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
upcoming_fight_details = fp.ParquetFile("./data/" + config["upcoming_fight_details_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
fight_stats = fp.ParquetFile("./data/" + config["fight_stats_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
fight_odds = fp.ParquetFile("./data/" + config["fighter_odds_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
event_odds = fp.ParquetFile("./data/" + config["event_odds_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
betfair_odds = fp.ParquetFile("./data/" + config["betfair_odds_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
betfair_upcoming_odds = fp.ParquetFile("./data/" + config["betfair_upcoming_odds_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()

# Read in the average odds model
average_odds_model = joblib.load("./results/average_odds_model.joblib")

###########################################################################################
# UPCOMING

# Add upcoming fights to fight stats
fight_stats = add_upcoming_to_fight_stats(fight_stats, upcoming_fight_details, events, upcoming_events)

# Add upcoming events
events = pd.concat([upcoming_events, events], axis=0)

# Add the date to the upcoming fights
upcoming_fight_details = upcoming_fight_details.merge(upcoming_events[["event", "date"]], on="event", how="left")

# Add upcoming fights to the results with empty results
fight_results = pd.concat([upcoming_fight_details.drop("url", axis=1), fight_results])



###########################################################################################
# CALCULATE ELOs

# Get the starting data
start_elo, elo_fight_data = prep_stats_for_elo_calc(fight_results, fight_stats, config, start_date="2000-01-01", remove_nas=False)

# Dderive the ELO for the params in the config
elo_data, elo_fight_data_welo = derive_start_elos(start_elo, elo_fight_data, elo_config)

end_elos = elo_fight_data_welo[['date', 'fighter1', 'fighter2', 'fighter1_end_elo', 'fighter2_end_elo']].to_pandas()

stacked_elos = pd.concat([
    end_elos.drop(['fighter2_end_elo'], axis=1).rename({'fighter1': 'fighter', 'fighter2': 'opponent', 'fighter1_end_elo': 'end_elo'}, axis=1),
    end_elos.drop(['fighter1_end_elo'], axis=1).rename({'fighter2': 'fighter', 'fighter1': 'opponent', 'fighter2_end_elo': 'end_elo'}, axis=1)
], axis=0)

# Get the max per fighter
max_elos = stacked_elos.groupby('fighter').agg({'end_elo': max}).sort_values('end_elo', ascending=False)