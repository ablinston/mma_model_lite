# This script fits XGBoost models to the future events data after refreshing data scrape

import gc
# import h2o
# from h2o.automl import H2OAutoML
# from hyperopt import hp, STATUS_OK, tpe, fmin, Trials # hyperparameter optimisation functions
# from hyperopt.atpe import suggest as atpe_suggest
import itertools
# import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import seaborn as sns
import xgboost as xgb

# Read in global objects and packages
import fastparquet as fp
import joblib
import numpy as np
import os
import pandas as pd

from mma_functions import evaluation
from mma_functions.cleaning import clean_target_column, standard_data_filter
from mma_functions.dataset import prep_data_for_model_training
from mma_functions.modelling import predict_xgboost_ensemble
from mma_functions.pipelines import btest_model_and_log_mlflow, clean_all_raw_data_and_save, merge_and_process_all_clean_data, scrape_all_unparsed_data
from mma_functions.config import bet_config, betfair_login_config, config, data_processing_config, elo_config, model_hyper_params, odds_scrape_config, test_fighters

# Replace the firefox location if we're not on windows
if not os.name == "nt":
    config['firefox_location'] = config['firefox_location_linux']

# Use this to skip update of data if it's already been done recently
skip_data_update = True
skip_save_data = True

# The length of the trailing window for the rolling logloss
rolling_window_months = 8

# The date to split the sample on when comparing logloss by method of victory
period_split_date = '2025-08-01'

# # Load files
mlflow.set_tracking_uri(uri="http://mlflow-mma-model.192.168.68.51.nip.io:8081")
mlflow.set_experiment('Model_Monitoring_v2')

# Load all model versions
rf_xgb_models = {}
model_versions = ['4.0.14']

for i in model_versions:
    rf_xgb_models.update({str(i): joblib.load('./models/xgb_forest_v' + str(i) + '.joblib')})


##########################################################################################
### Update the upcoming event and odds data ###
##########################################################################################

if not skip_data_update:
    
    # Run these scripts to scrape then clean the raw data
    # Run these scripts to scrape then clean the raw data
    scrape_all_unparsed_data(False, config, betfair_login_config)
    clean_all_raw_data_and_save(config)
    
    # Optional, can either rerun all the processing, or just update the odds. By default will rerun all processing if new events were found to have completed
    if True: # unparsed_events: Hard coded this to always run because fights can change if fighters drop out and otherwise they wont have the hist stats

        # Refresh the dataset
        backtesting_data = merge_and_process_all_clean_data(
                config,
                elo_config,
                data_processing_config,
                odds_scrape_config,
                model_hyper_params,
                test_fighter_list = test_fighters,
                skip_save_data = skip_save_data # whether to skip the save data and just return merged data
                )
        
else:
        
    backtesting_data = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()
       



##########################################################################################
### Feature selection using config ###
##########################################################################################

# Apply standard filters
backtesting_data = standard_data_filter(backtesting_data)

# Create an ID column to easily filter the data
backtesting_data['id'] = backtesting_data.index

# Calculate the year and month of the fight
backtesting_data['year'] = backtesting_data['date'].dt.year
backtesting_data['month_year'] = backtesting_data['date'].dt.to_period('M').astype(str)

# Calculate total fights of both fighters
backtesting_data['total_fights'] = backtesting_data['fighter1_hist_fights'] + backtesting_data['fighter2_hist_fights']

# Create a column showing if both fighters veterans or not
backtesting_data['fighter_history'] = np.where(
    (backtesting_data['fighter1_hist_fights'] > 3) & (backtesting_data['fighter2_hist_fights'] > 3),
    '1. Both > 3 fights',
    np.where(
        (backtesting_data['fighter1_hist_fights'] > 2) & (backtesting_data['fighter2_hist_fights'] > 2),
        '2. Both > 2 fights',
        np.where(
            (backtesting_data['fighter1_hist_fights'] > 3) | (backtesting_data['fighter2_hist_fights'] > 3),
            '3. Only one fighter > 3 fights',
            np.where(
                (backtesting_data['fighter1_hist_fights'] == 0) | (backtesting_data['fighter2_hist_fights'] == 0),
                '5. One noob',
                np.where(
                    (backtesting_data['fighter1_hist_fights'] == 0) & (backtesting_data['fighter2_hist_fights'] == 0),
                    '6. Both noobs',
                    '4. Other')))))

backtesting_data['favourite'] = np.where(
    ((backtesting_data['fighter1_backtesting_odds'] < backtesting_data['fighter2_backtesting_odds']) & (backtesting_data['outcome'] == 'W/L')) |
    ((backtesting_data['fighter1_backtesting_odds'] > backtesting_data['fighter2_backtesting_odds']) & (backtesting_data['outcome'] == 'L/W')),
    "Favourite win",
    np.where(~backtesting_data['outcome'].isin(['W/L', 'L/W']),
             "Other",
             "Non-favourite win"
             ))

# Group the method of victory so all the decisions can be looked at together
backtesting_data['method_group'] = np.where(
    backtesting_data['method'].str.startswith('Decision', na = False),
    'Decision',
    np.where(
        backtesting_data['method'].isin(['KO/TKO', 'Submission']),
        backtesting_data['method'],
        'Other'))

# Split the sample either side of the split date. Numbered so they sort into date order
backtesting_data['period'] = np.where(
    backtesting_data['date'] < pd.Timestamp(period_split_date),
    '1. Pre ' + period_split_date,
    '2. Post ' + period_split_date)

##########################################################################################
### Data prep for model ###
##########################################################################################

# Get the model input data
# Exclude fights that haven't happened yet. They carry the oot flag because of their date, but have
# no outcome, and clean_target_column raises on a missing outcome.
# Draws and no contests go too. A bet on one is voided and the stake comes back, so they say nothing
# about how the model is doing, and the model gives them so little probability that a couple of them
# landing in a window moves the logloss more than everything else put together
oot_data = backtesting_data[(backtesting_data['train_test_oot_flag'] == 'oot')
                            & backtesting_data['outcome'].notna()
                            & ~backtesting_data['outcome'].isin(['D/D', 'NC/NC'])]

X_oot, temp1, y_oot, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
    oot_data,
    oot_testing = True)



##########################################################################################
### Calculating metrics ###
##########################################################################################

# Set which columns and metrics we're interested in
breakdown_cols = ['weightclass', 'year', 'month_year', 'time_format', 'fighter_history', 'date']
breakdown_metrics= ['Avg WL only AUC', 'X Odds Avg WL only AUC', 'Hmean WL only Accuracy', 'X Odds Hmean WL only Accuracy', 'Logloss', 'X Odds Logloss', 'EV MSE', 'X Odds EV MSE', 'Sample Size']
target = 'outcome'

backtesting_params = {
    'min_prob_to_bet': bet_config['min_prob_to_bet'],
    'min_ev_to_bet': bet_config['min_ev_to_bet'],
    'max_bet_size': 100,
    'min_bet_size': 100,
    'constant_bets': True
}


# Loop through each model version and test
for model_version in model_versions:
    
    with mlflow.start_run() as run:
        
        # Get the model predictions
        prediction_probabilities, temp = predict_xgboost_ensemble(
            rf_xgb_models[str(model_version)], 
            X_oot, 
            [0] * len(X_oot), 
            model_hyper_params = model_hyper_params)
        
        # Merge the data on the train to ensure the same order of rows
        oot_actuals = X_oot[['id']].merge(backtesting_data, on = 'id', how = 'left')

        # The outcome as classes, the same way btest_model_and_log_mlflow does it
        oot_target = clean_target_column(oot_actuals['outcome'])

        # These two have to run before btest_model_and_log_mlflow, which converts its breakdown
        # columns to strings and so would leave the date column unusable for date arithmetic

        # Log the logloss over a trailing window, so a drift shows up as a trend
        rolling_logloss = evaluation.rolling_log_loss(
            oot_actuals,
            prediction_probabilities,
            oot_target,
            months = rolling_window_months)

        # Nothing to chart if the sample is too short or too thin for any window to qualify
        if len(rolling_logloss) == 0:
            print('No rolling logloss logged, no ' + str(rolling_window_months) + ' month window held enough fights')
        else:
            fig, ax = evaluation.plot_rolling_log_loss(rolling_logloss, rolling_window_months)
            mlflow.log_figure(fig, 'Rolling logloss/rolling_logloss.png')

            rolling_logloss.to_csv('results/temp/summary.csv', index = False)
            mlflow.log_artifact('results/temp/summary.csv', 'Rolling logloss')

        # Log the logloss by method of victory either side of the split date, to see whether
        # decisions in particular got harder to call, and whether they did for the odds too
        period_logloss = evaluation.log_loss_by_group_and_period(
            oot_actuals,
            prediction_probabilities,
            oot_target,
            'method_group',
            'period')

        print(period_logloss)

        period_logloss.to_csv('results/temp/summary.csv')
        mlflow.log_artifact('results/temp/summary.csv', 'Logloss by method and period')

        btest_model_and_log_mlflow(
            oot_actuals,
            prediction_probabilities, 
            breakdown_cols = breakdown_cols, 
            breakdown_metrics = breakdown_metrics, 
            mlflow_description = 'XGB Forest v' + str(model_version), 
            target_column = target,
            backtesting_params = backtesting_params)
        
        mlflow.log_param('OOT_start_date', min(oot_data['date']))
        mlflow.log_param('Last_event_date', max(backtesting_data['date']))
        mlflow.set_tag('Sample', 'OOT')
