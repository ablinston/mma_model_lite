# This script fits XGBoost models to the future events data after refreshing data scrape

import gc
import h2o
from h2o.automl import H2OAutoML
from hyperopt import hp, STATUS_OK, tpe, fmin, Trials # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
import itertools
import lightgbm as lgb
import mlflow
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

# Read in global objects and packages
exec(open('global.py').read())

# Replace the firefox location if we're not on windows
if not os.name == "nt":
    config['firefox_location'] = config['firefox_location_pi']

# Use this to skip update of data if it's already been done recently
skip_data_update = False

# Read in the data for training
forecasting_data = pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

# # Load files
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/3')
# Set the model location
# model_path = './results/final_xgboost_forest.joblib'
# Load files
rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/1')

##########################################################################################
### Update the upcoming event and odds data ###
##########################################################################################

if not skip_data_update:
    
    # Run these scripts to scrape then clean the raw data
    exec(open('./src/4_scrape_all_unparsed_data.py').read())
    exec(open('./src/5_data_cleaning.py').read())
    
    # Optional, can either rerun all the processing, or just update the odds. By default will rerun all processing if new events were found to have completed
    if True: # unparsed_events: Hard coded this to always run because fights can change if fighters drop out and otherwise they wont have the hist stats
        exec(open('./src/7_data_processing.py').read())
        # Refresh the dataset
        forecasting_data = merged_data.copy()
    else:
        pass
        # add in only odds update
    

# Apply standard filters
forecasting_data = standard_data_filter(forecasting_data, include_upcoming = True)

##########################################################################################
### Choose which events to predict ###
##########################################################################################

upcoming_fights = forecasting_data[forecasting_data['outcome'].isna()]
# upcoming_fights = forecasting_data[forecasting_data['event'] == 'ufc311:makhachevvs.moicano']

##########################################################################################
### Get the models predictions ###
##########################################################################################

# Get the data ready for the model
X_upcoming, temp1, temp6, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
    upcoming_fights, 
    config = {'oot_start_date': '2999-01-01', 'min_training_date': '1000-01-01', 'seed_all': 1}, 
    split_or_other_flag = False, 
    test_size = 0,
    upcoming_fights = True)

# Get the model predictions
prediction_probabilities, temp = rf_xgb_model.predict(X_upcoming)

# Summarise the model predictions and betting decisions using the config
betting_data = summarise_model_preductions(
    actuals_data_df = upcoming_fights, 
    prediction_probabilities = prediction_probabilities, 
    min_prob_to_bet = bet_config['min_prob_to_bet'], 
    min_ev_to_bet = bet_config['min_ev_to_bet'], 
    min_bet_size = bet_config['min_bet_size'], 
    max_bet_size = bet_config['max_bet_size'])

# Add the final bet size based on balance
if bet_config['constant_bets'] == False:
    betting_data['Final_bet'] = betting_data['bets'] * betting_data['bet_size'] * bet_config['start_balance']
else:
    betting_data['Final_bet'] = betting_data['bets'] * betting_data['bet_size']

betting_data['Potential_win'] = betting_data['bets'] * betting_data['Final_bet'] * np.where(betting_data['Bet_trigger'] == betting_data['fighter1'], betting_data['fighter1_best_closing_odds'] - 1, betting_data['fighter2_best_closing_odds'] - 1)

summary_data = betting_data[['date', 'event', 'fighter1', 'fighter2', 'fighter1_best_closing_odds', 'fighter2_best_closing_odds', 'Fighter1_win_prob', 'Fighter2_win_prob', 'Fighter1_win_EV', 'Fighter2_win_EV', 'Bet_trigger', 'Final_bet', 'Potential_win']]

print(
      summary_data
)

print(
      betting_data[(betting_data['bets'] == 1) & (betting_data['date'] == betting_data['date'].min())][['date', 'event', 'fighter1', 'fighter2', 'fighter1_best_closing_odds', 'fighter2_best_closing_odds', 'Bet_trigger', 'Final_bet', 'Potential_win']]
)

# Output the results
summary_data.to_csv('./results/upcoming_predictions.csv')
