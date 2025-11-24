# This script fits XGBoost models to the future events data after refreshing data scrape

import gc
# import h2o
# from h2o.automl import H2OAutoML
from hyperopt import hp, STATUS_OK, tpe, fmin, Trials # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
import itertools
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import seaborn as sns
import xgboost as xgb

# Read in global objects and packages
exec(open('global.py').read())

# Replace the firefox location if we're not on windows
if not os.name == "nt":
    config['firefox_location'] = config['firefox_location_pi']

# Use this to skip update of data if it's already been done recently
skip_data_update = False # Needs to be run again because bayes settings different
skip_save_data = False

# # Load files
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('Model_Monitoring_v2')

# Load all model versions
rf_xgb_models = {}
model_versions = [1, 2, '3.0.10']

for i in model_versions:
    rf_xgb_models.update({str(i): joblib.load('./models/xgb_forest_v' + str(i) + '.joblib')})


##########################################################################################
### Update the upcoming event and odds data ###
##########################################################################################

if not skip_data_update:
    
    # Run these scripts to scrape then clean the raw data
    # Run these scripts to scrape then clean the raw data
    scrape_all_unparsed_data(True, config, betfair_login_config)
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

# # For v1-3 need to rename columns
# backtesting_data.columns = backtesting_data.columns.str.replace("_sub_attempt", "_sub.att")

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

##########################################################################################
### Data prep for model ###
##########################################################################################

# Get the model input data
X_oot, temp1, y_oot, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
    backtesting_data[backtesting_data['train_test_oot_flag'] == 'oot'],
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
        
        btest_model_and_log_mlflow(
            X_oot[['id']].merge(backtesting_data, on = 'id', how = 'left'), # merge the data on the train to ensure the same order of rows, 
            prediction_probabilities, 
            breakdown_cols = breakdown_cols, 
            breakdown_metrics = breakdown_metrics, 
            mlflow_description = 'XGB Forest v' + str(model_version), 
            target_column = target,
            backtesting_params = backtesting_params)
        
        mlflow.log_param('OOT_start_date', min(backtesting_data[backtesting_data['train_test_oot_flag'] == 'oot']['date']))
        mlflow.log_param('Last_event_date', max(backtesting_data['date']))
        mlflow.set_tag('Sample', 'OOT')
