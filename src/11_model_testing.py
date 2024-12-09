# This script fits XGBoost models to the final data

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

# Set the model location
model_path = './results/final_xgboost_forest.joblib'

# Read in the data for training
backtesting_data = concat_parquets_fastparquet('./data/final_dataset/')
# Load files
rf_xgb_model = joblib.load(model_path)

##########################################################################################
### Feature selection using config ###
##########################################################################################

# Apply standard filters
backtesting_data = standard_data_filter(backtesting_data)

# Create an ID column to easily filter the data
backtesting_data['id'] = backtesting_data.index

# Calculate the year of the fight
backtesting_data['year'] = backtesting_data['date'].dt.year

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
    ((backtesting_data['fighter1_best_closing_odds'] < backtesting_data['fighter2_best_closing_odds']) & (backtesting_data['outcome'] == 'W/L')) |
    ((backtesting_data['fighter1_best_closing_odds'] > backtesting_data['fighter2_best_closing_odds']) & (backtesting_data['outcome'] == 'L/W')),
    "Favourite win",
    np.where(~backtesting_data['outcome'].isin(['W/L', 'L/W']),
             "Other",
             "Non-favourite win"
             ))

##########################################################################################
### Data prep for model ###
##########################################################################################

# Prep the data ready for model training as it was done in the overall model training
X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(backtesting_data, config, test_size = 0.1)

# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('final_models_oot_testing_v2')

# Set this model's description for MLflow
model_description = 'XGB forest - GT2 fights split dec exc, ' + str(len(rf_xgb_model)) + ' trees, ' + str(len(final_model_features)) + ' features'

##########################################################################################
### Calculating metrics ###
##########################################################################################

# Set which columns and metrics we're interested in
breakdown_cols = ['weightclass', 'year', 'fighter_history', 'favourite', 'matchup', 'date', 'total_fights']
breakdown_metrics= ['Avg WL only AUC', 'X Odds Avg WL only AUC', 'Hmean WL only Accuracy', 'X Odds Hmean WL only Accuracy', 'Sample Size']
target = 'outcome'


# Backtest for both constant terms and non-constant bets
for backtesting_params in [
        {
            'min_prob_to_bet': 0.5,
            'min_ev_to_bet': 0,
            'max_bet_size': 200,
            'min_bet_size': 85,
            'constant_bets': True
        },
        {
            'min_prob_to_bet': 0.5,
            'min_ev_to_bet': 0,
            'max_bet_size': 0.2,
            'min_bet_size': 0.085,
            'start_balance': 1000,
            'constant_bets': False
        }]:
    
    # Run the metrics on the training data
    with mlflow.start_run():
        
        # Get the model predictions
        prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_train, y_train)
        
        btest_model_and_log_mlflow(
            X_train[['id']].merge(backtesting_data, on = 'id', how = 'left'), # merge the data on the train to ensure the same order of rows
            prediction_probabilities, 
            breakdown_cols = breakdown_cols, 
            breakdown_metrics = breakdown_metrics, 
            mlflow_description = model_description + ' - Train data', 
            target_column = target,
            backtesting_params = backtesting_params)
        
        # Log the model as an artifact so we have a record of it
        mlflow.log_artifact(model_path, 'Model_object')
        mlflow.set_tag('Sample', 'Train')
        
    mlflow.end_run()
    
    # Run the metrics on the test data
    with mlflow.start_run():
        
        # Get the model predictions
        prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_test, y_test)
        
        btest_model_and_log_mlflow(
            X_test[['id']].merge(backtesting_data, on = 'id', how = 'left'), # merge the data on the train to ensure the same order of rows, 
            prediction_probabilities, 
            breakdown_cols = breakdown_cols, 
            breakdown_metrics = breakdown_metrics, 
            mlflow_description = model_description + '  - Test data', 
            target_column = target,
            backtesting_params = backtesting_params)
        
        mlflow.set_tag('Sample', 'Test')
        
    mlflow.end_run()
    
    # Run the metrics on the OOT data
    with mlflow.start_run():
        
        # Get the model input data
        X_oot, temp1, y_oot, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
            oot, 
            config = {'oot_start_date': '2999-01-01', 'min_training_date': '1000-01-01', 'seed_all': 1}, 
            split_or_other_flag = False, 
            test_size = 1e-99)
        
        # Get the model predictions
        prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_oot, y_oot)
        
        btest_model_and_log_mlflow(
            X_oot[['id']].merge(backtesting_data, on = 'id', how = 'left'), # merge the data on the train to ensure the same order of rows, 
            prediction_probabilities, 
            breakdown_cols = breakdown_cols, 
            breakdown_metrics = breakdown_metrics, 
            mlflow_description = model_description + '  - OOT data', 
            target_column = target,
            backtesting_params = backtesting_params)
        
        mlflow.set_tag('Sample', 'OOT')
        
    mlflow.end_run()


# # If the model is good, then log it in the registery
# if model_approved:
#     mlflow.register_model('runs:/61115eb8a09f496f815535119dad8900/artifacts/Model_object', name = 'xgb_forest_model')


# test = mlflow.pyfunc.load_model('models:/xgb_forest_model/5')
