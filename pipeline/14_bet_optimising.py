# This script takes the final model and looks to optimise betting strategies

import gc
from hyperopt import hp, STATUS_OK, tpe, fmin, Trials # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
import itertools
import mlflow
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

# Read in global objects and packages
import fastparquet as fp
import joblib
import numpy as np

from mma_functions.cleaning import standard_data_filter
from mma_functions.dataset import prep_data_for_model_training
from mma_functions.modelling import predict_xgboost_ensemble
from mma_functions.pipelines import btest_model_and_log_mlflow
from mma_functions.config import final_model_features

# Set the model location
model_path = "./results/final_xgboost_forest.joblib"

# Read in the data for training
backtesting_data = fp.ParquetFile("./data/final_dataset/final_dataset.parquet").to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

# Load files
rf_xgb_model = joblib.load(model_path)

##########################################################################################
### Feature selection using config ###
##########################################################################################

# Apply standard filters
backtesting_data = standard_data_filter(backtesting_data)

# Create an ID column to easily filter the data
backtesting_data["id"] = backtesting_data.index

# Calculate the year of the fight
backtesting_data["year"] = backtesting_data["date"].dt.year

# Calculate total fights of both fighters
backtesting_data["total_fights"] = backtesting_data["fighter1_hist_fights"] + backtesting_data["fighter2_hist_fights"]

# Create a column showing if both fighters veterans or not
backtesting_data["fighter_history"] = np.where(
    (backtesting_data["fighter1_hist_fights"] > 3) & (backtesting_data["fighter2_hist_fights"] > 3),
    "1. Both > 3 fights",
    np.where(
        (backtesting_data["fighter1_hist_fights"] > 2) & (backtesting_data["fighter2_hist_fights"] > 2),
        "2. Both > 2 fights",
        np.where(
            (backtesting_data["fighter1_hist_fights"] > 3) | (backtesting_data["fighter2_hist_fights"] > 3),
            "3. Only one fighter > 3 fights",
            np.where(
                (backtesting_data["fighter1_hist_fights"] == 0) | (backtesting_data["fighter2_hist_fights"] == 0),
                "5. One noob",
                np.where(
                    (backtesting_data["fighter1_hist_fights"] == 0) & (backtesting_data["fighter2_hist_fights"] == 0),
                    "6. Both noobs",
                    "4. Other")))))

backtesting_data["favourite"] = np.where(
    ((backtesting_data["fighter1_backtesting_odds"] < backtesting_data["fighter2_backtesting_odds"]) & (backtesting_data["outcome"] == "W/L")) |
    ((backtesting_data["fighter1_backtesting_odds"] > backtesting_data["fighter2_backtesting_odds"]) & (backtesting_data["outcome"] == "L/W")),
    "Favourite win",
    np.where(~backtesting_data["outcome"].isin(["W/L", "L/W"]),
             "Other",
             "Non-favourite win"
             ))


##########################################################################################
### Data prep for model ###
##########################################################################################

# Prep the data ready for model training as it was done in the overall model training
X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(backtesting_data)

# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("final_model_v2_bet_testing")

# Set this model's description for MLflow
model_description = "Odds to betfair 5pm " + str(len(rf_xgb_model)) + " trees, " + str(len(final_model_features)) + " features"

##########################################################################################
### Calculating metrics ###
##########################################################################################

# Set which columns and metrics we're interested in
breakdown_cols = ["weightclass", "year", "total_fights", "fighter_history"]
breakdown_metrics= ["Logloss", "EV MSE", "Avg WL only AUC", "X Odds Avg WL only AUC", "Hmean WL only Accuracy", "X Odds Hmean WL only Accuracy", "Sample Size"]
target = "outcome"

# Loop through all the minimum probabilities and ex[ected values to bet
for min_prob_to_bet in [0.3, 0.4, 0.49]: # np.arange(0.1, 0.7, 0.1):
    for min_ev_to_bet in [0, 0.1, 0.2]: # np.arange(0, 0.2, 0.05):
        for max_bet_size in [1]:#np.arange(0.1, 0.4, 0.02):
            for min_bet_size in [1]:#np.arange(0.01, 0.1, 0.01):
            
                backtesting_params = {
                    "min_prob_to_bet": min_prob_to_bet,
                    "min_ev_to_bet": min_ev_to_bet,
                    "max_bet_size": max_bet_size,
                    "min_bet_size": min_bet_size
                }
                
                # Run the metrics on the training data
                with mlflow.start_run():
                    
                    # Log parameters
                    mlflow.log_params(backtesting_params)
                    
                    # Get the model predictions
                    prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_train, y_train)
                    
                    btest_model_and_log_mlflow(
                        X_train[["id"]].merge(backtesting_data, on="id", how="left"), # merge the data on the train to ensure the same order of rows
                        prediction_probabilities, 
                        breakdown_cols=breakdown_cols, 
                        breakdown_metrics=breakdown_metrics, 
                        mlflow_description=model_description + " - Train data", 
                        target_column=target,
                        backtesting_params=backtesting_params)
                    
                    # Log the model as an artifact so we have a record of it
                    mlflow.log_artifact(model_path, "Model object")
                    mlflow.set_tag("Sample", "Train")
                    
                mlflow.end_run()
                
                # Run the metrics on the test data
                with mlflow.start_run():
                    
                    # Log parameters
                    mlflow.log_params(backtesting_params)
                    
                    # Get the model predictions
                    prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_test, y_test)
                    
                    btest_model_and_log_mlflow(
                        X_test[["id"]].merge(backtesting_data, on="id", how="left"), # merge the data on the train to ensure the same order of rows, 
                        prediction_probabilities, 
                        breakdown_cols=breakdown_cols, 
                        breakdown_metrics=breakdown_metrics, 
                        mlflow_description=model_description + "  - Test data", 
                        target_column=target,
                        backtesting_params=backtesting_params)
                    
                    mlflow.set_tag("Sample", "Test")
                    
                mlflow.end_run()
                
                # # Run the metrics on the OOT data
                # with mlflow.start_run():
                    
                #     # Log parameters
                #     mlflow.log_params(backtesting_params)
                    
                #     # Get the model input data
                #     X_oot, temp1, y_oot, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
                #         oot, 
                #         oot_testing = True)
                    
                #     # Get the model predictions
                #     prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_oot, y_oot)
                    
                #     btest_model_and_log_mlflow(
                #         X_oot[['id']].merge(backtesting_data, on = 'id', how = 'left'), # merge the data on the train to ensure the same order of rows, 
                #         prediction_probabilities, 
                #         breakdown_cols = breakdown_cols, 
                #         breakdown_metrics = breakdown_metrics, 
                #         mlflow_description = model_description + '  - OOT data', 
                #         target_column = target,
                #         backtesting_params = backtesting_params)
                    
                #     mlflow.set_tag('Sample', 'OOT')
                    
                # mlflow.end_run()
