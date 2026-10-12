# This script fits XGBoost models to the final data

import gc
# import h2o
# from h2o.automl import H2OAutoML
from hyperopt import hp, STATUS_OK, tpe, fmin, Trials # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
import itertools
# import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import pickle
import shap
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns
import xgboost as xgb

# Read in global objects and packages
import fastparquet as fp
import joblib
import numpy as np

from mma_functions.cleaning import standard_data_filter
from mma_functions.dataset import prep_data_for_model_training
from mma_functions.modelling import predict_xgboost_ensemble
from mma_functions.pipelines import btest_model_and_log_mlflow
from mma_functions.tracking import log_shap_values_mlflow
from mma_functions.config import config, model_hyper_params

# Set the model location
model_path = "./results/final_xgboost_forest.joblib"

# model_path = './models/xgb_forest_v4.0.14.joblib'
# # Set the version number to be used in outputs
new_version_number = '4.1.0d1'

# Read in the data for training
backtesting_data = fp.ParquetFile("./data/final_dataset/final_dataset.parquet").to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

# Load files
rf_xgb_model = joblib.load(model_path)
# rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/1')._model_impl.python_model.model

##########################################################################################
### Feature selection using config ###
##########################################################################################

# Apply standard filters
backtesting_data = standard_data_filter(backtesting_data)

# Create an ID column to easily filter the data
backtesting_data["id"] = backtesting_data.index

# Calculate the year and month of the fight
backtesting_data["year"] = backtesting_data["date"].dt.year
backtesting_data["month_year"] = backtesting_data["date"].dt.to_period("M").astype(str)

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

# Rename columns if needed
if "v3" in model_path:
    backtesting_data.columns = backtesting_data.columns.str.replace("_sub_attempt", "_sub.att")

# Prep the data ready for model training as it was done in the overall model training
X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(backtesting_data)

# Set up experiment training
mlflow.set_tracking_uri(uri="http://mlflow-mma-model.192.168.68.51.nip.io:8081")
mlflow.set_experiment("OOT_testing")

# Set this model's description for MLflow
model_description = "XGB forest v" + new_version_number + ", " + str(len(rf_xgb_model)) + " trees, " + str(len(rf_xgb_model[0]["features"])) + " features " + str(config["test_proportion"]) + " test size"

##########################################################################################
### Calculating metrics ###
##########################################################################################

# Set which columns and metrics we're interested in
breakdown_cols = ["weightclass", "year", "month_year", "time_format", "fighter_history", "date"]
breakdown_metrics= ["Avg WL only AUC", "X Odds Avg WL only AUC", "Hmean WL only Accuracy", "X Odds Hmean WL only Accuracy", "Logloss", "X Odds Logloss", "Sample Size"]
target = "outcome"


# Backtest for both constant terms and non-constant bets
for backtesting_params in [
        # {
        #     'min_prob_to_bet': 0.49,
        #     'min_ev_to_bet': 0.0,
        #     'max_bet_size': 0.05,
        #     'min_bet_size': 0.05,
        #     'start_balance': 2000,
        #     'constant_bets': False,
        #     'point_in_time_only_odds': True
        # }
        # # ,
        {
            "min_prob_to_bet": 0.485,
            "min_ev_to_bet": 0.01,
            "max_bet_size": 100,
            "min_bet_size": 100,
            "constant_bets": True,
            "point_in_time_only_odds": True
            # 'target_ev_after_cutoff': 0.05
        }
        ]:
    
    # Run the metrics on the training data
    with mlflow.start_run() as run:
        
        # Get the model predictions
        prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_train, y_train, model_hyper_params)

        # Log the SHAP values for the model
        log_shap_values_mlflow(rf_xgb_model, X_train)
        
        btest_model_and_log_mlflow(
            X_train[["id"]].merge(backtesting_data, on="id", how="left"), # merge the data on the train to ensure the same order of rows
            prediction_probabilities, 
            breakdown_cols=breakdown_cols, 
            breakdown_metrics=breakdown_metrics, 
            mlflow_description=model_description + " - Train data", 
            target_column=target,
            backtesting_params=backtesting_params)
        
        # Log the model as an artifact so we have a record of it
        mlflow.log_artifact(model_path, "Model_object")
        # mlflow.pyfunc.log_model(name = "Model", python_model = custom_xgboost_model(), artifacts={"model": model_path})
        
        # Save the run id for if we want to register the model
        run_id = run.info.run_id
        
        mlflow.set_tag("Sample", "Train")
        
    mlflow.end_run()
    
    # Run the metrics on the test data
    with mlflow.start_run():
        
        # Get the model predictions
        prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_test, y_test, model_hyper_params)
        
        # Log the SHAP values for the model
        log_shap_values_mlflow(rf_xgb_model, X_test)
        
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
    
    # Run the metrics on the OOT data
    with mlflow.start_run():
        
        # Get the model input data
        X_oot, temp1, y_oot, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
            oot,
            oot_testing=True)
        
        # Log the SHAP values for the model
        log_shap_values_mlflow(rf_xgb_model, X_oot)
        
        # Get the model predictions
        prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_oot, y_oot, model_hyper_params)
        
        btest_model_and_log_mlflow(
            X_oot[["id"]].merge(backtesting_data, on="id", how="left"), # merge the data on the train to ensure the same order of rows, 
            prediction_probabilities, 
            breakdown_cols=breakdown_cols, 
            breakdown_metrics=breakdown_metrics, 
            mlflow_description=model_description + "  - OOT data", 
            target_column=target,
            backtesting_params=backtesting_params)
        
        mlflow.set_tag("Sample", "OOT")
        
    mlflow.end_run()

# # If the model is good, then log it in the registery
# if model_approved:
#     run_id = 'aa217e4b1bf442729c4f86abfba56f98'
#     mlflow.register_model(f'runs:/{run_id}/Model', name = 'xgb_forest_model')


# test = mlflow.pyfunc.load_model('models:/xgb_forest_model/2')
