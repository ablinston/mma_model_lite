# This script fits XGBoost models to the final data

import mlflow

# Read in global objects and packages
import fastparquet as fp
import joblib
import numpy as np
import pandas as pd
from yaml import dump

from mma_functions.cleaning import standard_data_filter
from mma_functions.dataset import prep_data_for_model_training
from mma_functions.features import switch_fighter1_fighter2
from mma_functions.modelling import log_metrics_xgboost_ensemble_mlflow
from mma_functions.pipelines import train_ensemble_xgboost
from mma_functions.tracking import log_metrics_odds_mlflow
from mma_functions.config import bet_config, config, model_hyper_params

# Set the version number for the model this run will save
new_version_number = "4.4.0"

# Load the model to be refreshed
# rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/2')
rf_xgb_model = joblib.load(bet_config["model_to_use"])

# Everything about the model comes off the one being refreshed, so this is a frozen refit on newer
# data rather than a redevelopment. All entries in the ensemble carry the same params and features
params = rf_xgb_model[0]["params"].copy()
all_features = rf_xgb_model[0]["features"]
num_models = len(rf_xgb_model)

# Train this many trees beyond the size being refreshed, so the metrics can be read either side
# of it rather than at one size only
trees_either_side = 5

# Read in the data for training
data = fp.ParquetFile("./data/final_dataset/final_dataset.parquet").to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

# Apply standard filters
data = standard_data_filter(data)

##########################################################################################
### Feature selection using config ###
##########################################################################################

# Don't need to select columns now. The train function takes the feature selection as an input

# # Only select the columns needed for training and the test train split
# data = data[[col for col in data.columns if col in final_model_features + ['date', 'method', 'fighter1', 'fighter2', 'index', 'outcome', 'fighter1_backtesting_odds', 'fighter2_backtesting_odds']]]

if model_hyper_params["switch_all_fighters_and_stack"] == True:
    data = pd.concat([
        data,
        switch_fighter1_fighter2(data, [True] * len(data))
    ])
else:
    # Switch fighers if necessary
    if model_hyper_params["switch_fighters_with_odds"] == True:
        data = switch_fighter1_fighter2(data, data["fighter1_backtesting_odds"] > data["fighter2_backtesting_odds"])
    if model_hyper_params["switch_fighters_randomly"] == True: # Commented this out because I don't think we want random for the RF
        np.random.seed(seed=config["seed_all"] + 8451)
        data = switch_fighter1_fighter2(data, np.random.choice([True, False], size=len(data)))

##########################################################################################
### Data prep for model ###
##########################################################################################

# Prep the data ready for model training
X_train_master, X_test, y_train, y_test, y_test_bin, temp, oot = prep_data_for_model_training(data, split_or_other_flag=True)

# Remove other columns from training data as not required
X_train_master = X_train_master[all_features + ["fighter1_backtesting_odds", "fighter2_backtesting_odds", "notformodel_split_dec_or_other", "notformodel_fighter1", "notformodel_fighter2"]]

# Set up experiment training
mlflow.set_tracking_uri(uri="http://mlflow-mma-model.192.168.68.51.nip.io:8081")



# Save features used to a YAML file
with open("./configs/final_model_features.yaml", "w") as file:
    dump(list(set(all_features)), file, default_flow_style=False)


##########################################################################################
### XGBoost and Forest parameters
##########################################################################################

# Prep the data ready for model training
X_train, X_test, y_train, y_test, y_test_bin, temp, oot = prep_data_for_model_training(data, split_or_other_flag=True)


# Set hyperparameters for the forest
colsample_bynode = 1
subsample = 0.6


##########################################################################################
### Model training
##########################################################################################

# Train the models
xgboost_rf = train_ensemble_xgboost(
    X_train,
    y_train,
    X_columns_for_training=all_features,
    num_models=num_models + trees_either_side,
    num_boost_round=params["num_iterations"],
    colsample_bynode=colsample_bynode, # If it's only 1 tree, train the full XGboost model
    subsample=subsample,
    early_stopping_rounds=params["early_stopping_rounds"],
    params=params
    )

# Try each number of trees either side of the size being refreshed
mlflow.set_experiment("xgb_model_v" + new_version_number + "_refresh_trees")

# for num_trees in [1]:
for num_trees_i in range(max(1, num_models - trees_either_side), len(xgboost_rf) + 1):

    # Start a new model run in MLflow
    with mlflow.start_run():
        
        # Log the initial metrics
        mlflow.log_params(params)
        mlflow.log_param("num_trees", num_trees_i)
        mlflow.log_param("Forest col proportion", colsample_bynode)
        mlflow.log_param("Forest subsample", subsample)
        mlflow.set_tag("Sample", "Train")
                    
        # Log the metrics of the forest with that many trees in MLflow
        log_metrics_xgboost_ensemble_mlflow(xgboost_rf[0:(num_trees_i)], X_train, y_train, params)
        
        # Log the predictions just from the odds for info
        log_metrics_odds_mlflow(X_train, y_train)
        
    # Start a new model run in MLflow
    with mlflow.start_run():
        
        # Log the initial metrics
        mlflow.log_params(params)
        mlflow.log_param("num_trees", num_trees_i)
        mlflow.log_param("Forest col proportion", colsample_bynode)
        mlflow.log_param("Forest subsample", subsample)
        mlflow.set_tag("Sample", "Test")
                    
        # Log the metrics of the forest with that many trees in MLflow
        log_metrics_xgboost_ensemble_mlflow(xgboost_rf[0:(num_trees_i)], X_test, y_test, params)
        
        # Log the predictions just from the odds for info
        log_metrics_odds_mlflow(X_test, y_test)


# Keep the same number of models as the ensemble being refreshed
final_forest_size = num_models


# Save the model where the rest of the pipeline picks it up, the same as the 13 scripts do
joblib.dump(xgboost_rf[0:final_forest_size], "./results/final_xgboost_forest.joblib", compress=9)

# This is only the working copy. Once the metrics in MLflow pass, promote it by hand with:
# joblib.dump(xgboost_rf[0:final_forest_size], "./models/xgb_forest_v" + new_version_number + ".joblib", compress=9)


##########################################################################################
### Out of time testing ###
##########################################################################################

# Score the final ensemble on the oot sample once, after the forest size has been chosen on the test
# sample. Doing it per tree count would turn oot into another selection set and the number would stop
# being an honest estimate of how the model does on fights it has never seen
X_oot, temp1, y_oot, temp2, temp3, temp4, temp5 = prep_data_for_model_training(oot, oot_testing=True)

with mlflow.start_run():

    # Log the initial metrics
    mlflow.log_params(params)
    mlflow.log_param("num_trees", final_forest_size)
    mlflow.log_param("Forest col proportion", colsample_bynode)
    mlflow.log_param("Forest subsample", subsample)
    mlflow.set_tag("Sample", "OOT")

    # Log the metrics of the final forest in MLflow
    log_metrics_xgboost_ensemble_mlflow(xgboost_rf[0:final_forest_size], X_oot, y_oot, params)

    # Log the predictions just from the odds for info
    log_metrics_odds_mlflow(X_oot, y_oot)
