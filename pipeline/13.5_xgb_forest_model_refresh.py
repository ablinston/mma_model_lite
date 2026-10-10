# This script fits XGBoost models to the final data

import gc
import h2o
from h2o.automl import H2OAutoML
from hyperopt import hp, STATUS_OK, tpe, fmin, rand, Trials # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
from hyperopt.early_stop import no_progress_loss
import hyperopt
import itertools
import lightgbm as lgb
import mlflow
import pickle
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

# Read in global objects and packages
import fastparquet as fp
import joblib
import numpy as np
import pandas as pd
from yaml import dump

from mma_functions.cleaning import standard_data_filter
from mma_functions.dataset import prep_data_for_model_training
from mma_functions.features import switch_fighter1_fighter2
from mma_functions.modelling import log_metrics_xgboost_ensemble_mlflow, objective_xgboost_features_mlflow
from mma_functions.pipelines import train_ensemble_xgboost
from mma_functions.tracking import log_metrics_odds_mlflow
from mma_functions.config import bet_config, config, model_hyper_params

# Load the model to be refreshed
# rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/2')
rf_xgb_model = joblib.load(bet_config["model_to_use"])

# Set the new model version number
new_version_number = "2.0.13"

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
X_train_master, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, split_or_other_flag=True)

# Remove other columns from training data as not required
# Get a list of all the features needed
all_features = []
for i in range(0, len(rf_xgb_model)):
    all_features = all_features + rf_xgb_model[i]["features"]

# Add on manually any features we want in there
all_features = all_features + ["fighter1_date_minus_dob", "fighter2_date_minus_dob", "diff_date_minus_dob"]

all_features = list(set(all_features))

# Record how many features there are
number_of_features = len(all_features)

# Remove the rest from the data
X_train_master = X_train_master[all_features + ["fighter1_backtesting_odds", "fighter2_backtesting_odds", "notformodel_split_dec_or_other", "notformodel_fighter1", "notformodel_fighter2"]]

# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("xgb_model_v2_refresh")



##########################################################################################
### XGBoost optimise hyperparameters
##########################################################################################

# Find the optimal hyperparameters for this set of features

trials = Trials()

space = {
    # Set the feature objective params
    "max_evals": 500,
    "num_of_features": hp.choice("num_of_features", [number_of_features]),
    "split_dec_or_other": False, # model_hyper_params['split_or_other_exclude'], # Want it to be in the test data for cross validation regardless
    "hyperopt_early_stopping": 75
}

# Set the hyperopt parameters for the inner optimisation
params_inner = {
     "nthread": 3, # Only applies to CPU cores
     "learning_rate": hp.uniform("learning_rate", 0.005, 0.2), # Learning rate #  0.01-0.2 is considered low-moderate learning rates
     "min_child_weight": 15, # hp.quniform('min_child_weight', 8, 16, 1), # The min observations that should be left in a leaf node    
     "subsample": 0.85,
     "colsample_bytree": hp.quniform("colsample_bytree", 0.2, 0.5, 0.05),
     "max_depth": hp.quniform("max_depth", 1, 3, 1),
     "max_bin": 200, # used in the hist method to speed up, this says how many bins the vars should be split into
     "objective": "multi:softprob", # Multi classification objective
     "eval_metric": "mlogloss",       # Evaluation metric
     "tree_method": "hist",      # Use GPU accelerated algorithm
     "device": "cuda",           # Use CUDA / GPU
     "predictor": "gpu_predictor",   # Use GPU for prediction
     "verbosity": 1,
     "num_class": no_of_classes,
     "n_folds": 5,                   # Number of folds for the cross-validation
     "random_state": config["seed_all"] + 39,
     "num_iterations": 5000,
     "early_stopping_rounds": 75,
     # 'reg_alpha': hp.uniform('reg_alpha', 0, 1), # Lasso regularisation
     # 'reg_lambda': hp.uniform('reg_lambda', 0, 3), # ridge regularisation
     # 'gamma': hp.uniform('gamma', 0, 10), # controls when the nodes have a sufficient enough split
     "exclude_split_other_from_train": hp.choice("exclude_split_other_from_train", [True, False]),
     "switch_all_fighters_and_stack": model_hyper_params["switch_all_fighters_and_stack"],
     "win_loss_only": hp.choice("win_loss_only", [True, False]),
     "loss_function" : model_hyper_params["loss_function"],
     "trim_model_features": False, # hp.choice('trim_model_features', [True, False])
}

best = fmin(fn=objective_xgboost_features_mlflow,
            space=space,
            algo=atpe_suggest,
            max_evals=1,
            trials=trials)


# Save features used to a YAML file
with open("./configs/final_model_features.yaml", "w") as file:
    dump(list(set(all_features)), file, default_flow_style=False)


##########################################################################################
### XGBoost and Forest parameters
##########################################################################################

# Prep the data ready for model training
X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, split_or_other_flag=True)


params = trials.best_trial["result"]["params"]
params.update({
# params = {
    # 'nthread': 3, # Only applies to CPU cores
    # 'learning_rate': 0.0225, # Learning rate
    # 'min_child_weight': 15,                 # The min observations that should be left in a leaf node
    # 'split_dec_or_other': params['exclude_split_other_from_train'],
    # 'subsample': 0.85,
    # 'colsample_bytree': 0.95,
    # 'max_depth': 1,
    # # 'max_bin': 200, # used in the hist method to speed up, this says how many bins the vars should be split into
    # 'objective': 'multi:softprob', # Multi classification objective
    # 'eval_metric': 'mlogloss',       # Evaluation metric
    # 'tree_method': 'hist',      # Use GPU accelerated algorithm
    # 'device': 'cuda',           # Use CUDA / GPU
    # 'predictor': 'gpu_predictor',   # Use GPU for prediction
    # 'verbosity': 1,
    # 'num_class': no_of_classes,
    # 'random_state': config['seed_all'] + 38,
    # 'num_iterations': 5000,
    # 'early_stopping_rounds': 30,
    # 'reg_alpha': 0.3, # Lasso regularisation
    # 'reg_lambda': 0.02, # ridge regularisation
    # # 'gamma': 1.13,
    # 'win_loss_only': False,
    "switch_fighters_randomly": False if model_hyper_params["switch_all_fighters_and_stack"] else model_hyper_params["switch_fighters_randomly"], # Set whether each tree should have randomised fighters
    # 'loss_function' : model_hyper_params['loss_function'],
    "add_noise_to_odds": model_hyper_params["add_noise_to_odds"],
    "odds_noise_standard_dev": model_hyper_params["odds_noise_standard_dev"],
    "xgb_forest_bagging": model_hyper_params["xgb_forest_bagging"],
    "base_margin_cols": model_hyper_params["base_margin_cols"] 
}
)


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
    num_models=150,
    num_boost_round=5000,
    colsample_bynode=colsample_bynode, # If it's only 1 tree, train the full XGboost model
    subsample=subsample,
    early_stopping_rounds=30,
    params=params
    )

# Try different numbers of trees
mlflow.set_experiment("xgb_model_v" + new_version_number + "_refresh_trees")

# for num_trees in [1]:
for num_trees_i in list(range(1, 51, 2)) + list(range(50, len(xgboost_rf) + 1, 10)):
        
    if len(xgboost_rf) >= num_trees_i:
        
        # Start a new model run in MLflow
        with mlflow.start_run():
            
            # Log the initial metrics
            mlflow.log_params(params)
            mlflow.log_param("num_trees", num_trees_i)
            mlflow.log_param("Forest col proportion", colsample_bynode)
            mlflow.log_param("Forest subsample", subsample)
            mlflow.set_tag("Sample", "Train")
                        
            # Log the metrics of the forest with that many trees in MLflow
            log_metrics_xgboost_ensemble_mlflow(xgboost_rf[0:(num_trees_i)], X_train, y_train)
            
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
            log_metrics_xgboost_ensemble_mlflow(xgboost_rf[0:(num_trees_i)], X_test, y_test)
            
            # Log the predictions just from the odds for info
            log_metrics_odds_mlflow(X_test, y_test)


# Set the final number of models
final_forest_size = 2


# Save the model
joblib.dump(xgboost_rf[0:final_forest_size], "./results/final_xgboost_forest.joblib", compress=9)
