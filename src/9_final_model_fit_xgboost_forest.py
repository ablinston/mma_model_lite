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

# Read in the data for training
data = concat_parquets_fastparquet('./data/final_dataset/')

# Apply standard filters
data = standard_data_filter(data)

##########################################################################################
### Feature selection using config ###
##########################################################################################

# Only select the columns needed for training and the test train split
data = data[[col for col in data.columns if col in final_model_features + ['date', 'method', 'fighter1', 'fighter2', 'index', 'outcome', 'fighter1_best_closing_odds', 'fighter2_best_closing_odds']]]


##########################################################################################
### Data prep for model ###
##########################################################################################

# Prep the data ready for model training
X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, config, test_size = 0.1)

# Remove other columns from training data as not required
X_train = X_train[final_model_features]

# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('2014+_gt2_fights_xgboost_forest_v3')



##########################################################################################
### XGBoost and Forest parameters
##########################################################################################

params = {
    'nthread': 3, # Only applies to CPU cores
    'learning_rate': 0.26, # Learning rate
    'min_child_weight': 15,                 # The min observations that should be left in a leaf node    
    'subsample': 0.9,
    'max_depth': 1,
    # 'max_bin': 200, # used in the hist method to speed up, this says how many bins the vars should be split into
    'objective': 'multi:softprob', # Multi classification objective
    'eval_metric': 'auc',       # Evaluation metric
    'tree_method': 'hist',      # Use GPU accelerated algorithm
    'device': 'cuda',           # Use CUDA / GPU
    'predictor': 'gpu_predictor',   # Use GPU for prediction
    'verbosity': 1,
    'num_class': no_of_classes,
    'random_state': config['seed_all'] + 38,
    'num_iterations': 5000,
    'early_stopping_rounds': 30,
    'win_loss_only': False
}

# Set hyperparameters for the forest
colsample_bynode = 0.1
# subsample = 0.6


##########################################################################################
### Model training
##########################################################################################

# Try different numbers of trees
for num_trees in [150, 200, 250]:
    # Try different subsamples
    for subsample in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    
        # Start a new model run in MLflow
        with mlflow.start_run():
            
            # Log the initial metrics
            mlflow.log_params(params)
            mlflow.log_param('num_trees', num_trees)
            mlflow.log_param('Forest col proportion', colsample_bynode)
            mlflow.log_param('Forest subsample', subsample)
            
            # Train the models
            xgboost_rf = train_ensemble_xgboost(
                X_train,
                y_train,
                num_models = num_trees,
                num_boost_round = 5000,
                colsample_bynode = 1 if num_trees == 1 else colsample_bynode, # If it's only 1 tree, train the full XGboost model
                subsample = 1 if num_trees == 1 else subsample,
                early_stopping_rounds = 30,
                params = params
                )
            
            # Log the metrics in MLflow
            log_metrics_xgboost_ensemble_mlflow(xgboost_rf, X_test, y_test)
            
            # Log the predictions just from the odds for info
            log_metrics_odds_mlflow(X_test, y_test)


# Set the final number of models
final_forest_size = 200
final_subsample = 0.6

# Save the final model, choosing the number of trees
final_xgboost_rf = train_ensemble_xgboost(
    X_train,
    y_train,
    num_models = final_forest_size,
    num_boost_round = 5000,
    colsample_bynode = 1 if final_forest_size == 1 else colsample_bynode, # If it's only 1 tree, train the full XGboost model
    subsample = 1 if final_forest_size == 1 else final_subsample,
    early_stopping_rounds = 50,
    params = params
    )

# Save the model
joblib.dump(final_xgboost_rf, './results/final_xgboost_forest.joblib', compress = 9)
