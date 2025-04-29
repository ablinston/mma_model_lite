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
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

# Read in global objects and packages
exec(open('global.py').read())

# Read in the data for training
data = pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

# Apply standard filters
data = standard_data_filter(data)

##########################################################################################
### Feature selection using config ###
##########################################################################################

# Don't need to select columns now. The train function takes the feature selection as an input

# # Only select the columns needed for training and the test train split
# data = data[[col for col in data.columns if col in final_model_features + ['date', 'method', 'fighter1', 'fighter2', 'index', 'outcome', 'fighter1_backtesting_odds', 'fighter2_backtesting_odds']]]

if model_hyper_params['switch_all_fighters_and_stack'] == True:
    data = pd.concat([
        data,
        switch_fighter1_fighter2(data, [True] * len(data))
    ])


##########################################################################################
### Data prep for model ###
##########################################################################################

# Prep the data ready for model training
X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, split_or_other_flag = True)

# # Remove other columns from training data as not required
# X_train = X_train[final_model_features]

# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('2014+_gt2_fights_xgboost_forest_v5')



##########################################################################################
### XGBoost and Forest parameters
##########################################################################################

params = {
    'nthread': 3, # Only applies to CPU cores
    'learning_rate': 0.029, # Learning rate
    'min_child_weight': 15,                 # The min observations that should be left in a leaf node
    'split_dec_or_other': True,
    'subsample': 0.85,
    'colsample_bytree': 0.65,
    'max_depth': 1,
    # 'max_bin': 200, # used in the hist method to speed up, this says how many bins the vars should be split into
    'objective': 'multi:softprob', # Multi classification objective
    'eval_metric': 'mlogloss',       # Evaluation metric
    'tree_method': 'hist',      # Use GPU accelerated algorithm
    'device': 'cuda',           # Use CUDA / GPU
    'predictor': 'gpu_predictor',   # Use GPU for prediction
    'verbosity': 1,
    'num_class': no_of_classes,
    'random_state': config['seed_all'] + 38,
    'num_iterations': 5000,
    'early_stopping_rounds': 30,
    # 'reg_alpha': 0.3, # Lasso regularisation
    # 'reg_lambda': 0.02, # ridge regularisation
    # 'gamma': 1.13,
    'win_loss_only': True,
    'switch_fighters_randomly': False if model_hyper_params['switch_all_fighters_and_stack'] else model_hyper_params['switch_fighters_randomly'], # Set whether each tree should have randomised fighters
    'loss_function' : model_hyper_params['loss_function'],
    'add_noise_to_odds': model_hyper_params['add_noise_to_odds'],
    'odds_noise_standard_dev': model_hyper_params['odds_noise_standard_dev'],
    'xgb_forest_bagging': model_hyper_params['xgb_forest_bagging']
}

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
    X_columns_for_training = final_model_features,
    num_models = 100,
    num_boost_round = 5000,
    colsample_bynode = colsample_bynode, # If it's only 1 tree, train the full XGboost model
    subsample = subsample,
    early_stopping_rounds = 30,
    params = params
    )

# Try different numbers of trees
# for num_trees in [1]:
for num_trees_i in list(range(1, 51, 2)) + list(range(50, len(xgboost_rf) + 1, 20)):
        
    if len(xgboost_rf) >= num_trees_i:
        
        # Start a new model run in MLflow
        with mlflow.start_run():
            
            # Log the initial metrics
            mlflow.log_params(params)
            mlflow.log_param('num_trees', num_trees_i)
            mlflow.log_param('Forest col proportion', colsample_bynode)
            mlflow.log_param('Forest subsample', subsample)
            mlflow.set_tag('Sample', 'Train')
                        
            # Log the metrics of the forest with that many trees in MLflow
            log_metrics_xgboost_ensemble_mlflow(xgboost_rf[0:(num_trees_i + 1)], X_train, y_train)
            
            # Log the predictions just from the odds for info
            log_metrics_odds_mlflow(X_train, y_train)
            
        # Start a new model run in MLflow
        with mlflow.start_run():
            
            # Log the initial metrics
            mlflow.log_params(params)
            mlflow.log_param('num_trees', num_trees_i)
            mlflow.log_param('Forest col proportion', colsample_bynode)
            mlflow.log_param('Forest subsample', subsample)
            mlflow.set_tag('Sample', 'Test')
                        
            # Log the metrics of the forest with that many trees in MLflow
            log_metrics_xgboost_ensemble_mlflow(xgboost_rf[0:(num_trees_i + 1)], X_test, y_test)
            
            # Log the predictions just from the odds for info
            log_metrics_odds_mlflow(X_test, y_test)


# Set the final number of models
final_forest_size = 15


# Save the model
joblib.dump(xgboost_rf[0:final_forest_size], './results/final_xgboost_forest.joblib', compress = 9)
