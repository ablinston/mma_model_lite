# This script fits XGBoost models to the final data

import gc
# import h2o
# from h2o.automl import H2OAutoML
from hyperopt import hp, STATUS_OK, tpe, fmin, rand, Trials # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
from hyperopt.early_stop import no_progress_loss
import hyperopt
import itertools
import lightgbm as lgb
import mlflow
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb
import warnings

# Read in global objects and packages
exec(open('global.py').read())

np.warnings = warnings # to avoid ATPE error with hyperopt

# Read in the data for training
data = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()


##########################################################################################
### Filtering ###
##########################################################################################

# Apply standard filters
data = standard_data_filter(data)

if model_hyper_params['switch_all_fighters_and_stack'] == True:
    data = pd.concat([
        data,
        switch_fighter1_fighter2(data, [True] * len(data))
    ])
else:
    # Switch fighers if necessary
    if model_hyper_params['switch_fighters_with_odds'] == True:
        data = switch_fighter1_fighter2(data, data['fighter1_backtesting_odds'] > data['fighter2_backtesting_odds'])
    if model_hyper_params['switch_fighters_randomly'] == True: # Commented this out because I don't think we want random for the RF
        np.random.seed(seed = config['seed_all'] + 8451)
        data = switch_fighter1_fighter2(data, np.random.choice([True, False], size = len(data)))

# Compute what k-fold each fight is in for cross-validation
data['notformodel_fold_id'] = k_fold_id_by_runner(5, data['fighter1'], data['fighter2'])


##########################################################################################
### Data prep for model ###
##########################################################################################

# Prep the data ready for model training. Create a master for the train because we will vary the number of features
X_train_master, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, split_or_other_flag = True)


##########################################################################################
### Feature selection - Varies ###
##########################################################################################


# Read in the variable importance
var_importance = pd.read_csv('./results/shap_feature_importance.csv').sort_values('Mean SHAP Value', ascending = False)

# Ensure that the data columns are ordered in descending feature importance
# First list any columns not in the importance
[col for col in X_train_master.columns if col not in var_importance['Feature'].tolist()]

# Now select the columns
columns_in_order = var_importance['Feature'].tolist()
# Get rid of the last X results
columns_in_order = [col for col in columns_in_order if not ('_last_' in col and '_results' in col)]

# Remove adjsum columns that don't make as much sense
columns_in_order = [col for col in columns_in_order if not (('_adjsum' in col))] # or ('_adj' in col and '_minus' in col))]

# Add on other columns needed
columns_in_order = columns_in_order + [col for col in X_train_master.columns if 'notformodel' in col and col not in columns_in_order]
    
if model_hyper_params['odds_to_use'] == 'best_closing_odds':
    columns_in_order = [col for col in columns_in_order if 'mean_closing_odds' not in col]
else:
    columns_in_order = [col for col in columns_in_order if 'best_closing_odds' not in col]
    
# Remove true odds if not only using them to avoid too much emphasis on odds
if model_hyper_params['true_odds_only'] == True:
    columns_in_order = [col for col in columns_in_order if not ('true' not in col and 'odds' in col)]
else:
    columns_in_order = [col for col in columns_in_order if not ('true' in col and 'odds' in col)]
    
# Remove columns we wish to exclude
columns_in_order = [col for col in columns_in_order if not any(substr in col for substr in feature_control['exclude_strings'])]

# Ensure we always keep the requested columns by putting them at the top
columns_in_order = [col for col in feature_control['include_features'] if col in X_train_master.columns] + columns_in_order

# Add on the odds needed for custom loss and backtesting
columns_in_order += ['fighter1_backtesting_odds', 'fighter2_backtesting_odds']

# Remove any duplicates
columns_in_order = pd.Series(columns_in_order).drop_duplicates().tolist()

# Remove any columns not in the data
columns_in_order = [col for col in columns_in_order if col in X_train_master.columns]

# Select only the needed columns without duplicates
X_train_master = X_train_master.loc[:, ~X_train_master.columns.duplicated()].copy()[columns_in_order]

# Now remove any columns that aren't wanted in a training dataset
X_train_master = standard_feature_removal(X_train_master, open_odds_only = model_hyper_params['open_odds_only'], differentials_only = model_hyper_params['differential_features_only'], true_odds_only = model_hyper_params['true_odds_only'])

# Create the empty training data for the inner optimisations
X_train = None


# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('fights_gt2_post_data_fix_v20')


##########################################################################################
### XGBoost
##########################################################################################

trials = Trials()

space = {
    # Set the feature objective params
    'max_evals': 150,
    'num_of_features': hp.choice('num_of_features', [i for i in range(30, 700, 10)]), # , 1000, 2000, len(X_train_master)]),
    'split_dec_or_other': False, # model_hyper_params['split_or_other_exclude'], # Want it to be in the test data for cross validation regardless
    'hyperopt_early_stopping': 15
}

# Set the hyperopt parameters for the inner optimisation
params_inner = {
     'nthread': 3, # Only applies to CPU cores
     'learning_rate': hp.uniform('learning_rate', 0.002, 0.2), # Learning rate #  0.01-0.2 is considered low-moderate learning rates
     'min_child_weight': hp.quniform('min_child_weight', 8, 21, 1), # The min observations that should be left in a leaf node    
     'subsample': 0.85,
     'colsample_bytree': hp.quniform('colsample_bytree', 0.01, 0.4, 0.01),
     'max_depth': hp.quniform('max_depth', 1, 8, 1),
     'max_bin': 180, # used in the hist method to speed up, this says how many bins the vars should be split into
     'objective': 'multi:softprob', # Multi classification objective
     'eval_metric': 'mlogloss',       # Evaluation metric
     'tree_method': 'hist',      # Use GPU accelerated algorithm
     'device': 'cuda',           # Use CUDA / GPU
     'predictor': 'gpu_predictor',   # Use GPU for prediction
     'verbosity': 1,
     'num_class': no_of_classes,
     'random_state': config['seed_all'] + 39,
     'num_iterations': 5000,
     'early_stopping_rounds': 10,
     # 'reg_alpha': hp.uniform('reg_alpha', 0, 1), # L1 regularisation
     # 'reg_lambda': hp.uniform('reg_lambda', 0, 5), # L2 regularisation
     'gamma': hp.uniform('gamma', 0, 10), # controls when the nodes have a sufficient enough split
     'exclude_split_other_from_train': True, # hp.choice('exclude_split_other_from_train', [True, False]),  # Almost all best trials end up with this as True
     'switch_all_fighters_and_stack': model_hyper_params['switch_all_fighters_and_stack'],
     'win_loss_only': True, # hp.choice('win_loss_only', [True, False]), # Almost all best trials end up with this as True
     'loss_function' : model_hyper_params['loss_function'],
     'trim_model_features': True, # hp.choice('trim_model_features', [True, False]),
     'base_margin_cols': model_hyper_params['base_margin_cols'] 
}

best = fmin(fn = objective_xgboost_features_mlflow,
            space = space,
            algo = atpe_suggest,
            max_evals = 100,
            trials = trials)



# Find more accurate parameters
trials = Trials()

space = {
    # Set the feature objective params
    'max_evals': 1000,
    'num_of_features': hp.choice('num_of_features', [170]), # , 1000, 2000, len(X_train_master)]),
    'split_dec_or_other': False, # model_hyper_params['split_or_other_exclude'], # Want it to be in the test data for cross validation regardless
    'hyperopt_early_stopping': 50
}

best = fmin(fn = objective_xgboost_features_mlflow,
            space = space,
            algo = atpe_suggest,
            max_evals = 1,
            trials = trials)

# # Output the feature set if we want to use it for the model
# X_train.columns.to_series().to_csv('./results/2014+_feature_set_6.csv', index = False)
# Convert column names to a list
column_names = X_train_master.columns.tolist()[0:160]

# Ensure that both fighter1 and fighter2 versions of columns are in there
column_names = column_names + list_swap_fighter1_fighter2(column_names)

if trim_model_features:
    column_names = trim_column_list(column_names)

# Save to a YAML file
with open('./configs/final_model_features.yaml', 'w') as file:
    dump(pd.Series(column_names).unique().tolist(), file, default_flow_style=False)
