# This script fits XGBoost models to the final data

import gc
import h2o
from h2o.automl import H2OAutoML
from hyperopt import hp, STATUS_OK, tpe, fmin, rand, Trials # hyperparameter optimisation functions
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


##########################################################################################
### Filtering ###
##########################################################################################

# Apply standard filters
data = standard_data_filter(data)

# Switch fighers if necessary
if model_hyper_params['switch_fighters_with_odds'] == True:
    data = switch_fighter1_fighter2(data, data['fighter1_best_closing_odds'] > data['fighter2_best_closing_odds'])


# Don't include split or other decisions in training data. This flag will be used in functions later to exclude them from training data but not from test data (including test in CV data)
split_or_other_exclude = True

##########################################################################################
### Data prep for model ###
##########################################################################################

# Prep the data ready for model training. Create a master for the train because we will vary the number of features
X_train_master, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, config, split_or_other_flag = split_or_other_exclude, test_size = 0.1)


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

# Add on the split indicator if needed
if split_or_other_exclude == True:
    columns_in_order = columns_in_order + ['split_dec_or_other']
    
X_train_master = X_train_master[columns_in_order]

# Now remove any columns that aren't wanted in a training dataset
X_train_master = standard_feature_removal(X_train_master, differentials_only = False)

# Create the empty training data for the inner optimisations
X_train = None


# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('2014+_mens_fights_gt2_split_exc_f1_fav_feature_opt')


##########################################################################################
### XGBoost
##########################################################################################

trials = Trials()

space = {
    # Set the feature objective params
    'max_evals': 700,
    'num_of_features': 230, # hp.choice('num_of_features', [3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 175, 190, 200, 210, 230, 250, 270, 300, 500, 1000, 2000, len(X_train_master)]),
    'split_dec_or_other': split_or_other_exclude
}

# Set the hyperopt parameters for the inner optimisation
params_inner = {
     'nthread': 3, # Only applies to CPU cores
     'learning_rate': hp.quniform('learning_rate', 0.001, 0.301, 0.005), # Learning rate
     'min_child_weight': hp.quniform('min_child_weight', 15, 21, 3), # The min observations that should be left in a leaf node    
     'subsample': 0.85,
     'max_depth': hp.quniform('max_depth', 1, 5, 1),
     'max_bin': 255, # used in the hist method to speed up, this says how many bins the vars should be split into
     'objective': 'multi:softprob', # Multi classification objective
     'eval_metric': 'auc',       # Evaluation metric
     'tree_method': 'hist',      # Use GPU accelerated algorithm
     'device': 'cuda',           # Use CUDA / GPU
     'predictor': 'gpu_predictor',   # Use GPU for prediction
     'verbosity': 1,
     'num_class': no_of_classes,
     'n_folds': 8,                   # Number of folds for the cross-validation
     'random_state': config['seed_all'] + 39,
     'num_iterations': 5000,
     'early_stopping_rounds': 30,
     'exclude_split_other_from_train': split_or_other_exclude,
     'win_loss_only': hp.choice('win_loss_only', [True, False])
}

best = fmin(fn = objective_xgboost_features_mlflow,
            space = space,
            algo = rand.suggest, # Randomly explore the space
            max_evals = 1,
            trials = trials)



# # Output the feature set if we want to use it for the model
# X_train.columns.to_series().to_csv('./results/2014+_feature_set_6.csv', index = False)
# Convert column names to a list
column_names = X_train_master.columns.tolist()[0:230]
# Save to a YAML file
with open('./configs/final_model_features.yaml', 'w') as file:
    dump(column_names, file, default_flow_style=False)
