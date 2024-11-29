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


##########################################################################################
### Feature selection PART 1 ###
##########################################################################################


# Remove cols that shouldn't really affect the fight.
# Remove dates because they can overfit. Leave date minuses, which are days since
cols_to_remove = [col for col in data.columns if ('date' in col) & (('diff' not in col) & ('minus' not in col) & (col != 'date'))]

# Get rid of dob and reach. Seem arbitrary
cols_to_remove = cols_to_remove + ['fighter1_dob', 'fighter2_dob', 'fighter1_reach', 'fighter2_reach', 'diff_reach']

# # Remove cols that are just totals for a fighter, like total kicks etc. Leave percentages and ratios and per minute stats
cols_to_remove = cols_to_remove + [col for col in data.columns if ('_round' in col) and not (('percent' in col) or ('ratio' in col) or ('per_minute' in col) or ('per_fight' in col))]
cols_to_remove = cols_to_remove + [col for col in data.columns if ('allrounds' in col) and not (('percent' in col) or ('ratio' in col) or ('per_minute' in col) or ('per_fight' in col) or (('against' in col) and (('hist_round' in col) or ('hist_last_3_fights' in col))))]
# Remove the columns that are single results
cols_to_remove = cols_to_remove + [col for col in data.columns if ('result_' in col)]


data = data[[col for col in data.columns if not col in cols_to_remove]]


##########################################################################################
### Filtering ###
##########################################################################################

# Apply standard filters
data = standard_data_filter(data)

# Don't include split or other decisions in training data. This flag will be used in functions later to exclude them from training data but not from test data (including test in CV data)
split_or_other_exclude = False
win_loss_only = False

##########################################################################################
### Data prep for model ###
##########################################################################################

# Prep the data ready for model training
X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, config, split_or_other_flag = split_or_other_exclude)


##########################################################################################
### Feature selection PART 2 - Varies ###
##########################################################################################

# Read in the variable importance
var_importance = pd.read_csv('./results/shap_feature_importance.csv').sort_values('Mean SHAP Value', ascending = False)


# Remove the bottom 3/4 of importance
feature_to_remove = var_importance.iloc[200:len(var_importance),]['Feature'].tolist()

X_train = X_train[[col for col in X_train.columns if col not in feature_to_remove]]

# # Remove all features except for differentials
# X_train = X_train[[col for col in X_train.columns if 'diff' in col]]

# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('2014+_mens_fights_gt3_features_set_9')


##########################################################################################
### XGBoost
##########################################################################################

trials = Trials()

# Load previous trials if available
# try:
#     with open('./results/xgboost_trials.pkl', 'rb') as f:
#         trials = pickle.load(f)
#     print("Loaded saved trials!")
# except FileNotFoundError:
#     pass

space = {
    'nthread': 3, # Only applies to CPU cores
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)), # Learning rate
    'min_child_weight': hp.quniform('min_child_weight', 8, 20, 1), # 15,                 # The min observations that should be left in a leaf node    
    'subsample': 0.9,
    'max_depth': hp.quniform('max_depth', 1, 5, 1),
    'max_bin': 200, # used in the hist method to speed up, this says how many bins the vars should be split into
    'objective': 'multi:softprob', # Multi classification objective
    'eval_metric': 'auc',       # Evaluation metric
    'tree_method': 'hist',      # Use GPU accelerated algorithm
    'device': 'cuda',           # Use CUDA / GPU
    'predictor': 'gpu_predictor',   # Use GPU for prediction
    'verbosity': 1,
    'num_class': no_of_classes,
    'n_folds': 10,                   # Number of folds for the cross-validation
    'random_state': config['seed_all'] + 39,
    'num_iterations': 5000,
    'early_stopping_rounds': 30,
    'exclude_split_other_from_train': split_or_other_exclude,
    'win_loss_only': win_loss_only
}

best = fmin(fn = objective_cv_xgboost_mlflow,
            space = space,
            algo = tpe.suggest, # this is the most commonly used algorithm in hyperopt by atpe_suggest is more thorough. Might take 500-1000 iterations
            max_evals = 500,
            trials = trials)

# Save trials after each run
with open('./results/xgboost_trials.pkl', 'wb') as f:
    pickle.dump(trials, f)



# # Output the feature set if we want to use it for the model
# X_train.columns.to_series().to_csv('./results/2014+_feature_set_6.csv', index = False)
# Convert column names to a list
column_names = X_train.columns.tolist()
# Save to a YAML file
with open('./configs/final_model_features.yaml', 'w') as file:
    dump(column_names, file, default_flow_style=False)
