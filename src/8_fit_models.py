_'''

This script runs the training experiments, using different methodologies such as random forests
and GBM.

Before running the experiments in this script, ensure that the mlflow server is running

mlflow server --host 127.0.0.1 --port 18080 --backend-store-uri /Documents/mma_model/mlruns/

AB: I couldn't get this working when run in the python venv. Have to run it in the normal cmd prompt before running this script. There is a bat file to do this in the repo.


'''

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
data = pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

# Replace infinite values with large values
data.replace([np.inf], 99999, inplace = True)
data.replace([-np.inf], -99999, inplace = True)

# Filter the OOT data and training data to be ignored
oot = data[data['date'] >= config['oot_start_date']]
data = data[(data['date'] >= config['min_training_date']) & (data['date'] < config['oot_start_date'])]


##########################################################################################
### Feature selection ###
##########################################################################################

# Remove column that has results, or that contain information we don't want included.
data.drop(columns = ['method', 'fighter1', 'fighter2', 'index', 'level_0', 'date'], inplace = True)


# AB: Haven't implemented any feature cleaning as of yet because I suspect the position of the fighters
# in the card does have some predictive power

# Remove cols that are just totals for a fighter, like total kicks etc. Leave percentages and ratios and per minute stats
cols_to_remove = [col for col in data.columns if ('_round' in col) and not (('percent' in col) or ('ratio' in col) or ('per_minute' in col) or ('per_fight' in col))]
cols_to_remove.append([col for col in data.columns if ('allrounds' in col) and not (('percent' in col) or ('ratio' in col) or ('per_minute' in col) or ('per_fight' in col))])
# Remove the columns that are single results
cols_to_remove.append([col for col in data.columns if ('result_' in col)])

data = data[[col for col in data.columns if not col in cols_to_remove]]


# Now set a more strict set of criteria for including columns
# keep_strings = ['diff_', 'minus', 'odds', '_results', 'streak']

# Then test if

##########################################################################################
### Data prep for model ###
##########################################################################################

# Set target
target = 'outcome'

# Recode the target. Don't need to distinguish between D and NC
data[target] = np.where((data[target] == "D/D") | (data[target] == "NC/NC"), "OTH", data[target])

# Convert all dates to integers
date_cols = [col for col in data.columns if data.dtypes[col] == 'datetime64[ns]']
data[date_cols] = data[date_cols].astype('int64')

# Convert object columns into series' of dummies
obj_cols = [col for col in data.columns if data.dtypes[col] == 'object']
obj_cols.remove(target)
data = pd.get_dummies(data, columns = obj_cols)

# Convert the target into a labelled column ready for multiclass modelling
le = LabelEncoder()
data[target] =le.fit_transform(data[target])

# Save what the classes are
print(le.classes_)
with open('results/target_classes.txt', 'w') as f:
    for class_label in le.classes_:
        f.write(f'{class_label}\n')

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns = [target]),
    data[target],
    test_size = 0.1,
    random_state = config['seed_all']
)

# Convert the test output to a binarised data frame. Can help with AUC calculations
y_test_bin = label_binarize(y_test, classes = np.unique(y_test))

# Check data types
data.dtypes.value_counts()

# Set number of classes for training
no_of_classes = len(set(y_train))


# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('all_fights')


##########################################################################################
### Random Forest - For feature importance
##########################################################################################

# These initial RF runs are to get variable importance in order to check what the models will
# likely be using and to remove columns that aren't adding much value. This will speed up training
# later on when we get to hyperparameter tuning.

# Set the parameters for the model
params = {
    'nthread': 3, # Only applies to CPU cores
    'booster': 'gbtree',
    'num_boost_round': 1, # to tell the package to train only 1 random forest
    'colsample_bynode': 0.1, # fraction of columns to select for each tree
    'num_parallel_tree': 2000, # the size of the forest
    'learning_rate': 1, # Learning rate
    'min_child_weight': 50,                 # The min observations that should be left in a leaf node    
    'subsample': 0.75,
    # 'max_depth': 6,
    'max_bin': 200, # used in the hist method to speed up, this says how many bins the vars should be split into
    'objective': 'multi:softprob', # Multi classification objective
    'eval_metric': 'mlogloss',       # Evaluation metric
    'tree_method': 'hist',      # Use GPU accelerated algorithm
    'device': 'cuda',           # Use CUDA / GPU
    'predictor': 'gpu_predictor',   # Use GPU for prediction
    'verbosity': 1,
    'num_class': no_of_classes,
    'n_folds': 5,                   # Number of folds for the cross-validation
    'random_state': config['seed_all'] + 38
}

# Prepare data
train = xgb.DMatrix(X_train, label = y_train)

with mlflow.start_run():

    # Log the params
    mlflow.log_params(params)
    mlflow.log_param('model', "RandomForest")

    # Train the model
    rf_model = xgb.train(params, train)

    # Get feature importance scores
    importance = rf_model.get_score(importance_type='weight')

    # Sort features by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # Save the variable importance plot to csv
    with open('results/random_forest_variable_importance.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Feature", "Importance"])  # Write the header
        writer.writerows(sorted_importance)


##########################################################################################
### XGBoost - For feature importance
##########################################################################################

# This XGBoost model is traing using the best parameters and is just used to get feature importance

# Set the parameters for the model
params = {
    'nthread': 3, # Only applies to CPU cores
    'learning_rate': 0.022363235793641148, # Learning rate
    'min_child_weight': 15,                 # The min observations that should be left in a leaf node    
    'subsample': 0.85,
    'max_depth': 2,
    'max_bin': 200, # used in the hist method to speed up, this says how many bins the vars should be split into
    'objective': 'multi:softprob', # Multi classification objective
    'eval_metric': 'auc',       # Evaluation metric
    'tree_method': 'hist',      # Use GPU accelerated algorithm
    'device': 'cuda',           # Use CUDA / GPU
    'predictor': 'gpu_predictor',   # Use GPU for prediction
    'verbosity': 1,
    'num_class': no_of_classes,
    'n_folds': 5,                   # Number of folds for the cross-validation
    'random_state': config['seed_all'] + 38,
    'num_iterations': 5000,
    'early_stopping_rounds': 30
}

# Prepare data
train = xgb.DMatrix(X_train, label = y_train)

# Train the model
importance_model = xgb.train(params, train)

# Get feature importance scores
importance = importance_model.feature_importances

# Convert to a DataFrame
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})


# Sort features by importance
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Save the variable importance plot to csv
with open('results/xgboost_variable_importance_set_2.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Feature", "Importance"])  # Write the header
    writer.writerows(sorted_importance)


##########################################################################################
### Light GBM
##########################################################################################

# Use hyperopt package to automatically optimise hyperparameters

# Define distributions of hyperparameters 
space = {
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
    'num_iterations': hp.quniform('num_iterations', 30, 500, 1),
    'objective': 'multiclass',      # Binary classification
    'metric': 'multi_logloss', # Evaluation metric
    'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
    'bagging_freq': 2,          # Perform bagging every 5 iterations
    'n_jobs': 3,               # Use n CPU cores
    'verbose': -1,            # Suppress output
    'num_class': 3,
    'device': 'gpu',  # Use GPU
    'gpu_platform_id': 0,  # Platform ID, 0 for default
    'gpu_device_id': 0,  # Device ID, 0 for the first GPU
    'n_folds': 5, # Number of folds for the cross-validation
    'random_state': config['seed_all']
}

# Optimize
best = fmin(fn = objective_cv_light_gbm_mlflow, 
            space = space, 
            algo = tpe.suggest, # this is the most commonly used algorithm in hyperopt
            max_evals = 500)

# # Use the custom function to train and test different combinations of hyperparameters.
# train_and_test_light_gbms_mlflow(
#         X_train, 
#         X_test,
#         y_train,
#         y_test,
#         y_test_bin,
#         leaves = [30, 50],
#         learning_rates = [0.01, 0.02, 0.005],
#         rounds = [100, 200, 500])

# Results can be view in MLflow

##########################################################################################
### XGBoost
##########################################################################################

# Load previous trials if available
try:
    with open('./results/xgboost_trials.pkl', 'rb') as f:
        trials = pickle.load(f)
    print("Loaded saved trials!")
except FileNotFoundError:
    trials = Trials()

space = {
    'nthread': 3, # Only applies to CPU cores
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)), # Learning rate
    'min_child_weight': hp.quniform('min_child_weight', 10, 20, 1),                 # The min observations that should be left in a leaf node    
    'subsample': hp.uniform('subsample', 0.7, 1),
    'max_depth': hp.quniform('max_depth', 5, 9, 1),
    'max_bin': hp.quniform('max_bin', 100, 300, 1), # used in the hist method to speed up, this says how many bins the vars should be split into
    'objective': 'multi:softprob', # Multi classification objective
    'eval_metric': 'mlogloss',       # Evaluation metric
    'tree_method': 'hist',      # Use GPU accelerated algorithm
    'device': 'cuda',           # Use CUDA / GPU
    'predictor': 'gpu_predictor',   # Use GPU for prediction
    'verbosity': 1,
    'num_class': no_of_classes,
    'n_folds': 5,                   # Number of folds for the cross-validation
    'random_state': config['seed_all'] + 38,
    'num_iterations': hp.quniform('num_iterations', 100, 300, 1)
}

best = fmin(fn = objective_cv_xgboost_mlflow,
            space = space,
            algo = tpe.suggest, # this is the most commonly used algorithm in hyperopt by atpe_suggest is more thorough. Might take 500-1000 iterations
            max_evals = 200,
            trials = trials)

# Save trials after each run
with open('./results/xgboost_trials.pkl', 'wb') as f:
    pickle.dump(trials, f)

##########################################################################################
## Now remove Opp stance variables from the model and see if there is any change in prediction accuracy

# This users fewer variable hyperparameters and simply chooses some of the roughly best performing values
# from the first set of trials

mlflow.set_experiment('all_fights_no_opp_stance_rounds')

# Remove opp stance from the data
no_opp_stance_cols = [col for col in data.columns if not (("_opp_stance_round" in col) | ("_opp_stance_allround" in col))]
no_opp_stance_cols.remove('outcome')
X_train_master = X_train.copy() # Keep a copy of the training dataset full for later
X_train = X_train[no_opp_stance_cols]

# Load previous trials if available
del([trials])
try:
    with open('./results/xgboost_trials_no_opp_stance.pkl', 'rb') as f:
        trials = pickle.load(f)
    print("Loaded saved trials!")
except FileNotFoundError:
    trials = Trials()

space = {
    'nthread': 3, # Only applies to CPU cores
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)), # Learning rate
    'min_child_weight': 14,                 # The min observations that should be left in a leaf node    
    'subsample': 0.75,
    'max_depth': 6,
    'max_bin': 255, # used in the hist method to speed up, this says how many bins the vars should be split into
    'objective': 'multi:softprob', # Multi classification objective
    'eval_metric': 'mlogloss',       # Evaluation metric
    'tree_method': 'hist',      # Use GPU accelerated algorithm
    'device': 'cuda',           # Use CUDA / GPU
    'predictor': 'gpu_predictor',   # Use GPU for prediction
    'verbosity': 1,
    'num_class': no_of_classes,
    'n_folds': 5,                   # Number of folds for the cross-validation
    'random_state': config['seed_all'] + 38,
    'num_iterations': hp.quniform('num_iterations', 150, 300, 1)
}

# Run hyperopt to fit some models
best = fmin(
    fn = objective_cv_xgboost_mlflow,
    space = space,
    algo = tpe.suggest,
    max_evals = 200,
    trials = trials
)

# Save trials after each run
with open('./results/xgboost_trials_no_opp_stance.pkl', 'wb') as f:
    pickle.dump(trials, f)
    


##########################################################################################
### Auto ML
##########################################################################################

## AB: Seems to run out of memory. Hangs at 100% for hours.

# Initialize the H2O cluster. Save a thread if want to do other things on PC.
h2o.init(max_mem_size = '7G', nthreads = 3)

# Get the data into h2o format
data_h2o = h2o.H2OFrame(data)

# Split the data into train and test
train, test = data_h2o.split_frame(ratios = [0.8], seed = config['seed_all'])

# Define the target variable
target = 'outcome'
# Define the features to be used to predict
features = train.columns
features.remove(target)

# Initialise the model
automl_models = H2OAutoML(
    max_models = 10, 
    seed = config['seed_all'], 
    max_runtime_secs = 3000,
    sort_metric = 'AUC')

# Fit the models to the data
automl_models.train(x = features, y = target, training_frame = train)

automl_models.leaderboard
