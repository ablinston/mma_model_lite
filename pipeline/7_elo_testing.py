# This script is to derive a custom ELO rating for the fighters.

from hyperopt import hp, STATUS_OK, STATUS_FAIL, tpe, fmin, Trials, atpe # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
from functools import partial
import itertools
import mlflow
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb
import warnings

# imports
import fastparquet as fp
import numpy as np
from yaml import dump

from mma_functions.elo import objective_cv_elo_basic_mlflow, prep_stats_for_elo_calc
from mma_functions.tracking import get_params_by_run_name
from mma_functions.config import config

np.warnings = warnings # to avoid ATPE error with hyperopt

# Read in the data
fight_results = fp.ParquetFile("./data/" + config["fight_results_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
fight_stats = fp.ParquetFile("./data/" + config["fight_stats_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()



# Prep the data for the ELO calcs
start_elo, merged_data = prep_stats_for_elo_calc(fight_results, fight_stats, config, start_date="2000-01-01", remove_nas=False)

# # Begin calculation of the ELO
# elo_constants = {
#     'C': 400,  # Elo rating scaling constant
#     'K': 60,   # K-factor (rate at which the rating adjusts)
    
#     # Coefficients for fight metrics differences
#     'constant_control_proportion_diff': 0,  # Control proportion difference weight
#     'constant_sig_strikes_landed_per_minute_diff': 0,  # Significant strikes landed per minute difference weight
#     'constant_kds_diff': 0,  # Knockdowns difference weight
#     'constant_submissions_diff': 0,  # Submissions difference weight
#     'constant_knockouts_diff': 0,  # Knockouts difference weight
#     'constant_unanimous_decs_diff': 0,  # Unanimous decisions difference weight
#     'constant_split_decs_diff': 0,  # Split decisions difference weight
# }

# params = elo_constants | {
#     'objective': 'multi:softmax',  # Multiclass classification
#     'num_class': 3,  # Number of target classes
#     'tree_method': 'hist',      # Use GPU for training
#     'predictor': 'gpu_predictor',   # GPU predictor
#     # 'max_depth': 6,                 # Depth of the tree (adjustable)
#     'eta': 0.1,                     # Learning rate
#     'seed': config['seed_all'],                   # Ensure reproducibility
#     'num_iterations': 1000,
#     'early_stopping_rounds': 20,
#     'eval_metric': 'merror',       # Evaluation metric
#     'device': 'cuda'
# }

# # Create the elo data and the results data
# elo_data, results_data = derive_start_elos(start_elo, merged_data, elo_constants)

######################################################################################
## Begin optimising the ELO

<<<<<<< Updated upstream
experiment_name = "elo_testing_v1"
=======
experiment_name = "elo_testing_v2"
>>>>>>> Stashed changes

# Set up experiment training
mlflow.set_tracking_uri(uri="http://mlflow-mma-model.192.168.68.51.nip.io:8081")
mlflow.set_experiment(experiment_name)

######################################################################################
## Standard ELO with no customisation. Just test different C and K

# Set the parameters for the Decision tree model and ELO
elo_constants_hp = {
    "C": 400,  # Elo rating scaling constant
    "K": hp.quniform("K", 30, 140, 2),   # K-factor (rate at which the rating adjusts)
    # 'C': 200,  # Elo rating scaling constant
    # 'K': 54,
    
    "K_factor": 0, # This is an additional factor to adjust K depending on the number of fights someone has. 0 means no effect, because this is added onto the given K value
    "starting_elo": config["start_elo"],
    "shrinkage_half_life_days": 1000000, # Days for half the deviation from the starting ELO to decay away. Very large means no reversion
    
    # Coefficients for fight metrics differences
    "constant_control_proportion_diff": 0,
    "constant_sig_strikes_landed_per_minute_diff": 0,
    "constant_kds_diff": 0,
    "constant_submissions_diff": 0,
    "constant_knockouts_diff": 0,
    "constant_unanimous_decs_diff": 0,
    "constant_inverse_fight_length_proportion": 0,
    "constant_split_decs_diff": 0
}

trials = Trials()

best = fmin(
    fn=partial(objective_cv_elo_basic_mlflow, start_elo=start_elo, merged_data=merged_data),
    space=elo_constants_hp,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials)

######################################################################################
## Standard ELO trials with customisation

# C is fixed rather than searched. Only the ratio K/C affects anything, because every rating starts
# at 1500 so the spread scales with K, and the expected score only sees the gap divided by C. Doubling
# both gives an identical logloss to eight decimal places, so searching both wastes most of the trials
# on a flat ridge. 400 is the chess convention, which puts the ratings on a familiar scale.
# K is searched here as well as in the plain ELO run above, because the dominance constants multiply
# the same update and the two move together
elo_constants_hp = {
    "C": 400,  # Elo rating scaling constant. Redundant given K, so held fixed
    "K": hp.quniform("K", 30, 140, 2),   # K-factor (rate at which the rating adjusts). Flat bottom around 60 to 80 at C=400
    
    "K_factor": hp.quniform("K_factor", 0, 0.2, 0.001), # This is an additional factor to adjust K depending on the number of fights someone has. 0 means no effect
    
    "starting_elo": config["start_elo"],
    
    "shrinkage_half_life_days": hp.quniform("shrinkage_half_life_days", 300, 5000, 60),  # Days for half the deviation from the starting ELO to decay away. Larger means less reversion
    # Coefficients for fight metrics differences. The ranges start at zero so the search can turn one
    # off, and reach well above the old values because tanh squashes the top end, so the best scale
    # now sits around half of what it used to be
    "constant_control_proportion_diff": hp.quniform("constant_control_proportion_diff", 0, 2, 0.02),  # Control proportion difference weight
    "constant_sig_strikes_landed_per_minute_diff": hp.quniform("constant_sig_strikes_landed_per_minute_diff", 0, 1.2, 0.02),  # Indicator for which fighter landed more significant strikes
    "constant_kds_diff": hp.quniform("constant_kds_diff", 0, 1.2, 0.02),  # Indicator for which fighter had more knockdowns
    "constant_submissions_diff": hp.quniform("constant_submissions_diff", 0, 1.2, 0.02),  # Submissions difference weight
    "constant_inverse_fight_length_proportion": 0, #hp.quniform('constant_inverse_fight_length_proportion', 0, 0.2, 0.01), # The inverse of proportion of fight, so 25 / fight length. Potential values 1 to 300
    "constant_knockouts_diff": hp.quniform("constant_knockouts_diff", 0, 1.4, 0.02),  # Knockouts difference weight
    "constant_unanimous_decs_diff": hp.quniform("constant_unanimous_decs_diff", 0, 1.0, 0.02)  # Unanimous decisions difference weight
    # 'constant_split_decs_diff': hp.quniform('constant_split_decs_diff', 0, 1, 0.2),  # Split decisions difference weight
}

# Reset trials if using atpe after tpe
trials = Trials()

best = fmin(
    fn=partial(objective_cv_elo_basic_mlflow, start_elo=start_elo, merged_data=merged_data),
    space=elo_constants_hp,
    algo=tpe.suggest, # tpe.suggest,
    max_evals=3000,
    trials=trials
    )



######################################################################################
# Get the best fitting set of parameters and save them
# best_params = get_best_params_mlflow(experiment_name, metric_name = 'Overall accuracy')

# # Or choose a specific run to get
best_params = get_params_by_run_name(experiment_name, target_run_name="youthful-mule-443") # chose this because high KDs doesn't make as much sense and I don't like the outputs when kfactor isn't 0. Fighters end up dropping ELO even on wins and it didn't really look right

with open("./configs/elo_params.yaml", "w") as file:
    dump(best_params, file, default_flow_style=False)
