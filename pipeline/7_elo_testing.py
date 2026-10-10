# This script is to derive a custom ELO rating for the fighters.

from hyperopt import hp, STATUS_OK, STATUS_FAIL, tpe, fmin, Trials, atpe # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
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

experiment_name = "elo_testing_v11"

# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment(experiment_name)

######################################################################################
## Standard ELO with no customisation. Just test different C and K

# Set the parameters for the Decision tree model and ELO
elo_constants_hp = {
    "C": hp.quniform("C", 100, 600, 20),  # Elo rating scaling constant
    "K": hp.quniform("K", 10, 80, 2),   # K-factor (rate at which the rating adjusts)
    # 'C': 200,  # Elo rating scaling constant
    # 'K': 54,
    
    "K_factor": 0, # This is an additional factor to adjust K depending on the number of fights someone has. 0 means no effect, because this is added onto the given K value
    "starting_elo": config["start_elo"],
    "constants_shrinkage_factor": 0, # How strongly the ELOs revert to mean over time. 0 is no shrinkage
    
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
    fn=objective_cv_elo_basic_mlflow,
    space=elo_constants_hp,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials)

######################################################################################
## Standard ELO trials with customisation

elo_constants_hp = {
    "C": 140,  # Elo rating scaling constant
    "K": 76,   # K-factor (rate at which the rating adjusts)
    
    "K_factor": hp.quniform("K_factor", 0, 0.04, 0.001), # This is an additional factor to adjust K depending on the number of fights someone has. 0 means no effect
    
    "starting_elo": config["start_elo"],
    
    "constants_shrinkage_factor": hp.quniform("constants_shrinkage_factor", 0.01, 0.1, 0.0001),  # How strongly the ELOs revert to mean over time. 0 is no shrinkage
    # Coefficients for fight metrics differences
    "constant_control_proportion_diff": hp.quniform("constant_control_proportion_diff", 0.25, 0.5, 0.01),  # Control proportion difference weight ranges from -1.2 to 1.5
    "constant_sig_strikes_landed_per_minute_diff": hp.quniform("constant_sig_strikes_landed_per_minute_diff", 0.7, 0.9, 0.005),  # Indicator for which fighter landed more significant strikes
    "constant_kds_diff": 0.96, #hp.quniform('constant_kds_diff', 0.4, 0.5, 0.005),  # Indicator for which fighter had more knockdowns
    "constant_submissions_diff": 0.975, #hp.quniform('constant_submissions_diff', 0.85, 1, 0.005),  # Submissions difference weight
    "constant_inverse_fight_length_proportion": 0, #hp.quniform('constant_inverse_fight_length_proportion', 0, 0.2, 0.01), # The inverse of proportion of fight, so 25 / fight length. Potential values 1 to 300
    "constant_knockouts_diff": hp.quniform("constant_knockouts_diff", 0.95, 1.1, 0.001),  # Knockouts difference weight
    "constant_unanimous_decs_diff": hp.quniform("constant_unanimous_decs_diff", 0.45, 0.65, 0.005)  # Unanimous decisions difference weight
    # 'constant_split_decs_diff': hp.quniform('constant_split_decs_diff', 0, 1, 0.2),  # Split decisions difference weight
}

# Reset trials if using atpe after tpe
trials = Trials()

best = fmin(
    fn=objective_cv_elo_basic_mlflow,
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
