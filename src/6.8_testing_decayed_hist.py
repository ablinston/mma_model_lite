# This script tests out methods of calculating decayed stats


import csv
import gc
import itertools
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

# Read in global objects and packages
exec(open('global.py').read())


# Read in the data
events = fp.ParquetFile('./data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
upcoming_events = fp.ParquetFile('./data/' + config['upcoming_event_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fighters = fp.ParquetFile('./data/' + config['fighter_tott_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fight_results = fp.ParquetFile('./data/' + config['fight_results_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
upcoming_fight_details = fp.ParquetFile('./data/' + config['upcoming_fight_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fight_stats = fp.ParquetFile('./data/' + config['fight_stats_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fight_odds = fp.ParquetFile('./data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
event_odds = fp.ParquetFile('./data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
betfair_odds = fp.ParquetFile('./data/' + config['betfair_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
betfair_upcoming_odds = fp.ParquetFile('./data/' + config['betfair_upcoming_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()

# Read in the average odds model
average_odds_model = joblib.load('./results/average_odds_model.joblib')

###########################################################################################
# UPCOMING

# Add upcoming fights to fight stats
fight_stats = add_upcoming_to_fight_stats(fight_stats, upcoming_fight_details, events, upcoming_events)

# Add upcoming events
events = pd.concat([upcoming_events, events], axis = 0)

# Add the date to the upcoming fights
upcoming_fight_details = upcoming_fight_details.merge(upcoming_events[['event', 'date']], on = 'event', how = 'left')

# Add upcoming fights to the results with empty results
fight_results = pd.concat([upcoming_fight_details.drop('url', axis = 1), fight_results])


###########################################################################################
# FIGHT STATS PROCESSING

# Calculate the historical stats of each fighter at the fight date
fight_stats = fight_stats.sort_values(by=['fighter', 'date'], axis = 0)

# Copy the fight stats but for against a fighter, then merge onto stats
fight_stats_against = fight_stats.copy()
fight_stats_against.rename(columns = lambda x: x + "_against" if "attempt" in x else x, inplace = True)
fight_stats_against.rename(columns = lambda x: x + "_against" if "landed" in x else x, inplace = True)
fight_stats_against.rename(columns = lambda x: x + "_against" if "ctrl_minutes" in x else x, inplace = True)
fight_stats_against.rename(columns = lambda x: x + "_against" if "kd" in x else x, inplace = True)
fight_stats_against.rename(columns = lambda x: x + "_against" if "sub.att" in x else x, inplace = True)

# Split the bout into fighters
# Set the opponent to the fighter for merging
fight_stats_against[['fighter1', 'fighter2']] = fight_stats_against['bout'].str.split('vs\\.', expand = True)
fight_stats_against['fighter'] = np.where(fight_stats_against['fighter'] == fight_stats_against['fighter1'], fight_stats_against['fighter2'], fight_stats_against['fighter1'])
# Get only the columns we want
against_columns = [col for col in fight_stats_against.columns if "against" in col]
fight_stats_against = fight_stats_against[['event', 'bout', 'fighter'] + against_columns]

# Merge onto the fight stats
rows_before = len(fight_stats)
fight_stats = fight_stats.merge(fight_stats_against, on = ['event', 'bout', 'fighter'], how = 'left')

# Check that all the rows are correct
if rows_before == len(fight_stats):
    print('Row check PASSED')
else:
    print('Row check FAILED')
    raise SystemExit
del fight_stats_against

fight_stats_bayes_priors = bayesian_priors(
    fight_stats.sort_values('date').reset_index(drop = True).merge(fight_results[['event', 'bout', 'weightclass']], on = ['event', 'bout'], how = 'left'), # Join on the weightclass for grouping
    id_cols = ['event', 'bout', 'fighter'], 
    group_by = data_processing_config['bayesian_groupby'], 
    fight_span = data_processing_config['bayes_fightspan'],
    ignore_last_n_fights = 30 if data_processing_config['bayesian_groupby'] == [] else 10
)


# Now aggrgate
fight_stats_hist_1 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_X1_",
    X = 1,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_2 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_X2_",
    X = 2,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_3 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_X3_",
    X = 3,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_4 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_X4_",
    X = 4,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_5 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_X5_",
    X = 5,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_6 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_X6_",
    X = 6,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])

fight_stats_hist_7 = aggregate_last_X_fight_data(
    fight_stats,
    id_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_X7_",
    X = 7,
    per_minute_calc = True, 
    bayes_priors_df = fight_stats_bayes_priors,
    bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
).drop(columns = ['date'])



merged_data = merge_df_both_fighters(
    left_df =  fight_results[['event', 'bout', 'fighter1', 'fighter2', 'outcome', 'method']],
    df_list = [
        fight_stats_hist_1,
        fight_stats_hist_2,
        fight_stats_hist_3,
        fight_stats_hist_4,
        fight_stats_hist_5,
        fight_stats_hist_6,
        fight_stats_hist_7
    ]
).reset_index(drop = True)



###########################################################################################
# FIT MODEL

# Filter out missing
merged_data = merged_data[~merged_data['outcome'].isna()]
merged_data = merged_data[(merged_data['fighter1_hist_X7_fights'] > 2) & (merged_data['fighter2_hist_X7_fights'] > 2)]

merged_data['train_test_oot_flag'] = 'oot'

X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(merged_data, config, oot_testing = True)

X_train = X_train[[col for col in X_train.columns if 'notformodel' not in col and 'allrounds' in col]]

# Create the matrix for the model
train = xgb.DMatrix(X_train, label = y_train)

# Set the parameters for the model
params = {
    'nthread': 3, # Only applies to CPU cores
    'booster': 'gbtree',
    'num_boost_round': 1, # to tell the package to train only 1 random forest
    'colsample_bynode': np.sqrt(len(X_train.columns)) / len(X_train.columns), # fraction of columns to select for each tree
    'num_parallel_tree': 1500, # the size of the forest. For 16k features, and choosing sqrt, 2500 gives 99.998% chance all features selected at least once
    'learning_rate': 1, # Learning rate
    'min_child_weight': 15,                 # The min observations that should be left in a leaf node    
    'subsample': 0.85,
    # 'max_depth': 6,
    'max_bin': 175, # used in the hist method to speed up, this says how many bins the vars should be split into
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

rf_model = xgb.train(params, train)


# Get feature importance scores
importance = rf_model.get_score(importance_type='total_gain')

# Sort features by importance
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Save the variable importance plot to csv
with open('results/random_forest_agg_X_test_variable_importance.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Feature", "Importance"])  # Write the header
    writer.writerows(sorted_importance)

# TreeSHAP calculation
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

# # Plot SHAP feature importance
# shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
# plt.title("SHAP Feature Importance (TreeSHAP)")
# plt.savefig('results/treeshap_feature_importance.png')
# plt.show()

# Combine SHAP values with feature names into a DataFrame for each target class
# If you have a single-target classification, shap_values will be a single array
# For multi-class, it will be a list of arrays, one for each class

# Assuming `shap_values` has the shape (6792, 18000, 3)
# Take the mean absolute SHAP values across all samples and classes
mean_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))

# Create a DataFrame for better readability
shap_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Mean SHAP Value': mean_shap_values})

# Sort features by importance
shap_importance_df = shap_importance_df.sort_values(by='Mean SHAP Value', ascending=False).reset_index(drop=True)

# Display or save the table
print(shap_importance_df.head(10))  # Shows the top 10 most important features
# Optionally, save to CSV
shap_importance_df.to_csv('./results/rf_agg_X_test_shap_feature_importance.csv', index=False)