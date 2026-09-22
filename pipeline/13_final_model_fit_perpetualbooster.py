# This script fits XGBoost models to the final data


from perpetual import PerpetualBooster

# Read in global objects and packages
exec(open('global.py').read())

# Set the new model version number
new_version_number = "4.1.0"
number_of_features = 100
pb_budget = 0.2

# Read in the data for training
data = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

# Apply standard filters
data = standard_data_filter(data)

##########################################################################################
### Feature selection using config ###
##########################################################################################

# Read in the variable importance
var_importance = pd.read_csv('./results/shap_feature_importance.csv').sort_values('Mean SHAP Value', ascending = False)

# Ensure that the data columns are ordered in descending feature importance
# First list any columns not in the importance
[col for col in data.columns if col not in var_importance['Feature'].tolist()]

# Now select the columns
columns_in_order = var_importance['Feature'].tolist()
# Get rid of the last X results
columns_in_order = [col for col in columns_in_order if not ('_last_' in col and '_results' in col)]

    
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
columns_in_order = [col for col in feature_control['include_features'] if col in data.columns] + columns_in_order

# Remove any duplicates
columns_in_order = pd.Series(columns_in_order).drop_duplicates().tolist()

columns_in_order = trim_column_list(columns_in_order)

##########################################################################################
### Data prep for model ###
##########################################################################################

data['id'] = data.index.copy()

# Prep the data ready for model training
X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, split_or_other_flag = True)


# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('gt2_fights_xgboost_forest_v' + new_version_number)




##########################################################################################
### Model training
##########################################################################################

start = time.time()

# Initialize model
model = PerpetualBooster(objective = 'LogLoss', 
                         num_threads = config['num_threads'], 
                         budget = pb_budget, 
                         reset = True,
                         max_bin = 180, # default 256
                         timeout = 3600)

# Fit the model
model.fit(X_train[columns_in_order[0:number_of_features]], y_train)

train_time = time.time() - start

# Set which columns and metrics we're interested in
breakdown_cols = ['weightclass', 'year', 'month_year', 'time_format', 'fighter_history', 'date']
breakdown_metrics= ['Avg WL only AUC', 'X Odds Avg WL only AUC', 'Hmean WL only Accuracy', 'X Odds Hmean WL only Accuracy', 'Logloss', 'X Odds Logloss', 'Sample Size']
target = 'outcome'

backtesting_params = {
            'min_prob_to_bet': 0.485,
            'min_ev_to_bet': 0.05,
            'max_bet_size': 100,
            'min_bet_size': 100,
            'constant_bets': True,
            'point_in_time_only_odds': True
            # 'target_ev_after_cutoff': 0.05
        }

# Set this model's description for MLflow
model_description = 'PerpetualBooster'

probs = model.predict_proba(X_test[columns_in_order[0:number_of_features]])

with mlflow.start_run():
    
    mlflow.log_param('budget', pb_budget)
    mlflow.log_param('num_features', number_of_features)
    mlflow.log_param('train_time', train_time)
    
    btest_model_and_log_mlflow(
        X_test[['id']].merge(data, on = 'id', how = 'left'), # merge the data on the train to ensure the same order of rows, 
        probs, 
        breakdown_cols = breakdown_cols, 
        breakdown_metrics = breakdown_metrics, 
        mlflow_description = model_description + '  - Test data', 
        target_column = target,
        backtesting_params = backtesting_params)

# # Try different numbers of trees
# # for num_trees in [1]:
# for num_trees_i in list(range(1, 21, 2)) + list(range(25, 51, 5)) + list(range(60, len(xgboost_rf) + 1, 10)):
        
#     if len(xgboost_rf) >= num_trees_i:
        
#         # Start a new model run in MLflow
#         with mlflow.start_run():
            
#             # Log the initial metrics
#             mlflow.log_params(params)
#             mlflow.log_param('num_trees', num_trees_i)
#             mlflow.log_param('Forest col proportion', colsample_bynode)
#             mlflow.log_param('Forest subsample', subsample)
#             mlflow.set_tag('Sample', 'Train')
                        
#             # Log the metrics of the forest with that many trees in MLflow
#             log_metrics_xgboost_ensemble_mlflow(xgboost_rf[0:(num_trees_i)], X_train, y_train, model_hyper_params)
            
#             # Log the predictions just from the odds for info
#             log_metrics_odds_mlflow(X_train, y_train)
            
#         # Start a new model run in MLflow
#         with mlflow.start_run():
            
#             # Log the initial metrics
#             mlflow.log_params(params)
#             mlflow.log_param('num_trees', num_trees_i)
#             mlflow.log_param('Forest col proportion', colsample_bynode)
#             mlflow.log_param('Forest subsample', subsample)
#             mlflow.set_tag('Sample', 'Test')
                        
#             # Log the metrics of the forest with that many trees in MLflow
#             log_metrics_xgboost_ensemble_mlflow(xgboost_rf[0:(num_trees_i)], X_test, y_test, model_hyper_params)
            
#             # Log the predictions just from the odds for info
#             log_metrics_odds_mlflow(X_test, y_test)


# # Set the final number of models
# final_forest_size = 1


# # Save the model
# joblib.dump(xgboost_rf[0:final_forest_size], './results/final_xgboost_forest.joblib', compress = 9)
