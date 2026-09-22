# This script is testing out different bases for the bayes variables by using a subset of them and a random forest. Chose a RF because there
# are less hyperparameters to optimise.


import csv
import gc
import itertools
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb

from mma_functions.cleaning import standard_data_filter
from mma_functions.config import config, elo_config, load_configs, model_hyper_params, odds_scrape_config, test_fighters
from mma_functions.dataset import prep_data_for_model_training, standard_feature_removal
from mma_functions.modelling import create_custom_loss
from mma_functions.pipelines import merge_and_process_all_clean_data
from mma_functions.tracking import log_metrics_mlflow

# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('bayes_base_testing_rf_v4')

# Choose a new base for the bayes variables
for bayes_base_per_minute in [0, 25, 50]: # :range(25, 175, 25):
    for bayes_base_percent in [0, 10, 25, 50, 75, 100]: #range(50, 300, 50):
        # for bayes_fightspan in [50, 150]:
        
        # Read the config in fresh each time so the overwrites below don't carry over
        data_processing_config = load_configs()['data_processing_config']

        # Overwrite the base
        data_processing_config['bayes_alpha_plus_beta_rebase_per_minute'] = bayes_base_per_minute
        data_processing_config['bayes_alpha_plus_beta_rebase_percent'] = bayes_base_percent
        data_processing_config['bayesian_groupby'] = []
        # data_processing_config['bayes_fightspan'] = bayes_fightspan
        if bayes_base_percent == 0:
            data_processing_config['add_bayes_percentage_columns'] = False
            data_processing_config['add_adj_percentage_bayes_columns'] = False
        if bayes_base_per_minute == 0:
            data_processing_config['add_bayes_per_minute_columns'] = False
            data_processing_config['add_adj_per_minute_bayes_columns'] = False
    
        # Run the data processing
        data = merge_and_process_all_clean_data(
                config,
                elo_config,
                data_processing_config,
                odds_scrape_config,
                model_hyper_params,
                test_fighter_list = test_fighters,
                skip_save_data = True # whether to skip the save data and just return merged data
                )
    
        
        ##########################################################################################
        ### Data prep for model ###
        ##########################################################################################
        
        # Apply standard filters
        data = standard_data_filter(data)
        
        # Set target
        target = 'outcome'
        
        # # Switch fighers if necessary
        # if model_hyper_params['switch_fighters_with_odds'] == True:
        #     data = switch_fighter1_fighter2(data, data['fighter1_backtesting_odds'] > data['fighter2_backtesting_odds'])
        # if model_hyper_params['switch_fighters_randomly'] == True: # Commented this out because I don't think we want random for the RF
        #     np.random.seed(seed = config['seed_all'] + 9845)
        #     # Switch twice then concat to try to get some variance
        #     data = switch_fighter1_fighter2(data, np.random.choice([True, False], size = len(data)))
        #     # data2 = switch_fighter1_fighter2(data, np.random.choice([True, False], size = len(data)))
            
            # data = pd.concat([data, data2, data3], ignore_index = True)
        
        X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data)
        
        
        # Read in the baseline run and remove any columns with very low accuracy historically to speed up the RF
        var_importance = pd.read_csv('./results/shap_feature_importance_baseline.csv').sort_values('Mean SHAP Value', ascending = False)
        
        # Set the feature list for the random forests
        feature_list = var_importance['Feature']
        
        # Remove Bayes if not needed
        if bayes_base_per_minute == 0:
            feature_list = feature_list.str.replace('per_minute_bayes', 'per_minute')
        if bayes_base_percent == 0:
            feature_list = feature_list.str.replace('bayes_accuracy_percent', 'accuracy_percent')
        
        # Get unique values
        feature_list = list(feature_list.unique())

        X_train = X_train[[col for col in X_train.columns if col in feature_list[0:1000]]]
        
        # Now remove any columns that aren't wanted in a training dataset. At this point we don't remove non-true odds in order to see the rank in feature importance
        X_train = standard_feature_removal(X_train, differentials_only = model_hyper_params['differential_features_only'], true_odds_only = True, open_odds_only = False)
        
        # Remove all but a subgroup of features
        X_train = X_train[[col for col in X_train.columns if 'odds' not in col]].copy()
        
        # Check data types
        data.dtypes.value_counts()
        
        # Set number of classes for training
        no_of_classes = len(set(y_train)) 
        
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
            'colsample_bynode': np.sqrt(len(data.columns)) / len(data.columns), # fraction of columns to select for each tree
            'num_parallel_tree': 300, # the size of the forest. For 16k features, and choosing sqrt, 2500 gives 99.998% chance all features selected at least once
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
        
        # Remove any objects not for modelling
        X_train = X_train[[col for col in X_train if 'notformodel' not in col]]
        
        # Prepare data
        train = xgb.DMatrix(X_train, label = y_train)
        
        # Add the odds in case we're using custom loss
        # train.set_info(fighter1_backtesting_odds=X_train['fighter1_backtesting_odds'].values,
        #                fighter2_backtesting_odds=X_train['fighter2_backtesting_odds'].values)
        
        
        with mlflow.start_run():
        
            # Log the params
            mlflow.log_params(params)
            mlflow.log_param('model', "RandomForest")
            mlflow.log_param('bayes_base_per_minute', bayes_base_per_minute)
            mlflow.log_param('bayes_base_percent', bayes_base_percent)
            mlflow.log_param('bayes_groupby', data_processing_config['bayesian_groupby'])
            # mlflow.log_param('bayes_fightspan', bayes_fightspan)
            
            # Train the model
            if model_hyper_params['loss_function'] == 'ev_loss_function': # any eval metric above will be ignored
            
                # Pass the odds to the loss function
                custom_loss = create_custom_loss(X_train['fighter1_backtesting_odds'].values, X_train['fighter2_backtesting_odds'].values)
                # Train the model
                rf_model = xgb.train(params, train, obj = custom_loss)
                
            else:
                rf_model = xgb.train(params, train)
        
            # Get the test data
            test = xgb.DMatrix(X_test[X_train.columns], label = y_test)
        
            # Get the model predictions on the test dataset
            prediction_probabilities = rf_model.predict(test)
            
            # Log the metrics in MLflow
            log_metrics_mlflow(prediction_probabilities, y_test_df = y_test, fighter1_odds = X_test['fighter1_backtesting_odds'], fighter2_odds = X_test['fighter2_backtesting_odds'], classes = np.array([0, 1, 2]))
            
            mlflow.end_run()
        
    
