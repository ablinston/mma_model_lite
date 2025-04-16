# This script is testing out different bases for the bayes variables by using a subset of them and a random forest. Chose a RF because there
# are less hyperparameters to optimise.


import csv
import gc
import itertools
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

# Read in global objects and packages
exec(open('global.py').read())

# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('bayes_base_testing_rf')

# Choose a new base for the bayes variables
for bayes_base_per_minute, bayes_base_percent in [(10, 450), (110, 450)]:
    
    # Overwrite the base
    data_processing_config['bayes_alpha_plus_beta_rebase_per_minute'] = bayes_base_per_minute
    data_processing_config['bayes_alpha_plus_beta_rebase_percent'] = bayes_base_percent
    data_processing_config['bayesian_groupby'] = []

    # Run the data processing
    skip_save_data = True
    exec(open('./src/7_data_processing.py').read()) # make sure the read of global is commented out.
    data = merged_data.copy()

    
    ##########################################################################################
    ### Data prep for model ###
    ##########################################################################################
    
    # Apply standard filters
    data = standard_data_filter(data)
    
    # Set target
    target = 'outcome'
    
    # Switch fighers if necessary
    if model_hyper_params['switch_fighters_with_odds'] == True:
        data = switch_fighter1_fighter2(data, data['fighter1_backtesting_odds'] > data['fighter2_backtesting_odds'])
    if model_hyper_params['switch_fighters_randomly'] == True: # Commented this out because I don't think we want random for the RF
        np.random.seed(seed = config['seed_all'] + 9845)
        # Switch twice then concat to try to get some variance
        data = switch_fighter1_fighter2(data, np.random.choice([True, False], size = len(data)))
        # data2 = switch_fighter1_fighter2(data, np.random.choice([True, False], size = len(data)))
        
        # data = pd.concat([data, data2, data3], ignore_index = True)
    
    X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, config)
    
    
    # Read in the baseline run and remove any columns with very low accuracy historically to speed up the RF
    var_importance = pd.read_csv('./results/shap_feature_importance_baseline.csv').sort_values('Mean SHAP Value', ascending = False)
    
    X_train = X_train[[col for col in X_train.columns if col not in var_importance['Feature'][1000:len(var_importance)].tolist()]]
    
    # Now remove any columns that aren't wanted in a training dataset. At this point we don't remove non-true odds in order to see the rank in feature importance
    X_train = standard_feature_removal(X_train, differentials_only = model_hyper_params['differential_features_only'], true_odds_only = False, open_odds_only = False)
    
    # Remove all but a subgroup of features
    X_train = X_train[[col for col in X_train.columns if 'baye' in col and 'allrounds' in col]].copy()
    
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
        'num_parallel_tree': 250, # the size of the forest. For 16k features, and choosing sqrt, 2500 gives 99.998% chance all features selected at least once
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
        
    

