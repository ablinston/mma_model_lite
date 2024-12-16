# This function trains XGboost models on data with differing number of features to see how many are required. once a number of features has been set, it will use an fmin function to find the optimal hyperparameters using CV, log those in mlflow along with the accuracy metrics.

# This function requires the X_train and y_train data to be ready but named X_train_master and y_train, because the sub function that optimises hyperparameters uses X_train.

# X_train_master should have the columns sorted in order of importance. For n number of features, it will simply take the first n columns

def objective_xgboost_features_mlflow(
        params):

    # pdb.set_trace()
    # 
    # Check for duplicates
    if is_duplicate_run(params, parameter_to_check = 'num_of_features'):
        
        # If the run existing, find what the loss was and return it without logging in mlflow
        existing_metric = get_existing_metric(params, 'Avg WL only AUC')
        
        print(f"Duplicate run found, skipping logging. Accuracy: {existing_metric:.4f}")
        return {'loss': 1 - existing_metric, 'params': params, 'status': STATUS_OK}
        
    else:
        
        # # Copy the params to edit them
        # params_inner = params.copy()
        
        # # Set the hyperopt parameters for the inner optimisation
        # params_inner = {
        #      'nthread': 3, # Only applies to CPU cores
        #      'learning_rate': hp.quniform('learning_rate', 0.001, 0.301, 0.005), # Learning rate
        #      'min_child_weight': hp.quniform('min_child_weight', 8, 20, 1), # 15,                 # The min observations that should be left in a leaf node    
        #      'subsample': 0.85,
        #      'max_depth': hp.quniform('max_depth', 1, 5, 1),
        #      'max_bin': 200, # used in the hist method to speed up, this says how many bins the vars should be split into
        #      'objective': 'multi:softprob', # Multi classification objective
        #      'eval_metric': 'auc',       # Evaluation metric
        #      'tree_method': 'hist',      # Use GPU accelerated algorithm
        #      'device': 'cuda',           # Use CUDA / GPU
        #      'predictor': 'gpu_predictor',   # Use GPU for prediction
        #      'verbosity': 1,
        #      'num_class': no_of_classes,
        #      'n_folds': 5,                   # Number of folds for the cross-validation
        #      'random_state': config['seed_all'] + 39,
        #      'num_iterations': 5000,
        #      'early_stopping_rounds': 50,
        #      'exclude_split_other_from_train': hp.choice('exclude_split_other_from_train', [True, False]),
        #      'win_loss_only': hp.choice('win_loss_only', [True, False])  
        # }
        
        with mlflow.start_run():
            
            # Create the train dataset based on the number of features
            global X_train
            X_train = X_train_master.iloc[:, 0:params['num_of_features']]
            
            if params['split_dec_or_other'] == True:  
                # Add on any specific columns needed
                X_train = pd.concat([X_train, X_train_master[['split_dec_or_other']]], axis = 1)
            
            # Log parameters in MLflow
            mlflow.log_param('num_of_features', params['num_of_features'])
            mlflow.log_param('max_evals', params['max_evals'])
            
            # Set up the trials object for this inner hyperopt
            trials_inner = Trials()
            
            # Train models to find the best hyperparameters
            best = fmin(fn = objective_cv_xgboost,
                        space = params_inner,
                        algo = tpe.suggest, # this is the most commonly used algorithm in hyperopt by atpe_suggest is more thorough. Might take 500-1000 iterations
                        max_evals = params['max_evals'],
                        trials = trials_inner)
            
            # Log the parameters of the best model
            mlflow.log_params(best)
            
            # Log the metrics of the best fit
            mlflow.log_metrics(trials_inner.best_trial['result']['metrics'])

    # Return whatever metric we wish to optimise for the LightGBM. Do 1 - because it tries to minimise the value.
    return {'loss': 1 - trials_inner.best_trial['result']['metrics']['Avg WL only AUC'], 'params': params, 'status': STATUS_OK} 
