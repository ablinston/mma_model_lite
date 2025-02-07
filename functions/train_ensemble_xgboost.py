# This function will create a random forest of boosted trees.

def train_ensemble_xgboost(X, y, X_columns_for_training, num_models=10, num_boost_round=500, colsample_bynode=0.8, subsample=0.8, early_stopping_rounds=30, params=None, verbose_eval = False):
    """
    Trains an ensemble of boosted trees with random feature and row sampling for each model,
    ensuring that the validation sample consists of all data rows not included in the training data.

    Parameters:
    - X (pd.DataFrame): Feature matrix for training.
    - y (pd.Series or np.array): Target vector for training.
    - X_columns_for_training (list): a list of columns that should be used for training
    - num_models (int): Number of models (boosted trees) in the ensemble.
    - num_boost_round (int): Max number of boosting rounds for each model.
    - colsample_bynode (float): Fraction of columns to randomly sample for each tree.
    - subsample (float): Fraction of rows to sample for each model.
    - early_stopping_rounds (int): Number of rounds for early stopping.
    - params (dict): Parameters for the XGBoost model.

    Returns:
    - models (list): List of trained XGBoost models.
    """

    # pdb.set_trace()

    # Default parameters if none are provided
    if params is None:
        params = {
            'objective': 'multi:softprob',  # Set for multi-class classification; use 'binary:logistic' for binary
            'eval_metric': 'auc',
            'tree_method': 'hist',  # Efficient training with 'hist' method, use 'gpu_hist' if GPU is available
            'learning_rate': 0.2,
            'max_depth': 3,
            'colsample_bynode': colsample_bynode,
            'subsample': subsample,
            'verbosity': 1
        }

    # Set random seed for Numpy
    np.random.seed(config['seed_all'])

    # Add an id column to the X data
    X['id'] = X.index.copy()
    
    # Calculate new odds columns to be removed later
    X['train_fighter1_odds'] = X['fighter1_best_closing_odds'].to_numpy() # avoids pandas thinking they're the same column
    X['train_fighter2_odds'] = X['fighter2_best_closing_odds'].to_numpy()

    models = []
    for i in range(num_models):
        # pdb.set_trace()
        
        # # Switch fighters if needed
        # if params['switch_fighters_randomly'] == True:
        #     X_swapped, y_swapped = switch_fighter1_fighter2(X = X, y = y, filter_boolean = np.random.choice([True, False], size = len(X)))
        #     # Clean values
        #     X_swapped = clean_inf_and_large(X_swapped).copy()
        # else:
        X_swapped = X.copy()
        y_swapped = y.copy()
        
        model = {}
        
        # Randomly sample rows with replacement for the training set
        X_train_df, y_train_df = resample(X_swapped, y_swapped, replace=True, n_samples=int(len(X) * subsample), random_state=i)
        
        # # Switch fighters if needed
        if params['switch_fighters_randomly'] == True:
            X_train_df, y_train_df = switch_fighter1_fighter2(X = X_train_df, y = y_train_df, filter_boolean = np.random.choice([True, False], size = len(X_train_df)))
            # Clean values
            X_train_df = clean_inf_and_large(X_train_df).copy()
        
        # Calculate the number of columns to randomly select
        num_columns_to_select = int(colsample_bynode * X_train_df.shape[1])
        
        # Get a random selection of column names
        selected_columns = list(np.random.choice(X_columns_for_training, num_columns_to_select, replace = False))
        
        # Make sure both fighter1 and figher2 versions of columns in there or switch will fail
        selected_columns = list(set(selected_columns + list_swap_fighter1_fighter2(selected_columns)))
                
        X_train_df =  X_train_df[selected_columns + ['id', 'train_fighter1_odds', 'train_fighter2_odds']]
        
        # Filter win loss only if applicable
        if params['win_loss_only'] == True:
            X_train_df = X_train_df.copy()[y_train_df.isin([0, 2])]
            y_train_df = y_train_df.copy()[y_train_df.isin([0, 2])]

        # Create the validation set from all rows not included in the training set
        # Mask to get the validation data
        train_ids = X_train_df['id'].tolist()
        X_val = X_swapped.copy()[~X_swapped['id'].isin(train_ids).to_numpy()]
        y_val = y_swapped.copy()[~X_swapped['id'].isin(train_ids).to_numpy()]

        # Get the odds for the custom loss function if needed
        train_fighter1_odds = X_train_df['train_fighter1_odds'].copy()
        train_fighter2_odds = X_train_df['train_fighter2_odds'].copy()
        test_fighter1_odds = X_val['train_fighter1_odds'].copy()
        test_fighter2_odds = X_val['train_fighter2_odds'].copy()

        # Drop the ID column
        X_train_df.drop(['id', 'train_fighter1_odds', 'train_fighter2_odds'], axis = 1, inplace = True)
        X_val.drop(['id', 'train_fighter1_odds', 'train_fighter2_odds'], axis = 1, inplace = True)
        
        # Create the DMatrix for XGBoost with the bootstrap sample for training and the full remaining data for validation
        dmatrix_train = xgb.DMatrix(X_train_df, label=y_train_df)
        dmatrix_val = xgb.DMatrix(X_val[X_train_df.columns], label=y_val)
        
        # Train the model
        if model_hyper_params['loss_function'] == 'ev_loss_function': # any eval metric above will be ignored
            # Pass the odds to the loss function
            custom_loss = create_custom_loss(train_fighter1_odds.values, train_fighter2_odds.values)
            custom_loss_test = ev_loss_function_test(test_fighter1_odds.values, test_fighter2_odds.values)
            
            # Train an XGBoost model on the bootstrap sample with column sampling
            model['model'] = xgb.train(
                params,
                dmatrix_train,
                num_boost_round = num_boost_round,
                evals=[(dmatrix_val, 'eval')],
                early_stopping_rounds = early_stopping_rounds,
                obj = custom_loss,
                custom_metric = custom_loss_test,
                verbose_eval = verbose_eval)
        
        else:
            # Train an XGBoost model on the bootstrap sample with column sampling
            model['model'] = xgb.train(
                params,
                dmatrix_train,
                num_boost_round = num_boost_round,
                evals=[(dmatrix_val, 'eval')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval = verbose_eval)
            
        # Join the feature names into a single string and save them as a model attribute
        model['features'] = selected_columns
                
        # Add model to the list
        models.append(model)
        
        del X_train_df, X_swapped
        
        # print(f"Model {i + 1}/{num_models} trained with best iteration {model.best_iteration}.")

    return models
