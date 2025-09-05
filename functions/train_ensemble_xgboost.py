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
    
    # Get a full list of columns needed for switching fighters then cut down the data
    all_columns_for_diffs = all_columns_for_differentials(X_columns_for_training)
    X = X[[col for col in all_columns_for_diffs if col in X.columns] + [col for col in X.columns if 'notformodel' in col or ('odds' in col and col not in all_columns_for_diffs)]].copy()
    
    # Add an id column to the X data
    X['id'] = X.index.copy()
    
    # Calculate new odds columns to be removed later
    X['train_fighter1_odds'] = X['fighter1_backtesting_odds'].to_numpy() # avoids pandas thinking they're the same column
    X['train_fighter2_odds'] = X['fighter2_backtesting_odds'].to_numpy()

    models = []
    for i in range(num_models):
        
        # pdb.set_trace()
                
        model = {}
        
        # Randomly sample rows with replacement for the training set
        if subsample < 1:
            X_train_df, y_train_df = sample_fighters(
                X = X, 
                y = y, 
                n_rows = int(len(X) * subsample),
                rows_with_bagging = None if not params['xgb_forest_bagging'] else len(X))
            # X_train_df, y_train_df = resample(X, y, replace=True, n_samples=int(len(X) * subsample), random_state=i)
        else:
            X_train_df = X.copy()
            y_train_df = y.copy()
        
        # # Switch fighters if needed
        if params['switch_fighters_randomly'] == True:
            X_train_df, y_train_df = switch_fighter1_fighter2(X = X_train_df, y = y_train_df, filter_boolean = np.random.choice([True, False], size = len(X_train_df)))
            
        # Add noise to odds if required
        if params['add_noise_to_odds']:
            X_train_df = add_noise_to_odds(X_train_df, norm_sd = params['odds_noise_standard_dev'])
        
        # Remove split decisions if required
        if params['split_dec_or_other']:
            y_train_df = y_train_df[X_train_df['notformodel_split_dec_or_other'] == 0]
            X_train_df = X_train_df[X_train_df['notformodel_split_dec_or_other'] == 0]
        
        # Calculate the number of columns to randomly select
        if colsample_bynode < 1:
            num_columns_to_select = int(colsample_bynode * len(X_columns_for_training))
            
            # First, duplicate all the diff rows, otherwise 'fighter' rows have 2x the chance of being chosen
            diff_cols = [col for col in X_columns_for_training if 'fighter1' not in col and 'fighter2' not in col]
            
            # Get a random selection of column names
            selected_columns = list(set(np.random.choice(X_columns_for_training + diff_cols, num_columns_to_select, replace = False)))
        else:
            selected_columns = X_columns_for_training
        
        # # Make sure both fighter1 and figher2 versions of columns in there or switch will fail
        # selected_columns = list(set(selected_columns + list_swap_fighter1_fighter2(selected_columns)))
        
        # Filter win loss only if applicable
        if params['win_loss_only'] == True:
            X_train_df = X_train_df.copy()[y_train_df.isin([0, 2])]
            y_train_df = y_train_df.copy()[y_train_df.isin([0, 2])]

        # Create the validation set from all rows not included in the training set
        # Mask to get the validation data
        train_ids = X_train_df['id'].tolist()
        X_val = X.copy()[~X['id'].isin(train_ids).to_numpy()]
        y_val = y.copy()[~X['id'].isin(train_ids).to_numpy()]

        # Get the odds for the custom loss function if needed
        train_fighter1_odds = X_train_df['train_fighter1_odds'].to_numpy()
        train_fighter2_odds = X_train_df['train_fighter2_odds'].to_numpy()
        test_fighter1_odds = X_val['train_fighter1_odds'].to_numpy()
        test_fighter2_odds = X_val['train_fighter2_odds'].to_numpy()

        # Get the base margins
        base_margin_train = get_base_margin(X_train_df, params['base_margin_cols'])
        base_margin_val = get_base_margin(X_val, params['base_margin_cols'])

        X_train_df =  X_train_df[selected_columns + ['id', 'train_fighter1_odds', 'train_fighter2_odds']]

        # Drop the ID column
        cols_to_drop = [col for col in X_train_df.columns if 'train_fighter' in col or 'backtesting' in col or col == 'id' or 'notformodel' in col]
        X_train_df.drop(cols_to_drop, axis = 1, inplace = True)
        X_val.drop(cols_to_drop, axis = 1, inplace = True)
        
        # Create the DMatrix for XGBoost with the bootstrap sample for training and the full remaining data for validation
        dmatrix_train = xgb.DMatrix(X_train_df, label=y_train_df, base_margin = base_margin_train)
        dmatrix_val = xgb.DMatrix(X_val[X_train_df.columns], label=y_val, base_margin = base_margin_val)
        
        # Train the model
        if model_hyper_params['loss_function'] == 'ev_loss_function': # any eval metric above will be ignored
            # Pass the odds to the loss function
            custom_loss = create_custom_loss(train_fighter1_odds, train_fighter2_odds)
            custom_loss_test = ev_loss_function_test(test_fighter1_odds, test_fighter2_odds)
            
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
            
            # model['model'].model_features = X_train_df.columns
        
        else:
            # Train an XGBoost model on the bootstrap sample with column sampling
            model['model'] = xgb.train(
                params,
                dmatrix_train,
                num_boost_round = num_boost_round,
                evals=[(dmatrix_val, 'eval')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval = verbose_eval)
            
        # # Manually set the list of model features
        # model['model'].feature_names = list(X_train_df.columns)
        
        # Join the feature names into a single string and save them as a model attribute
        model['features'] = selected_columns
        
        # Save the parameters used
        model['params'] = params
        
        # Add model to the list
        models.append(model)
        
        del X_train_df
        gc.collect()
        print(f"Model {i + 1}/{num_models} trained.")

    return models
