# This function will create a random forest of boosted trees.

def train_ensemble_xgboost(X, y, num_models=10, num_boost_round=500, colsample_bynode=0.8, subsample=0.8, early_stopping_rounds=30, params=None, verbose_eval = False):
    """
    Trains an ensemble of boosted trees with random feature and row sampling for each model,
    ensuring that the validation sample consists of all data rows not included in the training data.

    Parameters:
    - X (pd.DataFrame): Feature matrix for training.
    - y (pd.Series or np.array): Target vector for training.
    - num_models (int): Number of models (boosted trees) in the ensemble.
    - num_boost_round (int): Max number of boosting rounds for each model.
    - colsample_bynode (float): Fraction of columns to randomly sample for each tree.
    - subsample (float): Fraction of rows to sample for each model.
    - early_stopping_rounds (int): Number of rounds for early stopping.
    - params (dict): Parameters for the XGBoost model.

    Returns:
    - models (list): List of trained XGBoost models.
    """

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

    models = []
    for i in range(num_models):
        
        model = {}
        
        # Randomly sample rows with replacement for the training set
        X_train_df, y_train_df = resample(X, y, replace=True, n_samples=int(len(X) * subsample), random_state=i)
        pdb.set_trace()
        # Calculate the number of columns to randomly select
        num_columns_to_select = int(colsample_bynode * X_train_df.shape[1])
        
        # Get a random selection of column names
        selected_columns = list(np.random.choice(X_train_df.columns, num_columns_to_select, replace = False))
        
        # Make sure both fighter1 and figher2 versions of columns in there or switch will fail
        selected_columns = selected_columns + pd.Series(selected_columns).replace('fighter2', 'fighterX').replace('fighter1', 'fighter2').replace('fighterX', 'fighter1').tolist()
        
        X_train_df =  X_train_df[list(set(selected_columns))]

        # Switch fighters if needed
        if params['switch_fighters_randomly'] == True:
            X_train_df, y_train_df = switch_fighter1_fighter2(X = X_train_df, y = y_train_df, filter_boolean = np.random.choice([True, False], size = len(X_train_df)))
        
        # Filter win loss only if applicable
        if params['win_loss_only'] == True:
            X_train_df = X_train_df[y_train_df.isin([0, 2])]
            y_train_df = y_train_df[y_train_df.isin([0, 2])]

        # Create the validation set from all rows not included in the training set
        # Mask to get the validation data
        train_indices = X_train_df.index
        X_val = X[~X.index.isin(train_indices)]
        y_val = y[~y.index.isin(train_indices)]

        # Create the DMatrix for XGBoost with the bootstrap sample for training and the full remaining data for validation
        dmatrix_train = xgb.DMatrix(X_train_df, label=y_train_df)
        dmatrix_val = xgb.DMatrix(X_val[selected_columns], label=y_val)
        
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

        # print(f"Model {i + 1}/{num_models} trained with best iteration {model.best_iteration}.")

    return models
