def create_dmatrix(X_train, y_train, train_idx, test_idx, base_margin_cols, win_loss_only = False, split_dec_or_other_filter = False):
    
    # pdb.set_trace()
    # Convert train_idx to a boolean mask
    mask = pd.Series(False, index=X_train.index)
    mask.iloc[train_idx] = True  # Set True for indices in train_idx
    
    # # Reset the index
    # X_train = X_train.reset_index(drop = True)
    # y_train = y_train.reset_index(drop = True)
    
    # Apply the combined filter to X_train and y_train
    if win_loss_only == False:
        if split_dec_or_other_filter:
            filtered_X_train = X_train.copy()[mask & ~(X_train['notformodel_split_dec_or_other'] == 1)]
            filtered_y_train = y_train[mask & ~(X_train['notformodel_split_dec_or_other'] == 1)]
        else:
            filtered_X_train = X_train.copy()[mask]
            filtered_y_train = y_train[mask]
    else:
        if split_dec_or_other_filter:
            filtered_X_train = X_train.copy()[mask & ~(X_train['notformodel_split_dec_or_other'] == 1) & (y_train.isin([0, 2, '0', '2']))] # These are the codes for W/L and L/W
            filtered_y_train = y_train[mask & ~(X_train['notformodel_split_dec_or_other'] == 1) & (y_train.isin([0, 2, '0', '2']))]
        else:
            filtered_X_train = X_train.copy()[mask & (y_train.isin([0, 2, '0', '2']))] # These are the codes for W/L and L/W
            filtered_y_train = y_train[mask & (y_train.isin([0, 2, '0', '2']))]

    # Save the odds for the loss for both test and train data
    train_fighter1_odds = filtered_X_train['fighter1_odds_for_loss']
    train_fighter2_odds = filtered_X_train['fighter2_odds_for_loss']
    
    test_fighter1_odds = X_train.iloc[test_idx]['fighter1_odds_for_loss']
    test_fighter2_odds = X_train.iloc[test_idx]['fighter2_odds_for_loss']

    # Create test
    X_test = X_train.copy().iloc[test_idx]

    # Get base margin cols if needed
    base_margin_train = get_base_margin(filtered_X_train, base_margin_cols)
    base_margin_test = get_base_margin(X_test, base_margin_cols)

    # Remove odds used for loss from training and test X data
    cols_to_remove = [col for col in X_train.columns if 'for_loss' in col or 'backtesting' in col or 'notformodel' in col]
    filtered_X_train.drop(cols_to_remove, axis = 1, inplace = True)
    X_test = X_test.drop(cols_to_remove, axis = 1)

    # Check if empty
    if len(filtered_X_train) == 0:
        pdb.set_trace()

   

    # Create the DMatrix
    if split_dec_or_other_filter:
        train_dmatrix = xgb.DMatrix(filtered_X_train, label=filtered_y_train, base_margin = base_margin_train)
        test_dmatrix = xgb.DMatrix(X_test, label=y_train.iloc[test_idx], base_margin = base_margin_test)
    else:
        train_dmatrix = xgb.DMatrix(filtered_X_train, label=filtered_y_train, base_margin = base_margin_train)
        test_dmatrix = xgb.DMatrix(X_test, label=y_train.iloc[test_idx], base_margin = base_margin_test)

    return train_dmatrix, test_dmatrix, train_fighter1_odds, train_fighter2_odds, test_fighter1_odds, test_fighter2_odds

# This function creates a dmatrix ready for XGBoost, but it also applies a filter to the training data (not test) to remove any split decisions or 'other' methods of victory.
def create_dmatrix_with_split_other_filter(X_train, y_train, train_idx, test_idx, win_loss_only = False, split_dec_or_other = False):
    pdb.set_trace()
    # Convert train_idx to a boolean mask
    mask = pd.Series(False, index=X_train.index)
    mask.iloc[train_idx] = True  # Set True for indices in train_idx

    # Apply the combined filter to X_train and y_train
    if win_loss_only == False:
        filtered_X_train = X_train[mask & ~(X_train['split_dec_or_other'] == 1)]
        filtered_y_train = y_train[mask & ~(X_train['split_dec_or_other'] == 1)]
    else:
        filtered_X_train = X_train[mask & ~(X_train['split_dec_or_other'] == 1) & (y_train.isin([0, 2]))] # These are the codes for W/L and L/W
        filtered_y_train = y_train[mask & ~(X_train['split_dec_or_other'] == 1) & (y_train.isin([0, 2]))]

    # Create the DMatrix
    train_dmatrix = xgb.DMatrix(filtered_X_train.drop(['split_dec_or_other'], axis = 1), label=filtered_y_train)
    test_dmatrix = xgb.DMatrix(X_train.iloc[test_idx].drop(['split_dec_or_other'], axis = 1), label=y_train.iloc[test_idx])
    return train_dmatrix, test_dmatrix

class custom_xgboost_model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Loads the model from the artifact path when the model is loaded."""
        model_path = context.artifacts["model"]
        self.model = joblib.load(model_path)  # Load the model from a joblib file

    def predict(self, context, model_input):
        """Uses the custom prediction function."""
        # Call the function directly instead of a method on the model
        return predict_xgboost_ensemble(self.model, model_input)  # Pass the model and input to the function

# This function trains LightGBM models using k-fold CV and summarises output metrics. Parameters are logged in MLflow as well as metrics.

def cv_light_gbm_mlflow(
        X_train,
        y_train,
        leaves = 30,
        learning_rate = 0.01,
        rounds = 30,
        feature_fraction = 1,
        bagging_fraction = 1,
        n_folds = 10,
        random_seed = 50):

    pdb.set_trace()
    # Set number of classes for training
    no_of_classes = len(set(y_train))

    # Create n folds of the training data
    kf = KFold(n_splits = n_folds, shuffle = True, random_state = random_seed)

    # Create lists that will be used to store models and test datasets
    avg_aucs = []
    wl_aucs = []
    lw_aucs = []
    accuracies = []

    # Print out progress
    print("Training CV models...")

    # Begin new model in MLflow
    with mlflow.start_run():

        mlflow.set_tag("Description", "Cross-validated Light GBM")

        # Now go through each fold
        for train_index, test_index in kf.split(X_train):

            # Create the folds training and test datasets
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Format the data for the model
            train = lgb.Dataset(X_train_fold, label = y_train_fold)

            params = {
                'objective': 'multiclass',      # Binary classification
                'metric': 'multi_logloss', # Evaluation metric
                'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
                'num_leaves': leaves,           # Controls the complexity of the model
                # 'min_data_in_leaf': 10,      # Sets a minimum sample size to prevent over-fitting
                'learning_rate': learning_rate,      # Smaller learning rate for better accuracy
                'feature_fraction': feature_fraction,    # Subsampling of features to speed up training
                'bagging_fraction': bagging_fraction,    # Subsampling of data to prevent overfitting
                'bagging_freq': 2,          # Perform bagging every 5 iterations
                'n_jobs': 3,               # Use n CPU cores
                'verbose': -1,            # Suppress output
                'num_class': no_of_classes
            }
            
            lightgbm_model = lgb.train(params, train, rounds)
                
            # Make predictions on the test set
            y_pred_prob = lightgbm_model.predict(X_test_fold, num_iteration = lightgbm_model.best_iteration)

            # Convert predicted probabilities to class predictions
            y_pred = np.argmax(y_pred_prob, axis=1)

            # Convert the actual results into a binarised format so that AUC functions can work for individual outcomes
            y_test_bin = label_binarize(y_test_fold, classes = np.unique(y_test_fold))

            # Evaluate the model based on accuracy and add it to the list of accuracies
            accuracies.append(accuracy_score(y_test_fold, y_pred))

            # Calculate the AUC
            avg_aucs.append(roc_auc_score(y_test_bin, y_pred_prob, multi_class = "ovr"))

            # Calculate the AUC of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L
            lw_aucs.append(roc_auc_score(y_test_bin[:, 0], y_pred_prob[:, 0]))
            wl_aucs.append(roc_auc_score(y_test_bin[:, 2], y_pred_prob[:, 2]))

        # Now all folds have been calculated, summarise the metrics

        # Log the parameters and outcome in MLflow
        mlflow.log_param('model', 'Light GBM')
        mlflow.log_param('num_leaves', leaves)
        mlflow.log_param('num_iterations', rounds)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('feature_fraction', feature_fraction)
        mlflow.log_param('bagging_fraction', bagging_fraction)
        mlflow.log_metric('accuracy', np.mean(accuracies))
        mlflow.log_metric('W/L AUC', np.mean(wl_aucs))
        mlflow.log_metric('L/W AUC', np.mean(lw_aucs))
        mlflow.log_metric('Avg AUC', np.mean(avg_aucs))
 
    # Return whatever metric we wish to optimise for the LightGBM
    return {'loss': 1 - np.mean(avg_aucs), 'params': params, 'status': STATUS_OK} 

def ev_loss_function(y_pred, y_true, odds_fighter_1, odds_fighter_2, return_loss = False, for_feval = False, input_probs = False):
    """
    Custom loss function for XGBoost that optimizes based on Expected Value (EV) from betting odds.

    Parameters:
    - y_true (np.array): True fight outcome (0 = Fighter 2 win, 1 = Draw, 2 = Fighter 1 win)
    - y_pred (np.array): Raw logits from XGBoost (before softmax transformation) i.e. NOT PROBABILITIES
    - odds_fighter_1 (np.array): Betting odds for Fighter 1
    - odds_fighter_2 (np.array): Betting odds for Fighter 2
    - return_loss (bool): If True, returns the actual EV loss for monitoring
    - for_feval (bool): If True, the function returns a tuple. This is for the xgb.train function
    - input_probs (bool): Set to True if y_pred is an array of probabilities rather than logits

    Returns:
    - grad (np.array): First derivative of the loss function
    - hess (np.array): Second derivative (Hessian) for XGBoost
    """
    # pdb.set_trace()
    # print(len(y_pred))
    
    # Convert from dmatrix if needed
    try:
        y_true = y_true.get_label()
    except:
        pass
    
    # if for_feval:
        # np.savetxt("E:/odds_fighter_1.csv", odds_fighter_1, delimiter=",")
        # np.savetxt("E:/y_true.csv", y_true, delimiter=",")
    #     probs = y_pred # Using the function for evaluation in xgb.train seems to pass probabilities through rather than logits
    # else:
    if input_probs:
        probs = y_pred.copy()
    else:
        # Convert logits into probabilities using softmax
        exp_preds = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))  # Stability trick
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    # np.savetxt("E:/odds_fighter_1.csv", odds_fighter_1, delimiter=",")

    # Extract individual probabilities
    p_fighter_1 = probs[:, 2]  # Probability of Fighter 1 win
    p_draw = probs[:, 1]  # Probability of Draw
    p_fighter_2 = probs[:, 0]  # Probability of Fighter 2 win

    # Expected value calculations
    ev_fighter_1 = p_fighter_1 * (odds_fighter_1 - 1) - p_fighter_2 - p_draw
    ev_fighter_2 = p_fighter_2 * (odds_fighter_2 - 1) - p_fighter_1 - p_draw
    ev_draw = -1 * (1 - p_draw) * np.ones_like(y_true)  # Draw results in a loss of stake

    # Now calculate the actual outcome
    ev_fighter1_actual = np.where(y_true == 2, odds_fighter_1 - 1, -1)
    ev_fighter2_actual = np.where(y_true == 0, odds_fighter_2 - 1, -1)
    ev_draw_actual = np.where(y_true == 1, 0, -1)

    # Select EV based on true outcome difference to the predicted EV
    # Removed draw from EV as will never be betting on it and have no odds for it
    ev_loss = np.stack((
        (ev_fighter2_actual - ev_fighter_2) ** 2,
        (ev_draw_actual - ev_draw) ** 2,
        (ev_fighter1_actual - ev_fighter_1) ** 2
        ), axis = 1)

    # Compute the gradient and Hessian (vectorized)
    
    # Calculate errors (difference between actual and predicted EV)
    error_fighter_2 = ev_fighter_2 - ev_fighter2_actual
    error_draw = ev_draw - ev_draw_actual
    error_fighter_1 = ev_fighter_1 - ev_fighter1_actual
    
    # Gradients for each class. Got these from chatgpt differentiation
    grad_fighter_2 = 2 * error_fighter_2 * p_fighter_2 * (1 - p_fighter_2)
    grad_draw = 2 * error_draw * p_draw * (1 - p_draw)
    grad_draw = np.zeros(len(p_fighter_1))
    grad_fighter_1 = 2 * error_fighter_1 * p_fighter_1 * (1 - p_fighter_1)
    
    # Combine gradients into a single matrix
    grad = np.stack([grad_fighter_2, grad_draw, grad_fighter_1], axis=1)

    # Hessians for each class
    hess_fighter_2 = 2 * p_fighter_2 * (1 - p_fighter_2) * (1 - 2 * p_fighter_2)
    hess_draw = 2 * p_draw * (1 - p_draw) * (1 - 2 * p_draw)
    hess_draw = np.zeros(len(p_fighter_1))
    hess_fighter_1 = 2 * p_fighter_1 * (1 - p_fighter_1) * (1 - 2 * p_fighter_1)
    
    # Combine Hessians into a single matrix
    hess = np.stack([hess_fighter_2, hess_draw, hess_fighter_1], axis=1)

    # Get the overall loss
    ev_loss = np.array(ev_loss, dtype=np.float64)
    overall_loss = np.mean(ev_loss)

    if for_feval:
    
        # Compute log loss for testing
        # return 'TEst_logloss', log_loss(y_true, probs)
        # print(ev_loss.mean())
        # joblib.dump(overall_loss, 'E:/overall_loss.pkl')
        return 'EV_MSE', overall_loss
    elif return_loss:
        return overall_loss
    return grad, hess

def create_custom_loss(odds_fighter_1, odds_fighter_2):
    """
    Wrapper function to pass odds dynamically to the loss function.
    
    Returns:
    - A custom loss function that can be used in XGBoost training.
    """
    # np.savetxt("E:/odds_fighter_1_train.csv", odds_fighter_1, delimiter=",")
    
    return lambda y_pred, y_true: ev_loss_function(y_pred, y_true, odds_fighter_1, odds_fighter_2)

def ev_loss_function_test(odds_fighter_1, odds_fighter_2):
    """
    Wrapper function to pass odds dynamically to the loss function.
    
    Returns:
    - A custom loss function that can be used in XGBoost training.
    """
    # pdb.set_trace()
    
    return lambda y_pred, y_true: ev_loss_function(y_pred, y_true, odds_fighter_1, odds_fighter_2, for_feval = True)

def get_base_margin(df, base_margin_cols):
    
    """
    This function takes a data frame and a list of columns of odds to be used as probabilities and returns.
    This function assumes there are 3 classes of outcome
    
    Parameters
    ----------
    df: the data frame
    
    base_margin_cols: an ordered list of the odds columns with the order corresponding to how the labels in the target variable to be modelled are ordered. Can also set a flat rate by inputing a dictionary instead of a column name like this `{'flat_prob': 0.01}`
    
    """
    
    # pdb.set_trace()
    
    # Check if there are any cols
    if base_margin_cols == None:        
        return np.zeros(3 * len(df))
    else:
    
        cols = []
        for spec in base_margin_cols:
            if isinstance(spec, dict) and "flat_prob" in spec:
                cols.append(np.full(len(df), spec["flat_prob"]))
            else:
                cols.append(1.0 / df[spec].to_numpy())
        
        # stack into array of shape (n_classes, n_rows) → transpose → (n_rows, n_classes)
        p = np.vstack(cols).T
    
        # Make into an array and clip to avoid 0 and 1
        eps = 1e-9
        p = np.clip(np.array(p), eps, 1 - eps)
    
        p = p / p.sum(axis=1, keepdims=True)  # ensure rows sum to 1
        
        # flatten row-major for XGBoost
        rv = np.log(p).ravel()
        
        # Check length
        stop_if_not(len(rv) == 3 * len(df), "Base margin length check")
        
        return rv  

# This function takes an h2o autoML results object and saves the outcomes to an MLflow experiment

def h2o_automl_to_mlflow(automl_models):
    
    # Loop through the models in the object
    for model in automl_models:
        # Begin a new model log in mlflow for the model
        with mlflow.start_run():
            # Log the stats and description for the model
            mlflow.log_param('model', x)
            mlflow.log_metric('RMSE', rmse)
            mlflow.log_metric('AUC', auc)

    return "Models saved to MLflow"

# This function trains LightGBM models using k-fold CV and summarises output metrics. Parameters are logged in MLflow as well as metrics.

def objective_cv_light_gbm_mlflow(
        params):

    # pdb.set_trace()

    # Create n folds of the training data
    kf = KFold(n_splits = params['n_folds'], shuffle = True, random_state = params['random_state'])

    # Create lists that will be used to store models and test datasets
    avg_aucs = []
    wl_aucs = []
    lw_aucs = []
    accuracies = []

    # Ensure formatting of params is correct
    for param in ['num_iterations', 'num_leaves']:
        params[param] = int(params[param])

    # Print out progress
    print("Training CV models...")

    # Begin new model in MLflow
    with mlflow.start_run():

        mlflow.set_tag("Description", "Cross-validated Light GBM")

        # Log the training parameters
        mlflow.log_param('model', 'Light GBM')
        mlflow.log_params(params)
        # mlflow.log_param('num_leaves', params['num_leaves'])
        # mlflow.log_param('num_iterations', params['num_rounds'])
        # mlflow.log_param('learning_rate', params['learning_rate'])
        # mlflow.log_param('feature_fraction', params['feature_fraction'])
        # mlflow.log_param('bagging_fraction', params['bagging_fraction'])

        # Now go through each fold
        for train_index, test_index in kf.split(X_train):

            # Create the folds training and test datasets
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Format the data for the model
            train = lgb.Dataset(X_train_fold, label = y_train_fold)
            
            lightgbm_model = lgb.train(params, train, params['num_iterations'])
                
            # Make predictions on the test set
            y_pred_prob = lightgbm_model.predict(X_test_fold, num_iteration = lightgbm_model.best_iteration)

            # Convert predicted probabilities to class predictions
            y_pred = np.argmax(y_pred_prob, axis=1)

            # Convert the actual results into a binarised format so that AUC functions can work for individual outcomes
            y_test_bin = label_binarize(y_test_fold, classes = np.unique(y_test_fold))

            # Evaluate the model based on accuracy and add it to the list of accuracies
            accuracies.append(accuracy_score(y_test_fold, y_pred))

            # Calculate the AUC
            avg_aucs.append(roc_auc_score(y_test_bin, y_pred_prob, multi_class = "ovr"))

            # Calculate the AUC of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L
            lw_aucs.append(roc_auc_score(y_test_bin[:, 0], y_pred_prob[:, 0]))
            wl_aucs.append(roc_auc_score(y_test_bin[:, 2], y_pred_prob[:, 2]))

        # Now all folds have been calculated, summarise the metrics

        # Log the outcome in MLflow
        mlflow.log_metric('accuracy', np.mean(accuracies))
        mlflow.log_metric('W/L AUC', np.mean(wl_aucs))
        mlflow.log_metric('L/W AUC', np.mean(lw_aucs))
        mlflow.log_metric('Avg AUC', np.mean(avg_aucs))
 
    # Return whatever metric we wish to optimise for the LightGBM
    return {'loss': 1 - np.mean(avg_aucs), 'params': params, 'status': STATUS_OK} 

# This function trains LightGBM models using k-fold CV and summarises output metrics. Parameters are logged in MLflow as well as metrics.

def objective_cv_randomforest_mlflow(
        params):

    # pdb.set_trace()

    # Create n folds of the training data
    kf = KFold(n_splits = params['n_folds'], shuffle = True, random_state = params['random_state'])

    # Create lists that will be used to store models and test datasets
    avg_aucs = []
    wl_aucs = []
    lw_aucs = []
    accuracies = []

    # Ensure formatting of params is correct
    for param in ['max_bin', 'min_child_weight', 'max_depth', 'num_iterations']:
        params[param] = int(params[param])

    # Force a random forest
    param.update({
        'booster': 'gbtree',
        'num_boost_round': 1, # to tell the package to train only 1 random forest
        'learning_rate': 1, # Learning rate
    })

    # Print out progress
    print("Training CV models...")

    # Begin new model in MLflow
    with mlflow.start_run():

        mlflow.set_tag("Description", "Cross-validated XGBoost")

        # Log the training parameters
        mlflow.log_param('model', 'XGBoost')
        mlflow.log_params(params)
        # mlflow.log_param('num_leaves', params['num_leaves'])
        # mlflow.log_param('num_iterations', params['num_rounds'])
        # mlflow.log_param('learning_rate', params['learning_rate'])
        # mlflow.log_param('feature_fraction', params['feature_fraction'])
        # mlflow.log_param('bagging_fraction', params['bagging_fraction'])

        # Now go through each fold
        for train_index, test_index in kf.split(X_train):

            # Create the folds training and test datasets
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Format the data for the model
            train = xgb.DMatrix(X_train_fold, label = y_train_fold)
            test = xgb.DMatrix(X_test_fold, label = y_test_fold)
            
            model = xgb.train(params, train, num_boost_round = params['num_iterations'])
                
            # Make predictions on the test set
            y_pred_prob = model.predict(test)

            # Convert predicted probabilities to class predictions
            y_pred = np.argmax(y_pred_prob, axis=1)

            # Convert the actual results into a binarised format so that AUC functions can work for individual outcomes
            y_test_bin = label_binarize(y_test_fold, classes = np.unique(y_test_fold))

            # Evaluate the model based on accuracy and add it to the list of accuracies
            accuracies.append(accuracy_score(y_test_fold, y_pred))

            # Calculate the AUC
            avg_aucs.append(roc_auc_score(y_test_bin, y_pred_prob, multi_class = "ovr"))

            # Calculate the AUC of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L
            lw_aucs.append(roc_auc_score(y_test_bin[:, 0], y_pred_prob[:, 0]))
            wl_aucs.append(roc_auc_score(y_test_bin[:, 2], y_pred_prob[:, 2]))

            del [X_train_fold, X_test_fold, y_train_fold, y_test_fold, train, test, model, y_pred_prob]

        # Now all folds have been calculated, summarise the metrics

        # Log the outcome in MLflow
        mlflow.log_metric('accuracy', np.mean(accuracies))
        mlflow.log_metric('W/L AUC', np.mean(wl_aucs))
        mlflow.log_metric('L/W AUC', np.mean(lw_aucs))
        mlflow.log_metric('Avg AUC', np.mean(avg_aucs))
 
    del kf
    
    gc.collect()

    # Return whatever metric we wish to optimise for the LightGBM. Do 1 - because it tries to minimise the value.
    return {'loss': 1 - np.mean(wl_aucs), 'params': params, 'status': STATUS_OK} 

# This function trains XGboost models using k-fold CV and summarises output metrics. Parameters are NOT logged in MLflow.

def objective_cv_xgboost(
        params):

    # pdb.set_trace()
    
    try:
    
        # Create lists that will be used to store models and test datasets
        avg_aucs = []
        wl_aucs = []
        lw_aucs = []
        accuracies = []
        wl_accuracies = []
        lw_accuracies = []
        avg_accuracies = []
        log_losses = []
        ev_mses = []
    
        # Ensure formatting of params is correct
        for param in ['max_bin', 'min_child_weight', 'max_depth', 'num_iterations']:
            params[param] = int(params[param])
        
        # Trim the model features for similar columns if set to
        if params['trim_model_features']:
            X_train_trim = X_train[trim_column_list(X_train.columns)].copy()
        else:
            X_train_trim = X_train.copy()
            
        ## Precompute DMatrix objects using ThreadPoolExecutor
        # If the training data is to have split decs and other results excluded, then use the custom function to do that
        # if params['exclude_split_other_from_train'] == False:
        with ThreadPoolExecutor(max_workers = config['num_threads']) as executor:
            futures = {executor.submit(create_dmatrix, 
                                       X_train, 
                                       y_train, 
                                       X_train[X_train['notformodel_fold_id'] != fold_id].index, 
                                       X_train[X_train['notformodel_fold_id'] == fold_id].index,
                                       params['base_margin_cols'],
                                       params['win_loss_only'], 
                                       params['exclude_split_other_from_train']): fold_id for fold_id in range(0, 5)}
            dmatrix_data = [future.result() for future in as_completed(futures)]   
        # else:
        #     with ThreadPoolExecutor(max_workers = config['num_threads']) as executor:
        #         futures = {executor.submit(create_dmatrix_with_split_other_filter, X_train, y_train, train_idx, test_idx, params['win_loss_only']): (train_idx, test_idx) for train_idx, test_idx in kf.split(X_train)}
        #         dmatrix_data = [future.result() for future in as_completed(futures)]   
    
        # dmatrix_data = []
        # for train_idx, test_idx in kf.split(X_train):
        #     result = create_dmatrix(X_train, y_train, train_idx, test_idx, params['win_loss_only'], params['exclude_split_other_from_train'])
        #     dmatrix_data.append(result)
    
        # Print out progress
        print("Training CV models...")
    
        # Loop through precomputed DMatrix objects for each fold
        for train_dmatrix, test_dmatrix, train_fighter1_odds, train_fighter2_odds, test_fighter1_odds, test_fighter2_odds in dmatrix_data:
            # pdb.set_trace()
            
            # Train the model
            if model_hyper_params['loss_function'] == 'ev_loss_function': # any eval metric above will be ignored
            
                # Pass the odds to the loss function
                custom_loss = create_custom_loss(train_fighter1_odds.to_numpy(), train_fighter2_odds.to_numpy())
                custom_loss_test = ev_loss_function_test(test_fighter1_odds.to_numpy(), test_fighter2_odds.to_numpy())
                
                # print(test_dmatrix.get_data().shape)
                try:
                    model = xgb.train(params, train_dmatrix,
                                      num_boost_round = params['num_iterations'],
                                      evals = [(test_dmatrix, 'eval')],
                                      early_stopping_rounds = params['early_stopping_rounds'],
                                      verbose_eval = False,
                                      obj = custom_loss,
                                      custom_metric = custom_loss_test
                                      ) # Logloss still displays with verbose but isn't being used
                except:
                    pdb.set_trace()
                
            else:
                # Train the model with early stopping
                model = xgb.train(params, train_dmatrix,
                                  num_boost_round = params['num_iterations'],
                                  evals = [(test_dmatrix, 'eval')],
                                  early_stopping_rounds = params['early_stopping_rounds'],
                                  verbose_eval = False)
                
            # Make predictions on the test set
            y_pred_prob = model.predict(test_dmatrix)
                    
            # Get the outcomes of the test data
            y_test_fold = test_dmatrix.get_label().astype(int)
    
            # Convert predicted probabilities to class predictions
            y_pred = np.argmax(y_pred_prob, axis = 1)
    
            # Convert actual results to binarized format
            y_test_bin = label_binarize(y_test_fold, classes=np.array([0, 1, 2]))
            y_pred_bin = label_binarize(y_pred, classes=np.array([0, 1, 2]))
    
            # Evaluate the model based on accuracy and add it to the list of accuracies
            accuracies.append(accuracy_score(y_test_fold, y_pred))
    
            # # Calculate the AUC
            # avg_aucs.append(roc_auc_score(y_test_bin, y_pred_prob, multi_class = "ovr"))
    
            # Calculate the AUC of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L
            wl_auc = roc_auc_score(y_test_bin[:, 2], y_pred_prob[:, 2])
            lw_auc = roc_auc_score(y_test_bin[:, 0], y_pred_prob[:, 0])
            lw_aucs.append(lw_auc)
            wl_aucs.append(wl_auc)
            
            # Calculate the AUC as the average of the W/L and L/W
            avg_aucs.append(np.mean([wl_auc, lw_auc]))
            
            # Calculate the accuracies
            wl_accuracy = (y_test_bin[:, 2] == y_pred_bin[:, 2]).mean()
            lw_accuracy = (y_test_bin[:, 0] == y_pred_bin[:, 0]).mean()
            lw_accuracies.append(lw_accuracy)
            wl_accuracies.append(wl_accuracy)
            
            # Average the two accuracies
            avg_accuracies.append(hmean([lw_accuracy, wl_accuracy]))
            
            # Calculate the log loss
            log_loss_fold = log_loss(y_test_fold, y_pred_prob)
            log_losses.append(log_loss_fold)
            
            # Get the logit predictions for use in custom loss function
            # if model_hyper_params['loss_function'] == 'ev_loss_function':
            ev_mse_fold = ev_loss_function(
                y_pred_prob,
                y_test_fold,
                test_fighter1_odds,
                test_fighter2_odds,
                return_loss = True,
                input_probs = True)
            ev_mses.append(ev_mse_fold)
    
            # Clean up memory
            del model, train_dmatrix, test_dmatrix, y_pred_prob, wl_auc, lw_auc, wl_accuracy, lw_accuracy, ev_mse_fold
            gc.collect()
             
            
            
        # Save the metrics
        metrics = {
            'Avg WL only AUC': np.mean(avg_aucs),
            'Hmean WL only Accuracy': np.mean(avg_accuracies),
            'Accuracy': np.mean(accuracies),
            'Logloss': np.mean(log_losses),
            'EV MSE': np.mean(ev_mses)
        }
                
        if model_hyper_params['loss_function'] == 'ev_loss_function':
            return {'loss': np.mean(ev_mses), 'params': params, 'status': STATUS_OK, 'metrics': metrics} 
        else:
            # Return whatever metric we wish to optimise for the LightGBM. Do 1 - because it tries to minimise the value.
            return {'loss': np.mean(log_losses), 'params': params, 'status': STATUS_OK, 'metrics': metrics} 
        
    except Exception as e:
        print(f"Skipping trial due to error: {e}")
        return {'loss': float('inf'), 'status': hyperopt.STATUS_FAIL}
        
        

# This function trains LightGBM models using k-fold CV and summarises output metrics. Parameters are logged in MLflow as well as metrics.

def objective_cv_xgboost_mlflow(
        params):

    # pdb.set_trace()

    # Create n folds of the training data
    kf = KFold(n_splits = params['n_folds'], shuffle = True, random_state = params['random_state'])

    # Create lists that will be used to store models and test datasets
    avg_aucs = []
    wl_aucs = []
    lw_aucs = []
    accuracies = []
    wl_accuracies = []
    lw_accuracies = []
    avg_accuracies = []
    log_losses = []

    # Ensure formatting of params is correct
    for param in ['max_bin', 'min_child_weight', 'max_depth', 'num_iterations']:
        params[param] = int(params[param])

    ## Precompute DMatrix objects using ThreadPoolExecutor
    # If the training data is to have split decs and other results excluded, then use the custom function to do that
    if params['exclude_split_other_from_train'] == False:
        with ThreadPoolExecutor(max_workers = config['num_threads']) as executor:
            futures = {executor.submit(create_dmatrix, X_train, y_train, train_idx, test_idx, params['win_loss_only']): (train_idx, test_idx) for train_idx, test_idx in kf.split(X_train)}
            dmatrix_data = [future.result() for future in as_completed(futures)]   
    else:
        with ThreadPoolExecutor(max_workers = config['num_threads']) as executor:
            futures = {executor.submit(create_dmatrix_with_split_other_filter, X_train, y_train, train_idx, test_idx, params['win_loss_only']): (train_idx, test_idx) for train_idx, test_idx in kf.split(X_train)}
            dmatrix_data = [future.result() for future in as_completed(futures)]   

    # Print out progress
    print("Training CV models...")

    # Begin new model in MLflow
    with mlflow.start_run():

        mlflow.set_tag("Description", "Cross-validated XGBoost")

        # Log the training parameters
        mlflow.log_param('model', 'XGBoost')
        mlflow.log_params(params)
        # mlflow.log_param('num_leaves', params['num_leaves'])
        # mlflow.log_param('num_iterations', params['num_rounds'])
        # mlflow.log_param('learning_rate', params['learning_rate'])
        # mlflow.log_param('feature_fraction', params['feature_fraction'])
        # mlflow.log_param('bagging_fraction', params['bagging_fraction'])

        # Loop through precomputed DMatrix objects for each fold
        for train_dmatrix, test_dmatrix in dmatrix_data:
            # pdb.set_trace()
            # Train the model with early stopping
            model = xgb.train(params, train_dmatrix,
                              num_boost_round = params['num_iterations'],
                              evals = [(test_dmatrix, 'eval')],
                              early_stopping_rounds = params['early_stopping_rounds'],
                              verbose_eval = False)
                
            # Make predictions on the test set
            y_pred_prob = model.predict(test_dmatrix)
            
            # Get the outcomes of the test data
            y_test_fold = test_dmatrix.get_label().astype(int)

            # Convert predicted probabilities to class predictions
            y_pred = np.argmax(y_pred_prob, axis = 1)

            # Convert actual results to binarized format
            y_test_bin = label_binarize(y_test_fold, classes=np.array([0, 1, 2]))
            y_pred_bin = label_binarize(y_pred, classes=np.array([0, 1, 2]))

            # Evaluate the model based on accuracy and add it to the list of accuracies
            accuracies.append(accuracy_score(y_test_fold, y_pred))

            # # Calculate the AUC
            # avg_aucs.append(roc_auc_score(y_test_bin, y_pred_prob, multi_class = "ovr"))

            # Calculate the AUC of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L
            wl_auc = roc_auc_score(y_test_bin[:, 2], y_pred_prob[:, 2])
            lw_auc = roc_auc_score(y_test_bin[:, 0], y_pred_prob[:, 0])
            lw_aucs.append(lw_auc)
            wl_aucs.append(wl_auc)
            
            # Calculate the AUC as the average of the W/L and L/W
            avg_aucs.append(np.mean([wl_auc, lw_auc]))
            
            # Calculate the accuracies
            wl_accuracy = (y_test_bin[:, 2] == y_pred_bin[:, 2]).mean()
            lw_accuracy = (y_test_bin[:, 0] == y_pred_bin[:, 0]).mean()
            lw_accuracies.append(lw_accuracy)
            wl_accuracies.append(wl_accuracy)
            
            # Average the two accuracies
            avg_accuracies.append(hmean([lw_accuracy, wl_accuracy]))
            
            # **Calculate LogLoss for the fold**
            log_loss_fold = log_loss(y_test_fold, y_pred_prob)
            log_losses.append(log_loss_fold)
            
            # Clean up memory
            del model, train_dmatrix, test_dmatrix, y_pred_prob, wl_auc, lw_auc, wl_accuracy, lw_accuracy
            gc.collect()
            
        # Now all folds have been calculated, summarise the metrics

        # Log the outcome in MLflow
        mlflow.log_metric('Accuracy', np.mean(accuracies))
        mlflow.log_metric('W/L AUC', np.mean(wl_aucs))
        mlflow.log_metric('L/W AUC', np.mean(lw_aucs))
        mlflow.log_metric('W/L Accuracy', np.mean(wl_accuracies))
        mlflow.log_metric('L/W Accuracy', np.mean(lw_accuracies))
        mlflow.log_metric('Avg WL only AUC', np.mean(avg_aucs))
        mlflow.log_metric('Hmean WL only Accuracy', np.mean(avg_accuracies))
        mlflow.log_metric('LogLoss', np.mean(log_losses)) 
 
    del kf

    # Return whatever metric we wish to optimise for. Do 1 - because it tries to minimise the value.
    return {'loss': np.mean(log_losses), 'params': params, 'status': STATUS_OK} 

# This function trains XGboost models on data with differing number of features to see how many are required. once a number of features has been set, it will use an fmin function to find the optimal hyperparameters using CV, log those in mlflow along with the accuracy metrics.

# This function requires the X_train and y_train data to be ready but named X_train_master and y_train, because the sub function that optimises hyperparameters uses X_train.

# X_train_master should have the columns sorted in order of importance. For n number of features, it will simply take the first n columns

def objective_xgboost_features_mlflow(
        params):

    # pdb.set_trace()
    
    # Check for duplicates
    if is_duplicate_run(params, parameter_to_check = 'num_of_features'):
        
        # pdb.set_trace()
        
        # If the run existing, find what the loss was and return it without logging in mlflow
        if model_hyper_params['loss_function'] == 'ev_loss_function':
            existing_metric, all_params = get_existing_metric_and_params({'num_of_features': params['num_of_features']}, 'EV MSE')
        else:
            existing_metric, all_params = get_existing_metric_and_params({'num_of_features': params['num_of_features']}, 'Logloss')
        
        print(f"Duplicate run found, skipping logging. Accuracy: {existing_metric:.4f}")
        return {'loss': existing_metric, 'params': all_params, 'status': STATUS_OK}
        
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
            
            # Find the features that we want
            column_names = X_train_master.columns[0:params['num_of_features']].to_list()
            
            # Ensure that both fighter1 and fighter2 versions of columns are in there
            column_names = column_names + pd.Series(column_names).replace('fighter2', 'fighterX').replace('fighter1', 'fighter2').replace('fighterX', 'fighter1').tolist()
            
            # Add on any columns needed for functions
            column_names = column_names + [col for col in X_train_master.columns if 'notformodel' in col]
            
            # Create the train dataset based on the number of features
            global X_train
            X_train = X_train_master.copy()[list(set(column_names))]
            
            # Add on odds columns only for the loss function
            X_train['fighter1_odds_for_loss'] = X_train_master['fighter1_backtesting_odds']
            X_train['fighter2_odds_for_loss'] = X_train_master['fighter2_backtesting_odds']
            
            # X_train = pd.concat([X_train, X_train_master[['notformodel_split_dec_or_other']]], axis = 1)
            
            X_train.reset_index(drop = True, inplace = True)
            y_train.reset_index(drop = True, inplace = True)
            
            # Log parameters in MLflow
            mlflow.log_param('num_of_features', params['num_of_features'])
            mlflow.log_param('max_evals', params['max_evals'])
            
            # Set up the trials object for this inner hyperopt
            trials_inner = Trials()
            
            # Train models to find the best hyperparameters
            # Allow 5 errors with trials
            error_count = 0
            
            while error_count < 5:
                try:
                    # First do some random ones to get a baseline
                    best = fmin(fn = objective_cv_xgboost,
                                space = params_inner,
                                algo = rand.suggest, # tpe is the most commonly used algorithm in hyperopt by atpe_suggest is more thorough. Might take 500-1000 iterations
                                max_evals = 100,
                                trials = trials_inner,
                                early_stop_fn = no_progress_loss(101))
                    break # exit loop if successful
                except:
                    print('Error encountered')
                    error_count += 1
                    # pdb.set_trace()
            
            while error_count < 5:
                try:
                    best = fmin(fn = objective_cv_xgboost,
                                space = params_inner,
                                algo = atpe_suggest, # tpe is the most commonly used algorithm in hyperopt by atpe_suggest is more thorough. Might take 500-1000 iterations
                                max_evals = params['max_evals'],
                                trials = trials_inner,
                                early_stop_fn = no_progress_loss(params['hyperopt_early_stopping']))
                    break # exit loop if successful
                except:
                    # pdb.set_trace()
                    print('Error encountered')
                    error_count += 1
                    # pdb.set_trace()
                    
            # Get the best params if errors occurred
            if error_count >= 5:
                best = trials_inner.best_trial['result']['params']
            
            # Log the parameters of the best model
            mlflow.log_params(best)
            
            # Log the metrics of the best fit
            mlflow.log_metrics(trials_inner.best_trial['result']['metrics'])
    
    # pdb.set_trace()
    
    # Add inner params
    params.update(best | trials_inner.best_trial['result']['params'])
    
    if model_hyper_params['loss_function'] == 'ev_loss_function':
        # Return whatever metric we wish to optimise for the LightGBM. Do 1 - because it tries to minimise the value.
        return {'loss': trials_inner.best_trial['result']['metrics']['EV MSE'], 'params': params, 'status': STATUS_OK}  #  Add on the inner params
    else:
        # Return whatever metric we wish to optimise for the LightGBM. Do 1 - because it tries to minimise the value.
        return {'loss': trials_inner.best_trial['result']['metrics']['Logloss'], 'params': params, 'status': STATUS_OK}

# This function predicts future UFC events

def predict_future_events(
        skip_data_update,
        skip_full_data_update,
        skip_save_data,
        skip_betfair_history,
        config,
        data_processing_config,
        model_hyper_params,
        betfair_login_config,
        test_fighters,
        output_path = None,
        skip_data_scrape = False
        ):
    
    """
    This function predicts all the future UFC events. It first scrapes the latest data, then if required, reprocesses
    the data to get the latest forecasting dataset, and then forecasts the upcoming fights based on the standard data filters
    
    Parameters
    ----------
    skip_data_update: bool
        whether to skip updating the data and go straight to predictions
    
    skip_full_data_update: bool
        whether to skip re-running the processing script on cleaned data. If set to False, only the odds will be updated
        
    skip_save_data: bool
        set whether the saving of processed data should be skipped i.e. only used for the predictions
        
    skip_betfair_history: bool
        set whether to skip downloading the latest betfair odds history data
    
    output_path: str
        the path where the outputs from the predictions will be saved. This is in addition to automatic outputs, which will be saved in ./results/
    
    """
    # pdb.set_trace()
    # Set the odds to use as upcoming instead of as used in the regular data
    data_processing_config['odds_suffix_for_true_odds'] = data_processing_config['upcoming_odds_suffix_for_true_odds']
    model_hyper_params['odds_to_use'] = data_processing_config['upcoming_odds_suffix_for_true_odds']

    ##########################################################################################
    ### Update the upcoming event and odds data ###
    ##########################################################################################

    if not skip_data_update:
        
        if not skip_data_scrape:
            # Run these scripts to scrape then clean the raw data.
            # Pass skip_fighter_tott_update so the slow full-fighter re-scrape is
            # skipped whenever we are also skipping the full dataset rebuild.
            scrape_all_unparsed_data(
                skip_betfair_history,
                config,
                betfair_login_config,
                skip_fighter_tott_update = skip_full_data_update
            )
            clean_all_raw_data_and_save(config)
        
            gc.collect();
        
        # Optional, can either rerun all the processing, or just update the odds. By default will rerun all processing if new events were found to have completed
        if skip_full_data_update:
            
            # Clear out memory
            # globals().clear()
            # # Read in global objects and packages
            # exec(open('global.py').read())
            # gc.collect();
            
            # # Make sure for the backtesting odds we use whatever upcoming because they're used to assess the model predictions later
            # model_hyper_params['odds_to_use'] = data_processing_config['upcoming_odds_suffix_for_true_odds']
            
            forecasting_data = update_odds_on_merged_data(
                    cleaned_fight_odds_data_path = './data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet'),
                    cleaned_event_odds_data_path = './data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet'),
                    cleaned_betfair_odds_data_path = './data/' + config['betfair_upcoming_odds_file_name'].replace('.parquet', '_clean.parquet'),
                    # events_path = './data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet'),
                    final_data_folder = config['final_data_path'],
                    average_odds_model_path = './results/average_odds_model.joblib',
                    config = config,
                    data_processing_config = data_processing_config,
                    upcoming_only = True
                    )
        else:
            # Refresh the dataset
            forecasting_data = merge_and_process_all_clean_data(
                    config,
                    elo_config,
                    data_processing_config,
                    odds_scrape_config,
                    model_hyper_params,
                    test_fighter_list = test_fighters,
                    skip_save_data = skip_save_data # whether to skip the save data and just return merged data
                    )
            
    else:
        forecasting_data = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas() # pq.ParquetDataset(config['final_data_path']).read().to_pandas()
        # add in only odds update
     
    gc.collect();
    # pdb.set_trace()
            
    # Apply standard filters
    forecasting_data = standard_data_filter(forecasting_data, include_upcoming = True, keep_rows_with_flag = True)

    ##########################################################################################
    ### Choose which events to predict ###
    ##########################################################################################

    upcoming_fights = forecasting_data[forecasting_data['outcome'].isna()]
    # upcoming_fights = forecasting_data[forecasting_data['event'] == 'ufc311:makhachevvs.moicano']

    upcoming_fights_for_model = upcoming_fights[upcoming_fights['notformodel_exclusion_reason'] == ''].copy()

    ##########################################################################################
    ### Get the models predictions ###
    ##########################################################################################

    # Get the data ready for the model
    X_upcoming, temp1, y_upcoming, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
        upcoming_fights_for_model,
        upcoming_fights = True)

    # Load files
    # rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/2')
    rf_xgb_model = joblib.load(bet_config['model_to_use'])

    # Get the model predictions
    # prediction_probabilities, temp = rf_xgb_model.predict(X_upcoming)
    prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_upcoming, [0] * len(y_upcoming), model_hyper_params)

    # Add formatted fighters
    upcoming_fights_for_model = add_formatted_fighter_names(upcoming_fights_for_model, on = ['fighter1', 'fighter2'], formatted_col_names = ['fighter1', 'fighter2'], config = config)
    upcoming_fights = add_formatted_fighter_names(upcoming_fights, on = ['fighter1', 'fighter2'], formatted_col_names = ['fighter1', 'fighter2'], config = config)

    gc.collect();
    # pdb.set_trace()
    # Summarise the model predictions and betting decisions using the config
    betting_data = summarise_model_predictions(
        actuals_data_df = upcoming_fights_for_model, 
        prediction_probabilities = prediction_probabilities, 
        min_prob_to_bet = bet_config['min_prob_to_bet'], 
        min_ev_to_bet = bet_config['min_ev_to_bet'], 
        min_bet_size = bet_config['min_bet_size'], 
        max_bet_size = bet_config['max_bet_size'],
        exchange_commission = bet_config['exchange_commission'],
        target_ev_after_cutoff = bet_config['min_ev_to_bet'])

    # Add the final bet size based on balance
    if bet_config['constant_bets'] == False:
        betting_data['Final_bet'] = np.round((betting_data['bets'] * betting_data['bet_size'] * bet_config['start_balance']) / 5) * 5
    else:
        betting_data['Final_bet'] = np.round((betting_data['bets'] * betting_data['bet_size']) / 5) * 5

    betting_data['Potential_win'] = np.round(betting_data['bets'] * betting_data['Final_bet'] * np.where(betting_data['Bet_trigger'] == betting_data['fighter1'], betting_data['fighter1_backtesting_odds'] - 1, betting_data['fighter2_backtesting_odds'] - 1), 2)

    betting_data['bet_assumed_odds'] = betting_data['bets'] * np.where(betting_data['Bet_trigger'] == betting_data['fighter1'], betting_data['fighter1_backtesting_odds'], betting_data['fighter2_backtesting_odds'])

    # Round odds columns
    for col in [cols for cols in betting_data.columns if 'odds' in cols and 'datetime' not in cols]:
        betting_data.loc[:, col] = np.round(betting_data[col], 2)

    betting_data.rename(columns = {'fighter1_notformodel_true_odds_asat_datetime': 'f1_scrape_date', 
                                   'fighter2_notformodel_true_odds_asat_datetime': 'f2_scrape_date',
                                   'Bet_target_exchange_odds': 'Bet_min_exchange_odds'}, inplace = True)

    # Flag whether odds Betfair imputed
    

    summary_data = betting_data[['date', 'event', 'f1_scrape_date', 'f2_scrape_date', 'fighter1', 'fighter1_backtesting_odds', 'Fighter1_win_prob', 'Fighter1_win_EV', 'fighter2', 'fighter2_backtesting_odds', 'Fighter2_win_prob', 'Fighter2_win_EV', 'Bet_trigger', 'bet_assumed_odds', 'Bet_min_exchange_odds', 'Final_bet', 'Potential_win']]

    # Add the fights that haven't been modelled along with exclusion reason for info
    summary_data = pd.concat([
         summary_data,
         upcoming_fights[upcoming_fights['notformodel_exclusion_reason'] != ''][['date', 'event', 'fighter1', 'fighter2', 'notformodel_exclusion_reason']]
    ])

    print(
          summary_data.to_string()
    )

    # Get only the bets to place
    bets_to_place = betting_data[(betting_data['bets'] == 1) & (betting_data['date'] == betting_data['date'].min())].copy()
    bets_to_place['bet_win_prob'] = np.where(bets_to_place['Bet_trigger'] == bets_to_place['fighter1'], bets_to_place['Fighter1_win_prob'], bets_to_place['Fighter2_win_prob'])

    try:
        del round
    except:
        pass

    for i in range(0, len(bets_to_place)):
        print("")
        print(f"Bet: {bets_to_place.iloc[i]['Bet_trigger']}")
        print(f"Win probability: {bets_to_place.iloc[i]['bet_win_prob']:.1f}%")
        print(f"EV: {bets_to_place.iloc[i]['bet_ev']:.2f} at {bets_to_place.iloc[i]['bet_assumed_odds']:.2f} odds")
        print(f"Min exchange odds (EV >= {bet_config['min_ev_to_bet']:.2f}): {bets_to_place.iloc[i]['Bet_min_exchange_odds']:.2f}")

    # Output the results
    summary_data.to_csv('./results/upcoming_predictions.csv')

    # If on Raspberry Pi, output to Gdrive
    if not output_path == None:
        
        summary_data.to_csv(f'{output_path}/upcoming_predictions_{datetime.today()}.csv')
        
        # Create text file with the bet summaries
        text_file_path = f"{output_path}/bets_to_place.txt"
        
        with open(text_file_path, "w") as file:
            for i in range(0, len(bets_to_place)):
                print("", file = file)
                print(f"Bet: {bets_to_place.iloc[i]['Bet_trigger']}", file = file)
                print(f"Win probability: {bets_to_place.iloc[i]['bet_win_prob']:.1f}%", file = file)
                print(f"EV: {bets_to_place.iloc[i]['bet_ev']:.2f} at {bets_to_place.iloc[i]['bet_assumed_odds']:.2f} odds", file = file)
                print(f"Min exchange odds (EV >= {bet_config['min_ev_to_bet']:.2f}): {bets_to_place.iloc[i]['Bet_min_exchange_odds']:.2f}", file = file)

    return summary_data

def predict_xgboost_ensemble(models, X_test_df, y_test_df = None, model_hyper_params = {}):
    """
    Generates multi-class predictions from an ensemble of XGBoost models.

    Parameters:
    - models (list): List of trained XGBoost models.
    - X (pd.DataFrame): Feature matrix for making predictions.

    Returns:
    - predictions (np.array): Averaged probabilities for each class from the ensemble.
    - predicted_classes (np.array): Predicted class labels for each instance based on averaged probabilities.
    """
    
    # pdb.set_trace()
    
    # Collect predictions from each model
    model_predictions = []
    
    # Check if columns need to be renamed
    if any('_sub.att' in feature for feature in models[0]['features']):
        X_test_df.columns = X_test_df.columns.str.replace('_sub_attempt', '_sub.att')
    
    for model in models:
        
        # Clean the test features
        X_test_df = clean_inf_and_large(X_test_df).copy()
        
        # Get base margin
        base_margin_test = get_base_margin(X_test_df, model_hyper_params['base_margin_cols'])
        
        # Check if this is a calibration model
       
        if 'calibrated' not in model.keys():
            model['calibrated'] = False
        
        # Get model predictions depending if a calibrated model or not
        if model['calibrated'] == True:
            preds = model['model'].predict_proba(X_test_df[model['features']])
        
        else:
            # Create the DMatrix for predictions
            dmatrix_test = xgb.DMatrix(X_test_df[model['model'].feature_names], label = y_test_df, base_margin = base_margin_test)
            
            preds = model['model'].predict(dmatrix_test)
            
        model_predictions.append(preds)
        
    # pdb.set_trace()
    # Stack predictions and average them
    model_predictions = np.array(model_predictions)  # Shape: (num_models, num_samples, num_classes)
    averaged_predictions = np.mean(model_predictions, axis=0)  # Average across models
    
    # Get predicted classes by taking the argmax across the class probabilities
    predicted_classes = np.argmax(averaged_predictions, axis=1)

    return averaged_predictions, predicted_classes

# Example usage
# Assuming you have your ensemble of models trained and a test dataset
# predictions, predicted_classes = predict_ensemble(ensemble_models, X_test)

# `predictions` will contain the averaged probabilities for each class
# `predicted_classes` will contain the predicted class labels based on the highest probability

'''
This function trains and tests LightGBM models for hyperparameter. Parameters are logged in MLflow as well as metrics.


'''

def train_and_test_light_gbms_mlflow(
        X_train, 
        X_test,
        y_train,
        y_test,
        y_test_bin,
        leaves,
        learning_rates,
        rounds):

    # Set number of classes for training
    no_of_classes = len(set(y_train))

    # Format the data for the model
    train = lgb.Dataset(X_train, label = y_train)
    test_data = lgb.Dataset(X_test, label = y_test, reference = train)

    # Combine these into a data frame with all possible combinations
    parameters = pd.DataFrame(list(itertools.product(leaves, learning_rates, rounds)),
                        columns = ['leaves', 'learning_rate', 'rounds'])

    # Now loop through each set of hyperparameters and record the errors
    for i in range(0, len(parameters)):
        with mlflow.start_run():
            params = {
                'objective': 'multiclass',      # Binary classification
                'metric': 'multi_logloss', # Evaluation metric
                'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
                'num_leaves': parameters['leaves'][i],           # Controls the complexity of the model
                # 'min_data_in_leaf': 10,      # Sets a minimum sample size to prevent over-fitting
                'learning_rate': parameters['learning_rate'][i],      # Smaller learning rate for better accuracy
                'feature_fraction': 1,    # Subsampling of features to speed up training
                'bagging_fraction': 0.8,    # Subsampling of data to prevent overfitting
                'bagging_freq': 5,          # Perform bagging every 5 iterations
                'n_jobs': 3,               # Use all CPU cores
                'verbose': -1,            # Suppress output
                'num_class': no_of_classes
            }

            # Print out progress
            print(f"Training model " + str(i + 1) + " of " + str(len(parameters)) + "...")

            num_round = parameters['rounds'][i]
            lightgbm_model = lgb.train(params, train, num_round, valid_sets=[train, test_data])

                        # Make predictions on the test set
            y_pred_prob = lightgbm_model.predict(X_test, num_iteration = lightgbm_model.best_iteration)

            # Convert predicted probabilities to class predictions
            y_pred = np.argmax(y_pred_prob, axis=1)

            # Evaluate the model based on accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy:.4f}')

            # Calculate the AUC
            auc = roc_auc_score(y_test_bin, y_pred_prob, multi_class = "ovr")

            # Calculate the AUC of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L
            loss_auc = roc_auc_score(y_test_bin[:, 0], y_pred_prob[:, 0])
            win_auc = roc_auc_score(y_test_bin[:, 2], y_pred_prob[:, 2])

            # Log the parameters and outcome in MLflow
            mlflow.log_param('model', 'Light GBM')
            mlflow.log_params(params)
            mlflow.log_param('num_iterations', parameters['rounds'][i])
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('W/L AUC', win_auc)
            mlflow.log_metric('L/W AUC', loss_auc)
            mlflow.log_metric('Avg AUC', auc)

            del([lightgbm_model, y_pred_prob, y_pred, accuracy])

    return "Light GBM model runs completed."

'''
This function trains and tests LightGBM models for hyperparameter. Parameters are logged in MLflow as well as metrics.


'''
def train_and_test_random_forests_mlflow(
        X_train, 
        X_test,
        y_train,
        y_test,
        y_test_bin,
        leaves,
        learning_rates,
        rounds):

    # Set number of classes for training
    no_of_classes = len(set(y_train))

    # Format the data for the model
    train = lgb.Dataset(X_train, label = y_train)
    test_data = lgb.Dataset(X_test, label = y_test, reference = train)

    # Combine these into a data frame with all possible combinations
    parameters = pd.DataFrame(list(itertools.product(leaves, learning_rates, rounds)),
                        columns = ['leaves', 'learning_rate', 'rounds'])

    # Now loop through each set of hyperparameters and record the errors
    for i in range(0, len(parameters)):
        with mlflow.start_run():
            params = {
                'objective': 'multiclass',      # Binary classification
                'metric': 'multi_logloss', # Evaluation metric
                'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
                'num_leaves': parameters['leaves'][i],           # Controls the complexity of the model
                # 'min_data_in_leaf': 10,      # Sets a minimum sample size to prevent over-fitting
                'learning_rate': parameters['learning_rate'][i],      # Smaller learning rate for better accuracy
                'feature_fraction': 1,    # Subsampling of features to speed up training
                'bagging_fraction': 0.8,    # Subsampling of data to prevent overfitting
                'bagging_freq': 5,          # Perform bagging every 5 iterations
                'n_jobs': 3,               # Use all CPU cores
                'verbose': -1,            # Suppress output
                'num_class': no_of_classes
            }

            # Print out progress
            print(f"Training model " + str(i + 1) + " of " + str(len(parameters)) + "...")

            num_round = parameters['rounds'][i]
            lightgbm_model = lgb.train(params, train, num_round, valid_sets=[train, test_data])

                        # Make predictions on the test set
            y_pred_prob = lightgbm_model.predict(X_test, num_iteration = lightgbm_model.best_iteration)

            # Convert predicted probabilities to class predictions
            y_pred = np.argmax(y_pred_prob, axis=1)

            # Evaluate the model based on accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy:.4f}')

            # Calculate the AUC
            auc = roc_auc_score(y_test_bin, y_pred_prob, multi_class = "ovr")

            # Calculate the AUC of wins and losses. Check the classes of the target column. Was 2 = L/W and 4 = W/L
            loss_auc = roc_auc_score(y_test_bin[:, 1], y_pred_prob[:, 1])
            win_auc = roc_auc_score(y_test_bin[:, 3], y_pred_prob[:, 3])

            # Log the parameters and outcome in MLflow
            mlflow.log_param('model', 'Light GBM')
            mlflow.log_param('num_leaves', parameters['leaves'][i])
            mlflow.log_param('num_iterations', parameters['rounds'][i])
            mlflow.log_param('learning_rate', parameters['learning_rate'][i])
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('W/L AUC', win_auc)
            mlflow.log_metric('L/W AUC', loss_auc)
            mlflow.log_metric('Avg AUC', auc)

            del([lightgbm_model, y_pred_prob, y_pred, accuracy])

    return "Light GBM model runs completed."

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
        if params['calibrate_models']: # Calibration doesn't work with EV loss function at the moment. Not built.
            
            # Add a dummy 1 value in case it's missing (that causes the fit to fail)
            X_dummy = np.zeros((1, X_train_df.shape[1]))  # or any constant
            y_dummy = np.array([1])  # the missing class
            
            # Convert to Data Frames
            X_dummy = pd.DataFrame(X_dummy, columns = X_train_df.columns)
            
            X_train_aug = pd.concat([X_train_df, X_dummy], axis = 0)
            y_train_aug = pd.concat([y_train_df, pd.Series(y_dummy)], axis = 0)
            
            # Train the model in a way compatible with SKLearn
            raw_model = xgb.XGBClassifier(params)
            raw_model.fit(X_train_aug, y_train_aug,
                          eval_set = [(X_val[X_train_df.columns], y_val)],
                          early_stopping_rounds = early_stopping_rounds,
                          verbose = verbose_eval)                
            
            # Create a calibration model
            model_cal = TemperatureScaler(raw_model)
            # model_cal = CalibratedClassifierCV(estimator = raw_model, 
            #                                    method = 'sigmoid', # This is a Platt calibration
            #                                    cv = 'prefit') # the model has already been trained
            
            # Now fit to the validation data
            model_cal.fit(X_val[X_train_df.columns], y_val)
            
            model['xgb_model'] = raw_model
            model['model'] = model_cal
        
        elif model_hyper_params['loss_function'] == 'ev_loss_function': # any eval metric above will be ignored
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
        model['calibrated'] = params['calibrate_models']
        
        # Add model to the list
        models.append(model)
        
        del X_train_df
        gc.collect()
        print(f"Model {i + 1}/{num_models} trained.")

    return models
