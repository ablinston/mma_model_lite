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
