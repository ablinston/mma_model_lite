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
