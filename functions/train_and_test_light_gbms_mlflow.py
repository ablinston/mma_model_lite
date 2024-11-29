
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
