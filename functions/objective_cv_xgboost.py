# This function trains XGboost models using k-fold CV and summarises output metrics. Parameters are NOT logged in MLflow.

def objective_cv_xgboost(
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

    # Loop through precomputed DMatrix objects for each fold
    for train_dmatrix, test_dmatrix in dmatrix_data:
        pdb.set_trace()
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

        # Clean up memory
        del model, train_dmatrix, test_dmatrix, y_pred_prob, wl_auc, lw_auc, wl_accuracy, lw_accuracy
        gc.collect()
         
    # Save the metrics
    metrics = {
        'Avg WL only AUC': np.mean(avg_aucs),
        'Hmean WL only Accuracy': np.mean(avg_accuracies),
        'Accuracy': np.mean(accuracies)
    }
        
    del kf

    # Return whatever metric we wish to optimise for the LightGBM. Do 1 - because it tries to minimise the value.
    return {'loss': 1 - np.mean(avg_aucs), 'params': params, 'status': STATUS_OK, 'metrics': metrics} 
