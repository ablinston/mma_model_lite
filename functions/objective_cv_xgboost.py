# This function trains XGboost models using k-fold CV and summarises output metrics. Parameters are NOT logged in MLflow.

def objective_cv_xgboost(
        params):

    # pdb.set_trace()
    
    try:
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
            futures = {executor.submit(create_dmatrix, X_train, y_train, train_idx, test_idx, params['win_loss_only'], params['exclude_split_other_from_train']): (train_idx, test_idx) for train_idx, test_idx in kf.split(X_train_trim)}
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
            
        del kf
    
        if model_hyper_params['loss_function'] == 'ev_loss_function':
            return {'loss': np.mean(ev_mses), 'params': params, 'status': STATUS_OK, 'metrics': metrics} 
        else:
            # Return whatever metric we wish to optimise for the LightGBM. Do 1 - because it tries to minimise the value.
            return {'loss': np.mean(log_losses), 'params': params, 'status': STATUS_OK, 'metrics': metrics} 
        
    except Exception as e:
        print(f"Skipping trial due to error: {e}")
        return {'loss': float('inf'), 'status': hyperopt.STATUS_FAIL}
        
        