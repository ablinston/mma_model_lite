# This function is used to train decision trees on the elo and assess how good it is at
# predicting the fight outcomes on it's own

# It relies on the following objects being available in memory:
    # - start_elo, merged_data: these are the standard outputs of the prep_stats_for_elo_calc() function
    # elo_n_folds to set the number of folds
    

def objective_cv_elo_dtree_mlflow(params):

    # pdb.set_trace()
    print("Params received:", params)    
    
    # Check for duplicates
    if is_duplicate_run(params):
        
        # If the run existing, find what the loss was and return it without logging in mlflow
        existing_metric = get_existing_metric(params, 'Avg of two accuracies')
        
        print(f"Duplicate run found, skipping logging. Accuracy: {existing_metric:.4f}")
        return {'loss': 1 - existing_metric, 'params': params, 'status': STATUS_OK}
        
    else:
    
        try:
        
            # Dderive the ELO for the selected constants
            elo_data, fight_data = derive_start_elos(start_elo, merged_data, params)
        
            # Create the training dataset. This only contains the outcome and ELO variables. Also filters for the training start date
            train = fight_data.to_pandas()
            train = train[train['date'] >= config['min_training_date']][['outcome', 'fighter1_start_elo', 'fighter2_start_elo']]
            train['elo_diff'] = train['fighter1_start_elo'] - train['fighter2_start_elo']
                
            # Create n folds of the training data
            kf = KFold(n_splits = elo_n_folds, shuffle = True, random_state = config['seed_all'])
        
            # Create lists that will be used to store models and test datasets
            wl_accuracies = []
            lw_accuracies = []
            accuracies = []
        
            # Print out progress
            print("Training CV models...")
            
            # Convert the target into a labelled column ready for multiclass modelling
            le = LabelEncoder()
            train['outcome'] = le.fit_transform(train['outcome'])
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                train.drop(columns = ['outcome']),
                train['outcome'],
                test_size = 0.1,
                random_state = config['seed_all']
            )
            
            # Precompute DMatrix objects using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers = config['num_threads']) as executor:
                futures = {executor.submit(create_dmatrix, X_train, y_train, train_idx, test_idx): (train_idx, test_idx) for train_idx, test_idx in kf.split(X_train)}
                dmatrix_data = [future.result() for future in as_completed(futures)]
            
            # Begin new model in MLflow
            with mlflow.start_run():
        
                mlflow.set_tag("Description", "Cross-validated Decision Tree")
        
                # Log the training parameters
                mlflow.log_param('model', 'Decision Tree')
                mlflow.log_params(params)
                mlflow.log_param('Fighter 1 Max ELO', train['fighter1_start_elo'].max())
                mlflow.log_param('Fighter 1 Min ELO', train['fighter1_start_elo'].min())
                
                # Loop through precomputed DMatrix objects for each fold
                for train_dmatrix, test_dmatrix in dmatrix_data:
        
                    # # For debugging
                    # train_dmatrix = xgb.DMatrix(X_train, label = y_train)
                    # test_dmatrix = xgb.DMatrix(X_test, label = y_test)
                    
                    # Train the model with early stopping
                    model = xgb.train(params, train_dmatrix,
                                      num_boost_round = params['num_iterations'],
                                      evals = [(test_dmatrix, 'eval')],
                                      early_stopping_rounds = params['early_stopping_rounds'],
                                      verbose_eval = True)
                        
                    # Make predictions on the test set
                    y_pred_prob = model.predict(test_dmatrix)
                    
                    # Get the outcomes of the test data
                    y_test_fold = test_dmatrix.get_label().astype(int)
        
                    # Convert actual results to binarized format
                    y_test_bin = label_binarize(y_test_fold, classes=np.unique(y_test_fold))
                    y_pred_bin = label_binarize(y_pred_prob, classes=np.unique(y_test_fold))
        
                    # Calculate the accuracy of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L (should be alphabetical)
                    wl_accuracy = (y_test_bin[:, 2] == y_pred_bin[:, 2]).mean()
                    lw_accuracy = (y_test_bin[:, 0] == y_pred_bin[:, 0]).mean()
                    lw_accuracies.append(lw_accuracy)
                    wl_accuracies.append(wl_accuracy)
                    
                    # Calculate the AUC as the average of the W/L and L/W
                    accuracies.append(np.mean([lw_accuracy, wl_accuracy]))
        
                    # Clean up memory
                    del model, train_dmatrix, test_dmatrix, y_pred_prob, wl_accuracy, lw_accuracy
                    gc.collect()
                    
                # Now all folds have been calculated, summarise the metrics
        
                # Log the outcome in MLflow
                mlflow.log_metric('W/L accuracy', np.mean(wl_accuracies))
                mlflow.log_metric('L/W accuracy', np.mean(lw_accuracies))
                mlflow.log_metric('Avg of two accuracies', np.mean(accuracies))
         
            # Return whatever metric we wish to optimise for the model
            return {'loss': 1 - np.mean(accuracies), 'params': params, 'status': STATUS_OK} 
        
        except:
            print("Error encountered. Model fit abandoned")
            return {'loss': float('inf'), 'params': params, 'status': STATUS_FAIL, 'error': str(e)}
