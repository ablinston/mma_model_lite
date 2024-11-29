# This function is used to predict fight outcomes based only on the elo and assess how good it is at
# predicting the fight outcomes on it's own. It takes the highest ELO as the predicted winner.

# It relies on the following objects being available in memory:
    # - start_elo, merged_data: these are the standard outputs of the prep_stats_for_elo_calc() function
  

def objective_cv_elo_basic_mlflow(params):

    # pdb.set_trace()
    print("Params received:", params)    
    
    # Check for duplicates
    if is_duplicate_run(params):
        
        # If the run existing, find what the loss was and return it without logging in mlflow
        existing_metric = get_existing_metric(params, 'H.Mean of two accuracies')
        
        print(f"Duplicate run found, skipping logging. Accuracy: {existing_metric:.4f}")
        return {'loss': 1 - existing_metric, 'params': params, 'status': STATUS_OK}
        
    else:
    
        try:
        
            # Begin new model in MLflow
            with mlflow.start_run():
                
                # Derive the ELO for the selected constants
                elo_data, fight_data = derive_start_elos(start_elo, merged_data, params)
            
                # Create the training dataset. This only contains the outcome and ELO variables. Also filters for the training start date
                train = fight_data.to_pandas()
                train = train[train['date'] >= config['min_training_date']][['outcome', 'fighter1_start_elo', 'fighter2_start_elo']]
        
                mlflow.set_tag("Description", "Basic prediction, highest ELO wins")
        
                # Log the training parameters
                mlflow.log_param('model', 'Basic')
                mlflow.log_params(params)
                mlflow.log_param('Fighter 1 Max ELO', train['fighter1_start_elo'].max())
                mlflow.log_param('Fighter 1 Min ELO', train['fighter1_start_elo'].min())
                        
                # Make predictions on the test set
                y_pred = np.where(train['fighter1_start_elo'] >= train['fighter2_start_elo'], 'W/L', 'L/W')
                correct_binary = y_pred == train['outcome']
            
                # Now calculate the accuracies
                wl_accuracy = np.mean(correct_binary[y_pred == 'W/L'])
                lw_accuracy = np.mean(correct_binary[y_pred == 'L/W'])
        
                # Log the outcome in MLflow
                mlflow.log_metric('W/L accuracy', wl_accuracy)
                mlflow.log_metric('L/W accuracy', lw_accuracy)
                mlflow.log_metric('H.Mean of two accuracies', hmean([wl_accuracy, lw_accuracy]))
                mlflow.log_metric('Overall accuracy', np.mean(correct_binary))
         
            # Return whatever metric we wish to optimise for the model
            return {'loss': 1 - np.mean(correct_binary), 'params': params, 'status': STATUS_OK} 
        
        except:
            print("Error encountered. Model fit abandoned")
            return {'loss': float('inf'), 'params': params, 'status': STATUS_FAIL, 'error': str(e)}