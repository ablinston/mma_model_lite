# This function takes a model and a test dataset and computes test statistics then logs them in mlflow.

def btest_model_and_log_mlflow(actuals_data, prediction_probabilities, breakdown_cols, breakdown_metrics, mlflow_description, target_column = 'outcome', backtesting_params = {'min_prob_to_bet': 0.5, 'min_ev_to_bet': 0.05, 'max_bet_size': 1}):

    """
    Tests a model by taking in some actuals data, a set of prediction probabilities, and any breakdown variables to use. It logs various metrics in MLflow as well as outputing some statistics

    Args:
        actuals_data (data frame): this is the data that the predictions are for. It should not contain dummy versions of columns to be uesd in breakdown cols
        prediction_probabilities (array): the probabilities for each class of the outcome
        breakdown_cols (list): a list of column names to create metric breakdowns for
        breakdown_metrics (list): the metrics to output for the breakdown. These names must align with metrics calculated in the calculate_metrics() function
        mlflow_description: a description to give for the run in MLflow

    Returns:
        None, but prints the breakdown outputs and logs everything in MLflow
    
    """
    
    # pdb.set_trace()
    
    ####
    # # Debug testing
    # actuals_data = data[data['id'].isin(X_train['id'].tolist())].copy()
    # X_data = X_train
    # y_data = y_train
    # prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_train, y_train)
    # breakdown_cols = ['weightclass']
    # breakdown_metrics  = ['Avg WL only AUC', 'X Odds Avg WL only AUC', 'Hmean WL only Accuracy', 'X Odds Hmean WL only Accuracy']
    # mlflow_description = 'test'
    # target_column = 'outcome'
    ####

    # Get the class predictions from the probabilities
    actuals_data['predictions'] = np.argmax(prediction_probabilities, axis = 1)
    actuals_data['y_pred_prob'] = list(prediction_probabilities)
    
    # Categorise the outcome variable to align with prediction outputs
    actuals_data['target'] = clean_target_column(actuals_data[target_column])
    
    # Create the MLflow description
    mlflow.set_tag("mlflow.note.content", mlflow_description)

    # Log the overall metrics in MLflow
    log_metrics_mlflow(prediction_probabilities, actuals_data['target'], actuals_data['fighter1_backtesting_odds'], actuals_data['fighter2_backtesting_odds'], classes = np.array([0, 1, 2]))
    
    # Log the overall odds metrics in MLflow
    log_metrics_odds_mlflow(actuals_data, actuals_data['target'])
    
    # # Log the parameters of the model
    # mlflow.log_params(model['params'])
    
    ## Now begin calculating the metrics for the various breakdowns
    
    # Initiate the results dictionary
    metrics_by_group = {}
    
    # Loop through the columns to be analysed
    for col in breakdown_cols:
        
        print('Calculating metrics for ' + col)
        
        # Initial the new dictionary for this column
        metrics_by_group.update({col: {}})
        # pdb.set_trace()
        # Convert the column to a string to ensure consistent outputs
        actuals_data[col] = actuals_data[col].apply(str)
        
        # Now loop through all the unique values within the column
        for group in np.unique(actuals_data[col]).tolist():
            
            # print(group)
            
            # Get the rows we're interested in
            row_filter = actuals_data[col] == group
            # print(row_filter.sum())
            
            # Use try except in case sample sizes too small
            try:
                # Calculate the metrics
                metrics = calculate_metrics(prediction_probabilities[row_filter], actuals_data['target'][row_filter], actuals_data['fighter1_backtesting_odds'][row_filter], actuals_data['fighter1_backtesting_odds'][row_filter])
                
                # Add odds metrics
                metrics.update(calculate_metrics_odds(actuals_data[row_filter], actuals_data['target'][row_filter]))
                
                # Add these metrics to the results dictionary
                metrics_by_group[col].update({group: metrics})
            except:
                print(f"Metrics for {col}: {group} couldn't be calculated.")
                continue
        # pdb.set_trace()
        # Get the outputs for each column and log them
        summary_df = metric_table_from_metrics_dictionary(metrics_by_group, col)[['group'] + breakdown_metrics]
        
        # Sort it
        summary_df.sort_values('group', inplace = True)
        
        # Calculate any differences with odds if available
        for sub_col in summary_df.columns:
            if 'X Odds ' + sub_col in summary_df.columns:
                summary_df['Diff ' + sub_col] = summary_df[sub_col] - summary_df['X Odds ' + sub_col]
        
        print(summary_df)
        
        # Save a temporary csv and log it as an artifact
        # Name the index
        summary_df.index.name = 'Group'
        summary_df.to_csv('results/temp/summary.csv')
        mlflow.log_artifact('results/temp/summary.csv', col + ' metrics')
        
        # Create chart of logloss and EV MSE
        if 'Logloss' in breakdown_metrics and 'X Odds Logloss' in breakdown_metrics:
            fig, ax = metric_chart(x = summary_df['group'], y_model = summary_df['Logloss'], y_odds = summary_df['X Odds Logloss'], x_label = 'Group', y_label = 'Logloss')
            mlflow.log_figure(fig, col +  " metrics/logloss_chart.png")
        
        if 'EV MSE' in breakdown_metrics and 'X Odds EV MSE' in breakdown_metrics:
            fig, ax = metric_chart(x = summary_df['group'], y_model = summary_df['EV MSE'], y_odds = summary_df['X Odds EV MSE'], x_label = 'Group', y_label = 'EV MSE')
            mlflow.log_figure(fig, col +  " metrics/ev_mse_chart.png")

    
    # Finally, get the backtesting results
    backtesting, backtesting_fight_output = backtest_model(actuals_data, prediction_probabilities, breakdown_cols, **backtesting_params)
   
    # Summarise the ROI for EV bands
    roi_by_ev = summarise_bet_roi_by_ev(backtesting_fight_output)
    
    # Create chart and log in MLflow
    fig = plot_roi_and_sample_size(roi_by_ev)
    mlflow.log_figure(fig, "Backtesting results/ROI_by_EV.png")
    
    # Plot and log the calibration curve
    fig = plot_calibration_curve(
        np.where(actuals_data['outcome'] == 'W/L', 1, 0),
        prediction_probabilities[:, 2] # Associated probabilities
    )
    mlflow.log_figure(fig, "Backtesting results/F1_Win_Calibration_Curve.png")
    
    # Plot and log the calibration curve
    fig = plot_calibration_curve(
        np.where(actuals_data['outcome'] == 'L/W', 1, 0),
        prediction_probabilities[:, 0] # Associated probabilities
    )
    mlflow.log_figure(fig, "Backtesting results/F2_Win_Calibration_Curve.png")
    
    # Log the overall backtesting results
    backtesting.to_csv('results/temp/summary.csv', index = True)
    mlflow.log_artifact('results/temp/summary.csv', 'Backtesting results')
    backtesting_fight_output.to_csv('results/temp/fight_backtesting.csv', index = True)
    mlflow.log_artifact('results/temp/fight_backtesting.csv', 'Backtesting results')
 
    # Chart profit and loss
    cumulative_profit_loss = backtesting_fight_output[backtesting_fight_output['bets'] == 1]['profit_loss'].cumsum()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pd.to_datetime(backtesting_fight_output[backtesting_fight_output['bets'] == 1]['date']).dt.date, cumulative_profit_loss)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit")
    plt.xticks(rotation=90)
    mlflow.log_figure(fig, "Backtesting results/Running_Cumulative_Profit_Loss.png")   
 
    # Log the total metrics in MLflow
    mlflow.log_params(backtesting_params)
    
    mlflow.log_metric('Backtesting_P_and_L', backtesting.loc['Total', 'profit_loss'])
    mlflow.log_metric('Backtesting_Win_Percent', backtesting.loc['Total', 'win_percent'])
    mlflow.log_metric('Backtesting_Max_Drawdown', backtesting.loc['Total', 'max_drawdown'])
    mlflow.log_metric('Backtesting_ROI', backtesting.loc['Total', 'roi'])
    mlflow.log_metric('Backtesting_Bets', backtesting.loc['Total', 'bets'])
    mlflow.log_metric('Backtesting_Percent_Bet_On', backtesting.loc['Total', 'percent_bet_on'])
    mlflow.log_metric('Backtesting_Normalised_Total_Percent_Return', 100 * ((1 + backtesting.loc['Total', 'roi'] / 100) ** backtesting.loc['Total', 'bets'] - 1))
    mlflow.log_metric('Backtesting_Average_Bet_Size', backtesting.loc['Total', 'bet_size'] / backtesting.loc['Total', 'bets'])
    mlflow.log_metric('Backtesting_Actual_Max_Bet_Size', backtesting.loc['Total', 'max_bet_size'])
    
    return backtesting
