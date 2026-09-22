def backtest_model(
        actuals_data, 
        prediction_probabilities, 
        breakdown_columns, 
        min_prob_to_bet, 
        min_ev_to_bet, 
        min_bet_size = 1, 
        max_bet_size = 1, 
        constant_bets = True, 
        start_balance = 1000,
        point_in_time_only_odds = True,
        target_ev_after_cutoff = 0,
        output_data_for_debug = True, ):
    """
    
    A function to backtest a model's predictions against actuals data that contain the odds'.
    
    This function assume constant betting amounts rather than fractions of a pot. This means it's easier to track the consistency over time and not be affected by early wins and losses affecting the overall metrics from a test'

    Parameters
    ----------
    actuals_data : data frame
        DESCRIPTION.
    prediction_probabilities : array
        DESCRIPTION.
    breakdown_cols : TYPE
        DESCRIPTION.
    min_prob_to_bet : float64
        the minimum probability of a fighter winning required to place the bet
    min_ev_to_bet : float64
        the minimum expected return required to place the bet (EV of 0 means no profit expected), and it is displayed in terms of betting units units
    min_bet_size : float64
        the minimum bet size in units. This should be left as 1 and only max varied unless wanting to model £ bets
    max_bet_size : float64
        the maximum bet size. Set this to 1 to give uniform betting regardless of the probability
    point_in_time_only_odds : bool
        A True or False flag to say whether we only use the backtesting odds to make bets. If False, it will also use the best Betfair odds after cutoff and make bets based on the min EV
    target_ev_after_cutoff: float64
        The minimum EV that the test will target for odds after the cutoff time only

    Returns
    -------
    None.

    """
    
    # ###
    # # Debug
    # actuals_data = X_train[['id']].merge(data, on = 'id', how = 'left')
    # prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_train, y_train)
    # breakdown_cols = ['weightclass', 'year']
    # min_prob_to_bet = 0.5
    # min_ev_to_bet = 0.1
    # ###
    
    # pdb.set_trace()
    
    breakdown_cols = ['bet_ev_band', 'Predict_to_win', 'Fighter_to_win'] + breakdown_columns.copy()
    
    # Use the new function to summarise the model predictions and bet triggers
    actuals_data, wl_trigger, lw_trigger = summarise_model_predictions(actuals_data, prediction_probabilities, min_prob_to_bet, min_ev_to_bet, min_bet_size, max_bet_size, return_triggers = True, point_in_time_only_odds = point_in_time_only_odds, target_ev_after_cutoff = target_ev_after_cutoff)
    
    # Record the odds that are to be used in backtesting. These may be edited next
    fighter1_backtesting_odds = actuals_data['fighter1_backtesting_odds'].copy()
    fighter2_backtesting_odds = actuals_data['fighter2_backtesting_odds'].copy()
    
    # Find any bets that occur after the cutoff if necessary by using target odds
    if not point_in_time_only_odds:
        
        breakdown_cols = ['bet_triggered_after_cutoff'] + breakdown_cols
        
        # Check where the odds meet the requirement and a bet isn't triggered at the cutoff point
        additional_wl_trigger = np.where((wl_trigger == 0) & (
            actuals_data['fighter1_notformodel_best_betfair_odds_after_cutoff'] >=
            actuals_data['fighter1_target_exchange_odds']) & (
            actuals_data['Fighter1_win_prob'] >= min_prob_to_bet),
                1, 0)
                
        additional_lw_trigger = np.where((lw_trigger == 0) & (
            actuals_data['fighter2_notformodel_best_betfair_odds_after_cutoff'] >=
            actuals_data['fighter2_target_exchange_odds']) & (
            actuals_data['Fighter2_win_prob'] >= min_prob_to_bet),
                1, 0)
        
        # Deal with double triggers
        all_triggers = additional_wl_trigger + additional_lw_trigger
        additional_wl_trigger = np.where(all_triggers == 2, 0, additional_wl_trigger)
        additional_lw_trigger = np.where(all_triggers == 2, 0, additional_lw_trigger)
                
        # Now add these triggers and the odds to the existing vectors
        fighter1_backtesting_odds = np.where(additional_wl_trigger == 1, actuals_data['fighter1_target_exchange_odds'], fighter1_backtesting_odds)
        
        fighter2_backtesting_odds = np.where(additional_lw_trigger == 1, actuals_data['fighter2_target_exchange_odds'], fighter2_backtesting_odds)
        
        wl_trigger = np.where(additional_wl_trigger == 1, 1, wl_trigger)
        lw_trigger = np.where(additional_lw_trigger == 1, 1, lw_trigger)
        
        actuals_data['bet_triggered_after_cutoff'] = np.where(all_triggers == 1, 'Bet triggered after cutoff', 'Usual bet at cutoff')
        
        # Correct the bet count and amount
        actuals_data['bets'] = np.where(additional_wl_trigger == 1, 1, actuals_data['bets'])
        actuals_data['bet_size'] = np.where(additional_wl_trigger == 1, actuals_data['fighter1_target_bet_size'],
                                            np.where(additional_lw_trigger == 1, actuals_data['fighter2_target_bet_size'], actuals_data['bet_size']))
        
    else:
        actuals_data['bet_triggered_after_cutoff'] = 0
        
    # Do a check that the inputs are consistent if doing a non constant bets. If using non constant bets, the bet sizing should be as proportions, not £ amounts
    if not constant_bets:
        stop_if_not((min_bet_size < 1) and (max_bet_size < 1), "Bet size for non-constant bets check")
        
    # Now calculate the returns if a bet was made
    actuals_data['profit_loss'] = ((fighter1_backtesting_odds - 1) * wl_trigger * (actuals_data['outcome'] == 'W/L') - wl_trigger * (actuals_data['outcome'] != 'W/L') + (fighter2_backtesting_odds - 1) * lw_trigger * (actuals_data['outcome'] == 'L/W') - lw_trigger * (actuals_data['outcome'] != 'L/W')) * actuals_data['bet_size']
    
    if constant_bets:
        pass
    else:
        # If bets are not constant, we need to work out the cumulative returns
        
        # Save the profit and loss
        actuals_data['profit_loss_proportion'] = actuals_data['profit_loss'].copy()
        
        # First add 1 to get the proportion of the balance still left
        actuals_data['profit_loss_multiplier'] = 1 + actuals_data['profit_loss_proportion']
    
        # Then convert to a cumulative product after sorting by date
        actuals_data.sort_values('date', inplace = True)
        
        # Create balances
        actuals_data['end_balance'] = start_balance * actuals_data['profit_loss_multiplier'].cumprod()
        actuals_data['start_balance'] = actuals_data['end_balance'].shift(1).fillna(start_balance)
        
        # Then calculate profit and loss
        actuals_data['profit_loss'] = actuals_data['end_balance'] - actuals_data['start_balance']
        
    # Calculate some further columns that will be used for stats
    actuals_data['total'] = 'Total'
    actuals_data['fights'] = 1
    actuals_data['wins'] = 1 * (actuals_data['profit_loss'] > 0)
    actuals_data['max_bet_size'] = actuals_data['bet_size']
    
    # If not doing constant bets, this script isn't able to break down results by anything as it stands, so reset the breakdown cols
    if not constant_bets:
        breakdown_cols = []
    
    ## Summarise the results
    results = actuals_data.groupby('total', observed = False).agg({'fights': lambda x: len(x), 'bets': 'sum', 'wins': 'sum', 'bet_size': 'sum', 'profit_loss': 'sum', 'max_bet_size': 'max'})
    for col in breakdown_cols:
        results = pd.concat([
            results,
            actuals_data.groupby(col, observed = False).agg({'fights': lambda x: len(x), 'bets': 'sum', 'wins': 'sum', 'bet_size': 'sum', 'profit_loss': 'sum', 'max_bet_size': 'max'})
        ])
    
    # Calculate percentage info
    results['percent_bet_on'] = results['bets'] / results['fights'] * 100
    results['win_percent'] = results['wins'] / results['bets'] * 100
    
    results['max_drawdown'] = np.nan
    # For the total, calculate the longest streaks
    results['max_losing_streak'] = np.nan
    if constant_bets:
        results['roi'] = results['profit_loss'] / results['bet_size'] * 100 # the profit and loss assumes every bet equal size
        results.loc['Total', 'max_drawdown'] = minimum_sum_series(actuals_data['profit_loss'])
        results.loc['Total', 'max_losing_streak'] = longest_streak(actuals_data['profit_loss'], -1)
    else:
        # Work out the ROI per fight, otherwise going to be very large
        results['roi'] = ((results['profit_loss'] / start_balance + 1) ** (1 / results['bets']) - 1) * 100 # the profit and loss assumes every bet equal size
        results.loc['Total', 'max_drawdown'] = minimum_sum_series(actuals_data['profit_loss_proportion'])

    # # Add the row names as a column
    results = pd.concat([pd.DataFrame({'Group': results.index}, index = results.index), results], axis=1)
    # results.rename(columns={'index': 'RowName'}, inplace=True)
    
    # Get fight level output
    actuals_data.sort_values('date', inplace = True)
    
    if 'y_pred_prob' in actuals_data.columns:
        fight_output = actuals_data[['date', 'fighter1', 'Fighter1_win_prob', 'Fighter1_win_EV', 'fighter1_backtesting_odds', 'fighter2', 'Fighter2_win_prob', 'Fighter2_win_EV', 'fighter2_backtesting_odds', 'outcome', 'target', 'predictions', 'bets', 'bet_ev', 'profit_loss', 'bet_triggered_after_cutoff' ,'kelly_criterion', 'bet_size', 'max_bet_size', 'total', 'fights', 'wins', 'y_pred_prob']]
    else:
        fight_output = None
    
    # Output the underlying data if required
    if output_data_for_debug:
        fight_output.to_csv('results/backtesting_data_for_checking.csv')
    
    # Mpve column positions for better searching
    results = move_column(results, 'profit_loss', 1)
    results = move_column(results, 'roi', 2)
    
    return results, fight_output

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
            
        if 'Hmean WL only Accuracy' in breakdown_metrics and 'X Odds Hmean WL only Accuracy' in breakdown_metrics:
            fig, ax = metric_chart(x = summary_df['group'], y_model = summary_df['Hmean WL only Accuracy'], y_odds = summary_df['X Odds Hmean WL only Accuracy'], x_label = 'Group', y_label = 'Hmean WL only Accuracy')
            mlflow.log_figure(fig, col +  " metrics/accuracy_chart.png")
            
        # pdb.set_trace()
    
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
