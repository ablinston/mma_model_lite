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
