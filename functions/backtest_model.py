def backtest_model(actuals_data, prediction_probabilities, breakdown_columns, min_prob_to_bet, min_ev_to_bet, min_bet_size = 1, max_bet_size = 1, constant_bets = True, start_balance = 1000, output_data_for_debug = True):
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
    
    breakdown_cols = ['bet_ev_band'] + breakdown_columns.copy()
    
    # Use the new function to summarise the model predictions and bet triggers
    actuals_data, wl_trigger, lw_trigger = summarise_model_preductions(actuals_data, prediction_probabilities, min_prob_to_bet, min_ev_to_bet, min_bet_size, max_bet_size, return_triggers = True)
    
    # Do a check that the inputs are consistent if doing a non constant bets. If using non constant bets, the bet sizing should be as proportions, not £ amounts
    if not constant_bets:
        stop_if_not((min_bet_size < 1) and (max_bet_size < 1), "Bet size for non-constant bets check")
    
    # # First calculate the Expected Values on a W/L result and L/W result, given in betting units
    # # Odds are decimal odds, so represent the payout for a 1 unit bet e.g. odds of 2 means a (2 - 1) = 1 profit.
    # ev_wl = (actuals_data['fighter1_best_closing_odds'] - 1) * prediction_probabilities[:, 2] - (1 - prediction_probabilities[:, 2])
    # ev_lw = (actuals_data['fighter2_best_closing_odds'] - 1) * prediction_probabilities[:, 0] - (1 - prediction_probabilities[:, 0])
    
    # # Now calculate bet triggers as 0 or 1 indicators
    # ev_wl_trigger = ev_wl > min_ev_to_bet
    # prob_wl_trigger = prediction_probabilities[:, 2] > min_prob_to_bet
    # wl_trigger = 1 * ev_wl_trigger * prob_wl_trigger
    
    # ev_lw_trigger = ev_lw > min_ev_to_bet
    # prob_lw_trigger = prediction_probabilities[:, 0] > min_prob_to_bet
    # lw_trigger = 1 * ev_lw_trigger * prob_lw_trigger
    
    # # Fix any instances where both triggers happen. Remove the bet because there must be an odds mistake
    # double_trigger = (lw_trigger + wl_trigger) > 1
    # lw_trigger[double_trigger] = 0
    # wl_trigger[double_trigger] = 0
    
    # # Set whether a bet is made
    # actuals_data['bets'] = wl_trigger + lw_trigger
    
    # # Add the EV for later breakdowns
    # actuals_data['bet_ev'] = np.where(actuals_data['bets'] == 0, np.nan, wl_trigger * ev_wl + lw_trigger * ev_lw)
    # #  Convert it into buckets
    # bucket_width = 0.1
    # col_min = round(np.floor(actuals_data['bet_ev'].min() / bucket_width) * bucket_width, 2)
    # col_max = round(np.ceil(actuals_data['bet_ev'].max() / bucket_width) * bucket_width, 2)
    # actuals_data['bet_ev_band'] = pd.cut(actuals_data['bet_ev'],
    #                                 bins=np.arange(col_min, col_max + bucket_width, bucket_width))
    
    
    # ## Determine the bet size. Begin with the minimum, then add on the confidence of the bet
    
    # # First get the probabilities of winning and odds, irrespective of fighter to win
    # win_prob = wl_trigger * prediction_probabilities[:, 2] + lw_trigger * prediction_probabilities[:, 0]
    # win_odds = wl_trigger * actuals_data['fighter1_best_closing_odds'] + lw_trigger * actuals_data['fighter2_best_closing_odds']
    
    # # Now calculate the kelly criterion
    # actuals_data['kelly_criterion'] = win_prob - (1 - win_prob) / (win_odds - 1)
    
    # # Perform a check that kelly is correct
    # stop_if_not((actuals_data['kelly_criterion'].max() <= 1) and (actuals_data['kelly_criterion'].min() >= 0),
    #             'Kelly criterion check')
    
    # # And use it to calculate how much to increase from the min
    # actuals_data['bet_size'] = actuals_data['bets'] * (min_bet_size + (max_bet_size - min_bet_size) * actuals_data['kelly_criterion'])

    # # Check no bets breach the max
    # stop_if_not(actuals_data['bet_size'].max() <= max_bet_size, 'Max bet check')
    
    # Now calculate the returns if a bet was made
    actuals_data['profit_loss'] = ((actuals_data['fighter1_best_closing_odds'] - 1) * wl_trigger * (actuals_data['outcome'] == 'W/L') - wl_trigger * (actuals_data['outcome'] != 'W/L') + (actuals_data['fighter2_best_closing_odds'] - 1) * lw_trigger * (actuals_data['outcome'] == 'L/W') - lw_trigger * (actuals_data['outcome'] != 'L/W')) * actuals_data['bet_size']
    
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
    results = actuals_data.groupby('total').agg({'fights': lambda x: len(x), 'bets': 'sum', 'wins': 'sum', 'bet_size': 'sum', 'profit_loss': 'sum', 'max_bet_size': 'max'})
    for col in breakdown_cols:
        results = pd.concat([
            results,
            actuals_data.groupby(col).agg({'fights': lambda x: len(x), 'bets': 'sum', 'wins': 'sum', 'bet_size': 'sum', 'profit_loss': 'sum', 'max_bet_size': 'max'})
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
    
    # Output the underlying data if required
    if output_data_for_debug:
        actuals_data.sort_values('date', inplace = True)
        actuals_data[['fighter1', 'fighter2', 'outcome', 'fighter1_best_closing_odds', 'fighter2_best_closing_odds', 'predictions', 'y_pred_prob', 'target', 'bets', 'bet_ev' ,'kelly_criterion', 'bet_size', 'max_bet_size', 'profit_loss', 'total', 'fights', 'wins']].to_csv('results/backtesting_data_for_checking.csv')
    
    return results
