def backtest_model(actuals_data, prediction_probabilities, breakdown_cols, min_prob_to_bet, min_ev_to_bet, bet_sizing = 1, output_data_for_debug = False):
    """
    
    A function to backtest a model's predictions against actuals data that contain the odds'

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
    min_prob_to_bet : float64
        the minimum expected return required to place the bet (EV of 0 means no profit expected)

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
    
    # First calculate the Expected Values on a W/L result and L/W result, given in betting units
    # Odds are decimal odds, so represent the payout for a 1 unit bet e.g. odds of 2 means a (2 - 1) = 1 profit.
    ev_wl = (actuals_data['fighter1_best_closing_odds'] - 1) * prediction_probabilities[:, 2] - (1 - prediction_probabilities[:, 2])
    ev_lw = (actuals_data['fighter2_best_closing_odds'] - 1) * prediction_probabilities[:, 0] - (1 - prediction_probabilities[:, 0])
    
    # Now calculate bet triggers as 0 or 1 indicators
    ev_wl_trigger = ev_wl > min_ev_to_bet
    prob_wl_trigger = prediction_probabilities[:, 2] > min_prob_to_bet
    wl_trigger = 1 * ev_wl_trigger * prob_wl_trigger
    
    ev_lw_trigger = ev_lw > min_ev_to_bet
    prob_lw_trigger = prediction_probabilities[:, 0] > min_prob_to_bet
    lw_trigger = 1 * ev_lw_trigger * prob_lw_trigger
    
    # Set whether a bet is made
    actuals_data['bets'] = wl_trigger + lw_trigger
    
    # Now calculate the returns if a bet was made
    actuals_data['profit_loss'] = wl_trigger * (actuals_data['outcome'] == 'W/L') - wl_trigger * (actuals_data['outcome'] != 'W/L') + lw_trigger * (actuals_data['outcome'] == 'L/W') - lw_trigger * (actuals_data['outcome'] != 'L/W')
    
    # Calculate some further columns that will be used for stats
    actuals_data['total'] = 'Total'
    actuals_data['fights'] = 1
    actuals_data['wins'] = 1 * (actuals_data['profit_loss'] > 0)
    
    ## Summarise the results
    results = actuals_data.groupby('total').agg({'fights': lambda x: len(x), 'bets': 'sum', 'wins': 'sum', 'profit_loss': 'sum'})
    for col in breakdown_cols:
        results = pd.concat([
            results,
            actuals_data.groupby(col).agg({'fights': lambda x: len(x), 'bets': 'sum', 'wins': 'sum', 'profit_loss': 'sum'})
        ])
    
    # Calculate percentage info
    results['percent_bet_on'] = results['bets'] / results['fights'] * 100
    results['win_percent'] = results['wins'] / results['bets'] * 100
    results['roi'] = results['profit_loss'] / results['bets'] * 100 # the profit and loss assumes every bet equal size
    
    # For the total, calculate the longest streaks
    results['max_losing_streak'] = np.nan
    results.loc['Total', 'max_losing_streak'] = longest_streak(actuals_data['profit_loss'], -1)
    
    results['max_drawdown'] = np.nan
    results.loc['Total', 'max_drawdown'] = minimum_sum_series(actuals_data['profit_loss'])
    
    # # Add the row names as a column
    # results = pd.concat([pd.DataFrame({'Group': results.index}), results], axis=1)
    results.rename(columns={'index': 'RowName'}, inplace=True)
    
    # Output the underlying data if required
    if output_data_for_debug:
        actuals_data[['fighter1', 'fighter2', 'outcome', 'fighter1_best_closing_odds', 'fighter2_best_closing_odds', 'predictions', 'y_pred_prob', 'target', 'bets', 'profit_loss', 'total', 'fights', 'wins']].to_csv('results/backtesting_data_for_checking.csv')
    
    return results
