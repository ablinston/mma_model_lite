# This function takes a dataset of actuals and prediction probabilities (which must correspond in rows to each other)
# It returns a data frame with added columns for the EVs of each bet and bet sizing
def summarise_model_predictions(actuals_data_df, prediction_probabilities, min_prob_to_bet, min_ev_to_bet, min_bet_size = 1, max_bet_size = 1, return_triggers = False):

    
    # ###
    # # Debug
    # actuals_data = X_train[['id']].merge(data, on = 'id', how = 'left')
    # prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_train, y_train)
    # breakdown_cols = ['weightclass', 'year']
    # min_prob_to_bet = 0.5
    # min_ev_to_bet = 0.1
    # ###
    
    actuals_data = actuals_data_df.copy()
    
    # pdb.set_trace()
    
    # Do a check that the inputs are consistent if doing a non constant bets. If using non constant bets, the bet sizing should be as proportions, not £ amounts

    # First calculate the Expected Values on a W/L result and L/W result, given in betting units
    # Odds are decimal odds, so represent the payout for a 1 unit bet e.g. odds of 2 means a (2 - 1) = 1 profit.
    ev_wl = (actuals_data['fighter1_backtesting_odds'] - 1) * prediction_probabilities[:, 2] - (1 - prediction_probabilities[:, 2])
    ev_lw = (actuals_data['fighter2_backtesting_odds'] - 1) * prediction_probabilities[:, 0] - (1 - prediction_probabilities[:, 0])
    
    # Now calculate bet triggers as 0 or 1 indicators
    ev_wl_trigger = ev_wl > min_ev_to_bet
    prob_wl_trigger = prediction_probabilities[:, 2] > min_prob_to_bet
    wl_trigger = 1 * ev_wl_trigger * prob_wl_trigger
    
    ev_lw_trigger = ev_lw > min_ev_to_bet
    prob_lw_trigger = prediction_probabilities[:, 0] > min_prob_to_bet
    lw_trigger = 1 * ev_lw_trigger * prob_lw_trigger
    
    # Fix any instances where both triggers happen. Remove the bet because there must be an odds mistake
    double_trigger = (lw_trigger + wl_trigger) > 1
    lw_trigger[double_trigger] = 0
    wl_trigger[double_trigger] = 0
    
    # Set whether a bet is made
    actuals_data['bets'] = wl_trigger + lw_trigger
    
    # Add the EV for later breakdowns
    actuals_data['bet_ev'] = np.where(actuals_data['bets'] == 0, np.nan, wl_trigger * ev_wl + lw_trigger * ev_lw)
    #  Convert it into buckets
    bucket_width = 0.1
    col_min = np.round(np.floor(actuals_data['bet_ev'].min() / bucket_width) * bucket_width, 2)
    col_max = np.round(np.ceil(actuals_data['bet_ev'].max() / bucket_width) * bucket_width, 2)
    actuals_data['bet_ev_band'] = pd.cut(actuals_data['bet_ev'],
                                    bins=np.arange(col_min, col_max + bucket_width, bucket_width))
    
    
    ## Determine the bet size. Begin with the minimum, then add on the confidence of the bet
    
    # First get the probabilities of winning and odds, irrespective of fighter to win
    win_prob = wl_trigger * prediction_probabilities[:, 2] + lw_trigger * prediction_probabilities[:, 0]
    win_odds = wl_trigger * actuals_data['fighter1_backtesting_odds'] + lw_trigger * actuals_data['fighter2_backtesting_odds']
    
    # Now calculate the kelly criterion
    actuals_data['kelly_criterion'] = win_prob - (1 - win_prob) / (win_odds - 1)
    
    # Perform a check that kelly is correct
    stop_if_not((actuals_data['kelly_criterion'].max() <= 1) and (actuals_data['kelly_criterion'].min() >= 0),
                'Kelly criterion check')
    
    # And use it to calculate how much to increase from the min
    actuals_data['bet_size'] = actuals_data['bets'] * (min_bet_size + (max_bet_size - min_bet_size) * actuals_data['kelly_criterion'])

    # Check no bets breach the max
    stop_if_not(actuals_data['bet_size'].max() <= max_bet_size, 'Max bet check')
    
    # Summarise key outputs
    actuals_data['Fighter1_win_prob'] = np.round(prediction_probabilities[:, 2] * 100, 1)
    actuals_data['Fighter1_win_EV'] = np.round(ev_wl, 2)
    actuals_data['Fighter2_win_prob'] = np.round(prediction_probabilities[:, 0] * 100, 1)
    actuals_data['Fighter2_win_EV'] = np.round(ev_lw, 2)
    actuals_data['Bet_trigger'] = np.where(wl_trigger == 1, actuals_data['fighter1'], np.where(lw_trigger == 1, actuals_data['fighter2'], 'None'))
    actuals_data['Predict_to_win'] = np.where((actuals_data['Fighter1_win_prob'] > actuals_data['Fighter2_win_prob']) & (actuals_data['fighter1_backtesting_odds'] > actuals_data['fighter2_backtesting_odds']), 
                                              'Underdog to win',
                                             np.where((actuals_data['Fighter2_win_prob'] > actuals_data['Fighter1_win_prob']) & (actuals_data['fighter2_backtesting_odds'] > actuals_data['fighter1_backtesting_odds']),
                                             'Underdog to win',
                                             'Favourite to win'))
    actuals_data['Bet_breakeven_odds'] = np.where(wl_trigger == 1, 100 / actuals_data['Fighter1_win_prob'],
                                                  np.where(lw_trigger == 1, 100 / actuals_data['Fighter2_win_prob'], np.nan))    # checked the maths and this is how it works out
    if return_triggers:
        return [actuals_data, wl_trigger, lw_trigger]
    else:
        return actuals_data
