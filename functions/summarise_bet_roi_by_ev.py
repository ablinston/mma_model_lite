def summarise_bet_roi_by_ev(df):
    """
    This function takes a backtesting output from backtest_model() and summarise the returns
    on investment from betting on all outcomes, and calculates the ROI for each EV band
    the model predicted.
    
    It assumes a fixed bet across all fighters of 1
    """
    
    # pdb.set_trace()
    
    # Split out data for fighter 1 and fighter 2
    f1_df = df[['Fighter1_win_EV', 'fighter1_backtesting_odds']].rename(columns = {'Fighter1_win_EV':'EV', 'fighter1_backtesting_odds': 'odds'})
    # Add on win
    f1_df['WL'] = np.where(df['outcome'] == "W/L", 1, 0)

    f2_df = df[['Fighter2_win_EV', 'fighter2_backtesting_odds']].rename(columns = {'Fighter2_win_EV':'EV', 'fighter2_backtesting_odds': 'odds'})
    # Add on win
    f2_df['WL'] = np.where(df['outcome'] == "L/W", 1, 0)
    
    # Stack the data to get two bets on every fight
    stacked_df = pd.concat([f1_df, f2_df]).reset_index(drop = True)

    # Now cut EV into bands
    stacked_df['EV_band'] = pd.cut(stacked_df['EV'], [-10, -0.25, -0.1, 0, 0.1, 0.25, 10], include_lowest = True)
    
    stacked_df['ROI_on_constant_bets'] = np.where(stacked_df['WL'] == 1, stacked_df['odds'] - 1, -1)
    
    # Summarise the output
    output_df = stacked_df.sort_values('EV').groupby('EV_band', observed = False)[['ROI_on_constant_bets']].mean() * 100
    
    # Add on the sample size
    output_df['Sample_Size'] = stacked_df.sort_values('EV').groupby('EV_band', observed = False)['ROI_on_constant_bets'].count()
    
    return output_df