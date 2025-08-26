def calculate_knocked_out_stats(fighter_results):
    
    ###########################################################################################
    # Calculate the date was last knocked out
    dates_knocked_out = fighter_results.copy()
    dates_knocked_out['knocked_out'] = np.where((dates_knocked_out['method'] == 'KO/TKO') & (dates_knocked_out['result'] == 'L'), 1, 0)
    dates_knocked_out['date_knocked_out'] = np.where(dates_knocked_out['knocked_out'] == 1, dates_knocked_out['date'], np.datetime64('NaT'))

    dates_knocked_out.sort_values(by = ['fighter', 'date'], inplace = True)

    # Calculate date last knocked out
    dates_knocked_out['date_last_knocked_out'] = dates_knocked_out.groupby(['fighter'])['date_knocked_out'].shift(1)
    dates_knocked_out['date_last_knocked_out'] = dates_knocked_out.groupby(['fighter'])['date_last_knocked_out'].ffill()

    # Calculate the number times knocked out previously
    dates_knocked_out['hist_times_knocked_out'] = dates_knocked_out.groupby(['fighter'])['knocked_out'].cumsum() - dates_knocked_out['knocked_out']

    # Fill in missing rounds with round 3
    dates_knocked_out['round'] = dates_knocked_out['round'].fillna(3)

    # Calculate how many times each round they've been knocked out
    for round in [1, 2, 3, 4, 5]:
        
        # Create a flag for whether the person was knocked out in that fight
        dates_knocked_out['round_knockout'] = np.where(
            (dates_knocked_out['round'] == round) &
            (dates_knocked_out['method'] == 'KO/TKO') & 
            (dates_knocked_out['result'] == 'L'),
            1, 0)
        
        # Aggregate this for fighters
        dates_knocked_out['hist_times_knocked_out_round_' + str(round)] = dates_knocked_out.groupby('fighter')['round_knockout'].cumsum() - dates_knocked_out['round_knockout']

    dates_knocked_out = dates_knocked_out[['event',
                                           'bout',
                                           'fighter', 
                                           'date_last_knocked_out', 
                                           'hist_times_knocked_out',
                                           'hist_times_knocked_out_round_1',
                                           'hist_times_knocked_out_round_2',
                                           'hist_times_knocked_out_round_3',
                                           'hist_times_knocked_out_round_4',
                                           'hist_times_knocked_out_round_5']]

    # Fill NAs with 0
    dates_knocked_out.loc[:, ['hist_times_knocked_out', 'hist_times_knocked_out_round_1', 'hist_times_knocked_out_round_2', 'hist_times_knocked_out_round_3', 'hist_times_knocked_out_round_4', 'hist_times_knocked_out_round_5']] = dates_knocked_out[['hist_times_knocked_out', 'hist_times_knocked_out_round_1', 'hist_times_knocked_out_round_2', 'hist_times_knocked_out_round_3', 'hist_times_knocked_out_round_4', 'hist_times_knocked_out_round_5']].fillna(0) 

    # Fill NAs in dates with arbitrary early date
    dates_knocked_out['date_last_knocked_out'] = pd.to_datetime(dates_knocked_out['date_last_knocked_out'].fillna('1900-01-01'))

    # Check number of rows hasn't changed
    stop_if_not(len(dates_knocked_out) == len(fighter_results))
    
    return dates_knocked_out
