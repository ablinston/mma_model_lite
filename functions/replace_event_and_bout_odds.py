# This function takes the cleaned odds data and replaces events and bouts with those from a fight stats dataset. It merges on fighter, opponent and date, but can also give a time delta for the date.

# You can set whether to take the odds data as priority or the stats data as prioirty when determining event and bout

def replace_event_and_bout_odds(odds_df, stats_df, join_on = ['fighter', 'date', 'opponent'], day_shift = 0, odds_event_priority = False):
    
    # pdb.set_trace()
    
    new_odds_df = odds_df.copy()
    
    # Perform the shift in date
    new_odds_df['date'] = new_odds_df['date'] + timedelta(days = day_shift)
    
    new_odds_df.rename(columns = {'event': 'event_old', 'bout': 'bout_old'}, inplace = True)
    new_odds_df = new_odds_df.merge(stats_df, how = 'left', left_on = join_on, right_on = ['event_date' if item == 'date' else item for item in join_on])
    
    if odds_event_priority:
        # Now fix the event and bout columns based on the matches
        new_odds_df['event'] = np.where(~new_odds_df['event_old'].isna(), 
                                        new_odds_df['event_old'],
                                        np.where(~new_odds_df['event'].isna(), 
                                                 new_odds_df['event'], 
                                                 new_odds_df['event_old']))
        new_odds_df['bout'] = np.where(~new_odds_df['bout_old'].isna(), 
                                       new_odds_df['bout_old'],
                                        np.where(~new_odds_df['bout'].isna(), 
                                                 new_odds_df['bout'], 
                                                 new_odds_df['bout_old']))
    else:
        # Now fix the event and bout columns based on the matches
        new_odds_df['event'] = np.where(~new_odds_df['event'].isna(), 
                                        new_odds_df['event'],
                                        new_odds_df['event_old'])
        new_odds_df['bout'] = np.where(~new_odds_df['bout'].isna(), 
                                       new_odds_df['bout'],
                                       new_odds_df['bout_old'])

    # Shift the date back to where it was originally
    new_odds_df['date'] = new_odds_df['date'] - timedelta(days = day_shift)
    
    # Drop extra columns
    new_odds_df.drop(columns = ['event_old', 'bout_old'], inplace = True)
    try:
        new_odds_df.drop(columns = ['event_date'], inplace = True)
    except:
        pass
    
    # Do some final checks on the data
    stop_if_not(new_odds_df['date'].tolist() == odds_df['date'].tolist(), 'Date check')
    stop_if_not(new_odds_df.shape == odds_df.shape, 'Dataset shape check')
    
    return new_odds_df
