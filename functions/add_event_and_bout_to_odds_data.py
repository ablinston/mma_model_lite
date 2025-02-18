# This function takes the odds data and finds the event and bout information from the event details and fight stats data

# It returns the odds data with the correct event and bout matched along with a correct date

def add_event_and_bout_to_odds_data(odds_df, event_details_df, stats_df, date_check = '2010-01-01'):
    
    # # debug
    # odds_df = parsed_odds_df.copy()
    # event_details_df = updated_event_details_df.copy()
    # stats_df = parsed_fight_stats_df.copy()
    # pdb.set_trace()
    
    # First create event and bout dummies if not there
    if 'event' not in odds_df.columns:
        odds_df['event'] = np.nan
    if 'bout' not in odds_df.columns:
        odds_df['bout'] = np.nan
        
    # Convert the event details date to a datetime
    event_dates = event_details_df.copy()[['event', 'date']]
    if pd.api.types.is_datetime64_any_dtype(event_dates['date']):
        event_dates.loc[:, 'event_date'] = event_dates['date']
    else:
        event_dates.loc[:, 'event_date'] = event_dates['date'].apply(lambda x: datetime.strptime(x, '%B %d, %Y'))
    
    # Merge the dates onto to stats data
    new_stats_df = stats_df.merge(event_dates[['event', 'event_date']], how = 'left', on = 'event')
        
    # Clean the bouts and fighter columns
    new_stats_df[['bout', 'fighter']] = clean_text_columns(new_stats_df[['bout', 'fighter']])
    odds_df[['fighter', 'opponent']] = clean_text_columns(odds_df[['fighter', 'opponent']])
    
    # Drop columns not needed and duplicates. Drop NA because they will affect the match
    new_stats_df = new_stats_df[['event', 'bout', 'fighter', 'event_date']].drop_duplicates()
    new_stats_df = new_stats_df.dropna()
    
    # Create an opponent column
    add_opponent_column(new_stats_df)
    
    ######################
    # Going in reverse priority, try to replace the event and bout from matches in the data
    
    # Create alternative dates on the odds data 
    new_odds_df = odds_df.copy()
    
    # Create a custom bout
    if 'bout' not in new_odds_df.columns:
        new_odds_df['bout'] = np.nan

    ######################
    # Do a merge based on opponent. This is a last resort where the date of some bouts is missing and so can't match
    
    # Create a data frame with opponents details.
    opponents_df = new_stats_df.copy()
    opponents_df = opponents_df[['event', 'bout', 'event_date', 'fighter', 'opponent']]
    
    # Remove all rows that have duplicates i.e. where fighters have fought more than once
    duplicates = opponents_df.duplicated(subset=['fighter', 'opponent'], keep=False)
    opponents_df = opponents_df[~duplicates]
    
    # Replace events and bouts with those from the opponents data
    new_odds_df = replace_event_and_bout_odds(odds_df = new_odds_df, stats_df = opponents_df, join_on = ['fighter', 'opponent'], day_shift = 0, odds_event_priority = False)
    
    # Now, working from 2 days down to one, fill the missing events
    new_odds_df = replace_event_and_bout_odds(odds_df = new_odds_df, stats_df = new_stats_df, join_on = ['fighter', 'date', 'opponent'], day_shift = 2, odds_event_priority = False)
    new_odds_df = replace_event_and_bout_odds(odds_df = new_odds_df, stats_df = new_stats_df, join_on = ['fighter', 'date', 'opponent'], day_shift = -2, odds_event_priority = False)
    new_odds_df = replace_event_and_bout_odds(odds_df = new_odds_df, stats_df = new_stats_df, join_on = ['fighter', 'date', 'opponent'], day_shift = 1, odds_event_priority = False)
    new_odds_df = replace_event_and_bout_odds(odds_df = new_odds_df, stats_df = new_stats_df, join_on = ['fighter', 'date', 'opponent'], day_shift = -1, odds_event_priority = False)
    new_odds_df = replace_event_and_bout_odds(odds_df = new_odds_df, stats_df = new_stats_df, join_on = ['fighter', 'date', 'opponent'], day_shift = 0, odds_event_priority = False)
        
    # Check the new data is a consistent length with the old data
    if len(new_odds_df) != len(odds_df):    
        print("ERROR: MATCHING EVENTS AND BOUTS TO ODDS CREATED TOO MANY RECORDS")
        return None
        
    # Do a check to see where we have not found UFC events
    ufc_bouts_missing_odds = new_stats_df[new_stats_df['event_date'] > date_check]['bout'][~new_stats_df['bout'].isin(new_odds_df['bout'].tolist())]
    ufc_events_missing_odds = new_stats_df[new_stats_df['event_date'] > date_check]['event'][~new_stats_df['event'].isin(new_odds_df['event'].tolist())]
    
    # Give a warning about any that haven't been matched to an event and bout
    if len(ufc_events_missing_odds) > 0:
        print(f"The following events after 01-01-2010 are missing odds data: {set(ufc_events_missing_odds)}")
    
    # Remove columns no longer needed    
    new_odds_df = new_odds_df[odds_df.columns]
        
    return new_odds_df
