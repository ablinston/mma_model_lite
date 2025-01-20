# This function takes the odds data and finds the event and bout information from the event details and fight stats data

# It returns the odds data with the correct event and bout matched along with a correct date

def add_event_and_bout_to_odds_data(odds_df, event_details_df, stats_df):
    
    # # debug
    # odds_df = parsed_odds_df.copy()
    # event_details_df = updated_event_details_df.copy()
    # stats_df = parsed_fight_stats_df.copy()
    # pdb.set_trace()
    
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
    
    # Create two alternative dates on the odds data 
    new_odds_df = odds_df.copy()
    new_odds_df['date-1'] = new_odds_df['date'] - timedelta(days = 1)
    new_odds_df['date+1'] = new_odds_df['date'] + timedelta(days = 1)
    
    # Now try to merge on the event and bout. Rename columns to avoid duplicated names
    new_odds_df.rename(columns = {'event': 'event_old', 'bout': 'bout_old'}, inplace = True)
    new_odds_df = new_odds_df.merge(new_stats_df, how = 'left', left_on = ['fighter', 'date', 'opponent'], right_on = ['fighter', 'event_date', 'opponent'])
    new_stats_df.rename(columns = {'event': 'event_m', 'bout': 'bout_m', 'event_date': 'event_date_m'}, inplace = True)
    new_odds_df = new_odds_df.merge(new_stats_df, how = 'left', left_on = ['fighter', 'date-1', 'opponent'], right_on = ['fighter', 'event_date_m', 'opponent'])
    new_stats_df.rename(columns = {'event_m': 'event_p', 'bout_m': 'bout_p', 'event_date_m': 'event_date_p'}, inplace = True)
    new_odds_df = new_odds_df.merge(new_stats_df, how = 'left', left_on = ['fighter', 'date+1', 'opponent'], right_on = ['fighter', 'event_date_p', 'opponent'])
    
    ## Now do a merge based on opponent. This is a last resort where the date of some bouts is missing and so can't match
    
    # Create a data frame with opponents details.
    new_stats_df.rename(columns = {'event_p': 'event', 'bout_p': 'bout', 'event_date_p': 'event_date'}, inplace = True)
    opponents_df = new_stats_df.copy()
    # opponents_df[['fighter1', 'fighter2']] = opponents_df['bout'].str.split('vs.', expand = True)
    # opponents_df['opponent'] = np.where(opponents_df['fighter1'] == opponents_df['fighter'], opponents_df['fighter2'], opponents_df['fighter1'])
    opponents_df = opponents_df[['event', 'bout', 'event_date', 'fighter', 'opponent']]
    opponents_df.columns = [f'{col}_opp' for col in opponents_df.columns]
    
    # Remove all rows that have duplicates i.e. where fighters have fought more than once
    duplicates = opponents_df.duplicated(subset=['fighter_opp', 'opponent_opp'], keep=False)
    opponents_df = opponents_df[~duplicates]
        
    # Now merge based on just opponent
    new_odds_df = new_odds_df.merge(opponents_df, how = 'left', left_on = ['fighter', 'opponent'], right_on = ['fighter_opp', 'opponent_opp'])
    
    # Now create the event and bout columns based on the matches
    new_odds_df['event'] = np.where(~new_odds_df['event'].isna(), new_odds_df['event'],
                                    np.where(~new_odds_df['event_m'].isna(), new_odds_df['event_m'], 
                                             np.where(~new_odds_df['event_p'].isna(), new_odds_df['event_p'], new_odds_df['event_opp'])))
    new_odds_df['bout'] = np.where(~new_odds_df['bout'].isna(), new_odds_df['bout'],
                                    np.where(~new_odds_df['bout_m'].isna(), new_odds_df['bout_m'], 
                                             np.where(~new_odds_df['bout_p'].isna(), new_odds_df['bout_p'], new_odds_df['bout_opp'])))
    
    # And replace dates with actual event dates where possible
    new_odds_df['date'] = np.where(~new_odds_df['event_date'].isna(), new_odds_df['event_date'], 
                                   np.where(~new_odds_df['date'].isna(), new_odds_df['date'], new_odds_df['event_date_opp']))
    
    # Check the new data is a consistent length with the old data
    if len(new_odds_df) != len(odds_df):    
        print("ERROR: MATCHING EVENTS AND BOUTS TO ODDS CREATED TOO MANY RECORDS")
        return None
    
    # Finally, if no event has been found, just use the old event name
    new_odds_df['event'] = np.where(new_odds_df['event'].isna(), new_odds_df['event_old'], new_odds_df['event'])
    
    # Do a check to see where we have not found UFC events
    ufc_bouts_missing_odds = new_stats_df[new_stats_df['event_date'] > '2010-01-01']['bout'][~new_stats_df['bout'].isin(new_odds_df['bout'].tolist())]
    ufc_events_missing_odds = new_stats_df[new_stats_df['event_date'] > '2010-01-01']['event'][~new_stats_df['event'].isin(new_odds_df['event'].tolist())]
    
    # Give a warning about any that haven't been matched to an event and bout
    if len(ufc_events_missing_odds) > 0:
        print(f"The following events after 01-01-2010 are missing odds data: {set(ufc_events_missing_odds)}")
    
    # Remove columns no longer needed
    # new_odds_df.drop(['event', 'event_m', 'event_p', 'bout_m', 'bout_p', 'event_date', 'event_date_m', 'event_date_p', \
    #                   'date+1', 'date-1', 'fighter_m', 'fighter_p', 'fighter', 'event_opp', 'bout_opp', \
    #                       'event_date_opp', 'fighter_opp', 'opponent_opp'], axis = 1, inplace = True)
    
    new_odds_df = new_odds_df[['open_odds', 'closing_odds_low', 'closing_odds_high', 'event', 'bout', 'date', 'fighter', 'opponent']]
        
    return new_odds_df
