def betfair_add_upcoming_bouts_details(betfair_df, upcoming_bouts_df, threshold = 75):
    """
    This function takes a data frame of the latest betfair odds for upcoming bouts, and matches them with upcoming bouts information from UFC stats to get a cleaned data frame
    
    Parameters:
    
    betfair_df: the upcoming betfair df data frame created
    upcoming_bouts_df: the upcoming events
    threshold: the % match minimum in order to match with the betfair exch odds
    
    """

    # pdb.set_trace()
    
    df = betfair_df.copy()
    
    # Find the closest names for the fighter
    df['fighter'] = df['fighter'].apply(lambda x: get_best_match(
        x,
        choices = upcoming_bouts_df['fighter1'].tolist() + upcoming_bouts_df['fighter2'].tolist(),
        threshold = threshold  # use token_sort_ratio for good name matching
    ))
    
    # Remove those not matched
    df = df[~df['fighter'].isna()]
    
    # Now get the event details
    fighter_events = pd.concat([
        upcoming_bouts_df[['fighter1', 'event', 'bout']].rename(columns = {'fighter1': 'fighter'}),
        upcoming_bouts_df[['fighter2', 'event', 'bout']].rename(columns = {'fighter2': 'fighter'})
        ])
    
    # Keep the date as the date and time of the odds
    df.rename(columns = {'gmt_datetime': 'notformodel_betfair_odds_datetime'}, inplace = True)
    
    # Merge these onto the odds data
    df = df[['fighter', 'last_price_traded', 'notformodel_betfair_odds_datetime']].merge(fighter_events, on = 'fighter', how = 'inner')
    
    # Clean names of column
    df.rename(columns = {'last_price_traded': 'betfair_exch_odds'}, inplace = True)

    return df