# This function processes the cleaned odds data so that it is ready for merging with the other datasets

def process_odds_data(fight_df, event_df, betfair_df, odds_scrape_config, average_odds_model = None, impute_missing_averages = True):
    
    pdb.set_trace()
     
    new_event_df = event_df.copy()
    
    # Convert opening odds to decimal
    fight_df.loc[:, ['open_odds', 'closing_odds_low', 'closing_odds_high']] = fight_df[['open_odds', 'closing_odds_low', 'closing_odds_high']].map(american_odds_to_decimal)
    new_event_df[odds_scrape_config['events_colnames']] = new_event_df[odds_scrape_config['events_colnames']].map(american_odds_to_decimal)
    
    # Where we have more than 4 odds values for a fight, calculate the mean
    new_event_df = new_event_df[new_event_df[odds_scrape_config['events_colnames']].isna().sum(axis = 1) <= 5]
    new_event_df['mean_closing_odds'] = np.nanmean(new_event_df[odds_scrape_config['events_colnames']], axis = 1)
    
    # # Remove any duplicates. New odds data will be on top so keep that
    # new_event_df[new_event_df.duplicated(subset = ['event', 'bout', 'fighter'], keep = False)].sort_values(['event', 'bout', 'fighter'])
    new_event_df = new_event_df.drop_duplicates(subset = ['event', 'bout', 'fighter'])
    
    # Merge these means onto the fight df
    new_fight_df = fight_df.merge(new_event_df[['event', 'bout', 'fighter', 'mean_closing_odds']], on = ['event', 'bout', 'fighter'], how = 'left')
    
    # Check row number
    stop_if_not(len(new_fight_df) == len(fight_df), 'Odds merge check')
    
    if impute_missing_averages:
        # Fill in missing odds with modelled results
        new_fight_df.loc[new_fight_df['mean_closing_odds'].isna(), 'mean_closing_odds'] = average_odds_model.predict(new_fight_df[average_odds_model.features])[new_fight_df['mean_closing_odds'].isna()]
    
    # Record the best odds as the highest (as that is what will give the biggest return)
    new_fight_df['best_closing_odds'] = new_fight_df['closing_odds_high'].copy()
    
    # Calculate the odds change
    new_fight_df['odds_change'] = new_fight_df['mean_closing_odds'] - new_fight_df['open_odds']
    
    # Keep only the columns we need, and remove duplicates (these duplicates are where the event and bout are empty)
    new_fight_df = new_fight_df[['event', 'bout', 'fighter', 'open_odds', 'mean_closing_odds', 'best_closing_odds', 'odds_change', 'notformodel_date_fighter_odd_scrape']].drop_duplicates(subset = ['event', 'bout', 'fighter'])
    
    # Process the betfair data ready for merging
    betfair_to_merge = betfair_df[~betfair_df['event'].isna()].drop_duplicates(subset = ['event', 'bout', 'fighter'])
    
    # Merge on the betfair odds
    new_fight_df = new_fight_df.merge(betfair_to_merge[['event', 'bout', 'fighter', 'betfair_exch_odds', 'notformodel_betfair_odds_datetime']], on = ['event', 'bout', 'fighter'],  how = 'left')
    
    # Add on a flag to say whether the odds have changed by more than 0.1
    new_fight_df['odds_change_gt_0'] = np.where(new_fight_df['odds_change'] < 0, -1, np.where(new_fight_df['odds_change'] > 0, 1, 0))     
    
    # Check the number of rows
    stop_if_not(len(new_fight_df) == len(fight_df), "Process odds length check")
    
    return new_fight_df
