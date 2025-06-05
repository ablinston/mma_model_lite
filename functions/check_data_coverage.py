# This function checks five data frames containing events, fight results and stats. It checks that the same
# events are covered in each one.

# It returns a data frame detailing the checks done and whether they passed (True) or not (False)

def check_data_coverage(event_details_df, fight_details_df, fight_results_df, fight_stats_df, fight_odds_df, fighter_details_df, fighter_tott_df, event_odds_df, betfair_odds_df, upcoming_events_df, upcoming_fight_details_df, betfair_upcoming_odds_df, config):
    
    # # Debug
    # event_details = parsed_event_details_df.copy()
    # fight_details = parsed_fight_details_df.copy()
    # fight_results = parsed_fight_results_df.copy()
    # fight_stats = parsed_fight_stats_df.copy()
    # fight_odds = parsed_odds_df.copy()
    # fighter_details = parsed_fighter_details_df.copy()
    # fighter_tott = parsed_fighter_tott_df.copy()
    
    # Create a flag to say whether checks passed
    all_checks_passed = True
    # pdb.set_trace()
    # Make copies so we don't overwrite any of the original data frames
    event_details = event_details_df.copy()
    fight_details = fight_details_df.copy()
    fight_results = fight_results_df.copy()
    fight_stats = fight_stats_df.copy()
    fight_odds = fight_odds_df.copy()
    event_odds = event_odds_df.copy()
    fighter_details = fighter_details_df.copy()
    fighter_tott = fighter_tott_df.copy()
    betfair_odds = betfair_odds_df.copy()
    upcoming_events = upcoming_events_df.copy()
    upcoming_fight_details = upcoming_fight_details_df.copy()
    betfair_upcoming_odds = betfair_upcoming_odds_df.copy()
    
    # Clean all the datasets
    cleaned_dict = clean_all_raw_data(event_details, upcoming_events, upcoming_fight_details, fighter_tott, fight_results, fight_stats, fight_odds, event_odds, betfair_odds, betfair_upcoming_odds)
    
    # Change the column names to lower
    for df in [event_details, fight_details, fight_results, fight_stats, fight_odds, fighter_details, fighter_tott, event_odds_df, betfair_odds, betfair_upcoming_odds]:
        df.columns = df.columns.str.lower()
                
            
    ##################################
    #### Data cleaning
    ##################################
    
    # # Clean all the columns that we wish to use
    # for df in [event_details, fight_details, fight_results, fight_stats, fight_odds, fighter_details, fighter_tott, event_odds]:
    #     for col in ['event', 'bout', 'fighter']:
    #         try:
    #             df[[col]] = clean_text_columns(df[[col]])
    #         except:
    #             continue
            
    # Process the odds data to combine events
    fight_odds.loc[:, ['open_odds', 'closing_odds_low', 'closing_odds_high']] = fight_odds[['open_odds', 'closing_odds_low', 'closing_odds_high']].apply(pd.to_numeric, errors = 'coerce')
    fight_odds = process_odds_data(cleaned_dict['fighter_odds'].drop_duplicates(subset = ['event', 'bout', 'fighter']), cleaned_dict['event_odds'], cleaned_dict['betfair_odds'], odds_scrape_config, impute_missing_averages = False)
    
    ## The events data is the first updated, so is assumed to be complete
    # Check that all the events have some fight details in the data
    missing_events_details = event_details['event'][~event_details['event'].isin(fight_details['event'].tolist())]
    
    if (len(missing_events_details) > 0):
        print(f"Error: The following events are missing from the fight_details data: {missing_events_details}")
        all_checks_passed = False
                    
    # Check that all events are covered in the fight results data
    missing_events_results = event_details['event'][~event_details['event'].isin(fight_results['event'].tolist())]
    
    if (len(missing_events_results) > 0):
        print(f"Error: The following events are missing from the fight_results data: {missing_events_results}")
        all_checks_passed = False
        
    # Check that all events are covered in the fight stats data
    missing_events_stats = event_details['event'][~event_details['event'].isin(fight_stats['event'].tolist())]
    
    if (len(missing_events_stats) > 0):
        print(f"Error: The following events are missing from the fight_stats data: {missing_events_stats}")
        all_checks_passed = False

    ## Next, check the fight results details and stats have the same bouts in them
    
    # Get a series detailing all bouts covered in the data
    all_bouts = pd.Series(list(set(fight_details['bout'].tolist() + fight_results['bout'].tolist() + fight_stats['bout'].tolist())))
    
    missing_bouts_on_results = list(set(all_bouts[~all_bouts.isin(fight_results['bout'].tolist())]))
    missing_bouts_on_details = list(set(all_bouts[~all_bouts.isin(fight_details['bout'].tolist())]))
    missing_bouts_on_stats = list(set(all_bouts[~all_bouts.isin(fight_stats['bout'].tolist())]))
    
    if (len(missing_bouts_on_details) > 0):
        print(f"Error: The following bouts are missing from the fight_stats data: {missing_bouts_on_details}")
        all_checks_passed = False
    if (len(missing_bouts_on_results) > 0):
        print(f"Error: The following bouts are missing from the fight_stats data: {missing_bouts_on_results}")
        all_checks_passed = False
    if (len(missing_bouts_on_stats) > 0):
        print(f"Error: The following bouts are missing from the fight_stats data: {missing_bouts_on_stats}")
        all_checks_passed = False
        
    # Check the event dates on the odds data to see if any are missing
    # We only consider events after 2010 as betting odds generally aren't available in early UFC
    # event_details['date'] = event_details['date'].apply(lambda x: datetime.strptime(x, '%B %d, %Y'))
    post_2010_events = event_details['event'][event_details['date'] > config['start_date_validation_checks']]
    
    # Only keep UFC events
    post_2010_events = pd.Series([event for event in post_2010_events if 'ufc' in event])
    
    missing_events_from_odds = post_2010_events[~post_2010_events.isin(fight_odds['event'].tolist())]
    
    if (len(missing_events_from_odds) > 0):
        print(f"Error: The following events are missing from the odds data: {missing_events_from_odds}")
        all_checks_passed = False
    
    # Check if there are any bouts missing from the odds data
    post_2010_bouts = fight_stats['bout'][fight_stats['event'].isin(post_2010_events)]
    
    missing_bouts_from_odds = set(post_2010_bouts[~post_2010_bouts.isin(fight_odds['bout'].tolist())])
    
    if (len(missing_bouts_from_odds) > 0):
        print(f"Warning: {len(missing_bouts_from_odds)} UFC bouts out of {len(all_bouts)} are missing from the odds data.")
        # all_checks_passed = False # Not setting to false because some bouts have no odds data because the fighters names are wrong. See UFC 298 for e.g.
        
    # Find which recent events have missing average odds data
    recent_events = event_details['event'][event_details['date'] > '2024-01-01']
    
    # Find which events are missing mean closing odds
    missing_events_average_odds = recent_events[~recent_events.isin(fight_odds[~fight_odds['mean_closing_odds'].isna()]['event'])].tolist()
    if (len(missing_events_average_odds) > 0):
        print(f"Warning: The following events are missing from the event odds data: {missing_events_average_odds}")
        
    # Find which fighters appear most in missed bouts
    # Split each element into two lists
    list1, list2 = zip(*(char.split('vs.') for char in missing_bouts_from_odds))
    # Combine both lists into a single list
    combined_list = list(list1) + list(list2)
    # Count occurrences of each item
    pd.Series(list(list1) + list(list2)).value_counts()
        
    ## Check that all the fighters have fighter details data
    # Add a fighter column
    fighter_details['fighter'] = fighter_details['first'] + fighter_details['last']
    
    # Get a list of all fighters after 2010
    all_fighters = pd.Series(list(set(fight_stats[fight_stats['event'].isin(post_2010_events)]['fighter'])))
    
    ## AB: commented this out as it doesn't seem to matter if some missing. The tott data is what's used
    # # Find which fighters are missing
    # missing_fighter_details = all_fighters[~all_fighters.isin(fighter_details['fighter'].to_list())].to_list()
    
    # if (len(missing_fighter_details) > 0):
    #     print(f"Error: Missing fighters from fighter_details data: {missing_fighter_details}")
    
    # Find which fighters are missing from tott data
    missing_fighter_tott = all_fighters[~all_fighters.isin(fighter_tott['fighter'].to_list())].to_list()
    if (len(missing_fighter_tott) > 0):
        print(f"Error: The following fighters are missing from the fighter_tott data: {missing_fighter_tott}")
        all_checks_passed = False
    
    # Check where we have missing Betfair odds
    merged_betfair_odds = cleaned_dict['fight_stats_agg'][['event', 'bout', 'fighter', 'date']].merge(
        cleaned_dict['betfair_odds'][['event', 'bout', 'fighter', 'betfair_exch_odds']],
        on = ['event', 'bout', 'fighter'],
        how = 'left'
    )
    
    # Check how many fighters after 2017 have missing betfair odds
    merged_betfair_odds = merged_betfair_odds[merged_betfair_odds['date'] > config['min_training_date']]
    # Find the percentage missing where it is a UFC event
    percent_missing_betfair = merged_betfair_odds[merged_betfair_odds['event'].str[0:3] == "ufc"]['betfair_exch_odds'].isna().mean() * 100
    
    print(f"There are {percent_missing_betfair} % of fighters with missing Betfair exchange odds in UFC events since {config['min_training_date']}")
    
    # Get the most missing fighters
    missing_betfair_fighter_counts = merged_betfair_odds[merged_betfair_odds['betfair_exch_odds'].isna()]['fighter'].value_counts()
    
    print("The most missing fighters from Betfair exchange odds are:")
    print(missing_betfair_fighter_counts.head(20))

    # te = merged_betfair_odds[merged_betfair_odds['betfair_exch_odds'].isna()]

    # Check how many upcoming bouts have missing odds
    merged_upcoming = cleaned_dict['upcoming_fight_details'].merge(cleaned_dict['betfair_upcoming_odds'], on = ['event', 'bout'], how = 'left')
    
    missing_percent_upcoming = merged_upcoming['betfair_exch_odds'].isna().mean() * 100
    
    print(f"There are {missing_percent_upcoming} % missing Betfair odds from upcoming fights")
    if missing_percent_upcoming > 50:
        all_checks_passed = False
    
    return all_checks_passed
