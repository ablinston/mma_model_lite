# This function checks five data frames containing events, fight results and stats. It checks that the same
# events are covered in each one.

# It returns a data frame detailing the checks done and whether they passed (True) or not (False)

def check_data_coverage(event_details_df, fight_details_df, fight_results_df, fight_stats_df, fight_odds_df, fighter_details_df, fighter_tott_df, config):
    
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
    fighter_details = fighter_details_df.copy()
    fighter_tott = fighter_tott_df.copy()

    # Change the column names to lower
    for df in [event_details, fight_details, fight_results, fight_stats, fight_odds, fighter_details, fighter_tott]:
        df.columns = df.columns.str.lower()
                
    # Add event and bout data onto odds
    fight_odds = add_event_and_bout_to_odds_data(fight_odds, event_details, fight_stats)
        
    # Clean all the columns that we wish to use
    for df in [event_details, fight_details, fight_results, fight_stats, fight_odds, fighter_details, fighter_tott]:
        for col in ['event', 'bout', 'fighter']:
            try:
                df[[col]] = clean_text_columns(df[[col]])
            except:
                continue
    
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
    event_details['date'] = event_details['date'].apply(lambda x: datetime.strptime(x, '%B %d, %Y'))
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
    
    return all_checks_passed
