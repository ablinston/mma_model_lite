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
    # pdb.set_trace()
    # Find which fighters are missing from tott data
    missing_fighter_tott = all_fighters[~all_fighters.isin(fighter_tott['fighter'].to_list())].to_list()
    if (len(missing_fighter_tott) > 0):
        print(f"Error: The following fighters are missing from the fighter_tott data: {missing_fighter_tott}")
        if (len(missing_fighter_tott) > 6):
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
    # if missing_percent_upcoming > 50:
    #     all_checks_passed = False
    
    return all_checks_passed

def clean_all_raw_data(events, upcoming_events, upcoming_fight_details, fighters, fight_results, fight_stats, fighter_odds, event_odds, betfair_odds, betfair_upcoming_odds):
    
    """
    This function takes all the raw scraped data and cleans them, producing datasets which will then be saved.
    
    Returns: dict of cleaned datasets
    
    """
    # pdb.set_trace()
    # Convert all column names to lowercase
    for df in [events, upcoming_events, upcoming_fight_details, fight_results, fight_stats, fighters, fighter_odds, event_odds, betfair_odds, betfair_upcoming_odds]:
        df.columns = df.columns.str.lower()
        
    # Convert fighter data to numeric
    # Replace spaces and inches, then split and convert to cm
    fighters[['feet', 'inches']] = fighters['height'].str.replace(' ', '').str.replace('"', '').str.replace('-', '').str.split("'", expand = True)
    fighters.drop(columns = ['height'], inplace = True)
    fighters['height'] = pd.to_numeric(fighters['feet']) * 30.48 + pd.to_numeric(fighters['inches']) * 2.54
    
    fighters = to_numeric(fighters, ['weight', 'reach'])
    fight_stats = to_numeric(fight_stats, ['kd'])
    
    # Get cleaned versions of the columns
    fighters['Fighter'] = fighters['fighter'].copy()
    
    
    ##################
    ## Betfair initial processing
    
    # Remove any duplicates from the betfair data, de-prioritising Yes/No
    betfair_odds['yesno'] = np.where(betfair_odds['fighter'].str.replace('.', '').str.lower().isin(['yes', 'no']), 1, 0)    
    # # Convert Yes/No fighters into names on betfair odds # MANY WERE JUST DUPLICATES OF BOUTS WITH PROPER DATA
    # betfair_odds = betfair_yesno_to_names(betfair_odds)
    betfair_odds = betfair_odds[betfair_odds['yesno'] == 0]
    
    # Now sort and remove duplicates
    # betfair_odds.sort_values(['yesno'], inplace = True)
    betfair_odds = betfair_odds.drop_duplicates(subset = ['fighter', 'bout', 'event_date'])
    betfair_odds.drop(columns = ['yesno'], inplace = True)
    
    # Replace text in betfair odds bout to be consistent with other data
    betfair_odds['bout'] = betfair_odds['bout'].str.replace(" v ", " vs. ")
    
    # Get the surnames on the betfair odds data for use later, before cleaning the spaces from the names
    add_opponent_column(betfair_odds)
    betfair_odds['fighter_surname'] = betfair_odds['fighter'].str.split().str[-1]
    betfair_odds['opponent_surname'] = betfair_odds['opponent'].str.split().str[-1]
    
    ################################
    
    # Remove spaces in events to avoid merge errors
    for df in [events, upcoming_events, upcoming_fight_details, fight_results, fight_stats, fighters, fighter_odds, event_odds, betfair_odds, betfair_upcoming_odds]:
        try:
            df[['event']] = clean_text_columns(df[['event']])
        except:
            pass
        try:
            df[['bout']] = clean_text_columns(df[['bout']])
        except:
            pass
        try:
            df[['fighter']] = clean_text_columns(df[['fighter']])
        except:
            pass
        try:
            df[['opponent']] = clean_text_columns(df[['opponent']])
        except:
            pass
    
    # Save the fighter formatted names separately
    fighter_names = fighters[['fighter', 'Fighter']].drop_duplicates(subset = ['fighter'])
    fighters.drop(columns = ['Fighter'], inplace = True)
    
    # Compute the control time in minutes
    fight_stats['ctrl_minutes'] = time_to_minutes_decimal(fight_stats['ctrl'])
    
    # Format dates
    fighters['dob'] = pd.to_datetime(fighters['dob'].str.replace("-", ""), format = '%b %d, %Y')
    events['date'] = pd.to_datetime(events['date'], format = '%B %d, %Y')
    upcoming_events['date'] = pd.to_datetime(upcoming_events['date'], format = '%B %d, %Y')
    
    # Add the event date to the fight results
    fight_results = fight_results.merge(events[['event', 'date']], on = 'event', how = 'left')
    
    # Reformat weightclass
    fight_results['weightclass'] = fight_results['weightclass'].str.replace('UFC', '').str.replace('Bout', '').str.replace('Title', '').str.replace(' ', '').str.replace('Interim', '')
    upcoming_fight_details['weightclass'] = upcoming_fight_details['weightclass'].str.replace('UFC', '').str.replace('Bout', '').str.replace('Title', '').str.replace(' ', '').str.replace('Interim', '')
    
    # Clean the methods of victory
    fight_results['method'] = fight_results['method'].str.replace(' ', '')
    fight_results['method'] = np.where(fight_results['method'].isin(config['methods']), fight_results['method'] , 'Other')
    fight_results['method'] = np.where(fight_results['method'] == 'Decision-Majority', 'Decision-Split', fight_results['method'])
    
    # Compute rounds
    fight_results['rounds'] = fight_results['time format'].str[0:1]
    
    # Compute numeric values for strikes and tds
    fight_stats = split_of_columns(fight_stats, ['sig.str.', 'total str.', 'td', 'head', 'body', 'leg', 'distance', 'clinch', 'ground'])
    
    # Split fight into fighters
    fight_results[['fighter1', 'fighter2']] = fight_results['bout'].str.split('vs\\.', expand = True)
    upcoming_fight_details[['fighter1', 'fighter2']] = upcoming_fight_details['bout'].str.split('vs\\.', expand = True)
    
    # Aggregate the fight stats to be at fight-fighter level
    # Merge on the final round time from the fight results
    fight_results['fight_minutes'] = time_to_minutes_decimal(fight_results['time'])
    fight_stats = to_numeric(fight_stats, ['round', 'sub.att'])
    fight_results = to_numeric(fight_results, ['round'])
    # Event is needed in the join, otherwise a rematch picks up the first fight's round time
    fight_stats = fight_stats.merge(fight_results[['event', 'bout', 'round', 'fight_minutes']], on = ['event', 'bout', 'round'], how = 'left')
    # Set fight time (5 minute round assumed)
    fight_stats['fight_minutes'] = fight_stats['fight_minutes'].fillna(config['round_time_default'])
    
    # Remove columns not needed
    fighters.drop(columns = ['feet', 'inches', 'url'], inplace = True)
    fight_results.drop(columns = ['time', 'referee', 'details', 'url', 'fight_minutes'], inplace = True)
    fight_stats.drop(columns = ['ctrl'], inplace = True)
    
    # Find the columns to be aggregated
    agg_cols = ['kd', 'ctrl_minutes', 'fight_minutes', 'sub.att'] + [col for col in fight_stats.columns if '_attempt' in col or '_landed' in col]
    id_cols = ['event', 'bout', 'fighter']
    
    fight_stats_agg = fight_stats[id_cols].drop_duplicates()
    
    # Group and calculate sums for each round and the total
    for round in range(1, 6):
        prefix = "round" + str(round) + "_"
        # Filter for that round then rename the columns for that round
        new_stats = fight_stats[id_cols + agg_cols + ['round']][fight_stats['round'] == round].drop(columns = ['round']).rename(
            columns = dict(zip(agg_cols, [prefix + col for col in agg_cols])))
        # Get rid of duplicates. AB: Don't know why they're in there and a general drop_duplicates() doesn't work, but they have the same stats
        new_stats.drop_duplicates(subset = id_cols, inplace = True)
        # Merge onto the data
        fight_stats_agg = fight_stats_agg.merge(new_stats, on = id_cols, how = 'left')
    
    del([new_stats, prefix])
    
    # Fill the NAs in the fight stats with 0 for landed and attempts
    stats_cols = [col for col in fight_stats_agg.columns.to_list() if col not in id_cols]
    fight_stats_agg.loc[:, stats_cols] = fight_stats_agg.loc[:, stats_cols].fillna(0)
    
    # Calculate the total for all rounds then merge onto data
    fight_stats_all_round = fight_stats.groupby(id_cols)[agg_cols].sum().reset_index().rename(columns = dict(zip(agg_cols, ['allrounds_' + col for col in agg_cols])))
    fight_stats_agg = fight_stats_agg.merge(fight_stats_all_round, on = id_cols, how = 'left')
    del(fight_stats_all_round)
    
    # Check the row numbers
    if (len(fight_stats_agg) == len(fight_stats[id_cols].drop_duplicates())):
        print('Row check PASSED')
    else:
        print('Row check FAILED')
        raise SystemExit
    
    fight_stats_agg['fights'] = 1
    # Add the date to the fight stats
    fight_stats_agg = fight_stats_agg.merge(events[['event', 'date']], on = 'event', how = 'left')
    
    # Fix column formats
    fight_stats_agg['fighter'] = fight_stats_agg['fighter'].astype(str)
    
    # Remove duplicate fighters. Sort first to keep records without NAs
    fighters = fighters.sort_values(by = 'reach')
    fighters = fighters.drop_duplicates(subset = ['fighter'])
    fighters = fighters.sort_values(by = 'fighter')
    
    # Remove stats where fighter = '0'. These seem to be fights where there are no stats available so not useful
    fight_stats_agg = fight_stats_agg[fight_stats_agg['fighter'] != '0']
    
    ###########################################
    ## Fighter odds cleaning
    
    # Add upcoming to events df
    all_events = pd.concat([upcoming_events, events], axis = 0)
    # Add upcoming to fight stats for new events and bouts
    fight_stats_all = add_upcoming_to_fight_stats(fight_stats, upcoming_fight_details, events, upcoming_events)
    
    # List out the odds columns
    odds_columns = ['open_odds', 'closing_odds_low', 'closing_odds_high']
    
    # Convert the odds columns into numeric
    fighter_odds[odds_columns] = fighter_odds[odds_columns].apply(pd.to_numeric, errors = 'coerce')
    
    # Count how many rows are n/a for each event, focussing on open odds as they will most likely be used in the model
    fighter_odds_nas = fighter_odds[fighter_odds['event'].apply(lambda x: 'UFC' in str(x))].groupby('event').agg(
        date = ('date', 'first'),
        na_count = ('closing_odds_low', lambda x: x.isna().sum()),
        row_count = ('fighter', lambda x: x.notna().sum()))
    
    # Output this to check later to see which UFC events have NAs
    fighter_odds_nas[fighter_odds_nas['na_count'] > 0].to_csv('./data/checks_fighter_odds_nas.csv')
    
    # Now remove NAs from the data frame
    fighter_odds = fighter_odds[(fighter_odds['closing_odds_low'].notna()) & ~(fighter_odds['event'] == 'Future Events(')]
    
    # Clean the name columns. Not bothering with event because that column may not match the UFC events columns
    fighter_odds.loc[:, ['fighter', 'opponent']] = clean_text_columns(fighter_odds[['fighter', 'opponent']])
    
    # Add the correct event and bout to the odds data
    fighter_odds = add_event_and_bout_to_odds_data(fighter_odds, all_events, fight_stats_all)
    
    
    ###########################################
    ## Event odds cleaning
    
    event_odds_cols = odds_scrape_config['events_colnames']
    
    # # Convert odds to numeric
    # event_odds[event_odds_cols] = event_odds[event_odds_cols].apply(pd.to_numeric, errors = 'coerce')
    
    # Add event and bout to the data
    event_odds = add_event_and_bout_to_odds_data(event_odds, all_events, fight_stats_all, date_check = '2018-01-01')
    
    # Remove empty rows
    event_odds = event_odds.dropna(subset = event_odds_cols, how = 'all')
    event_odds = event_odds.dropna(subset = 'event')
    
    
    
    ###########################################
    ## Betfair odds cleaning
    
    # Replace fighter names with those that appear in stats where the dates match
    betfair_clean_fighter_names(betfair_odds, all_events, fight_stats_all, fuzz_threshold = 70, day_threshold = 2)
    
    # Change column name for matching
    betfair_odds.rename(columns = {'event_date': 'date'}, inplace = True)
        
    # Add on events and bouts
    betfair_odds = add_event_and_bout_to_odds_data(betfair_odds, all_events, fight_stats_all, date_check = '2017-01-01')
    
    ## Now clean the UPCOMING BOUTS
    # pdb.set_trace()
    # First get the bout information and sync to the stats data
    betfair_upcoming_odds = betfair_add_upcoming_bouts_details(betfair_upcoming_odds, upcoming_fight_details)
    
    
    ###########################################
    # Remove multiple fights on same date - some events had fighters fight multiple times. This messes up the matching and some functions so we'll get rid of them
    
    # Take the fight results data as the primary one
    fight_results = remove_duplicate_fighter_pairs(fight_results)
    
    # # Also remove fights where we have no stats # KEPT IN FOR ELO RANKS
    # fight_results = fight_results[(fight_results['event'].isin(fight_stats_agg['event'])) & (fight_results['bout'].isin(fight_stats_agg['bout']))]
    
    # Now remove those results from the other datasets
    fight_stats_agg = fight_stats_agg[(fight_stats_agg['event'].isin(fight_results['event'].tolist())) & (fight_stats_agg['bout'].isin(fight_results['bout'].tolist()))]
    
    # Remove odds but not for more recent events
    comparison_date = datetime.combine(date.today() - timedelta(days=5), datetime.min.time())
    
    fighter_odds = fighter_odds[((fighter_odds['event'].isin(fight_results['event'].tolist())) & (fighter_odds['bout'].isin(fight_results['bout'].tolist()))) | (fighter_odds['date'] > comparison_date)]
    
    event_odds = event_odds[((event_odds['event'].isin(fight_results['event'].tolist())) & (event_odds['bout'].isin(fight_results['bout'].tolist()))) | (event_odds['date'] > comparison_date)]
    
    ###########################################
    
    # Overwrite any outputs that need it
    return {
        'events': events,
        'upcoming_events': upcoming_events,
        'fighters': fighters,
        'upcoming_fight_details': upcoming_fight_details,
        'fight_results': fight_results,
        'fight_stats_agg': fight_stats_agg,
        'fighter_odds': fighter_odds,
        'event_odds': event_odds,
        'betfair_odds': betfair_odds,
        'fighter_names': fighter_names,
        'betfair_upcoming_odds': betfair_upcoming_odds
    }

    
    

def clean_all_raw_data_and_save(config):
    
    """
    This function takes all the raw scraped data and cleans them, then saves to file
    
    Returns: nothing
    
    """
    # pdb.set_trace()
        
    # Read in the data
    events = fp.ParquetFile('./data/' + config['event_details_file_name']).to_pandas()
    upcoming_events = fp.ParquetFile('./data/' + config['upcoming_event_details_file_name']).to_pandas()
    upcoming_fight_details = fp.ParquetFile('./data/' + config['upcoming_fight_details_file_name']).to_pandas()
    fighters = fp.ParquetFile('./data/' + config['fighter_tott_file_name']).to_pandas()
    fight_results = fp.ParquetFile('./data/' + config['fight_results_file_name']).to_pandas()
    fight_stats = fp.ParquetFile('./data/' + config['fight_stats_file_name']).to_pandas()
    fighter_odds = fp.ParquetFile('./data/' + config['fighter_odds_file_name']).to_pandas()
    event_odds = fp.ParquetFile('./data/' + config['event_odds_file_name']).to_pandas()
    betfair_odds = fp.ParquetFile('./data/' + config['betfair_odds_file_name']).to_pandas()
    betfair_upcoming_odds = fp.ParquetFile('./data/' + config['betfair_upcoming_odds_file_name']).to_pandas()
    
    # Data clearning
    cleaned_dict = clean_all_raw_data(events, upcoming_events, upcoming_fight_details, fighters, fight_results, fight_stats, fighter_odds, event_odds, betfair_odds, betfair_upcoming_odds)
    
    ###########################################
    
    # Save the outputs
    
    fp.write('./data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['events'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['upcoming_event_details_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['upcoming_events'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fighter_tott_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fighters'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['upcoming_fight_details_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['upcoming_fight_details'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fight_results_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fight_results'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fight_stats_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fight_stats_agg'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fighter_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['event_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['betfair_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['betfair_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['betfair_upcoming_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['betfair_upcoming_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fighter_names_file_name'],
             cleaned_dict['fighter_names'],
             compression = 'ZSTD')

    return None

# This function cleans infinite values and large numbers

def clean_inf_and_large(data):
    
    data = data.infer_objects(copy=False)  # Ensures data types remain unchanged
    data.replace([np.inf, -np.inf], [99999, -99999], inplace=True)
    
    # Replace large float values with 99999
    data[data.select_dtypes(include=[np.float64]) > 99999] = 99999
    data[data.select_dtypes(include=[np.float64]) < -99999] = -99999
    
    return data

def clean_target_column(target_column):
    """
    Cleans the target column from a text entry to a categorised, allowing consistent formatting across function

    Args:
        target_column (series): the input column
    
    Returns
        A series to be input as the new column

    """
    # pdb.set_trace()
    
    # Recode anything that isn't Wins or losses as other
    cleaned_target = np.where(target_column.isna(), np.nan, np.where((target_column == "D/D") | (target_column == "NC/NC"), "OTH", target_column))
    
    # Convert the target into a labelled column ready for multiclass modelling
    oe = OrdinalEncoder(categories = [['L/W', 'OTH', 'W/L']])
    cleaned_target = oe.fit_transform(np.array(cleaned_target).reshape(-1, 1)).astype(int).ravel()
    
    # # Stop if the classes aren't as expected. These are hardcoded in other functions so need to be in this order
    # stop_if_not(le.classes_.tolist() == ['L/W', 'OTH', 'W/L'], "Target class check")

    return cleaned_target

# This function cleans a text column. It's used for the matching columns to ensure they're all done in a consistent way
def clean_text_columns(df):
    
    # Loop through the columns. This is faster than .apply with lambda
    for col in df.columns:
        df.loc[:, col] = df[col].str.replace(' ', '')
        # Remove capitals and special characters
        df.loc[:, col] = df[col].str.lower()
        df.loc[:, col] = df[col].str.replace('-', '')
        df.loc[:, col] = df[col].str.replace("'", "")

        # If looking at bouts, need to put in a placeholder for vs. before getting rid of full stops
        if col in ['bout', 'BOUT']:
            df.loc[:, col] = df[col].str.replace('vs.', 'vs##')
        df.loc[:, col] = df[col].str.replace('.', '')
        if col in ['bout', 'BOUT']:
            df.loc[:, col] = df[col].str.replace('vs##', 'vs.')
        df.loc[:, col] = df[col].str.replace('saint', 'st')
        
        # Clean any names that are equivalent.
        # This is to get alignment between the fighter names on UFCstats/fighters and bout stats
        if col in ['fighter', 'FIGHTER']:
            df.loc[:, col] = df[col].str.replace('montserendon', 'montserratrendon')
            df.loc[:, col] = df[col].str.replace('michaelaswell', 'michaelaswelljr').str.replace('jrjr', 'jr')
            df.loc[:, col] = df[col].str.replace('zachreese', 'zacharyreese')
            df.loc[:, col] = df[col].str.replace('shaquemerock', 'shemrock')
            df.loc[:, col] = df[col].str.replace('patrickyfreire', 'patriciofreire')
            df.loc[:, col] = df[col].str.replace('kaikamakaiii', 'kaikamaka')
            
    return df

# This function takes the final dataset (after processing) and logs some key metrics so they can be tracked with each update

def log_final_data_stats(df, log_csv_path, config):
    
    # ## For debug
    # df = forecasting_data.copy()
    # log_csv_path = 
    
    # pdb.set_trace()
    
    # Define which key columns we want to track NAs for
    # For all these cols, don't need to put fighter1_ fighter2_ prefix, 'diff' as they will be added later
    na_cols = ['open_odds', 'best_closing_odds', 'mean_closing_odds', 'betfair_exch_odds', 'start_elo', 'reach', 'hist_times_knocked_out', 'hist_round3_body_landed_against_per_minute', 'hist_last_3_fights_round1_distance_accuracy_percent', 'hist_wins_per_fight', 'hist_opp_stance_round1_leg_accuracy_percent_against', 'hist_opp_type_allrounds_sig.str._landed_against',  'hist_last_3_fights_round1_distance_bayes_accuracy_percent', 'hist_opp_stance_round1_kd_per_minute_bayes_adjmean', 'hist_last_3_fights_round1_body_landed_against_per_minute_bayes_adjsum', 'date_minus_dob', 'subs_per_attempt']
    
    # Remove any cols to track if the col isn't in the data frame
    na_cols = [col for col in na_cols if ('fighter1_' + col) in df.columns]

    # Start a new data frame for the final data stats
    results_df = pd.DataFrame({'Date': [datetime.today()]}, columns = ['Date'])

    for fighter in ['fighter1_', 'fighter2_']:  
        
        # Choose only fights that are after validation checks date, and where we have a fight history to avoid NAs
        test_df = df[(df['date'] >= config['start_date_validation_checks']) & (df[fighter + 'hist_fights'] > 0)][[fighter + col for col in na_cols]]
    
        # Calculate the missing values
        missing_values_percentage = test_df.isna().sum() / len(test_df) * 100
        # Calculate column means (ignoring NaNs)
        column_means = test_df.mean(skipna = True)
        
        # Convert the result into a DataFrame and transpose
        results_df = pd.concat([
                results_df,
                pd.DataFrame([missing_values_percentage.tolist() + column_means.tolist()], 
                             columns = [col + '_NA_Percent' for col in missing_values_percentage.index.tolist()] + [col + '_Mean' for col in column_means.index.tolist()])
            ],
            axis = 1)

    # Add this onto the existing csv
    existing_results = pd.read_csv(log_csv_path)
    
    results_df = pd.concat([results_df, existing_results])
    
    # Now overwrite the results
    results_df.to_csv(log_csv_path, index=False)

    return None

# This is the master function for processing all the clean data, calculating variables and merging together

def merge_and_process_all_clean_data(
        config,
        elo_config,
        data_processing_config,
        odds_scrape_config,
        model_hyper_params,
        test_fighter_list = [],
        skip_save_data = False # whether to skip the save data and just return merged data
        ):
    
    # pdb.set_trace()

    # Read in the data
    events = fp.ParquetFile('./data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    upcoming_events = fp.ParquetFile('./data/' + config['upcoming_event_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    fighters = fp.ParquetFile('./data/' + config['fighter_tott_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    fight_results = fp.ParquetFile('./data/' + config['fight_results_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    upcoming_fight_details = fp.ParquetFile('./data/' + config['upcoming_fight_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    fight_stats = fp.ParquetFile('./data/' + config['fight_stats_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    fight_odds = fp.ParquetFile('./data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    event_odds = fp.ParquetFile('./data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    betfair_odds = fp.ParquetFile('./data/' + config['betfair_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    betfair_upcoming_odds = fp.ParquetFile('./data/' + config['betfair_upcoming_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()

    # Read in the average odds model
    average_odds_model = joblib.load('./results/average_odds_model.joblib')

    ###########################################################################################
    # UPCOMING

    # Add upcoming fights to fight stats
    fight_stats = add_upcoming_to_fight_stats(fight_stats, upcoming_fight_details, events, upcoming_events)

    # Add upcoming events
    events = pd.concat([upcoming_events, events], axis = 0)

    # Add the date to the upcoming fights
    upcoming_fight_details = upcoming_fight_details.merge(upcoming_events[['event', 'date']], on = 'event', how = 'left')

    # Add upcoming fights to the results with empty results
    fight_results = pd.concat([upcoming_fight_details.drop('url', axis = 1), fight_results])



    ###########################################################################################
    # CALCULATE ELOs

    fight_results = add_start_elo_to_fight_results(fight_results, fight_stats, config, elo_config)

    # Recode missing to the default starting. There will be some missing if their first fight was a DQ, D/D or NC.

    # Check there are no missing ELOs after the start date
    missing_elo_check = fight_results[fight_results['date'] > '2000-01-01'][pd.isna(fight_results['fighter1_start_elo'])]

    # Output to file before treating
    missing_elo_check.to_csv('./data/missing_elo_check.csv')

    fight_results['fighter1_start_elo'] = fight_results['fighter1_start_elo'].fillna(config['start_elo'])
    fight_results['fighter2_start_elo'] = fight_results['fighter2_start_elo'].fillna(config['start_elo'])


    ###########################################################################################
    # FIGHT STATS PROCESSING

    # Calculate the historical stats of each fighter at the fight date
    fight_stats = fight_stats.sort_values(by=['fighter', 'date'], axis = 0)
    
    # Create subs attempted and landed versions
    fight_stats.columns = fight_stats.columns.str.replace('_sub.att', '_sub_attempt')
    
    # Get submission success
    fight_stats = add_submissions_landed(fight_stats, fight_results)

    # Copy the fight stats but for against a fighter, then merge onto stats
    fight_stats_against = fight_stats.copy()
    fight_stats_against.rename(columns = lambda x: x + "_against" if "attempt" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "landed" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "ctrl_minutes" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "kd" in x else x, inplace = True)

    # Split the bout into fighters
    # Set the opponent to the fighter for merging
    fight_stats_against[['fighter1', 'fighter2']] = fight_stats_against['bout'].str.split('vs\\.', expand = True)
    fight_stats_against['fighter'] = np.where(fight_stats_against['fighter'] == fight_stats_against['fighter1'], fight_stats_against['fighter2'], fight_stats_against['fighter1'])
    # Get only the columns we want
    against_columns = [col for col in fight_stats_against.columns if "against" in col]
    fight_stats_against = fight_stats_against[['event', 'bout', 'fighter'] + against_columns]

    # Merge onto the fight stats
    rows_before = len(fight_stats)
    fight_stats = fight_stats.merge(fight_stats_against, on = ['event', 'bout', 'fighter'], how = 'left')

    # Check that all the rows are correct
    if rows_before == len(fight_stats):
        print('Row check PASSED')
    else:
        print('Row check FAILED')
        raise SystemExit
    del fight_stats_against

    ###########################################################################################
    # Get the columns of stats
    
    if data_processing_config['add_bayes_percentage_columns'] or data_processing_config['add_bayes_per_minute_columns']:
        
        # FIrst calculate the Bayesian priors that will be used for Bayesian smoothing
        fight_stats_bayes_priors = bayesian_priors(
            fight_stats.sort_values('date').reset_index(drop = True).merge(fight_results[['event', 'bout', 'weightclass']], on = ['event', 'bout'], how = 'left'), # Join on the weightclass for grouping
            id_cols = ['event', 'bout', 'fighter', 'date'], 
            group_by = data_processing_config['bayesian_groupby'], 
            fight_span = data_processing_config['bayes_fightspan'],
            ignore_last_n_fights = 30 if data_processing_config['bayesian_groupby'] == [] else 10
        )
        
        # Use constant Bayesian priors if required
        if data_processing_config['constant_bayes_priors'] == True:
            
            # Get the single row for constant bayes using the max date
            max_date = fight_stats_bayes_priors[fight_stats_bayes_priors['date'] < pd.to_datetime(data_processing_config['constant_bayes_priors_date'])]['date'].max()
            constant_bayes = fight_stats_bayes_priors[fight_stats_bayes_priors['date'] == max_date].head(1)
            
            # Now set these as the bayes stats for everyone
            fight_stats_bayes_priors = fight_stats_bayes_priors[['event', 'bout']].merge(
                constant_bayes.drop(columns = ['event', 'bout', 'date']), how = 'cross')
        else:
            fight_stats_bayes_priors.drop(columns = ['date'], inplace = True)
        
    else:
        fight_stats_bayes_priors = None

    # Now aggrgate
    fight_stats_hist = aggregate_last_X_fight_data(
        fight_stats,
        id_cols = ['date', 'event', 'bout', 'fighter'],
        group_by = ['fighter'],
        prefix = "hist_",
        X = 100,
        per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date'])

    # Check that all first fights are 0 stats
    stat_columns = [col for col in fight_stats.columns if col not in ['date', 'event', 'bout', 'fighter']]
    fight_stats_hist_check = fight_stats_hist[fight_stats_hist['hist_fights'] == 0]
    if fight_stats_hist_check[['hist_' + s for s in stat_columns]].max().max() == 0:
        print('Cumulative aggregation check PASSED')
    else:
        print('Cumulative aggregation check FAILED')
        raise SystemExit
    del fight_stats_hist_check



    ###########################################################################################
    # ADJUSTED STATISTICS
    
    if data_processing_config['add_bayes_per_minute_columns'] and data_processing_config['add_adj_per_minute_bayes_columns']:
        
        fight_stats_bayes = calculate_per_minute_bayesian_stats(fight_stats, fight_stats_bayes_priors, excluded_strings = ['percent', 'fight_length', 'fight_minute', 'ratio', 'per_minute'], bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute'])
        
        # Adding on adjusted stats to the fight stats
        fight_stats_plus_adj = add_adjusted_stats(fight_stats_bayes, fight_stats_hist, columns_with = ['per_minute_bayes'], exclude_columns_with = [])
        
        # Replace the fight stats with one with the adj but remove all the extra bayes info (will be added by aggregation later)
        fight_stats = fight_stats_plus_adj[[col for col in fight_stats_plus_adj if 'bayes' not in col or ('bayes' in col and 'adj' in col)]]
    
    if data_processing_config['add_adj_per_minute_columns']:
        fight_stats = add_adjusted_stats(fight_stats, fight_stats_hist, columns_with = ['per_minute'], exclude_columns_with = ['bayes'])
    
    
    ###########################################################################################
    # Calculate accuracy percentages
    
    fight_stats = calculate_adjusted_percentages(
            fight_stats, 
            fight_stats_hist, 
            fight_stats_bayes_priors,
            data_processing_config)
    
    # Only keep the adj percentage columns. We don't want to aggregate the rest, just calc later from agg counts
    fight_stats = fight_stats[[col for col in fight_stats.columns if not ('accuracy_percent' in col and '_adj' not in col)]]
    
    ###########################################################################################
    # Reaggregate the data
    fight_stats_hist = aggregate_last_X_fight_data(
        fight_stats,
        id_cols = ['date', 'event', 'bout', 'fighter'],
        group_by = ['fighter'],
        prefix = "hist_",
        X = 100,
        per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date'])


    ###########################################################################################
    # Calculate average fight length, win length and loss length
    fight_stats_hist['hist_fight_length_minutes_avg'] = np.where(fight_stats_hist['hist_fights'] == 0, np.nan, fight_stats_hist['hist_allrounds_fight_minutes'] / fight_stats_hist['hist_fights'])

    stop_if_not(len(fight_stats) == len(fight_stats_hist), "Agg stats check")
        
        
    ###########################################################################################
    # Get the stats of the last 3 fights only
    fight_stats_hist_last_3 = aggregate_last_X_fight_data(
        fight_stats,
        id_cols = ['date', 'event', 'bout', 'fighter'],
        group_by = ['fighter'],
        prefix = "hist_last_3_fights_",
        X = 3,
        per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date'])

    # Check that all first fights are NaN
    stat_columns = [col for col in fight_stats_hist_last_3.columns if col not in ['date', 'event', 'bout', 'fighter']]
    fight_stats_hist_last_3_check = fight_stats_hist_last_3[fight_stats_hist_last_3['hist_last_3_fights_fights'] == 0]

    if np.isnan(fight_stats_hist_last_3_check[[s for s in stat_columns if s != 'hist_last_3_fights_fights']].max(skipna = True).max(skipna = True)):
        print('Cumulative aggregation check PASSED')
    else:
        print('Cumulative aggregation check FAILED')
        raise SystemExit
    del fight_stats_hist_last_3_check

    # fight_stats_hist_last_2 = aggregate_last_X_fight_data(
    #     fight_stats,
    #     id_cols = ['date', 'event', 'bout', 'fighter'],
    #     group_by = ['fighter'],
    #     prefix = "hist_last_2_fights_",
    #     X = 2,
    # per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
    #     per_minute_calc = True, 
    #     bayes_priors_df = fight_stats_bayes_priors,
    #     bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    # ).drop(columns = ['date'])
    
    # fight_stats_hist_last_1 = aggregate_last_X_fight_data(
    #     fight_stats,
    #     id_cols = ['date', 'event', 'bout', 'fighter'],
    #     group_by = ['fighter'],
    #     prefix = "hist_last_1_fights_",
    #     X = 1,
    # per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
    #     per_minute_calc = True, 
    #     bayes_priors_df = fight_stats_bayes_priors,
    #     bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    # ).drop(columns = ['date'])
    

    ###########################################################################################
    # Get the fight stats of each fighter against only specific types of fighters
    
    fight_stats = add_opponent_column(fight_stats.copy())
    
    # Get the historic records of each fighter for each opponent stance
    fight_stats_hist_opponent_stance = calculate_hist_opp_stance_stats(fight_stats, fighters, fight_stats_bayes_priors, data_processing_config)


    ###########################################################################################
    # Calculate per minute stats for the many hist datasets

    # Check all the row numbers are correct
    stop_if_not(
        len(fight_stats) == len(fight_stats_hist_opponent_stance) == len(fight_stats_hist_last_3),
        "All hist stats data length check")

    


    ###########################################################################################
    # Restructure the result data into fighter level i.e. duplicate for each fighter then concat

    temp_df = pd.DataFrame()
    temp_df[['fighter1_result', 'fighter2_result']] = fight_results['outcome'].str.split('/', expand = True)
    fighter1 = fight_results.copy().drop(['fighter1_start_elo', 'fighter2_start_elo'], axis = 1)
    fighter1['result'] = temp_df['fighter1_result']
    fighter1['fighter'] = fight_results['fighter1']
    fighter2 = fight_results.copy().drop(['fighter1_start_elo', 'fighter2_start_elo'], axis = 1)
    fighter2['result'] = temp_df['fighter2_result']
    del(temp_df)
    fighter2['fighter'] = fight_results['fighter2']
    fighter1.drop(columns = ['fighter1', 'fighter2', 'outcome'], inplace = True)
    fighter2.drop(columns = ['fighter1', 'fighter2', 'outcome'], inplace = True)
    fighter_results = pd.concat([fighter1, fighter2], ignore_index = True)
    del([fighter1, fighter2])
    fighter_results.sort_values(by = ['fighter', 'date'], inplace = True)

    # Change NC result to be one letter for the sake of strings of results later
    fighter_results['result'] = np.where(fighter_results['result'] == 'NC', 'N', fighter_results['result'])

    # Check we have the correct number of rows
    if len(fighter_results) == 2 * len(fight_results):
        print('Row number check PASSED')
    else:
        print('Merge row check FAILED')
        raise SystemExit

    ###########################################################################################
    # Calculate number of previous wins/losses - NONE OF THESE HAD STRONG ENOUGH PREDICTIVE POWER

    fighter_results = calculate_win_loss_stats(fighter_results)

    # Check that the win and loss finshes make sense
    check = fighter_results[(fighter_results['hist_wins'] < fighter_results['hist_wins_finishes']) | (fighter_results['hist_losses'] < fighter_results['hist_losses_finishes'])]

    stop_if_not(len(check) == 0, "Hist finishes and wins/losses check")

    ###########################################################################################
    # Record the last 10 results
    for n in list(range(1, 11)):
        fighter_results['result_' + str(n) +'_fights_prior'] = fighter_results.groupby(['fighter'])['result'].shift(n).fillna('')
        if n == 2:
            fighter_results['last_' + str(n) + '_results'] = fighter_results['result_' + str(n) +'_fights_prior'] + fighter_results['result_' + str(n - 1) +'_fights_prior']
        elif n > 2:
            fighter_results['last_' + str(n) + '_results'] = fighter_results['result_' + str(n) +'_fights_prior'] + fighter_results['last_' + str(n - 1) + '_results']
        else:
            pass

    ###########################################################################################
    # Get the fight stats of each fighter against only specific types of fighters

    # First define each fighter type
    fight_stats_hist['fighter_type'] = define_fighter_type(fight_stats_hist, fighter_results, config)

    # Add this fighter type to the stats data
    fight_stats = fight_stats.merge(fight_stats_hist[['event', 'bout', 'fighter', 'fighter_type']], on = ['event', 'bout', 'fighter'], how = 'left')

    # Now create a dataset with only opponents of that type
    types = fight_stats_hist['fighter_type'].str.replace(' ', '').unique().tolist()
    types = [type for type in types if type is not None]

    # Merge on opponent type
    fight_stats['opponent_type'] = fight_stats[['event', 'bout', 'opponent']].merge(
        fight_stats_hist[['event', 'bout', 'fighter', 'fighter_type']], 
        left_on = ['event', 'bout', 'opponent'], 
        right_on = ['event', 'bout', 'fighter'],
        how = 'left')['fighter_type']

    # Get the historic records of each fighter for each opponent type
    fight_stats_hist_opponent_type = aggregate_last_X_fight_data(
        fight_stats,
        id_cols = ['date', 'event', 'bout', 'fighter', 'opponent', 'fighter_type', 'opponent_type'],
        group_by = ['fighter', 'opponent_type'],
        prefix = 'hist_opp_type_',
        X = 100,
        per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date', 'opponent', 'fighter_type', 'opponent_type'])


    # Check that we have the correct number of rows
    stop_if_not(len(fight_stats_hist_opponent_type) == len(fight_stats_hist), 'Opponent type row count check')

    # test = fight_stats_hist_opponent_type[fight_stats_hist_opponent_type['fighter']=='ConorMcGregor'][['bout', 'fighter', 'opponent', 'opponent_type', 'hist_opp_type_allrounds_fight_minutes']]
    
    ###########################################################################################
    # Calculate the date was last knocked out
    
    dates_knocked_out = calculate_knocked_out_stats(fighter_results)

    ###########################################################################################
    # Calculate win and loss streaks
    fighter_results['hist_win_streak'] = 0
    fighter_results['hist_loss_streak'] = 0
    for streak in list(range(1, 11)):
        fighter_results['hist_win_streak'] = np.where(fighter_results['last_10_results'].str[-streak:] == ('W' * streak),
                                                      streak,
                                                      fighter_results['hist_win_streak'])
        fighter_results['hist_loss_streak'] = np.where(fighter_results['last_10_results'].str[-streak:] == ('L' * streak),
                                                       streak,
                                                       fighter_results['hist_loss_streak'])

    fighter_results['hist_win_streak'].value_counts()
    # Check the fighters that have been on win streaks of 10
    fighter_results[fighter_results['hist_win_streak'] == 10]['fighter'].unique()
    fighter_results[fighter_results['hist_loss_streak'] == 5]['fighter'].unique()

    # Calculate results by method
    methods = list(set(fighter_results['method'].dropna()))
    row_count = len(fighter_results)
    for result in ['W', 'L']:
        for method in methods:
            # First calculate a simple flag for whether the win or loss was by that method
            fighter_results['hist_' + result + '_' + method] = np.where((fighter_results['method'] == method) & (fighter_results['result'] == result),                                                                   1, 0)
            # Then sum these for previous fights ignoring the current one
            fighter_results['hist_' + result + '_' + method] = fighter_results.groupby(['fighter'])['hist_' + result + '_' + method].cumsum() - fighter_results['hist_' + result + '_' + method]

    # Cleanup columns not needed
    fighter_results = fighter_results[[col for col in fighter_results.columns if not (('last_' in col and '_result' in col) or 'fights_prior' in col)]]


    # Check row count before and after merge
    if len(fighter_results) == row_count:
        print('Row number check PASSED')
    else:
        print('Merge row check FAILED')
        raise SystemExit
    del(row_count)

    ###########################################################################################
    # Remove result as that isn't historical
    fighter_results.drop(columns = ['result', 'method', 'weightclass', 'round', 'rounds', 'time format', 'date'], inplace = True)

        

    # Add on the date of the last fight
    fight_stats['date_last_fight'] = fight_stats.groupby(['fighter'])['date'].shift(1)

    # Date check
    fight_stats_no_na = fight_stats[fight_stats['date_last_fight'].notna() & (fight_stats['date_last_fight'] > fight_stats['date'])]
    if len(fight_stats_no_na) == 0:
        print("Date of last fight check PASSED")
    else:
        print("Date of last fight check FAILED")
        raise SystemExit
    del(fight_stats_no_na)

    ###########################################################################################
    # FIGHT RESULTS PROCESSING

    # Filter for fight formats to be included
    fight_results = fight_results[((fight_results['time format'].isna()) | (fight_results['time format'].isin(config['time_format_filter']))) & fight_results['weightclass'].isin(config['weightclass_filter'])]

    # # Add elevation data to fight results
    # events['fight_elevation'] = get_elevation_series(events['location'], config['google_maps_api_key'])

    # # Merge onto the fight results
    # fight_results = fight_results.merge(
    #     events[['event', 'fight_elevation']],
    #     on = 'event',
    #     how = 'left'
    #     )

    # # Check there are no missing elevations
    # stop_if_not(fight_results['fight_elevation'].isna().sum() == 0, "Event elevation missing data check")

    ###########################################################################################
    # FIGHT ODDS PROCESSING

    # Fix the events and bouts using data including upcoming events. Doing this again after done in cleaning because events and stats been edited
    fight_odds = add_event_and_bout_to_odds_data(fight_odds, events, fight_stats)
    event_odds = add_event_and_bout_to_odds_data(event_odds, events, fight_stats, date_check = '2018-01-01')

    # Add opponent to the upcoming odds. It already has the event and bout matched on.
    if len(betfair_upcoming_odds) > 0:
        add_opponent_column(betfair_upcoming_odds)

        # Combine the upcoming odds with the historic odds
        betfair_odds = pd.concat([betfair_odds, betfair_upcoming_odds], axis = 0)

    fight_odds = process_odds_data(fight_odds.drop_duplicates(subset = ['event', 'bout', 'fighter']), event_odds, betfair_odds, odds_scrape_config, average_odds_model)


    ###########################################################################################
    # MERGE DATA

    merged_data = merge_df_both_fighters(
        left_df =  fight_results[['event', 'bout', 'fighter1', 'fighter2', 'outcome', 'method', 'weightclass', 'date', 'time format', 'fighter1_start_elo', 'fighter2_start_elo']],
        df_list = [
            fight_stats[['event', 'bout', 'fighter', 'date_last_fight']],
            dates_knocked_out,
            fight_stats_hist,
            # fight_stats_hist_last_1,
            # fight_stats_hist_last_2,
            fight_stats_hist_last_3,
            fight_stats_hist_opponent_stance,
            fight_stats_hist_opponent_type,
            fighter_results,
            fight_odds
        ]
    ).reset_index(drop = True)

    # Save rows for checks later
    merged_original_rows = len(merged_data)
    
    # Merge on the fighter information
    merged_data = merged_data.merge(fighters.add_prefix('fighter1_'), 
                                    left_on = ['fighter1'], 
                                    right_on = ['fighter1_fighter'],
                                    how = 'left')
    merged_data = merged_data.merge(fighters.add_prefix('fighter2_'), 
                                    left_on = ['fighter2'], 
                                    right_on = ['fighter2_fighter'],
                                    how = 'left')

    merged_data.drop(columns = ['fighter2_fighter', 'fighter1_fighter'], inplace = True)

    # Check for no duplicate columns that have been merged
    if len([col for col in merged_data.columns if "_x" in col]) == 0:
        print('Duplicate column check PASSED')
    else:
        duplicates = [col for col in merged_data.columns if "_x" in col]
        print(f'Duplicate column name check FAILED: {duplicates}')
        raise SystemExit

    # Check we have the right number of rows
    if len(merged_data) == len(fight_results):
        print("Merged data rows check PASSED")
    else:
        print("Merged data rows check FAILED")
        raise SystemExit

    # Check for duplicate column names
    duplicate_columns = merged_data.columns[merged_data.columns.duplicated()]

    # Cleanup
    del betfair_odds, fight_results, fight_stats, dates_knocked_out, fight_stats_hist, fight_stats_hist_last_3, fight_stats_hist_opponent_stance, fight_stats_hist_opponent_type,  fighter_results, fight_odds, events, event_odds, fighters
                                        
    gc.collect();
    # pdb.set_trace()

    ###########################################################################################
    # FURTHER FEATURE PROCESSING

    # Calculate all true odds
    merged_data = calculate_all_true_odds_columns(merged_data, impute_missing_betfair = data_processing_config['impute_missing_betfair'], odds_suffix_for_true_odds = data_processing_config['odds_suffix_for_true_odds'], impute_one_sided_missing_odds = data_processing_config['impute_one_sided_missing_odds'])

    # Fix model hyperparams where necessary to account for imputation being used
    if (data_processing_config['impute_missing_betfair'] == True) & (model_hyper_params['odds_to_use'] == 'betfair_exch_odds'):
        model_hyper_params['odds_to_use'] = 'betfair_exch_odds_imputed'

    # Create a QA dataset showing where Betfair odds and scraped odds are showing large differences
    odds_qa_df = merged_data[((
        merged_data['fighter1_betfair_exch_odds'] < 1.9) & (merged_data['fighter1_best_closing_odds'] > 2.1)) | ((
            merged_data['fighter1_betfair_exch_odds'] > 2.1) & (merged_data['fighter1_best_closing_odds'] < 1.9)) | ((
                merged_data['fighter1_betfair_exch_odds'] < 1.9) & (merged_data['fighter2_betfair_exch_odds'] < 1.9))][['fighter1', 'fighter2', 'date', 'fighter1_betfair_exch_odds', 'fighter1_best_closing_odds', 'fighter2_betfair_exch_odds', 'fighter2_best_closing_odds']]

    odds_qa_df.to_csv('./data/qa_merged_betfair_odds_out_of_sync.csv')   
                
    ###########################################################################################
    # PERCENTAGE CALCULATIONS - MOVED TO BEFORE MERGE
    
    if data_processing_config['add_bayes_percentage_columns']:
        merged_data = calculate_percentages(merged_data, fight_stats_bayes_priors, bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_percent'])
        
        # # Calculate bayes adj accuracy if required
        # if data_processing_config['add_adj_percentage_bayes_columns']:
        #     merged_data = add_()
    else:
        merged_data = calculate_percentages(merged_data, fight_stats_bayes_priors = None, bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_percent'])

    ###########################################################################################
    # RATIO CALCULATIONS

    # Get the columns for attempted strikes but not against
    head_columns = [col for col in merged_data.columns if '_head_attempt' in col and 'against' not in col and 'per_minute' not in col]
    body_columns = [col for col in merged_data.columns if 'body_attempt' in col and 'against' not in col and 'per_minute' not in col]
    leg_columns = [col for col in merged_data.columns if 'leg_attempt' in col and 'against' not in col and 'per_minute' not in col]

    # Do checks on columns
    stop_if_not(check_identical_excl_string(head_columns, body_columns, 'head', 'body'))
    stop_if_not(check_identical_excl_string(head_columns, leg_columns, 'head', 'leg'))

    # Calculate ratios of attempts on body and leg
    head_body_ratios = pd.DataFrame(
        safe_divide_np(merged_data[head_columns].to_numpy(), merged_data[body_columns].to_numpy(), zero_zero_fill = np.nan),
        columns = [col.replace('head_attempt', 'head_body_attempt_ratio') for col in head_columns])

    merged_data = pd.concat([merged_data, head_body_ratios], axis = 1)

    head_leg_ratios = pd.DataFrame(
        safe_divide_np(merged_data[head_columns].to_numpy(), merged_data[leg_columns].to_numpy(), zero_zero_fill = np.nan),
        columns = [col.replace('head_attempt', 'head_leg_attempt_ratio') for col in head_columns])

    merged_data = pd.concat([merged_data, head_leg_ratios], axis = 1)

    del(head_body_ratios, head_leg_ratios, head_columns, body_columns, leg_columns)

    # Do a check to see if row number has stayed the same
    stop_if_not(len(merged_data) == merged_original_rows)

    # Calculate ratio of striking to takedowns by looking at distance striking compared with td attempts, and distance compared with ground and pound
    for fighter in ['fighter1', 'fighter2']:
        
        merged_data[fighter + '_hist_total_distance_strikes_over_td_attempts'] = safe_divide_np(
            merged_data[fighter + '_hist_allrounds_distance_attempt'].to_numpy(), 
            merged_data[fighter + '_hist_allrounds_td_attempt'].to_numpy(), 
            zero_zero_fill = np.nan)
        
        merged_data[fighter + '_hist_total_distance_over_ground_strikes'] = safe_divide_np(
            merged_data[fighter + '_hist_allrounds_distance_attempt'].to_numpy(),
            merged_data[fighter + '_hist_allrounds_ground_attempt'].to_numpy(),
            zero_zero_fill = np.nan)

    gc.collect();

    ###########################################################################################
    # ROUND DIFFERENTIAL CALCULATIONS

    # Look at the changes in per minute striking from round to round to gauge some measure of stamina and accuracy
    for fighter in ['fighter1', 'fighter2']:
        for round in range(2, 6):
            merged_data[fighter + '_round' + str(round) + 'to' + str(round - 1) + '_total_strike_per_minute_ratio'] = \
            merged_data[fighter + '_hist_round' + str(round) + '_total str._attempt_per_minute'] / merged_data[fighter + '_hist_round' + str(round - 1) + '_total str._attempt_per_minute']

            merged_data[fighter + '_round' + str(round) + 'to' + str(round - 1) + '_total_strike_accuracy_ratio'] = \
            merged_data[fighter + '_hist_round' + str(round) + '_total str._accuracy_percent'] / merged_data[fighter + '_hist_round' + str(round - 1) + '_total str._accuracy_percent']
            
            merged_data[fighter + '_last_3_fights_round' + str(round) + 'to' + str(round - 1) + '_total_strike_per_minute_ratio'] = \
            merged_data[fighter + '_hist_last_3_fights_round' + str(round) + '_total str._attempt_per_minute'] / merged_data[fighter + '_hist_last_3_fights_round' + str(round - 1) + '_total str._attempt_per_minute']

            merged_data[fighter + '_last_3_fights_round' + str(round) + 'to' + str(round - 1) + '_total_strike_accuracy_ratio'] = \
            merged_data[fighter + '_hist_last_3_fights_round' + str(round) + '_total str._accuracy_percent'] / merged_data[fighter + '_hist_last_3_fights_round' + str(round - 1) + '_total str._accuracy_percent']
            
    # Do a check to see if row number has stayed the same
    stop_if_not(len(merged_data) == merged_original_rows)

    ###########################################################################################
    # PER FIGHT CALCULATIONS

    # To see what columns are left
    # merged_data[[col for col in merged_data.columns if '_attempt' not in col and '_landed' not in col and '_percent' not in col and '_minutes' not in col]].columns.tolist()

    numerator_strings = ['times_knocked_out']

    # Do each fighters per fight calcs separately
    for fighter in ['fighter1', 'fighter2']:
        # Get the column names
        cols = []
        for i in range(0, len(numerator_strings)):
            cols = cols + [col for col in merged_data.columns if fighter in col and numerator_strings[i] in col]
        new_data = merged_data[cols].div(merged_data[fighter + '_hist_fights'], axis = 0)
        # Deal with any infinities
        new_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        new_data.columns = [col + '_per_fight' for col in cols]
        
        # Merge with the data
        merged_data = pd.concat([merged_data, new_data], axis = 1)

    del([new_data, cols, numerator_strings])

    ###########################################################################################
    # OTHER 'PER' CALCULATIONS

    # This is a set of custom features to identify what kind of fighter someone is

    for fighter in ['fighter1', 'fighter2']:
        # Strikes per TD attempt
        merged_data[fighter + '_sig.str._per_td_attempt'] = merged_data[fighter + '_hist_allrounds_sig.str._attempt'] / merged_data[fighter + '_hist_allrounds_td_attempt']
        # Strikes per sub attempt
        merged_data[fighter + '_sig.str.per_sub_attempt'] = merged_data[fighter + '_hist_allrounds_sig.str._attempt'] / merged_data[fighter + '_hist_allrounds_sub_attempt']
        # Knockdowns per strike landed
        merged_data[fighter + '_kd_per_sig.str._landed'] = merged_data[fighter + '_hist_allrounds_kd'] / merged_data[fighter + '_hist_allrounds_sig.str._landed']
        # Sub attempt per td
        merged_data[fighter + '_sub_attempt.per_td_landed'] = merged_data[fighter + '_hist_allrounds_sub_attempt'] / merged_data[fighter + '_hist_allrounds_td_landed']
            
        # Subs per attempt
        merged_data[fighter + '_subs_per_attempt'] = merged_data[fighter + '_hist_wins_Submission'] / merged_data[fighter + '_hist_allrounds_sub_attempt']
        
        # Recode Infs as missing, because could give misleading results
        new_columns = [fighter + '_sig.str._per_td_attempt', fighter + '_sig.str.per_sub_attempt', fighter + '_kd_per_sig.str._landed', fighter + '_sub_attempt.per_td_landed', fighter + '_subs_per_attempt']
        
        merged_data[new_columns] = merged_data[new_columns].fillna(-np.inf)
        # merged_data[new_columns] = merged_data[new_columns].replace([-np.inf, np.inf], np.nans)


    ###########################################################################################
    # TIME SINCE CALCULATIONS

    # Calculate days since last KO, Win, Loss, Fight
    # Set the columns to be taken from date
    cols = ['_date_last_fight', '_date_last_knocked_out', '_date_last_loss', '_date_last_win', '_dob']
    # Do each fighters per fight calcs separately
    for fighter in ['fighter1', 'fighter2']:
        for col in cols:
            merged_data[fighter + '_date_minus' + col] = (merged_data['date'] - pd.to_datetime(merged_data[fighter + col]).copy()).dt.days

            # Fill in NAs with arbitrary early date
            # merged_data[fighter + '_date_minus' + col] = merged_data[fighter + '_date_minus' + col].fillna(99999).astype(int)

    # pd.DataFrame(merged_data.columns).to_csv("te.csv")
    ###########################################################################################
    # FIGHTER DIFFERENTIAL CALCULATIONS
    
    merged_data = add_fighter_differentials(merged_data)
    
    gc.collect();

    ###########################################################################################
    # Formatting tweaks
    # Remove any white spaces in names
    merged_data.columns = merged_data.columns.str.replace(" ", "_")

    # Add a womens flag
    merged_data['womens'] = np.where(merged_data['weightclass'].apply(lambda x: np.where('omen' in x, True, False)), 1, 0)

    # Add the matchup
    merged_data['matchup'] = merged_data['fighter1_fighter_type'] + '-vs-' + merged_data['fighter2_fighter_type']


    # Add what odds are to be used in model training and backtesting
    merged_data['fighter1_backtesting_odds'] = merged_data['fighter1_' + model_hyper_params['odds_to_use']].copy()
    merged_data['fighter2_backtesting_odds'] = merged_data['fighter2_' + model_hyper_params['odds_to_use']].copy()

    # Add on odds to be used for base
    merged_data['notformodel_wl_base_margin'] = merged_data['fighter1_' + data_processing_config['odds_suffix_for_base_margin']].copy()
    merged_data['notformodel_lw_base_margin'] = merged_data['fighter2_' + data_processing_config['odds_suffix_for_base_margin']].copy()
    

    ###########################################################################################
    # FLAG TRAIN/TEST/OOT
    # pdb.set_trace()
    np.random.seed(seed = config['seed_all'] + 715)
    merged_data = flag_train_test_oot(merged_data, config, test_fighter_list)



    ###########################################################################################
    # OUTPUT DATA

    # Remove columns not needed, or that are all na
    cols_to_remove = [col for col in merged_data.columns if any(substring in col for substring in ['level_0'])]
    # cols_to_remove.append('date')

    merged_data.drop(columns = cols_to_remove, inplace = True)

    # # Drop any columns that are all NA. Commented out as it affects
    # merged_data.dropna(axis = 1, how = 'all', inplace = True)

    # print(merged_data.columns)
    merged_data.reset_index(drop = True, inplace = True)
    merged_data.drop(columns = cols_to_remove, inplace = True, errors = 'ignore')
    del(cols_to_remove)
    
    if not skip_save_data:
        
        # Save the log of the final data
        log_final_data_stats(merged_data, config['final_data_stats_log'], config)
        
        # Save the final data
        # split_by_year_and_save_fastparquet(merged_data, 'date', './data/final_dataset/', compression='ZSTD')
        fp.write('./data/final_dataset/final_dataset.parquet', merged_data, compression = 'ZSTD')
        
        # Output a single fight for checking purposes
        merged_data[(merged_data['fighter1'] == 'khabibnurmagomedov') & (merged_data['fighter2'] == 'conormcgregor')].T.to_csv('./data/final_dataset_checking.csv', index=True)
        
        merged_data[(merged_data['fighter1'] == 'antoniosilva') | (merged_data['fighter2'] == 'antoniosilva')].T.to_csv('./data/final_dataset_fighter_checking.csv')
    
    return merged_data

# This function applies filters to the data for records we aren't interested in modelling and testing.

def standard_data_filter(df, include_upcoming = False, keep_rows_with_flag = False):

    # pdb.set_trace()    

    # Create an empty flag to exception reasons
    df['notformodel_exclusion_reason'] = ''
    
    # Remove any unrealistic odds
    df.loc[(df['fighter1_true_closing_odds'] >= 15) | (df['fighter2_true_closing_odds'] >= 15), 'notformodel_exclusion_reason'] = 'Odds out of range'
    
    df.loc[(df['fighter1_hist_fights'] <= 2) | (df['fighter2_hist_fights'] <= 2), 'notformodel_exclusion_reason'] = 'Fights < 3'

    # Find anyone with missing true odds
    df.loc[(df['fighter1_true_closing_odds'].isna()) | (df['fighter2_true_closing_odds'].isna()), 'notformodel_exclusion_reason'] = 'Missing odds'

    df.loc[(df['womens'] == 1), 'notformodel_exclusion_reason'] = 'Womens'

    if not keep_rows_with_flag:
        df = df[df['notformodel_exclusion_reason'] == ''].drop(columns = ['notformodel_exclusion_reason']).copy()

    # filtered_df = df[(df['womens'] == 0) & 
    #                  (df['fighter1_hist_fights'] > 2) &
    #                  (df['fighter2_hist_fights'] > 2)]
    # # Remove any data were odds are missing
    # filtered_df = filtered_df[(~filtered_df['fighter1_true_closing_odds'].isna()) & (~filtered_df['fighter2_true_closing_odds'].isna())]
    
    # # Remove any unrealistic odds
    # filtered_df = filtered_df[(filtered_df['fighter1_true_closing_odds'] < 15) & (filtered_df['fighter2_true_closing_odds'] < 15)]
    
    if not include_upcoming:
        df = df[(~df['outcome'].isna())]
        
    return df

def undo_clean_inf(df, threshold=99999):
    """
    Replaces all values >= threshold with np.inf and all values <= -threshold with -np.inf.
    Operates directly on numeric columns for maximum speed.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        threshold (int, optional): The threshold value. Defaults to 99999.

    Returns:
        pd.DataFrame: A modified DataFrame with replacements.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Ignore dates
    numeric_cols = [col for col in numeric_cols if ('date' not in col and 'dob' not in col) or 'diff' in col]
    
    arr = df[numeric_cols].values  # Convert to NumPy array for speed

    # Vectorized replacement
    arr[arr >= threshold] = np.inf
    arr[arr <= -threshold] = -np.inf

    df[numeric_cols] = arr  # Assign back to DataFrame
    return df

def update_data_summary(results_csv_path, data_frames, summary_date, save_output = False):
    """
    Updates the results dataframe with summary statistics of the provided dataframes.
    Each date will be a column with the summary for each dataframe.

    Parameters:
    - results_df (pd.DataFrame): The dataframe where the summary will be added.
    - data_frames (dict): A dictionary of dataframe names as keys and the dataframes as values.
    - summary_dates (list): A list of dates for each summary to be logged. Example: ['2024-12-09', '2024-12-10']

    Returns:
    - pd.DataFrame: The updated results dataframe with new summary columns.
    """
    
    # pdb.set_trace()
    
    results_df = pd.read_csv(results_csv_path, index_col = 0)
    
    # Create an empty list to store the summary data for each dataframe
    summary_data = []
    
    for df_name, df in data_frames.items():
        # Get the number of rows and columns
        num_rows, num_cols = df.shape
        
        # Count the number of NaNs in each column
        na_count = df.isna().sum()
        
        # Create a dictionary for the summary of this dataframe
        summary_row = {
            'date': summary_date,
            'dataframe': df_name,
            'rows': num_rows,
            'columns': num_cols,
            **na_count.to_dict()  # Add NaN counts as columns
        }
        
        # Append the summary row to the summary_data list
        summary_data.append(summary_row)
    
    # Convert the summary data to a DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Ensure the dates all correct
    summary_df['date'] = summary_date
    
    # Append the new summary to the results_df
    updated_results_df = pd.concat([results_df, summary_df], ignore_index=True)

    # Save to file before outputting
    if save_output:
        updated_results_df.to_csv(results_csv_path, index = True)

    return updated_results_df
