# This function takes existing odds data and a list of fighters and adds the new fights to the data. 
# It requires a list of dates for the events as well as the list of fighters that fought.

def update_odds_data(existing_odds_data, existing_event_urls, existing_event_odds, new_fighter_list, new_event_date_list, config, fast_scrape = True, num_threads = 1):

    # pdb.set_trace()
    
    # Read the fighter odds urls
    fighter_odds_urls = pd.read_csv(config['fighter_odds_urls'])

    # Scrape all the odds data for the fighters in the list
    new_odds_data, new_events_urls = parallel_scrape_odds_fast(
        new_fighter_list, 
        num_threads = config['num_threads'], 
        firefox_location = config['firefox_location'], 
        gecko_driver = config['gecko_driver_location'],
        odds_urls_overide_df = fighter_odds_urls,
        event_urls = True,
        fast = fast_scrape)
        
    # Get the date list and expand it to include dates on either side due to timezone differences potentially causing discrepencies
    full_date_list = list(new_event_date_list.copy())
    for date in new_event_date_list:
        full_date_list.append(date + timedelta(days = 1))
        full_date_list.append(date - timedelta(days = 1))
        full_date_list.append(date + timedelta(days = 2))
        full_date_list.append(date - timedelta(days = 2))
        
    # Filter only the fights that are on the specified dates
    new_odds_data = new_odds_data[new_odds_data['date'].isin(full_date_list)]
    
    # Add on the date of the odds scrape
    new_odds_data['notformodel_date_fighter_odd_scrape'] = datetime.now(timezone.utc) # Give the datetime in GMT
    
    # Add this onto the existing odds data
    new_odds_data = pd.concat([new_odds_data, existing_odds_data])
    
    # Remove any duplicate records
    new_odds_data = new_odds_data.drop_duplicates(subset = ['event', 'date', 'fighter', 'opponent'], keep = 'first', inplace = False)
    
    ###############################
    # Deal with missing odds data
    ###############################
    
    # Filter rows with NaT in 'date'
    nat_date_rows = new_odds_data[new_odds_data['date'].isna()]

    # Total count of rows with NaT in 'date'
    total_nat_count = nat_date_rows.shape[0]

    # Count and percentage for rows where 'event' contains "UFC"
    ufc_nat_count = nat_date_rows[nat_date_rows['event'].str.contains("UFC", case=False, na=False)].shape[0]
    ufc_nat_percent = (ufc_nat_count / total_nat_count * 100) if total_nat_count > 0 else 0

    # Count and percentage for rows where 'event' does not contain "UFC"
    non_ufc_nat_count = nat_date_rows[~nat_date_rows['event'].str.contains("UFC", case=False, na=False)].shape[0]
    non_ufc_nat_percent = (non_ufc_nat_count / total_nat_count * 100) if total_nat_count > 0 else 0

    print(f"Rows with NaT in 'date' and 'event' containing 'UFC': {ufc_nat_count} ({ufc_nat_percent:.2f}%)")
    print(f"Rows with NaT in 'date' and 'event' not containing 'UFC': {non_ufc_nat_count} ({non_ufc_nat_percent:.2f}%)")
    
    ###############################
    ### Update events data
    ###############################
    
    # Get a list of events that haven't already been parsed
    new_events_to_scrape = new_events_urls[(~new_events_urls['event_url'].isin(existing_event_urls['event_url'].tolist())) | (new_events_urls['event_date'] >= date.today())]
        
    # Scrape the new event odds
    new_events_odds = parallel_scrape_event_odds_fast(
        new_events_to_scrape, 
        num_threads = config['num_threads'], 
        firefox_location = config['firefox_location'], 
        gecko_driver = config['gecko_driver_location'],
        fast = fast_scrape)
    
    # Concat and replace duplicates
    final_event_urls = pd.concat([new_events_to_scrape, existing_event_urls]).drop_duplicates()
    final_event_odds = pd.concat([new_events_odds, existing_event_odds]).drop_duplicates(['date', 'fighter', 'opponent'], keep = 'first')
    # Ensure format of final cols correct
    final_event_odds.loc[:, odds_scrape_config['events_colnames']] = final_event_odds[odds_scrape_config['events_colnames']].apply(pd.to_numeric, errors = 'coerce')

        
    return new_odds_data, final_event_urls, final_event_odds
