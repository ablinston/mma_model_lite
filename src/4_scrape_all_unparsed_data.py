# This script will update the existing parsed data with new UFC events and fighters.
# It will scrape the data and perform consistency checks to ensure all datasets are consistent before
# saving them and overwriting existing files.

exec(open('global.py').read())

### check if there are any unparsed events ###

# read existing event details
parsed_event_details_df = fp.ParquetFile('./data/' + config['event_details_file_name']).to_pandas()
# get list of parsed event names
list_of_parsed_events = list(parsed_event_details_df['EVENT'])

# get soup
soup = get_soup(config['completed_events_all_url'])
# parse event details
new_event_details_df = parse_event_details(soup)

# get list of all event names
list_of_all_events = list(new_event_details_df[new_event_details_df['DATE'] != datetime.today().strftime("%B %d, %Y")]['EVENT']) # ignore events today

# find list event names that have not been parsed
list_of_unparsed_events = [event for event in list_of_all_events if event not in list_of_parsed_events]



###############################################################################
##### Parse upcoming fight data #####
###############################################################################

# Get a list of upcoming events
upcoming_events_df = parse_event_details(get_soup(config['upcoming_events_url']), upcoming = True)

# # If missed the event start time
# upcoming_events_df = pd.concat([new_event_details_df.iloc[[0]], upcoming_events_df], axis = 0, ignore_index = True).reset_index(drop = True)

# Only look at the first two to save time
upcoming_events_df = upcoming_events_df.iloc[0:2]

# Parse upcoming fight details
upcoming_fight_details_df = update_fight_details_data(
    existing_fights_data = pd.DataFrame(),
    new_events_url_list = upcoming_events_df['URL'].to_list(),
    config = config,
    num_threads = 3)



###############################################################################
### parse all missing events ###
###############################################################################

# check if there are any unparsed events
unparsed_events = False
# if list_of_unparsed_events is empty then all available events have been parsed
if not list_of_unparsed_events:
    print('All available events have been parsed.')
else:
    # set unparsed_events to true
    unparsed_events = True
    # show list of unparsed events
    print(list_of_unparsed_events)

# if unparsed_events = True
# the code below continues to run to parse all missing events
# new data is added to existing data and is written to file

# Read existing data files
parsed_fight_details_df = fp.ParquetFile('./data/' + config['fight_details_file_name']).to_pandas()
parsed_fight_results_df = fp.ParquetFile('./data/' + config['fight_results_file_name']).to_pandas()
parsed_fight_stats_df = fp.ParquetFile('./data/' + config['fight_stats_file_name']).to_pandas()
parsed_odds_df = fp.ParquetFile('./data/' + config['fighter_odds_file_name']).to_pandas()

if unparsed_events == True:
    
    ##### Add new events to events data #####
    updated_event_details_df = pd.concat([parsed_event_details_df, new_event_details_df[new_event_details_df['EVENT'].isin(list_of_unparsed_events)]])
    

    ##### Parse fight details #####

    # define list of urls of missing fights to parse
    list_of_unparsed_events_urls =  list(updated_event_details_df['URL'].loc[(updated_event_details_df['EVENT'].isin(list_of_unparsed_events))])

    # Create new data frame including the new events
    new_parsed_fight_details_df = update_fight_details_data(
        existing_fights_data = parsed_fight_details_df,
        new_events_url_list = list_of_unparsed_events_urls,
        config = config,
        num_threads = 3)

    ##### Parse fight results and fight stats #####
    
    # define list of urls of fights to parse
    list_of_unparsed_fight_details_urls = list(new_parsed_fight_details_df[new_parsed_fight_details_df['EVENT'].isin(list_of_unparsed_events)]['URL'])

    new_parsed_fight_results_df, new_parsed_fight_stats_df = update_fight_results_and_stats_data(
        parsed_fight_results_df, 
        parsed_fight_stats_df, 
        list_of_unparsed_fight_details_urls, 
        config, 
        num_threads = config['num_threads']
        )

    # Run a fix on the bouts in the details data
    new_parsed_fight_details_df = fix_fight_details_bout(new_parsed_fight_details_df, new_parsed_fight_stats_df)


###############################################################################
##### Parse new fighter odds data #####
###############################################################################

# define list of urls of fighters we want new odds data for
try:
    list_of_unparsed_fighter_odds = list(set(new_parsed_fight_stats_df[new_parsed_fight_stats_df['EVENT'].isin(list_of_unparsed_events)]['FIGHTER'].to_list()))
    list_of_unparsed_event_dates = list(updated_event_details_df[updated_event_details_df['EVENT'].isin(list_of_unparsed_events)]['DATE']) + upcoming_events_df['DATE'].to_list()
except:
    list_of_unparsed_fighter_odds = []
    list_of_unparsed_event_dates = []

# Add on the upcoming events too
list_of_unparsed_fighter_odds = list(set(list_of_unparsed_fighter_odds + fighter_list_from_bouts(upcoming_fight_details_df['BOUT'])))

list_of_unparsed_event_dates = list_of_unparsed_event_dates + upcoming_events_df['DATE'].to_list()

# Convert the dates
list_of_unparsed_event_dates = [datetime.strptime(x, '%B %d, %Y') for x in list_of_unparsed_event_dates]

new_parsed_odds_df = update_fighter_odds_data(parsed_odds_df, list_of_unparsed_fighter_odds, list_of_unparsed_event_dates, config, num_threads = config['num_threads'])
    
    
        


###############################################################################
##### Parse new fighter data #####
###############################################################################

parsed_fighter_details_df = fp.ParquetFile('./data/' + config['fighter_details_file_name']).to_pandas()
parsed_fighter_tott_df = fp.ParquetFile('./data/' + config['fighter_tott_file_name']).to_pandas()

# Scrape the full list of fighters from the UFC site again
# generate list of urls for fighter details
list_of_alphabetical_urls = generate_alphabetical_urls()

# Go through list of alphabetical urls
all_fighter_details_df = parallel_scrape(
    url_list = list_of_alphabetical_urls,
    parse_function = parse_fighter_details,
    num_threads = config['num_threads'],
    fighter_details_column_names = config['fighter_details_column_names'])

# Clean out any 'www' to ensure we find all the required URLs
all_fighter_details_df['URL'] = all_fighter_details_df['URL'].str.replace('www,', '')
parsed_fighter_tott_df['URL'] = parsed_fighter_tott_df['URL'].str.replace('www,', '')

# Find the unparsed urls
unparsed_fighter_urls = all_fighter_details_df['URL'][~all_fighter_details_df['URL'].isin(parsed_fighter_tott_df['URL'].to_list())].to_list()

# Now parse the unparsed URLs
if len(unparsed_fighter_urls) > 0:
    
    # Concatenate the old and new scraped data
    new_parsed_fighter_tott_df = pd.concat([
        parallel_scrape(
            url_list = unparsed_fighter_urls,
            # url_list = all_fighter_details_df['URL'].tolist(),
            parse_function = get_fighter_data,
            num_threads = config['num_threads'],
            fighter_tott_column_names = config['fighter_tott_column_names']),
        parsed_fighter_tott_df
    ])
else:
    new_parsed_fighter_tott_df = parsed_fighter_tott_df


###############################################################################
##### Data consistency checks #####
###############################################################################

if unparsed_events:
    
    ## Run checks to ensure each of the datasets are consistent and cover the same events
    checks_passed = check_data_coverage(
        updated_event_details_df, 
        new_parsed_fight_details_df, 
        new_parsed_fight_results_df, 
        new_parsed_fight_stats_df, 
        new_parsed_odds_df,
        all_fighter_details_df,
        new_parsed_fighter_tott_df,
        config)
    
    # Check new rows have been added
    stop_if_not(
        (len(updated_event_details_df) >= len(parsed_event_details_df)) & (
            len(new_parsed_fight_details_df) >= len(parsed_fight_details_df)) & (
                len(new_parsed_fight_results_df) >= len(parsed_fight_results_df)) & (
                    len(new_parsed_fight_stats_df) >= len(parsed_fight_stats_df)) & (
                        len(new_parsed_odds_df) >= len(parsed_odds_df)) & (
                            len(new_parsed_fighter_tott_df) >= len(parsed_fighter_tott_df)),
                            'Row change checks')
    
    # AB: If the fighter_tott has missing info, just run the whole thing rather than update old one
    
    if checks_passed:
        
        # Log the aspects of these files
        data_summary = update_data_summary(
            results_csv_path = './data/raw_data_summary.csv', 
            data_frames = {
                'updated_event_details_df': updated_event_details_df, 
                'parsed_fight_details_df': new_parsed_fight_details_df, 
                'parsed_fight_results_df': new_parsed_fight_results_df, 
                'parsed_odds_df':new_parsed_odds_df, 
                'parsed_fighter_tott_df': new_parsed_fighter_tott_df,
                'parsed_fight_stats_df': new_parsed_fight_stats_df},
            summary_date = datetime.today(),
            save_output = True)
        
        # Manually check if you're happy with the above then save
        
        # write new fight details to file
        fp.write('./data/' + config['event_details_file_name'], updated_event_details_df, compression = 'ZSTD')
        fp.write('./data/' + config['upcoming_event_details_file_name'], upcoming_events_df, compression = 'ZSTD')
        fp.write('./data/' + config['fight_details_file_name'], new_parsed_fight_details_df, compression = 'ZSTD')
        fp.write('./data/' + config['upcoming_fight_details_file_name'], upcoming_fight_details_df, compression = 'ZSTD')
        fp.write('./data/' + config['fight_results_file_name'], new_parsed_fight_results_df, compression = 'ZSTD')
        fp.write('./data/' + config['fight_stats_file_name'], new_parsed_fight_stats_df, compression = 'ZSTD')
        fp.write('./data/' + config['fighter_odds_file_name'], new_parsed_odds_df, compression = 'ZSTD')
        fp.write('./data/' + config['fighter_details_file_name'], all_fighter_details_df, compression = 'ZSTD')
        fp.write('./data/' + config['fighter_tott_file_name'], new_parsed_fighter_tott_df, compression = 'ZSTD')
        
    else:
        print("Data checks not passed. Data not saved. See previous output messages for more information.")

else:

    ## Run checks to ensure each of the datasets are consistent and cover the same events
    checks_passed = check_data_coverage(
        parsed_event_details_df, 
        parsed_fight_details_df, 
        parsed_fight_results_df, 
        parsed_fight_stats_df, 
        new_parsed_odds_df,
        all_fighter_details_df,
        parsed_fighter_tott_df,
        config)
        
    #######################
    ## Save files if all has run successfully
    #######################
    
    if checks_passed:
        
        # write new fight details to file
        fp.write('./data/' + config['upcoming_event_details_file_name'], upcoming_events_df, compression = 'ZSTD')
        fp.write('./data/' + config['upcoming_fight_details_file_name'], upcoming_fight_details_df, compression = 'ZSTD')

        fp.write('./data/' + config['fighter_odds_file_name'], new_parsed_odds_df, compression = 'ZSTD')
        
    else:
        print("Data checks not passed. Data not saved. See previous output messages for more information.")

    