# This script will update the existing parsed data with new UFC events and fighters.
# It will scrape the data and perform consistency checks to ensure all datasets are consistent before
# saving them and overwriting existing files.
import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.filters import market_filter, price_projection, streaming_market_filter, streaming_market_data_filter
import bz2
from pathlib import Path
import queue
import threading
import time

from ablinston_ml import stop_if_not
from datetime import datetime, timedelta
import fastparquet as fp
import pandas as pd

from mma_functions.betfair import betfair_get_all_upcoming_odds, update_betfair_odds
from mma_functions.cleaning import check_data_coverage, update_data_summary
from mma_functions.dataset import fighter_list_from_bouts
from mma_functions.odds_scraping import get_fighter_data, update_odds_data
from mma_functions.parsing import fix_fight_details_bout, parse_event_details, parse_fighter_details, update_fight_details_data, update_fight_results_and_stats_data
from mma_functions.scraping import generate_alphabetical_urls, get_soup, parallel_scrape
from mma_functions.config import betfair_data_config, betfair_login_config, config, odds_scrape_config

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
list_of_all_events = list(new_event_details_df['EVENT'])

# find list event names that have not been parsed
list_of_unparsed_events = [event for event in list_of_all_events if event not in list_of_parsed_events]



###############################################################################
##### Parse upcoming fight data #####
###############################################################################

# Get a list of upcoming events
upcoming_events_df = parse_event_details(get_soup(config['upcoming_events_url']), upcoming = True)

# # If missed the event start time because it is today, move it to the upcoming data
if pd.to_datetime(new_event_details_df.iloc[0]['DATE']) == pd.to_datetime(datetime.today().strftime('%Y-%m-%d')):
    upcoming_events_df = pd.concat([new_event_details_df.iloc[[0]], upcoming_events_df], axis = 0, ignore_index = True).reset_index(drop = True)
    new_event_details_df = new_event_details_df.iloc[1:, :]
    list_of_unparsed_events = []

# Only look at the first two to save time
upcoming_events_df = upcoming_events_df.iloc[0:1]

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
parsed_event_odds_urls = fp.ParquetFile('./data/event_odds_urls.parquet').to_pandas()
parsed_event_odds_df = fp.ParquetFile('./data/' + config['event_odds_file_name']).to_pandas()
parsed_betfair_odds_df = fp.ParquetFile('./data/' + config['betfair_odds_file_name']).to_pandas()

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

new_parsed_odds_df, new_parsed_event_odds_urls, new_parsed_event_odds_df = update_odds_data(
    parsed_odds_df, 
    parsed_event_odds_urls, 
    parsed_event_odds_df,
    list_of_unparsed_fighter_odds, 
    list_of_unparsed_event_dates, 
    config, 
    fast_scrape = odds_scrape_config['fast_scrape_update'],
    num_threads = config['num_threads'])
    
# Find which fighters didn't get odds properly by looking at the dates of people found
# Offsets of -1, 0, and +1 days
offsets = [timedelta(days=delta) for delta in [-1, 0, 1]]
# List comprehension to apply all offsets to all dates
expanded_dates = sorted(set(date + offset for date in list_of_unparsed_event_dates for offset in offsets))
found_fighters = new_parsed_odds_df[new_parsed_odds_df['date'].isin(expanded_dates)]['fighter'].tolist()
missing_fighters = [fighter for fighter in list_of_unparsed_fighter_odds if fighter not in found_fighters]

# Scrape the odds again but a slow scrape
if len(missing_fighters) > 0:
    new_parsed_odds_df, new_parsed_event_odds_urls, new_parsed_event_odds_df = update_odds_data(
        new_parsed_odds_df, 
        new_parsed_event_odds_urls, 
        new_parsed_event_odds_df,
        missing_fighters, 
        list_of_unparsed_event_dates, 
        config, 
        fast_scrape = False,
        num_threads = config['num_threads'])

###############################################################################
##### Update the Betfair odds #####
###############################################################################

# Initiate the trading object
trading = betfairlightweight.APIClient(betfair_login_config['username'], betfair_login_config['password'], app_key = betfair_login_config['delay_feed_app_key'], certs = betfair_login_config['certs_location'])

trading.login()

if 'skip_betfair_history' not in globals():
    skip_betfair_history = False

if not skip_betfair_history:
    # Update the odds
    new_parsed_betfair_odds_df = update_betfair_odds(
        parsed_betfair_odds_df, 
        betfair_login_config, 
        download_folder = betfair_data_config['download_folder'], 
        decompressed_output_dir = betfair_data_config['decompressed_output_dir'], 
        decompressed_temp_output_dir = betfair_data_config['decompressed_temp_output_dir']
    )
else:
    new_parsed_betfair_odds_df = parsed_betfair_odds_df

time.sleep(5)

# Get the upcoming Betfair odds data
upcoming_betfair_odds = betfair_get_all_upcoming_odds(betfair_login_config, betfair_data_config)

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
        new_parsed_event_odds_df,
        new_parsed_betfair_odds_df,
        upcoming_events_df,
        upcoming_fight_details_df,
        upcoming_betfair_odds,
        config)
    
    # Check new rows have been added
    stop_if_not(
        (len(updated_event_details_df) >= len(parsed_event_details_df)) & (
            len(new_parsed_fight_details_df) >= len(parsed_fight_details_df)) & (
                len(new_parsed_fight_results_df) >= len(parsed_fight_results_df)) & (
                    len(new_parsed_fight_stats_df) >= len(parsed_fight_stats_df)) & (
                        len(new_parsed_odds_df) >= len(parsed_odds_df)) & (
                            len(new_parsed_fighter_tott_df) >= len(parsed_fighter_tott_df)) & (
                                len(new_parsed_event_odds_df) >= len(parsed_event_odds_df)),
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
                'parsed_odds_df': new_parsed_odds_df, 
                'parsed_event_odds_df' : new_parsed_event_odds_df,
                'parsed_fighter_tott_df': new_parsed_fighter_tott_df,
                'parsed_fight_stats_df': new_parsed_fight_stats_df,
                'parsed_betfair_odds_df': new_parsed_betfair_odds_df},
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
        fp.write('./data/event_odds_urls.parquet', new_parsed_event_odds_urls, compression = 'ZSTD')
        fp.write('./data/' + config['event_odds_file_name'], new_parsed_event_odds_df, compression = 'ZSTD')
        
        fp.write('./data/' + config['betfair_odds_file_name'], new_parsed_betfair_odds_df, compression = 'ZSTD')
        fp.write('./data/' + config['betfair_upcoming_odds_file_name'], upcoming_betfair_odds, compression = 'ZSTD')
        
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
        new_parsed_event_odds_df,
        new_parsed_betfair_odds_df,
        upcoming_events_df,
        upcoming_fight_details_df,
        upcoming_betfair_odds,
        config)
        
    #######################
    ## Save files if all has run successfully
    #######################
    
    if checks_passed:
        
        # write new fight details to file
        fp.write('./data/' + config['upcoming_event_details_file_name'], upcoming_events_df, compression = 'ZSTD')
        fp.write('./data/' + config['upcoming_fight_details_file_name'], upcoming_fight_details_df, compression = 'ZSTD')
        
        fp.write('./data/' + config['betfair_odds_file_name'], new_parsed_betfair_odds_df, compression = 'ZSTD')
        fp.write('./data/' + config['betfair_upcoming_odds_file_name'], upcoming_betfair_odds, compression = 'ZSTD')

        fp.write('./data/' + config['fighter_odds_file_name'], new_parsed_odds_df, compression = 'ZSTD')
        fp.write('./data/event_odds_urls.parquet', new_parsed_event_odds_urls, compression = 'ZSTD')
        fp.write('./data/' + config['event_odds_file_name'], new_parsed_event_odds_df, compression = 'ZSTD')
        
    else:
        print("Data checks not passed. Data not saved. See previous output messages for more information.")

    
