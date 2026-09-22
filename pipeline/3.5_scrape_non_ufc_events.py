# This script scrapes data on non-UFC events that are listed on UFCstats.com such as Strikeforce and WEC (acquired by Zuffa)

from datetime import datetime, timedelta
import fastparquet as fp
import pandas as pd

from mma_functions.cleaning import check_data_coverage
from mma_functions.parsing import fix_fight_details_bout, parse_fighter_details, update_fight_details_data, update_fight_results_and_stats_data
from mma_functions.scraping import generate_alphabetical_urls, parallel_get_all_event_urls_from_fighters, parallel_scrape, parallel_scrape_event_details
from mma_functions.config import config

# read existing event details
parsed_event_details_df = fp.ParquetFile('./data/' + config['event_details_file_name']).to_pandas()

parsed_fighter_details_df = fp.ParquetFile('./data/' + config['fighter_details_file_name']).to_pandas()
parsed_fighter_tott_df = fp.ParquetFile('./data/' + config['fighter_tott_file_name']).to_pandas()

parsed_fight_details_df = fp.ParquetFile('./data/' + config['fight_details_file_name']).to_pandas()
parsed_fight_results_df = fp.ParquetFile('./data/' + config['fight_results_file_name']).to_pandas()
parsed_fight_stats_df = fp.ParquetFile('./data/' + config['fight_stats_file_name']).to_pandas()
parsed_odds_df = fp.ParquetFile('./data/' + config['fighter_odds_file_name']).to_pandas()


# Get a full list of event urls that are linked in fighter data
full_list = parallel_get_all_event_urls_from_fighters(parsed_fighter_tott_df['URL'].to_list(), config['fighter_result_headers'])

# Find unique set of urls that are not covered in the currently scraped data
full_list_u = list(set(full_list))
non_ufc_events = [url for url in full_list_u if url not in parsed_event_details_df['URL'].tolist()]

# Get the event details for all the unparsed events
unparsed_event_details_df = parallel_scrape_event_details(non_ufc_events, num_threads = config['num_threads'])

# Filter out any events that haven't happened yet
unparsed_event_details_df['date_formatted'] = pd.to_datetime(unparsed_event_details_df['DATE'])
unparsed_event_details_df = unparsed_event_details_df[unparsed_event_details_df['date_formatted'] < (datetime.now() - timedelta(days=1))]

# Updated the list of events
non_ufc_events = unparsed_event_details_df['URL'].to_list()

###############################################################################
### parse all missing events ###
###############################################################################

##### Parse fight details #####

# Create new data frame including the new events
parsed_fight_details_df_new = update_fight_details_data(
    existing_fights_data = parsed_fight_details_df,
    new_events_url_list = non_ufc_events,
    config = config,
    # debug = True,
    num_threads = 3)

##### Parse fight results and fight stats #####

# define list of urls of fights to parse
list_of_unparsed_fight_details_urls = list(parsed_fight_details_df_new[~parsed_fight_details_df_new['URL'].isin(parsed_fight_details_df['URL'].tolist())]['URL'])

parsed_fight_results_df, parsed_fight_stats_df = update_fight_results_and_stats_data(
    parsed_fight_results_df, 
    parsed_fight_stats_df, 
    list_of_unparsed_fight_details_urls, 
    config, 
    num_threads = config['num_threads']
    )

# Manual check for full fight history before UFC
kk = parsed_fight_stats_df[parsed_fight_stats_df['FIGHTER'] == 'Jose Aldo']

##### Parse new fighter odds data #####

# Get a list of all the events that have been added
list_of_unparsed_events = parsed_fight_details_df_new['EVENT'][~parsed_fight_details_df_new['EVENT'].isin(parsed_fight_details_df['EVENT'])].unique().tolist()

# Add it to the parsed events df
updated_event_details_df = pd.concat([parsed_event_details_df, unparsed_event_details_df])

# define list of urls of fighters we want new odds data for
list_of_unparsed_fighter_odds = list(set(parsed_fight_stats_df[parsed_fight_stats_df['EVENT'].isin(list_of_unparsed_events)]['FIGHTER']))
list_of_unparsed_event_dates = unparsed_event_details_df['DATE'].to_list()

# Convert the dates
list_of_unparsed_event_dates = [datetime.strptime(x, '%B %d, %Y') for x in list_of_unparsed_event_dates]

# Get the new odds data
parsed_odds_df = update_fighter_odds_data(parsed_odds_df, list_of_unparsed_fighter_odds, list_of_unparsed_event_dates, config, num_threads = config['num_threads'])


###############################################################################
##### Fight details finalised #####

# Add the new fight details to the old
parsed_fight_details_df = pd.concat([parsed_fight_details_df, parsed_fight_details_df_new])


# Run a fix on the bouts in the details data
parsed_fight_details_df = fix_fight_details_bout(parsed_fight_details_df, parsed_fight_stats_df)


# Drop any duplicates introduced
parsed_fight_results_df = parsed_fight_results_df.drop_duplicates()
parsed_fight_stats_df = parsed_fight_stats_df.drop_duplicates()
parsed_fight_details_df = parsed_fight_details_df.drop_duplicates()
updated_event_details_df = updated_event_details_df.drop_duplicates()

###############################################################################
##### Parse new fighter data #####
###############################################################################

# Scrape the full list of fighters from the UFC site again
# generate list of urls for fighter details
list_of_alphabetical_urls = generate_alphabetical_urls()

# Go through list of alphabetical urls
all_fighter_details_df = parallel_scrape(
    url_list = list_of_alphabetical_urls,
    parse_function = parse_fighter_details,
    num_threads = config['num_threads'],
    fighter_details_column_names = config['fighter_details_column_names'])



###############################################################################
##### Data consistency checks #####
###############################################################################

## Run checks to ensure each of the datasets are consistent and cover the same events
checks_passed = check_data_coverage(
    updated_event_details_df, 
    parsed_fight_details_df, 
    parsed_fight_results_df, 
    parsed_fight_stats_df, 
    parsed_odds_df,
    all_fighter_details_df,
    parsed_fighter_tott_df,
    config)


#######################
## Save files if all has run successfully
#######################

if checks_passed:
    
    # write new fight details to file
    fp.write('./data/' + config['event_details_file_name'], updated_event_details_df, compression = 'GZIP')
    fp.write('./data/' + config['fight_details_file_name'], parsed_fight_details_df, compression = 'GZIP')
    fp.write('./data/' + config['fight_results_file_name'], parsed_fight_results_df, compression = 'GZIP')
    fp.write('./data/' + config['fight_stats_file_name'], parsed_fight_stats_df, compression = 'GZIP')
    fp.write('./data/' + config['fighter_odds_file_name'], parsed_odds_df, compression = 'GZIP')
    fp.write('./data/' + config['fighter_details_file_name'], all_fighter_details_df, compression = 'GZIP')
    fp.write('./data/' + config['fighter_tott_file_name'], parsed_fighter_tott_df, compression = 'GZIP')
    
else:
    print("Data checks not passed. Data not saved. See previous output messages for more information.")
