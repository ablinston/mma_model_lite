
'''
Overview
this notebook parses all past ufc fight stats when run, it does not include upcoming fights


scrape ufc fight stats
get all event details, name, url, date, location for all ufc events
for each event, get fight details all fights on card
parse each fight to get fight stats of both fighters
'''

# imports
import itertools
import re
import string
from tqdm.notebook import tqdm_notebook
from concurrent.futures import ThreadPoolExecutor, as_completed
import fastparquet as fp
import pandas as pd

from mma_functions.parsing import parse_event_details, parse_fight_details, parse_organise_fight_results_and_stats
from mma_functions.scraping import get_soup, parallel_scrape
from mma_functions.config import config

# define url to parse
events_url = config['completed_events_all_url']

# get soup
soup = get_soup(events_url)

# parse event details
all_event_details_df = parse_event_details(soup)

# show event details
print(all_event_details_df)

# write event details to file
fp.write('./data/' + config['event_details_file_name'], all_event_details_df)

# define list of urls of fights to parse
# Get previously parsed list
#parsed_events = fp.ParquetFile(config['parsed_events_log']).to_pandas()

# Find all events that haven't been parsed
#all_event_details_df.merge(parsed_events, how = 'outer', indicator = True)

list_of_events_urls = list(all_event_details_df['URL'])

# create empty df to store fight details
all_fight_details_df = pd.DataFrame(columns = config['fight_details_column_names'])

all_fight_details_df = parallel_scrape(
    url_list = list_of_events_urls,
    parse_function = parse_fight_details,
    num_threads = config['num_threads']
)

# write fight details to file
fp.write('./data/' + config['fight_details_file_name'], all_fight_details_df)

# define list of urls of fights to parse
list_of_fight_details_urls = list(all_fight_details_df['URL'])

# create empty df to store fight results
all_fight_results_df = pd.DataFrame(columns=config['fight_results_column_names'])
# create empty df to store fight stats
all_fight_stats_df = pd.DataFrame(columns=config['fight_stats_column_names'])

# Scrape each fights data in parallel

with ThreadPoolExecutor(max_workers = config['num_threads']) as executor:

    future_to_url = {executor.submit(get_soup, url): url for url in list_of_fight_details_urls}

    # Now iterate through each completed future
    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            # Get the fight data
            fight_result, fight_stats = parse_organise_fight_results_and_stats(
                future.result(),
                url,
                config['fight_results_column_names'],
                config['totals_column_names'],
                config['significant_strikes_column_names']      
            )
            # Add this fight data onto the main data frame
            all_fight_results_df = pd.concat([all_fight_results_df, fight_result])
            all_fight_stats_df = pd.concat([all_fight_stats_df, fight_stats])
            print(f'Scraped url: {url}')
        except Exception as exc:
            print(f'Error scraping url: {url} : {exc}')

# write to file
fp.write('./data/' + config['fight_results_file_name'], all_fight_results_df, compression = 'GZIP')
# write to file
fp.write('./data/' + config['fight_stats_file_name'], all_fight_stats_df, compression = 'GZIP')
