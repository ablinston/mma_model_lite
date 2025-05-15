# This is a script to search bz2 files for a specific fighter string. This is what I used for searching for missing odds in investigative work.

import bz2
import os
import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.filters import market_filter, price_projection
import bz2
from pathlib import Path
import orjson
import os


exec(open('global.py').read())

# Set the download folder. This should be empty to save duplicated searching
download_folder = './data/raw_betfair/historic/'
decompressed_output_dir = './data/raw_betfair/decompressed_fight_files/'


# Open the connection with Betfair. This requires the betfair config which is not tracked by git because it contains passwords
trading = betfairlightweight.APIClient(betfair_login_config['username'], betfair_login_config['password'], app_key = betfair_login_config['delay_feed_app_key'], certs = './data/betfair_certs')

trading.login()



############################################
## HISTORIC DATA DOWNLOAD

# ChatGPT says data is generally available 5 days after and event

# get file list
file_list = trading.historic.get_file_list(
    "Other Sports",
    "Basic Plan",
    from_day=3,
    from_month=11,
    from_year=2017,
    to_day=5,
    to_month=11,
    to_year=2017,
    market_types_collection=["MATCH_ODDS"],
    # countries_collection=["GB"],
    file_type_collection=["L", "M"],
)

# download the files
for file in file_list:
    print(file)
    download = trading.historic.download_file(file_path = file, store_directory = download_folder)
    print(download)













# === CONFIGURATION ===
folder_path = download_folder  # <- CHANGE THIS
search_string = 'bisping'     # <- CHANGE THIS

# === SEARCH FUNCTION ===
def search_bz2_files(folder, target):
    matches = []

    for filename in os.listdir(folder):
        if filename.endswith('.bz2'):
            full_path = os.path.join(folder, filename)
            try:
                with bz2.open(full_path, 'rt', encoding='utf-8', errors='ignore') as file:
                    for i, line in enumerate(file, 1):
                        if target.lower() in line.lower():
                            matches.append((filename, i, line.strip()))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return matches

# === RUN SEARCH AND PRINT RESULTS ===
results = search_bz2_files(folder_path, search_string)

print(f"\nFound {len(results)} match(es):\n")
for filename, line_number, line in results:
    print(f"[{filename}] Line {line_number}: {line}")











# create listener
listener = StreamListener(max_latency=None)

# ## Go through each file in the decompressed folder and load the data into data frames

# # Create an empty list for the datasets
# betfair_odds_list = []

# # Loop through files
# for file in list_of_decompressed_filepaths[1000:1010]:
     

# Create historical stream (update file_path to your file location)
stream = trading.streaming.create_historical_generator_stream(
    file_path = decompressed_output_dir + '1.133115411',
    listener = listener,
)

# Convert the list of series into a data frame
betfair_odds_df = betfair_stream_to_df(stream = stream)

# Get the last odds before the chosen cutoff time
betfair_odds_summary_df = betfair_bout_get_chosen_odds(betfair_odds_df, betfair_data_config['odds_cutoff_time'])

# Save the output to the list as a dictionary for faster combination later
betfair_odds_list.append(betfair_odds_summary_df)