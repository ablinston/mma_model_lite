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
# decompressed_output_dir = './data/raw_betfair/temp_decompressed_files/'


# Open the connection with Betfair. This requires the betfair config which is not tracked by git because it contains passwords
trading = betfairlightweight.APIClient(betfair_login_config['username'], betfair_login_config['password'], app_key = betfair_login_config['delay_feed_app_key'], certs = betfair_login_config['certs_location'])

trading.login()



############################################
## HISTORIC DATA DOWNLOAD

# ChatGPT says data is generally available 5 days after and event

# get file list
file_list = trading.historic.get_file_list(
    "Other Sports",
    "Basic Plan",
    from_day=1,
    from_month=4,
    from_year=2025,
    to_day=30,
    to_month=4,
    to_year=2024,
    market_types_collection=["MATCH_ODDS"],
    # countries_collection=["GB"],
    file_type_collection=["L", "M"],
)

# download the files
for file in file_list:
    print(file)
    download = trading.historic.download_file(file_path = file, store_directory = download_folder)
    print(download)



############################################
## PROCESS DOWNLOADED FILES


bz2_file_log = betfair_filter_bz2_for_fights(download_folder, decompressed_output_dir)



############################################
## STREAMING HISTORIC DATA DOWNLOADS

## DONT NEED TO DO THIS NOW. IT'S DONE ABOVE
# Decompress all the files downloaded from Betfair. Run in CMD
# for /r "E:\Repos\mma_model_lite\data\raw_betfair\" %f in (*.bz2) do "C:\Program Files\7-Zip\7z.exe" x "%f" -o"E:\Repos\mma_model_lite\data\raw_betfair\decompressed_files" -y


# # create listener
# listener = StreamListener(max_latency=None)

# # ## Go through each file in the decompressed folder and load the data into data frames

# # # Create an empty list for the datasets
# # betfair_odds_list = []

# # # Loop through files
# # for file in list_of_decompressed_filepaths[1000:1010]:

#     print(file)    

    # # Create historical stream (update file_path to your file location)
    # stream = trading.streaming.create_historical_generator_stream(
    #     file_path = decompressed_output_dir + '1.172434143',
    #     listener = listener,
    # )
    
    # # Convert the list of series into a data frame
    # betfair_odds_df = betfair_stream_to_df(stream = stream)
    
    # # Get the last odds before the chosen cutoff time
    # betfair_odds_summary_df = betfair_bout_get_chosen_odds(betfair_odds_df, betfair_data_config['odds_cutoff_time'])

#     # Save the output to the list as a dictionary for faster combination later
#     betfair_odds_list.append(betfair_odds_summary_df)

# # betfair_odds = pd.concat(betfair_odds_list, axis = 0)

list_of_decompressed_filepaths = [decompressed_output_dir + file for file in os.listdir(decompressed_output_dir)]

betfair_odds = parallel_run_function(
        func = betfair_file_to_final_odds,
        input_list = list_of_decompressed_filepaths,
        max_workers = 1, # DO NOT USE MORE CORES FOR PROCESS. THE ODDS GO OFF MAYBE BECAUSE OF STREAM?
        collapse_output_to_df = True,
        odds_cutoff_time = betfair_data_config['odds_cutoff_time'],
        average_last_n = betfair_data_config['last_n_bets_to_average'],
        trading = trading)


# Change the date to a proper date to avoid parquet errors
betfair_odds['event_date'] = pd.to_datetime(betfair_odds['event_date'], errors = 'coerce')

# Output the data
fp.write('./data/' + config['betfair_odds_file_name'], betfair_odds, compression = 'ZSTD')
