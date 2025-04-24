import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.filters import market_filter, price_projection
import bz2
from pathlib import Path
import orjson
import os


exec(open('global.py').read())

# Open the connection with Betfair. This requires the betfair config which is not tracked by git because it contains passwords
trading = betfairlightweight.APIClient(betfair_login_config['username'], betfair_login_config['password'], app_key = betfair_login_config['delay_feed_app_key'], certs = './data/betfair_certs')

trading.login()



############################################
## STREAMING SPORTS


# 1. Get MMA event type ID (only needed once)
event_types = trading.betting.list_event_types()

for e in event_types:
    print(f"{e.event_type.id}: {e.event_type.name}")
    
mma_event_type = next(e for e in event_types if "Mixed Martial Arts" in e.event_type.name)

sport_ids = pd.DataFrame({
    'Sport': [event_type_object.event_type.name for event_type_object in event_types],
    'ID': [event_type_object.event_type.id for event_type_object in event_types]
}).set_index('Sport').sort_index()

sport_ids

event_id = '26420387'

# Define a market filter
market_types_filter = betfairlightweight.filters.market_filter(event_type_ids=event_id)

# Request market types
market_types = trading.betting.list_market_types(
        filter=market_types_filter
)

te = trading.betting.list_market_catalogue(
    max_results=100,
    market_projection=["EVENT"])[0]

ufc_markets = trading.betting.list_market_catalogue(
    filter=market_filter(
        event_type_ids=[mma_event_type.event_type.id]
    ),
    max_results=100,
    market_projection=["EVENT", "COMPETITION", "MARKET_DESCRIPTION"]
)

te = ufc_markets[13]

print(te.description)
# Create a DataFrame of market types
market_types_mooney_valley = pd.DataFrame({
    'Market Type': [market_type_object.market_type for market_type_object in market_types],
})

market_types_mooney_valley




############################################
## HISTORIC DATA DOWNLOAD

# get file list
file_list = trading.historic.get_file_list(
    "Other Sports",
    "Basic Plan",
    from_day=1,
    from_month=4,
    from_year=2025,
    to_day=20,
    to_month=4,
    to_year=2025,
    market_types_collection=["MATCH_ODDS"],
    countries_collection=["GB"],
    file_type_collection=["M"],
)

# download the files
for file in file_list:
    print(file)
    download = trading.historic.download_file(file_path = file, store_directory = './data/raw_betfair/')
    print(download)



############################################
## PROCESS DOWNLOADED FILES


bz2_file_log = betfair_filter_bz2_for_fights('./data/raw_betfair/historic/', './data/raw_betfair/decompressed_fight_files/')



############################################
## STREAMING HISTORIC DATA DOWNLOADS

## DONT NEED TO DO THIS NOW. IT'S DONE ABOVE
# Decompress all the files downloaded from Betfair. Run in CMD
# for /r "E:\Repos\mma_model_lite\data\raw_betfair\" %f in (*.bz2) do "C:\Program Files\7-Zip\7z.exe" x "%f" -o"E:\Repos\mma_model_lite\data\raw_betfair\decompressed_files" -y


# create listener
listener = StreamListener(max_latency=None)

## Go through each file in the decompressed folder and load the data into data frames

# Create an empty list for the datasets
betfair_odds_list = []

# Loop through files
for file in os.listdir(output_dir):

    # Create historical stream (update file_path to your file location)
    stream = trading.streaming.create_historical_generator_stream(
        file_path = output_dir + file,
        listener = listener,
    )

    # Convert the list of series into a data frame
    betfair_odds_df = betfair_stream_to_df(stream, historic = True)
    
    # Get the last odds before the chosen cutoff time
    betfair_odds_summary_df = betfair_bout_get_chosen_odds(betfair_odds_df, betfair_data_config['odds_cutoff_time'])

    # Save the output to the list as a dictionary for faster combination later
    betfair_odds_list.append(betfair_odds_summary_df)

betfair_odds = pd.concat(betfair_odds_list, axis = 0)
