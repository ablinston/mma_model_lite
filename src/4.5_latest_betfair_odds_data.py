import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.filters import market_filter, price_projection, streaming_market_filter, streaming_market_data_filter
import os
import logging
import queue
import threading

import betfairlightweight
from betfairlightweight.filters import (
    streaming_market_filter,
    streaming_market_data_filter,
)

exec(open('global.py').read())




############################################
## Betfair standard setup

# Open the connection with Betfair. This requires the betfair config which is not tracked by git because it contains passwords
trading = betfairlightweight.APIClient(betfair_login_config['username'], betfair_login_config['password'], app_key = betfair_login_config['delay_feed_app_key'], certs = './data/betfair_certs')

trading.login()

# create queue
output_queue = queue.Queue()

# create stream listener
listener = betfairlightweight.StreamListener(output_queue=output_queue)

# create stream
stream = trading.streaming.create_stream(listener=listener)


############################################
## Get market catalogue to get runner names and event names

# Build your market filter
market_catalogue_filter = {
    "eventTypeIds": [betfair_data_config['betfair_mma_event_id']],
    "marketTypeCodes": ["MATCH_ODDS"]
}

market_catalogues = trading.betting.list_market_catalogue(
    filter = market_catalogue_filter,
    market_projection=["RUNNER_DESCRIPTION", "MARKET_DESCRIPTION", "EVENT"],
    max_results=100
)



############################################
## STREAMING MMA

market_filter = streaming_market_filter(
    event_type_ids=["7"], country_codes=["GB"], market_types=["WIN"]
)
market_data_filter = streaming_market_data_filter(
    fields=["EX_BEST_OFFERS", "EX_MARKET_DEF"], ladder_levels=3
)


# create filters (GB WIN racing)
market_filter = streaming_market_filter(
    event_type_ids=[betfair_data_config['betfair_mma_event_id']], market_types=["MATCH_ODDS"]
)
market_data_filter = streaming_market_data_filter(
    fields=["EX_LTP", "EX_ALL_OFFERS", "EX_MARKET_DEF"], ladder_levels = 3 # Last traded price and market definitions
)

# subscribe
streaming_unique_id = stream.subscribe_to_markets(
    market_filter=market_filter,
    market_data_filter=market_data_filter,
    conflate_ms=10000,  # send update every 1000ms
)

# start stream in a new thread (in production would need err handling)
t = threading.Thread(target=stream.start, daemon=True)
t.start()

# Take a snapshot of the market right now
market_books = listener.snap()

# Save the results to a df
betfair_latest_future_odds = betfair_stream_to_df(stream_snapshot = market_books, market_catalogues = market_catalogues)



