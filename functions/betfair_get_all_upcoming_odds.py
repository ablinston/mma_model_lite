# This function takes a Betfair stream and saves the data to a data frame

def betfair_get_all_upcoming_odds(betfair_login_config, betfair_data_config):
    
    """
    This function will get all the odds from upcoming MMA fights from betfair
    
    
    """
    
    # pdb.set_trace()
    
    
    ############################################
    ## Betfair standard setup
    
    # Open the connection with Betfair. This requires the betfair config which is not tracked by git because it contains passwords
    trading = betfairlightweight.APIClient(betfair_login_config['username'], betfair_login_config['password'], app_key = betfair_login_config['delay_feed_app_key'], certs = betfair_login_config['certs_location'])
    
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
    
    # Attempt to get the catalogue
    for attempt in range(10):
        try:
            market_catalogues = trading.betting.list_market_catalogue(
                filter = market_catalogue_filter,
                market_projection=["RUNNER_DESCRIPTION", "MARKET_DESCRIPTION", "EVENT"],
                max_results=100
            )
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Attempt {attempt + 1} for getting catalogue failed. Retrying.")
            time.sleep(5)
    
    
    
    ############################################
    ## STREAMING MMA
    
    # create filters
    market_filter = streaming_market_filter(
        event_type_ids=[betfair_data_config['betfair_mma_event_id']], market_types=["MATCH_ODDS"]
    )
    market_data_filter = streaming_market_data_filter(
        fields=["EX_LTP", "EX_ALL_OFFERS", "EX_MARKET_DEF", "EX_BEST_OFFERS_DISP"], ladder_levels = 3 # Last traded price and market definitions
    )
    
    # Try to get the stream
    for attempt in range(10):
        
        try:
            # subscribe
            streaming_unique_id = stream.subscribe_to_markets(
                market_filter=market_filter,
                market_data_filter=market_data_filter,
                conflate_ms=10000,  # send update every 1000ms
            )
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Attempt {attempt + 1} for getting odds stream failed. Retrying.")
            time.sleep(5)
    
    # start stream in a new thread (in production would need err handling)
    t = threading.Thread(target=stream.start, daemon=True)
    t.start()
    
    # Give a short delay for the connection
    time.sleep(2)
    
    # Try to get the snapshot
    for attempt in range(20):
        
        try:
            # Take a snapshot of the market right now
            market_books = listener.snap()
            
            # Check we retreived the market books
            if len(market_books) > 0:
                break
            else:
                error()
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Attempt {attempt + 1} for getting snapshot failed. Retrying.")
            time.sleep(5)
            
    # pdb.set_trace()
    # Stop the stream now we have the snapshot
    stream.stop()
    
    # Save the results to a df
    betfair_latest_future_odds = betfair_stream_to_df(
        stream_snapshot = market_books, 
        market_catalogues = market_catalogues, 
        upcoming_min_best_offer = betfair_data_config['upcoming_min_best_offer'], 
        upcoming_odds_to_use = betfair_data_config['upcoming_odds_to_use'])
    
    # fp.write('./data/' + config['betfair_upcoming_odds_file_name'],
    #          betfair_latest_future_odds,
    #          compression = 'ZSTD')
  

    # Return the dataframe of odds
    return betfair_latest_future_odds
