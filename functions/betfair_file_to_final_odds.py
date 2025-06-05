def betfair_file_to_final_odds(file_path, odds_cutoff_time, average_last_n):
    
    """
    This function takes a path to a betfair file that has been extracted from bz2 and turns it into a dataframe
    with chosen odds given a set cutoff date
    
    """
    # pdb.set_trace()
    
    # create listener
    listener = StreamListener(max_latency=None)
    
    # Create historical stream (update file_path to your file location)
    stream = trading.streaming.create_historical_generator_stream(
        file_path = file_path,
        listener = listener,
    )

    # Convert the list of series into a data frame
    betfair_odds_df = betfair_stream_to_df(stream = stream)
    
    if len(betfair_odds_df) > 0:
        # Get the last odds before the chosen cutoff time
        betfair_odds_summary_df = betfair_bout_get_chosen_odds(betfair_odds_df, odds_cutoff_time, average_last_n)
    
        return betfair_odds_summary_df
    else:
        return pd.DataFrame()
            