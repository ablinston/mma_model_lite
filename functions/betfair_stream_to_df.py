# This function takes a Betfair stream and saves the data to a data frame

def betfair_stream_to_df(stream = None, stream_snapshot = None, inplay = False, include_prop_bets = False, market_catalogues = None, upcoming_min_best_offer = 0, upcoming_odds_to_use = 'last_price_traded'):
    
    # pdb.set_trace()
    
    # Initial checks
    if stream is None and stream_snapshot is None:
        print("Need a valid input stream")
        return None
    
    if stream_snapshot is not None and market_catalogues is None:
        print("stream_snapshot input needs a market catalogue for runner and market names")
        return None
    
    # Process the market catalogues if present
    if market_catalogues is not None:
        runners_dict = {k: v for d in map(betfair_get_runner_info_from_marketcatalogue, market_catalogues) for k, v in d.items()}
    
    if stream is not None:
        # create generator
        gen = stream.get_generator()
        all_market_books = gen()
    else:
        all_market_books = [stream_snapshot]
    
    # Create an empty odds object to be populated with the odds info
    betfair_odds_list = []
    
    # Create a file for each marketbook
    for market_books in all_market_books:
        for market_book in market_books:
                        
            # Get a list of the runners names and put them into a dictionary if we don't have the market catalogue
            if market_catalogues is None:
                runners_dict = {}
                for runner in market_book.market_definition.runners: # Runner names are contained in the marketbook
                    runners_dict = runners_dict | {runner.selection_id: runner.name}
            
            # Now loop through the runners and get the odds data
            for runner in market_book.runners:
                
                # pdb.set_trace()
                
                # Get the odds to use
                if upcoming_odds_to_use == 'best_offer':
                    
                    # Use the minium to filter out offers
                    offers = [item for item in runner.ex.available_to_back if item.get('size', 0) > upcoming_min_best_offer]
                    
                    # Best offer is the first in the list
                    best_odds = offers[0]['price']
                    
                else:
                    best_odds = runner.last_price_traded
                
                # Add a dictionary to the odds list for this runner
                betfair_odds_list.append({
                    'GMT_datetime': market_book.publish_time, # is technically in UTC but that is the same as GMT in 24H clock
                    'market_id': market_book.market_id,
                    'bout': market_book.market_definition.event_name,
                    'status': market_book.status,
                    'inplay': market_book.inplay,
                    'runner': runners_dict[runner.selection_id],
                    'last_price_traded': best_odds
                })
    
    # Get the date of the event
    output_df = pd.DataFrame(betfair_odds_list)
    
    # Get rid of NA bets
    output_df = output_df[~output_df['last_price_traded'].isna()]
    
    # # Remove any bets after the cutoff time ## THE LIVE STREAMS ONLY GIVE ONE SET OF ODDS PER FIGHT
    # if cutoff_time is not None:
    #     cutoff_date = datetime.strptime(str(output_df['GMT_datetime'].dt.date.max()) + "-" + cutoff_time, "%Y-%m-%d-%H:%M")
    #     output_df = output_df[output_df['GMT_datetime'] <= cutoff_date]
        
    # pdb.set_trace()
    
    # If no prop bets should be included, filter them out by looking for the fighters only in the runner column
    if not include_prop_bets:
        
        try:
            # Exclude common strings from prop bets
            exclusions = output_df['runner'].apply(lambda x: 1 if 'under' in x.lower() or 'over' in x.lower() or 'points' in x.lower() or '/' in x or ':' in x or ' round ' in x.lower() or ' by ' in x.lower() or 'draw ' in x.lower() or x.lower() == 'draw' else 0)
            output_df = output_df[exclusions == 0]
            
            # If this isn't a live stream, each marketbook will have only 2 fighters, so take the two most common
            if stream_snapshot is None:
                # Get the two most common bets. Assume these are the fighters
                two_modes = output_df['runner'].value_counts().head(2).index.tolist()
                
                # Filter only the two most common bets
                output_df = output_df[output_df['runner'].isin(two_modes)]
                
            output_df.rename(columns = {'runner': 'fighter'}, inplace = True)
        except:
            pdb.set_trace()
    
    # Filter out in play
    if not inplay:
        output_df = output_df[(output_df['inplay'] == False) & (output_df['status'] == "OPEN")]
            
    # Convert the list of series into a data frame
    return output_df
